"""
Kanal bazlı MAE + MI sınıflandırıcı eğitimi.

Önce önbellek üretilmeli:  python -m src.prepare_data
Çalıştırma (proje kök klasöründen):
    python -m src.train
    python -m src.train --epochs 2 --limit-batches 20     # hızlı deneme
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import PTBXLProcessedDataset
from src.losses import MultiTaskLoss
from src.model import ECGMaskedAutoencoder, random_lead_mask

CHECKPOINT_PATH = "models/mae_ecg_checkpoint.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def classification_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return {
        "auroc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "acc": float((preds == labels).mean()),
        "sens": tp / max(int((labels == 1).sum()), 1),
        "spec": tn / max(int((labels == 0).sum()), 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_drop=None, seed=0, max_batches=None):
    """
    Sabit tohumlu kanal kopması altında değerlendirme.
    n_drop=None: örnek başına U{0..max_drop} kanal (eğitim dağılımı); int: her örnekte tam o kadar kanal.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    max_drop = model.lead_dropout.max_drop
    totals = {"loss": 0.0, "rec": 0.0, "cls": 0.0}
    all_probs, all_labels = [], []
    n_batches = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        lead_mask = random_lead_mask(x.shape[0], x.shape[1], max_drop, n_drop, gen).to(device)
        recon, logits, lead_mask = model(x, lead_mask=lead_mask)
        loss, parts = criterion(recon, logits, x, y, lead_mask)
        totals["loss"] += loss.item()
        totals["rec"] += parts["rec"]
        totals["cls"] += parts["cls"]
        all_probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        n_batches += 1
    results = {k: v / max(n_batches, 1) for k, v in totals.items()}
    results.update(classification_metrics(np.concatenate(all_labels), np.concatenate(all_probs)))
    return results


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ECGMaskedAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Kanal bazlı MAE + MI/Normal sınıflandırıcı eğitimi")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-drop", type=int, default=4, help="Eğitimde örnek başına en fazla düşürülecek kanal")
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal Loss odak parametresi")
    parser.add_argument("--alpha", default="auto",
                        help="Focal Loss MI sınıf ağırlığı: 'auto' = 1 - eğitimdeki MI oranı, 'none' = ağırlıksız, ya da sayı")
    parser.add_argument("--patience", type=int, default=7, help="Doğrulama AUROC iyileşmezse erken durdurma")
    parser.add_argument("--limit-batches", type=int, default=None, help="Epoch başına batch sınırı (deneme için)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Eğitim cihazı: {device}")

    train_set = PTBXLProcessedDataset("train", args.processed_dir)
    val_set = PTBXLProcessedDataset("val", args.processed_dir)
    pos_frac = train_set.positive_fraction()
    print(f"Eğitim: {len(train_set)} kayıt (MI oranı {pos_frac:.3f})   Doğrulama: {len(val_set)} kayıt")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    if args.alpha == "auto":
        alpha = 1.0 - pos_frac
    elif args.alpha == "none":
        alpha = None
    else:
        alpha = float(args.alpha)
    print(f"Focal Loss: gamma={args.gamma}, alpha={alpha if alpha is None else round(alpha, 3)}")

    config = dict(in_channels=12, seq_len=1000, patch_size=50, embed_dim=128, depth=4,
                  decoder_depth=2, num_heads=4, num_classes=1, max_drop=args.max_drop)
    model = ECGMaskedAutoencoder(**config).to(device)
    criterion = MultiTaskLoss(args.lambda_rec, args.lambda_cls, args.gamma, alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    history_path = os.path.splitext(args.checkpoint)[0] + "_history.csv"
    history = []
    best_auroc, bad_epochs = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "rec": 0.0, "cls": 0.0}
        n_batches = 0
        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches is not None and i >= args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            recon, logits, lead_mask = model(x)  # LeadDropout eğitim modunda maskeyi üretir
            loss, parts = criterion(recon, logits, x, y, lead_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sums["loss"] += loss.item()
            sums["rec"] += parts["rec"]
            sums["cls"] += parts["cls"]
            n_batches += 1
        scheduler.step()
        train_stats = {k: v / max(n_batches, 1) for k, v in sums.items()}

        val = evaluate(model, val_loader, criterion, device, n_drop=None, seed=0, max_batches=args.limit_batches)
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train loss {train_stats['loss']:.4f} (rec {train_stats['rec']:.4f}, cls {train_stats['cls']:.4f})  |  "
              f"val loss {val['loss']:.4f} rec {val['rec']:.4f} AUROC {val['auroc']:.4f} F1 {val['f1']:.4f}  "
              f"({time.time() - t0:.0f} sn)")

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        if val["auroc"] > best_auroc:
            best_auroc, bad_epochs = val["auroc"], 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "val_metrics": val, "focal_alpha": alpha, "focal_gamma": args.gamma}, args.checkpoint)
            print(f"  ✓ En iyi model kaydedildi (val AUROC {best_auroc:.4f}) -> {args.checkpoint}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Erken durdurma: {args.patience} epoch boyunca doğrulama AUROC iyileşmedi.")
                break

    print(f"Eğitim bitti. En iyi val AUROC: {best_auroc:.4f}. Test için: python -m src.evaluate")


if __name__ == "__main__":
    main()
