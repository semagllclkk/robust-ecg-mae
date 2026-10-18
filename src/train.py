"""
Çift başlıklı MAE (veya karşılaştırma için 1D-ResNet) eğitimi — TÜBİTAK Raporu İP3.

Önce önbellek üretilmeli:  python -m src.prepare_data
Çalıştırma (proje kök klasöründen):
    python -m src.train                                   # MAE  -> models/mae_ecg_checkpoint.pth
    python -m src.train --arch resnet1d                   # 1D-ResNet temel model -> models/resnet1d_baseline.pth
    python -m src.train --epochs 1 --limit-batches 20     # hızlı deneme
"""
import argparse
import csv
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PTBXLProcessedDataset, population_sigma
from src.einthoven import einthoven_complete
from src.lead_dropout import LeadDropoutCollate, apply_lead_mask, full_keep_mask, random_keep_mask
from src.losses import MultiTaskLoss
from src.metrics import ClassificationMeter, ReconstructionMeter, best_f1_threshold
from src.model import build_model

CHECKPOINT_PATH = "models/mae_ecg_checkpoint.pth"
BASELINE_PATH = "models/resnet1d_baseline.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast(device, enabled):
    if enabled and device.type in ("mps", "cuda"):
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def _float_outputs(out):
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}


def model_forward(model, x_masked, M, target):
    """Genlik bilgisi alan modellere sağlam derivasyonların σ'sı da verilir (kopanlarınki modelde maskelenir)."""
    if getattr(model, "amplitude_aware", False):
        return model(x_masked, M, sigma=target["sigma"])
    return model(x_masked, M)


def recon_target(model, x, target):
    """
    Rekonstrüksiyon hedefi: z-score modelde z-score sinyal; genlik farkındalıklı modelde fiziksel ölçek
    (mV / σ_pop). Kayıp her iki durumda da yalnızca kopan kanallarda MSE'dir (Rapor Md. 2.4).
    """
    if getattr(model, "physical_recon", False):
        mv = x * target["sigma"].unsqueeze(-1) + target["mu"].unsqueeze(-1)
        return mv / model.pop_sigma.view(1, -1, 1)
    return x


def recon_to_mv(recon, pop_sigma):
    """
    Gerçekçi mV rekonstrüksiyonu: kopan derivasyonun gerçek genliği bilinemez, popülasyon σ'sı kullanılır
    (0.5–45 Hz filtre sonrası ortalama ≈ 0). Hem z-score çıkışlı hem fiziksel ölçekli modeller için geçerli.
    """
    return recon * pop_sigma.view(1, -1, 1)


def mixed_dropout_mask(batch_size, generator):
    """Doğrulama senaryosu: eğitim dağılımıyla aynı (k ∈ [1,4], %30 kritik kopma), sabit tohumlu."""
    return random_keep_mask(batch_size, critical_prob=0.3, generator=generator)


@torch.no_grad()
def evaluate(model, loader, criterion, device, mask_fn, seed=0, threshold=0.5, amp=True, max_batches=None,
             pop_sigma=None):
    """
    mask_fn(batch_size, generator) -> (B, 12) M maskesi (1 = sağlam, 0 = koptu). None ise hiçbir kanal kopmaz.
    Döndürür: kayıp bileşenleri + teşhis metrikleri + rekonstrüksiyon metrikleri (+ ham olasılıklar)
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    totals = {"loss": 0.0, "rec": 0.0, "cls": 0.0, "wall": 0.0}
    cls_meter = ClassificationMeter()
    rec_meter, ein_meter, oracle_meter = ReconstructionMeter(), ReconstructionMeter(), ReconstructionMeter()
    pop_sigma = torch.as_tensor(pop_sigma if pop_sigma is not None else population_sigma(), device=device)
    physical = getattr(model, "physical_recon", False)
    n_batches = 0
    for i, (x, target) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        B = x.shape[0]
        M = (mask_fn(B, gen) if mask_fn else full_keep_mask(B)).to(device)
        x = x.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        with autocast(device, amp):
            out = model_forward(model, apply_lead_mask(x, M), M, target)
        out = _float_outputs(out)
        loss, parts = criterion(out, recon_target(model, x, target), target)
        totals["loss"] += loss.item()
        for k in ("rec", "cls", "wall"):
            totals[k] += parts[k]
        cls_meter.update(out, target)
        if out["recon"] is not None:
            sigma, mu = target["sigma"].unsqueeze(-1), target["mu"].unsqueeze(-1)
            true_mv = x * sigma + mu
            dropped, valid = M < 0.5, target["sigma"] > 1e-6
            real_mv = recon_to_mv(out["recon"], pop_sigma)
            rec_meter.update(real_mv, true_mv, dropped, valid)
            # Risk 1 B planı: kopan ekstremite derivasyonları sağlam olanlardan Einthoven ile (kopanların içeriği kullanılmaz)
            ein_mv = real_mv.clone()
            ein_mv[:, :6] = torch.where(M[:, :6].unsqueeze(-1) > 0.5, real_mv[:, :6], einthoven_complete(true_mv, M))
            ein_meter.update(ein_mv, true_mv, dropped, valid)
            if not physical:  # yalnızca karşılaştırma: kopan kanalın gerçek genliği biliniyormuş gibi (oracle)
                oracle_meter.update(out["recon"] * sigma + mu, true_mv, dropped, valid)
        n_batches += 1
    results = {k: v / max(n_batches, 1) for k, v in totals.items()}
    results.update(cls_meter.result(threshold))
    results.update(rec_meter.result())
    results.update({f"ein_{k}": v for k, v in ein_meter.result().items()})
    results.update({f"oracle_{k}": v for k, v in oracle_meter.result().items()})
    results["_labels"], results["_probs"] = cls_meter.arrays()
    return results


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("arch", "mae"), ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Çift başlıklı MAE / 1D-ResNet eğitimi")
    parser.add_argument("--arch", choices=["mae", "resnet1d"], default="mae")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--min-drop", type=int, default=1, help="LeadDropout: batch başına en az kopan kanal (Rapor: 1)")
    parser.add_argument("--max-drop", type=int, default=4, help="LeadDropout: batch başına en fazla kopan kanal (Rapor: 4)")
    parser.add_argument("--critical-prob", type=float, default=0.3, help="Kopmanın tek bir kalp duvarına odaklanma olasılığı")
    parser.add_argument("--alpha-mse", type=float, default=0.4, help="Rapor Md. 2.4: α")
    parser.add_argument("--beta-focal", type=float, default=0.6, help="Rapor Md. 2.4: β")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal Loss odak parametresi")
    parser.add_argument("--focal-alpha", default="auto",
                        help="Focal Loss MI sınıf ağırlığı: 'auto' = 1 - eğitimdeki MI oranı, 'none' = ağırlıksız, ya da sayı")
    parser.add_argument("--patience", type=int, default=5, help="Doğrulama AUROC iyileşmezse erken durdurma")
    parser.add_argument("--no-amp", action="store_true", help="bfloat16 karışık hassasiyeti kapat")
    parser.add_argument("--limit-batches", type=int, default=None, help="Epoch başına batch sınırı (deneme için)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--amplitude-aware", action="store_true",
                        help="Sağlam kanalların genliğini modele ver, rekonstrüksiyonu fiziksel ölçekte üret")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or (CHECKPOINT_PATH if args.arch == "mae" else BASELINE_PATH)
    amp = not args.no_amp
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Eğitim cihazı: {device}   mimari: {args.arch}   bfloat16: {amp}")

    train_set = PTBXLProcessedDataset("train", args.processed_dir)
    val_set = PTBXLProcessedDataset("val", args.processed_dir)
    pos_frac = train_set.positive_fraction()
    print(f"Dengeli alt kümeler — eğitim: {len(train_set)} kayıt (MI oranı {pos_frac:.3f})   "
          f"doğrulama: {len(val_set)} kayıt")

    if args.arch == "mae":
        # LeadDropout DataLoader seviyesinde (Rapor Md. 2.2): her batch için k ∈ [min_drop, max_drop]
        collate = LeadDropoutCollate(args.min_drop, args.max_drop, args.critical_prob)
        print(f"LeadDropout: batch başına k ∈ [{args.min_drop}, {args.max_drop}], kritik kopma olasılığı {args.critical_prob}")
    else:
        collate = None  # klasik mimari: kopma simülasyonu olmadan eğitilir
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    if args.focal_alpha == "auto":
        focal_alpha = 1.0 - pos_frac
    elif args.focal_alpha == "none":
        focal_alpha = None
    else:
        focal_alpha = float(args.focal_alpha)

    if args.arch == "mae":
        config = dict(in_channels=12, seq_len=1000, embed_dim=128, depth=2, num_heads=4)
        if args.amplitude_aware:
            config.update(amplitude_aware=True, pop_sigma=population_sigma(args.processed_dir).tolist())
        alpha_mse = args.alpha_mse
    else:
        config = dict(in_channels=12)
        alpha_mse = 0.0  # 1D-ResNet rekonstrüksiyon yapmaz
    model = build_model(args.arch, config).to(device)
    criterion = MultiTaskLoss(alpha_mse, args.beta_focal, args.gamma, focal_alpha)
    print(f"Kayıp: {alpha_mse}·MSE(maskeli) + {args.beta_focal}·Focal(γ={args.gamma}, "
          f"α={focal_alpha if focal_alpha is None else round(focal_alpha, 3)})")
    print(f"Parametre sayısı: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_loader), args.limit_batches or len(train_loader))
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    history_path = os.path.splitext(checkpoint_path)[0] + "_history.csv"
    history = []
    best_auroc, bad_epochs = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "rec": 0.0, "cls": 0.0, "wall": 0.0}
        n_batches = 0
        for i, batch in enumerate(train_loader):
            if args.limit_batches is not None and i >= args.limit_batches:
                break
            if collate is not None:
                x_masked, x, M, target = batch
            else:
                x, target = batch
                M = full_keep_mask(x.shape[0])
                x_masked = x
            x_masked, x, M = x_masked.to(device), x.to(device), M.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            with autocast(device, amp):
                out = model_forward(model, x_masked, M, target)
            loss, parts = criterion(_float_outputs(out), recon_target(model, x, target), target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sums["loss"] += loss.item()
            for k in ("rec", "cls", "wall"):
                sums[k] += parts[k]
            n_batches += 1
            if (i + 1) % 50 == 0:
                print(f"  adım {i + 1}/{steps_per_epoch}  loss {sums['loss'] / n_batches:.4f} "
                      f"rec {sums['rec'] / n_batches:.4f}  ({time.time() - t0:.0f} sn)", flush=True)
        train_stats = {k: v / max(n_batches, 1) for k, v in sums.items()}

        val = evaluate(model, val_loader, criterion, device, mixed_dropout_mask, seed=0, amp=amp,
                       max_batches=args.limit_batches)
        print(f"Epoch {epoch:02d}/{args.epochs}  train loss {train_stats['loss']:.4f} "
              f"(rec {train_stats['rec']:.4f}, cls {train_stats['cls']:.4f}, duvar {train_stats['wall']:.4f})  |  "
              f"val AUROC {val['auroc']:.4f} F1 {val['f1']:.4f} duvar-AUROC {val['wall_auroc_macro']:.4f}  "
              f"RMSE {val['rmse_mv']:.4f} mV R² {val['r2']:.4f} r {val['pearson_r']:.4f}  ({time.time() - t0:.0f} sn)",
              flush=True)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
               **{f"val_{k}": v for k, v in val.items() if not k.startswith("_")}}
        history.append(row)
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        if val["auroc"] > best_auroc:
            best_auroc, bad_epochs = val["auroc"], 0
            threshold = best_f1_threshold(val["_labels"], val["_probs"])
            torch.save({"arch": args.arch, "model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "val_metrics": {k: v for k, v in val.items() if not k.startswith("_")},
                        "threshold": threshold, "focal_alpha": focal_alpha, "focal_gamma": args.gamma,
                        "alpha_mse": alpha_mse, "beta_focal": args.beta_focal,
                        "lead_dropout": {"min_drop": args.min_drop, "max_drop": args.max_drop,
                                         "critical_prob": args.critical_prob}}, checkpoint_path)
            print(f"  ✓ En iyi model kaydedildi (val AUROC {best_auroc:.4f}, F1-eşiği {threshold:.2f}) -> {checkpoint_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Erken durdurma: {args.patience} epoch boyunca doğrulama AUROC iyileşmedi.")
                break

    print(f"Eğitim bitti. En iyi val AUROC: {best_auroc:.4f}. Test için: python -m src.evaluate")


if __name__ == "__main__":
    main()
