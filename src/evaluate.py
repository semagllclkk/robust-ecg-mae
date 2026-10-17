"""
Test seti (strat_fold 10) değerlendirmesi: 0–4 kanal kopması altında MI/Normal başarımı ve
kopan kanalların rekonstrüksiyon hatası + örnek bir rekonstrüksiyon grafiği.

Çalıştırma (proje kök klasöründen):
    python -m src.evaluate
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import PTBXLProcessedDataset
from src.losses import MultiTaskLoss
from src.model import random_lead_mask
from src.prepare_data import LEAD_NAMES
from src.train import CHECKPOINT_PATH, evaluate, get_device, load_checkpoint


def plot_reconstruction(model, dataset, device, n_drop, out_path, idx=0, seed=0):
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)
    lead_mask = random_lead_mask(1, 12, n_drop=n_drop, generator=torch.Generator().manual_seed(seed)).to(device)
    model.eval()
    with torch.no_grad():
        recon, logits, _ = model(x, lead_mask=lead_mask)
    x, recon, dropped = x[0].cpu().numpy(), recon[0].cpu().numpy(), lead_mask[0].cpu().numpy()
    prob = torch.sigmoid(logits).item()

    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    for c, ax in enumerate(axes):
        ax.plot(x[c], color="black", linewidth=0.8, label="Orijinal")
        if dropped[c]:
            ax.plot(recon[c], color="crimson", linewidth=0.8, label="Rekonstrüksiyon")
            ax.set_facecolor("#fdecec")
        ax.set_ylabel(LEAD_NAMES[c] + (" ✗" if dropped[c] else ""), rotation=0, labelpad=20)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Örnek (100 Hz)")
    fig.suptitle(f"ecg_id {dataset.ecg_ids[idx]}  |  gerçek: {'MI' if y.item() == 1 else 'Normal'}  |  "
                 f"P(MI) = {prob:.2f}  |  kopuk kanal: {n_drop}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test seti sağlamlık değerlendirmesi")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--plot", default="data/raw/reconstruction_test.png")
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Model yüklendi: {args.checkpoint} (epoch {ckpt['epoch']}, val AUROC {ckpt['val_metrics']['auroc']:.4f})")

    test_set = PTBXLProcessedDataset("test", args.processed_dir)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    criterion = MultiTaskLoss(gamma=ckpt["focal_gamma"], alpha=ckpt["focal_alpha"])
    print(f"Test: {len(test_set)} kayıt (MI oranı {test_set.positive_fraction():.3f})\n")

    print(f"{'kopuk kanal':>11} | {'AUROC':>6} {'F1':>6} {'Acc':>6} {'Duyarl.':>7} {'Özgül.':>7} | {'MSE (kopuk)':>11}")
    for n_drop in range(0, model.lead_dropout.max_drop + 1):
        r = evaluate(model, loader, criterion, device, n_drop=n_drop, seed=123)
        mse = f"{r['rec']:.4f}" if n_drop > 0 else "-"
        print(f"{n_drop:>11} | {r['auroc']:.4f} {r['f1']:.4f} {r['acc']:.4f} {r['sens']:>7.4f} {r['spec']:>7.4f} | {mse:>11}")

    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    plot_reconstruction(model, test_set, device, n_drop=model.lead_dropout.max_drop, out_path=args.plot)
    print(f"\nRekonstrüksiyon grafiği kaydedildi: {args.plot}")


if __name__ == "__main__":
    main()
