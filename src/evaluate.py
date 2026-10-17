"""
Test seti (strat_fold 10) stres testleri — TÜBİTAK Raporu Hedef 4, Hipotez ve İP6.

Senaryolar: kopma yok; rastgele 1/2/3/4 kanal kopması; kritik duvar kopmaları
(Anterior V1–V4, Inferior II/III/aVF, Lateral I/aVL/V5/V6).
Karar eşiği doğrulama setinde belirlenip checkpoint'e yazılmıştır (test sızıntısı yok).

Çalıştırma (proje kök klasöründen):
    python -m src.evaluate
    python -m src.evaluate --baseline models/resnet1d_baseline.pth     # 1D-ResNet ile FNR karşılaştırması
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PTBXLProcessedDataset
from src.ecg_constants import LEAD_NAMES, WALL_GROUPS, WALL_NAMES
from src.losses import MultiTaskLoss
from src.model import random_lead_mask, wall_lead_mask
from src.train import BASELINE_PATH, CHECKPOINT_PATH, evaluate, get_device, load_checkpoint

TARGET_F1 = 0.88
TARGET_RMSE = 0.15
TARGET_R2 = 0.80
TARGET_FNR_REDUCTION = 0.25


def scenarios():
    s = {"0 kopuk (tam 12 kanal)": None}
    for k in range(1, 5):
        s[f"rastgele {k} kanal"] = (lambda B, g, k=k: random_lead_mask(B, n_drop=k, generator=g))
    for wall in WALL_NAMES:
        s[f"kritik {wall} ({', '.join(WALL_GROUPS[wall])})"] = (lambda B, g, w=wall: wall_lead_mask(B, w))
    return s


def run_all(model, ckpt, loader, device):
    criterion = MultiTaskLoss(ckpt.get("alpha_mse", 0.4), ckpt.get("beta_focal", 0.6),
                              ckpt["focal_gamma"], ckpt["focal_alpha"])
    results = {}
    for name, mask_fn in scenarios().items():
        r = evaluate(model, loader, criterion, device, mask_fn, seed=123, threshold=ckpt["threshold"])
        results[name] = {k: v for k, v in r.items() if not k.startswith("_")}
    return results


def print_table(title, results):
    print(f"\n=== {title} ===")
    print(f"{'senaryo':38s} | {'AUROC':>6} {'F1':>6} {'Duyar.':>6} {'Özgül.':>6} {'FNR':>6} {'DuvarAUC':>8} | "
          f"{'RMSE(z)':>7} {'RMSE(mV)':>8} {'R²':>6} {'r':>6}")
    for name, r in results.items():
        rec = (f"{r['rmse_z']:7.4f} {r['rmse_mv']:8.4f} {r['r2']:6.3f} {r['pearson_r']:6.3f}"
               if not np.isnan(r["rmse_z"]) else f"{'-':>7} {'-':>8} {'-':>6} {'-':>6}")
        print(f"{name:38s} | {r['auroc']:6.4f} {r['f1']:6.4f} {r['sens']:6.4f} {r['spec']:6.4f} {r['fnr']:6.4f} "
              f"{r['wall_auroc_macro']:8.4f} | {rec}")


def check_targets(results, baseline=None):
    print("\n=== Rapor hedefleri (1–4 kanal kopması) ===")
    drop = {k: v for k, v in results.items() if not k.startswith("0 kopuk")}
    min_f1 = min(r["f1"] for r in drop.values())
    max_rmse_z = max(r["rmse_z"] for r in drop.values())
    max_rmse_mv = max(r["rmse_mv"] for r in drop.values())
    min_r2 = min(r["r2"] for r in drop.values())
    ok = lambda b: "✓" if b else "✗"
    print(f"{ok(min_f1 > TARGET_F1)} En düşük F1 = {min_f1:.4f}  (hedef > {TARGET_F1})")
    print(f"{ok(max_rmse_mv < TARGET_RMSE)} En yüksek RMSE = {max_rmse_mv:.4f} mV  (hedef < {TARGET_RMSE}; "
          f"z-score biriminde {max_rmse_z:.4f})")
    print(f"{ok(min_r2 > TARGET_R2)} En düşük R² = {min_r2:.4f}  (hedef > {TARGET_R2})")
    if baseline is not None:
        reductions = []
        for name in drop:
            fb, fm = baseline[name]["fnr"], results[name]["fnr"]
            reductions.append((fb - fm) / fb if fb > 0 else 0.0)
        mean_red = float(np.mean(reductions))
        print(f"{ok(mean_red >= TARGET_FNR_REDUCTION)} 1D-ResNet'e göre ortalama yanlış negatif azalması = "
              f"%{100 * mean_red:.1f}  (hipotez ≥ %{100 * TARGET_FNR_REDUCTION:.0f})")
        return mean_red
    return None


def plot_reconstruction(model, dataset, device, out_path, wall="Anterior"):
    """Bir MI test kaydında belirli bir duvarın tüm derivasyonları koptuğunda rekonstrüksiyon (mV)."""
    idx = int(np.flatnonzero(dataset.labels == 1)[0])
    x, target = dataset[idx]
    lead_mask = wall_lead_mask(1, wall).to(device)
    with torch.no_grad():
        out = model(x.unsqueeze(0).to(device), lead_mask=lead_mask)
    mu, sigma = target["mu"].numpy()[:, None], target["sigma"].numpy()[:, None]
    orig = x.numpy() * sigma + mu
    recon = out["recon"][0].float().cpu().numpy() * sigma + mu
    dropped = lead_mask[0].cpu().numpy()
    p_mi = torch.sigmoid(out["mi_logit"]).item()

    t = np.arange(orig.shape[1]) / 100
    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    for c, ax in enumerate(axes):
        ax.plot(t, orig[c], color="black", linewidth=0.8, label="Orijinal (filtreli)")
        if dropped[c]:
            ax.plot(t, recon[c], color="crimson", linewidth=0.9, label="MAE rekonstrüksiyonu")
            ax.set_facecolor("#fdecec")
        ax.set_ylabel(LEAD_NAMES[c] + (" ✗" if dropped[c] else ""), rotation=0, labelpad=22)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[int(np.flatnonzero(dropped)[0])].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Zaman (sn)")
    fig.suptitle(f"ecg_id {dataset.ecg_ids[idx]} (gerçek: MI)  |  {wall} duvarı kanalları kopuk  |  P(MI) = {p_mi:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test seti stres testleri")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--baseline", default=None, help=f"1D-ResNet checkpoint'i (örn. {BASELINE_PATH})")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = get_device()
    test_set = PTBXLProcessedDataset("test", args.processed_dir)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(f"Test: {len(test_set)} kayıt (MI oranı {test_set.positive_fraction():.3f})")

    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"MAE: {args.checkpoint} (epoch {ckpt['epoch']}, val AUROC {ckpt['val_metrics']['auroc']:.4f}, "
          f"eşik {ckpt['threshold']:.2f})")
    results = run_all(model, ckpt, loader, device)
    print_table("Çift başlıklı MAE", results)

    baseline = None
    if args.baseline:
        bmodel, bckpt = load_checkpoint(args.baseline, device)
        baseline = run_all(bmodel, bckpt, loader, device)
        print_table("1D-ResNet (karşılaştırma)", baseline)

    fnr_reduction = check_targets(results, baseline)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_results.json"), "w") as f:
        json.dump({"mae": results, "resnet1d": baseline, "fnr_reduction": fnr_reduction,
                   "threshold": ckpt["threshold"]}, f, indent=2, ensure_ascii=False)
    plot_path = os.path.join(args.out_dir, "reconstruction_anterior.png")
    plot_reconstruction(model, test_set, device, plot_path)
    print(f"\nSonuçlar: {args.out_dir}/test_results.json   Grafik: {plot_path}")


if __name__ == "__main__":
    main()
