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

from src.dataset import PTBXLProcessedDataset, population_sigma
from src.ecg_constants import LEAD_NAMES, WALL_GROUPS, WALL_NAMES
from src.einthoven import einthoven_complete
from src.losses import MultiTaskLoss
from src.lead_dropout import random_keep_mask, wall_keep_mask
from src.train import BASELINE_PATH, CHECKPOINT_PATH, evaluate, get_device, load_checkpoint, model_forward

TARGET_F1 = 0.88
TARGET_RMSE = 0.15
TARGET_R2 = 0.80
TARGET_FNR_REDUCTION = 0.25


def scenarios():
    s = {"0 kopuk (tam 12 kanal)": None}
    for k in range(1, 5):
        s[f"rastgele {k} kanal"] = (lambda B, g, k=k: random_keep_mask(B, n_drop=k, generator=g))
    for wall in WALL_NAMES:
        s[f"kritik {wall} ({', '.join(WALL_GROUPS[wall])})"] = (lambda B, g, w=wall: wall_keep_mask(B, w))
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
    print("Rekonstrüksiyon (kopan kanallar, mV): gerçekçi = kopan kanalın genliği bilinmeden (API ile aynı); "
          "+Einthoven = Risk 1 B planı; oracle = genlik biliniyormuş gibi (yalnızca karşılaştırma)")
    print(f"{'senaryo':38s} | {'AUROC':>6} {'F1':>6} {'Duyar.':>6} {'Özgül.':>6} {'FNR':>6} {'DuvarAUC':>8} | "
          f"{'RMSE':>6} {'R²':>6} {'r':>6} | {'+Ein RMSE':>9} {'R²':>6} | {'oracle R²':>9}")
    for name, r in results.items():
        if np.isnan(r["rmse_mv"]):
            rec = f"{'-':>6} {'-':>6} {'-':>6} | {'-':>9} {'-':>6} | {'-':>9}"
        else:
            oracle = f"{r['oracle_r2']:9.3f}" if not np.isnan(r.get("oracle_r2", float("nan"))) else f"{'-':>9}"
            rec = (f"{r['rmse_mv']:6.4f} {r['r2']:6.3f} {r['pearson_r']:6.3f} | "
                   f"{r['ein_rmse_mv']:9.4f} {r['ein_r2']:6.3f} | {oracle}")
        print(f"{name:38s} | {r['auroc']:6.4f} {r['f1']:6.4f} {r['sens']:6.4f} {r['spec']:6.4f} {r['fnr']:6.4f} "
              f"{r['wall_auroc_macro']:8.4f} | {rec}")


def check_targets(results, baseline=None):
    print("\n=== Rapor hedefleri (1–4 kanal kopması; rekonstrüksiyon gerçekçi koşulda) ===")
    drop = {k: v for k, v in results.items() if not k.startswith("0 kopuk")}
    ok = lambda b: "✓" if b else "✗"
    min_f1 = min(r["f1"] for r in drop.values())
    print(f"{ok(min_f1 > TARGET_F1)} En düşük F1 = {min_f1:.4f}  (hedef > {TARGET_F1})")
    for prefix, label in [("", "MAE"), ("ein_", "MAE + Einthoven (Risk 1 B planı)")]:
        rmse = max(r[f"{prefix}rmse_mv"] for r in drop.values())
        r2 = min(r[f"{prefix}r2"] for r in drop.values())
        print(f"{ok(rmse < TARGET_RMSE)} En yüksek RMSE = {rmse:.4f} mV  (hedef < {TARGET_RMSE})  [{label}]")
        print(f"{ok(r2 > TARGET_R2)} En düşük R² = {r2:.4f}  (hedef > {TARGET_R2})  [{label}]")
    if baseline is not None:
        # Hipotez: 1–4 kanal koptuğunda yanlış negatif oranı. Tüm kopma senaryolarında havuzlanmış FNR
        # (toplam FN / toplam MI; senaryolarda MI sayısı eşit olduğundan FNR'lerin ortalaması) karşılaştırılır.
        fnr_mae = float(np.mean([results[n]["fnr"] for n in drop]))
        fnr_base = float(np.mean([baseline[n]["fnr"] for n in drop]))
        pooled_red = (fnr_base - fnr_mae) / fnr_base if fnr_base > 0 else 0.0
        print(f"{ok(pooled_red >= TARGET_FNR_REDUCTION)} 1D-ResNet'e göre yanlış negatif azalması = %{100 * pooled_red:.1f}  "
              f"(havuzlanmış FNR: MAE {fnr_mae:.4f}, 1D-ResNet {fnr_base:.4f}; hipotez ≥ %{100 * TARGET_FNR_REDUCTION:.0f})")
        print("   Senaryo bazında (1D-ResNet FNR → MAE FNR, 1D-ResNet özgüllüğü):")
        for n in drop:
            fb, fm = baseline[n]["fnr"], results[n]["fnr"]
            red = (fb - fm) / fb if fb > 0 else 0.0
            print(f"     {n:36s} {fb:.3f} → {fm:.3f}  (%{100 * red:+.1f})   özgüllük {baseline[n]['spec']:.3f}")
        return pooled_red
    return None


def plot_reconstruction(model, dataset, device, out_path, wall="Anterior", use_einthoven=False):
    """
    Bir MI test kaydında belirli bir duvarın tüm derivasyonları koptuğunda rekonstrüksiyon (mV).
    use_einthoven=True: kopan ekstremite derivasyonları Einthoven kısıtıyla hesaplanır (Risk 1 B planı).
    """
    idx = int(np.flatnonzero(dataset.labels == 1)[0])
    x, target = dataset[idx]
    M = wall_keep_mask(1, wall).to(device)
    with torch.no_grad():
        tgt = {k: v.unsqueeze(0).to(device) for k, v in target.items()}
        out = model_forward(model, x.unsqueeze(0).to(device) * M.unsqueeze(-1), M, tgt)
    mu, sigma = target["mu"].numpy()[:, None], target["sigma"].numpy()[:, None]
    orig = x.numpy() * sigma + mu
    # Gerçekçi: kopan kanalın genliği bilinmez, popülasyon σ kullanılır (API ile aynı)
    recon = out["recon"][0].float().cpu().numpy() * population_sigma()[:, None]
    dropped = (M[0] < 0.5).cpu().numpy()
    label = "MAE rekonstrüksiyonu"
    if use_einthoven:
        limb = einthoven_complete(torch.from_numpy(orig).unsqueeze(0).float(), M.cpu())[0].numpy()
        recon[:6] = np.where(dropped[:6, None], limb, recon[:6])
        label = "MAE + Einthoven kısıtı"
    p_mi = torch.sigmoid(out["mi_logit"]).item()

    t = np.arange(orig.shape[1]) / 100
    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    for c, ax in enumerate(axes):
        ax.plot(t, orig[c], color="black", linewidth=0.8, label="Orijinal (filtreli)")
        if dropped[c]:
            ax.plot(t, recon[c], color="crimson", linewidth=0.9, label=label)
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
        drop = [k for k in results if not k.startswith("0 kopuk")]
        pooled = ({m: float(np.mean([r[k]["fnr"] for k in drop])) for m, r in (("mae", results), ("resnet1d", baseline))}
                  if baseline is not None else None)
        json.dump({"mae": results, "resnet1d": baseline, "fnr_reduction": fnr_reduction, "fnr_pooled": pooled,
                   "threshold": ckpt["threshold"]}, f, indent=2, ensure_ascii=False)
    plot_path = os.path.join(args.out_dir, "reconstruction_anterior.png")
    plot_reconstruction(model, test_set, device, plot_path, wall="Anterior")
    plot_path2 = os.path.join(args.out_dir, "reconstruction_inferior_einthoven.png")
    plot_reconstruction(model, test_set, device, plot_path2, wall="Inferior", use_einthoven=True)
    print(f"\nSonuçlar: {args.out_dir}/test_results.json   Grafikler: {plot_path}, {plot_path2}")


if __name__ == "__main__":
    main()
