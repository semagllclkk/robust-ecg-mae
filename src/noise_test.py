"""
Gürültülü derivasyon stres testi — TÜBİTAK Raporu Amaç (Md. 1.2): "1 ila 4 derivasyonun tamamen koptuğu
veya parazitlendiği klinik şartlarda dahi tanı başarımını kaybetmeyen" sistem.

Gürültü kaynağı: MIT-BIH Noise Stress Test Database (NSTDB, PhysioNet) gerçek gürültü kayıtları
  bw = taban çizgisi kayması (solunum / hareket), ma = kas artefaktı (tremor), em = elektrot hareketi
Gürültü ham (mV) sinyale eklenir, ardından raporun ön işleme hattı (0.5–45 Hz Butterworth + Z-score)
uygulanır — gerçekte gürültü kayıt sırasında bindiği için. SNR = 10·log10(P_EKG / P_gürültü), P_EKG
derivasyonun filtrelenmiş temiz sinyal gücüdür.

Karşılaştırılan durumlar (parazitli kanallar sisteme bildirilmez):
  MAE            : çift başlıklı MAE, gürültüyü bilmeden (gerçek saha koşulu)
  1D-ResNet      : klasik karşılaştırma modeli, aynı koşulda
  MAE+tespit     : ek analiz — her derivasyon diğerlerinden MAE ile yeniden üretilir, gözlenen sinyalle
                   uyuşmayan derivasyonlar (doğrulama setinde kalibre edilmiş eşik) koptu sayılıp maskelenir
  MAE+oracle     : üst sınır — parazitli kanallar kusursuz bilinip maskelenseydi

Çalıştırma (proje kök klasöründen; önce NSTDB data/nstdb/ altına indirilmeli):
    python -m src.noise_test --baseline models/resnet1d_baseline.pth
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly

from src.ecg_constants import FS, LEAD_NAMES
from src.metrics import ClassificationMeter
from src.prepare_data import find_ptbxl_root, load_record
from src.preprocessing import preprocess_ecg
from src.train import CHECKPOINT_PATH, get_device, load_checkpoint

NOISE_TYPES = {"bw": "Taban kayması", "ma": "Kas artefaktı", "em": "Elektrot hareketi"}
SNR_LEVELS = [12, 6, 0]
NOISY_COUNTS = [1, 2, 3, 4]
DETECT_PERCENTILE = 99.0   # temiz doğrulama kayıtlarında derivasyon başına yanlış alarm ≈ %1
MAX_MASKED = 4             # Rapor: en fazla 4 kanal kopması tolere edilir


def load_noise(nstdb_dir):
    """NSTDB gürültü kayıtları, 360 Hz -> 100 Hz, kanal başına ortalaması çıkarılmış."""
    noise = {}
    for name in NOISE_TYPES:
        sig, fields = wfdb.rdsamp(os.path.join(nstdb_dir, name))
        sig = resample_poly(sig, FS, fields["fs"], axis=0).T          # (2, N)
        noise[name] = sig - sig.mean(axis=1, keepdims=True)
    return noise


def load_split(split, processed_dir, ptbxl_dir):
    """Dengeli MI/Normal alt kümesinin ham (mV) kayıtları ve etiketleri."""
    meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"), index_col="ecg_id")
    meta = meta[(meta["split"] == split) & (meta["label"] >= 0) & (meta["balanced"] == 1)]
    root = find_ptbxl_root(ptbxl_dir)
    raw = np.stack([load_record(root, p) for p in meta["filename_lr"]]).astype(np.float64)
    target = {
        "mi": torch.tensor(meta["label"].to_numpy(dtype=np.float32)),
        "walls": torch.tensor(meta[["AMI", "IMI", "LMI"]].to_numpy(dtype=np.float32)),
    }
    return raw, target


def preprocess_all(raw):
    out = [preprocess_ecg(r) for r in raw]
    return np.stack([o[0] for o in out]), np.stack([o[2] for o in out])  # z, sigma


def add_noise(raw, sigma_clean, noise, snr_db, k, rng):
    """Her kayıtta rastgele k derivasyona belirtilen SNR'da gerçek gürültü ekler. Döndürür: (gürültülü ham, maske)"""
    noisy = raw.copy()
    noisy_mask = np.zeros(raw.shape[:2], dtype=bool)
    L = raw.shape[2]
    for i in range(len(raw)):
        for lead in rng.choice(12, size=k, replace=False):
            ch = rng.integers(noise.shape[0])
            start = rng.integers(noise.shape[1] - L)
            seg = noise[ch, start:start + L]
            seg = seg - seg.mean()
            p_sig = sigma_clean[i, lead] ** 2
            p_noise = seg.var()
            if p_sig <= 0 or p_noise <= 0:
                continue
            noisy[i, lead] += seg * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
            noisy_mask[i, lead] = True
    return noisy, noisy_mask


@torch.no_grad()
def _sigma_kw(model, sigma, s, e, device):
    if getattr(model, "amplitude_aware", False):
        return {"sigma": torch.from_numpy(sigma[s:e]).float().to(device)}
    return {}


def predict(model, z, M, device, sigma, batch=128):
    """Sınıflandırma çıktıları (mi_logit, wall_logits), CPU tensörleri olarak."""
    mi, walls = [], []
    for s in range(0, len(z), batch):
        x = torch.from_numpy(z[s:s + batch]).float().to(device)
        m = torch.from_numpy(M[s:s + batch]).float().to(device)
        out = model(x * m.unsqueeze(-1), m, **_sigma_kw(model, sigma, s, s + batch, device))
        mi.append(out["mi_logit"].float().cpu())
        walls.append(out["wall_logits"].float().cpu())
    return {"mi_logit": torch.cat(mi), "wall_logits": torch.cat(walls)}


# Tutarlılık kontrolünde birlikte maskelenen derivasyon grupları: her grupta farklı bölgelerden 3 kanal
# (ekstremite ön/arka, prekordiyal sağ/sol karışık), böylece her kanal kalan 9 kanaldan yeniden üretilir.
CHECK_GROUPS = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]  # (I,aVL,V3) (II,aVF,V4) (III,V1,V5) (aVR,V2,V6)


@torch.no_grad()
def consistency_scores(model, z, device, sigma, batch=16):
    """
    Derivasyon başına tutarlılık skoru: derivasyonun grubu maskelenip kalan 9 kanaldan MAE ile yeniden
    üretilir; gözlenen (z-score) sinyalle ortalama karesel farkı alınır. Yüksek skor = diğer kanallarla uyuşmuyor.
    """
    scores = np.zeros(z.shape[:2], dtype=np.float32)
    G = len(CHECK_GROUPS)
    group_masks = torch.ones(G, 12, device=device)
    for g, leads in enumerate(CHECK_GROUPS):
        group_masks[g, leads] = 0.0
    for s in range(0, len(z), batch):
        x = torch.from_numpy(z[s:s + batch]).float().to(device)            # (b, 12, L)
        b = x.shape[0]
        x_rep = x.repeat_interleave(G, dim=0)                                # (b*G, 12, L)
        m_rep = group_masks.repeat(b, 1)                                     # (b*G, 12)
        kw = _sigma_kw(model, sigma, s, s + batch, device)
        if kw:
            kw["sigma"] = kw["sigma"].repeat_interleave(G, dim=0)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
            recon = model(x_rep * m_rep.unsqueeze(-1), m_rep, **kw)["recon"].float()
        if getattr(model, "physical_recon", False):  # fiziksel ölçek -> z (μ ≈ 0, filtre sonrası)
            sig = kw["sigma"].unsqueeze(-1).clamp(min=1e-6)
            recon = recon * model.pop_sigma.view(1, -1, 1) / sig
        err = ((recon - x_rep) ** 2).mean(dim=-1).reshape(b, G, 12)           # (b, G, 12)
        sel = (1 - group_masks).unsqueeze(0)                                  # yalnızca maskelenen kanalların hatası
        scores[s:s + b] = (err * sel).sum(dim=1).cpu().numpy()
    return scores


def detect_mask(scores, thresholds):
    """Eşiği aşan derivasyonları (en fazla 4, en yüksek skorlu) koptu say. Döndürür: M (1 = kullan)."""
    M = np.ones_like(scores)
    over = scores > thresholds[None, :]
    ratio = np.where(over, scores / thresholds[None, :], 0.0)
    for i in range(len(scores)):
        flagged = np.flatnonzero(over[i])
        if len(flagged) > MAX_MASKED:
            flagged = flagged[np.argsort(-ratio[i, flagged])[:MAX_MASKED]]
        M[i, flagged] = 0.0
    return M


def cls_metrics(out, target, threshold):
    meter = ClassificationMeter()
    meter.update(out, target)
    r = meter.result(threshold)
    return {k: r[k] for k in ("auroc", "f1", "sens", "spec", "fnr")}


def main():
    parser = argparse.ArgumentParser(description="Gürültülü derivasyon stres testi (NSTDB)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--ptbxl-dir", default="data/ptbxl")
    parser.add_argument("--nstdb-dir", default="data/nstdb")
    parser.add_argument("--out-dir", default="results/noise")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    t0 = time.time()
    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    thr = ckpt["threshold"]
    baseline, bthr = (load_checkpoint(args.baseline, device) if args.baseline else (None, None))
    if baseline is not None:
        bthr = bthr["threshold"]
    noise = load_noise(args.nstdb_dir)

    # 1) Tespit eşiklerinin kalibrasyonu: temiz doğrulama kayıtları (test setine dokunulmaz)
    val_raw, _ = load_split("val", args.processed_dir, args.ptbxl_dir)
    val_z, val_sigma = preprocess_all(val_raw)
    thresholds = np.percentile(consistency_scores(model, val_z, device, val_sigma), DETECT_PERCENTILE, axis=0)
    print("Tespit eşikleri (doğrulama, %{:.0f}. yüzdelik): {}".format(
        DETECT_PERCENTILE, {n: round(float(t), 3) for n, t in zip(LEAD_NAMES, thresholds)}), flush=True)

    # 2) Test seti
    raw, target = load_split("test", args.processed_dir, args.ptbxl_dir)
    z_clean, sigma_clean = preprocess_all(raw)
    ones = np.ones(z_clean.shape[:2], dtype=np.float32)
    clean = {"MAE": cls_metrics(predict(model, z_clean, ones, device, sigma_clean), target, thr)}
    if baseline is not None:
        clean["1D-ResNet"] = cls_metrics(predict(baseline, z_clean, ones, device, sigma_clean), target, bthr)
    clean_scores = consistency_scores(model, z_clean, device, sigma_clean)
    clean_false_alarm = float((clean_scores > thresholds[None, :]).any(axis=1).mean())
    print(f"Temiz test: {clean}   tespit yanlış alarm (kayıt başına) %{100 * clean_false_alarm:.1f}", flush=True)

    results = []
    for ntype in NOISE_TYPES:
        for snr in SNR_LEVELS:
            for k in NOISY_COUNTS:
                rng = np.random.default_rng([args.seed, list(NOISE_TYPES).index(ntype), snr + 100, k])
                noisy_raw, noisy_mask = add_noise(raw, sigma_clean, noise[ntype], snr, k, rng)
                z, sigma_noisy = preprocess_all(noisy_raw)
                row = {"noise": ntype, "snr_db": snr, "k": k}
                row["MAE"] = cls_metrics(predict(model, z, ones, device, sigma_noisy), target, thr)
                if baseline is not None:
                    row["1D-ResNet"] = cls_metrics(predict(baseline, z, ones, device, sigma_noisy), target, bthr)
                M_det = detect_mask(consistency_scores(model, z, device, sigma_noisy), thresholds)
                row["MAE+tespit"] = cls_metrics(predict(model, z, M_det, device, sigma_noisy), target, thr)
                flagged = M_det < 0.5
                row["tespit"] = {
                    "duyarlilik": float((flagged & noisy_mask).sum() / noisy_mask.sum()),
                    "kesinlik": float((flagged & noisy_mask).sum() / max(flagged.sum(), 1)),
                }
                M_oracle = (~noisy_mask).astype(np.float32)
                row["MAE+oracle"] = cls_metrics(predict(model, z, M_oracle, device, sigma_noisy), target, thr)
                results.append(row)
                base_f1 = f"{row['1D-ResNet']['f1']:.3f}" if baseline is not None else "-"
                print(f"{NOISE_TYPES[ntype]:18s} SNR {snr:3d} dB  k={k} | F1  MAE {row['MAE']['f1']:.3f}  "
                      f"ResNet {base_f1}  MAE+tespit {row['MAE+tespit']['f1']:.3f}  "
                      f"oracle {row['MAE+oracle']['f1']:.3f} | tespit duyarlılık %{100 * row['tespit']['duyarlilik']:.0f} "
                      f"kesinlik %{100 * row['tespit']['kesinlik']:.0f}  ({time.time() - t0:.0f} sn)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "noise_results.json"), "w") as f:
        json.dump({"clean": clean, "clean_detection_false_alarm": clean_false_alarm,
                   "detect_thresholds": dict(zip(LEAD_NAMES, thresholds.tolist())),
                   "threshold": thr, "baseline_threshold": bthr, "results": results}, f, indent=2, ensure_ascii=False)
    summarize(results, clean, args.out_dir)
    plot_example(raw, sigma_clean, noise, args.out_dir)
    print(f"\nSonuçlar: {args.out_dir}/noise_results.json, noise_f1.png, noise_example.png  ({time.time() - t0:.0f} sn)")


def summarize(results, clean, out_dir):
    variants = [v for v in ["MAE", "1D-ResNet", "MAE+tespit", "MAE+oracle"] if v in results[0]]
    print("\n=== Özet: F1 (k = 1–4 parazitli kanal ortalaması) ===")
    print(f"{'gürültü':18s} {'SNR':>6} | " + " ".join(f"{v:>11}" for v in variants))
    print(f"{'temiz':18s} {'-':>6} | " + " ".join(f"{clean[v]['f1']:11.3f}" if v in clean else f"{'-':>11}" for v in variants))
    fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(15, 4.2), sharey=True)
    for ax, ntype in zip(axes, NOISE_TYPES):
        for v in variants:
            f1 = [np.mean([r[v]["f1"] for r in results if r["noise"] == ntype and r["snr_db"] == s]) for s in SNR_LEVELS]
            ax.plot(SNR_LEVELS, f1, marker="o", label=v)
        for s in SNR_LEVELS:
            vals = [np.mean([r[v]["f1"] for r in results if r["noise"] == ntype and r["snr_db"] == s]) for v in variants]
            print(f"{NOISE_TYPES[ntype]:18s} {s:4d}dB | " + " ".join(f"{x:11.3f}" for x in vals))
        ax.axhline(0.88, color="gray", linestyle="--", linewidth=1, label="Hedef F1 = 0.88")
        ax.set_title(NOISE_TYPES[ntype])
        ax.set_xlabel("SNR (dB) — sağa doğru gürültü azalır")
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("F1 (1–4 parazitli kanal ortalaması)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Gürültülü derivasyon stres testi (NSTDB gerçek gürültü, PTB-XL test seti)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "noise_f1.png"), dpi=120)
    plt.close(fig)


def plot_example(raw, sigma_clean, noise, out_dir, lead=1, snr=6):
    """Bir kaydın aynı derivasyonuna üç gürültü türünün eklenmiş hâli (filtre sonrası, mV)."""
    t = np.arange(raw.shape[2]) / FS
    fig, axes = plt.subplots(len(NOISE_TYPES) + 1, 1, figsize=(12, 8), sharex=True)
    clean_f = preprocess_ecg(raw[0])
    axes[0].plot(t, clean_f[0][lead] * clean_f[2][lead] + clean_f[1][lead], color="black", linewidth=0.8)
    axes[0].set_title(f"Temiz ({LEAD_NAMES[lead]}, filtreli)")
    for ax, ntype in zip(axes[1:], NOISE_TYPES):
        one = raw[:1].copy()
        seg = noise[ntype][0, 5000:5000 + raw.shape[2]]
        seg = seg - seg.mean()
        one[0, lead] += seg * np.sqrt(sigma_clean[0, lead] ** 2 / (seg.var() * 10 ** (snr / 10)))
        z, mu, sg = preprocess_ecg(one[0])
        ax.plot(t, z[lead] * sg[lead] + mu[lead], color="crimson", linewidth=0.8)
        ax.set_title(f"{NOISE_TYPES[ntype]}, SNR {snr} dB (filtre sonrası)")
    for ax in axes:
        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Zaman (sn)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "noise_example.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
