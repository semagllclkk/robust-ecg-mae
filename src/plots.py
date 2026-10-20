"""
İP6 grafikleri (TÜBİTAK Raporu İP6: "deneysel başarım metriklerinin çıkarılması ... grafiklerin oluşturulması").

Tüm şekiller mevcut sonuç dosyalarından çizilir; model veya GPU gerekmez:
    models/mae_ecg_checkpoint_history.csv, models/resnet1d_baseline_history.csv   eğitim eğrileri
    results/test_results.json, results/v1/test_results.json                        test stres testleri
    results/noise/noise_results.json                                                gürültü testi
Çıktı: results/figures/*.png (300 dpi, bildiri ve rapor için)

Çalıştırma: python -m src.plots
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures"
V1_RESULTS = "results/v1/test_results.json"  # önceki (z-score çıkışlı) modelin test sonuçları

# Renkler: doğrulanmış kategorik palet (sabit sıra), açık yüzey; metin rengi seri renginden bağımsız
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#5f5e5a", "#e4e3df"
C_MAE, C_BASE, C_3 = "#2a78d6", "#eb6834", "#1baf7a"     # MAE, 1D-ResNet, üçüncü seri
TARGET = "#5f5e5a"

SCENARIOS = [
    ("0 kopuk (tam 12 kanal)", "12 kanal"),
    ("rastgele 1 kanal", "1 kanal"),
    ("rastgele 2 kanal", "2 kanal"),
    ("rastgele 3 kanal", "3 kanal"),
    ("rastgele 4 kanal", "4 kanal"),
    ("kritik Anterior (V1, V2, V3, V4)", "Anterior\n(V1–V4)"),
    ("kritik Inferior (II, III, aVF)", "Inferior\n(II, III, aVF)"),
    ("kritik Lateral (I, aVL, V5, V6)", "Lateral\n(I, aVL, V5, V6)"),
]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2,
    })


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def _grouped_bars(ax, labels, series, width=0.38, fmt="{:.3f}", label_series=None):
    """series: [(ad, renk, değerler)]. Çubuklar arasında 2 px yüzey boşluğu; seçili serilerde değer etiketi."""
    x = np.arange(len(labels))
    n = len(series)
    for i, (name, color, vals) in enumerate(series):
        offs = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offs, vals, width, color=color, label=name, edgecolor=SURFACE, linewidth=1.5)
        if label_series is None or name in label_series:
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                                fontsize=7, color=INK_2, zorder=5,
                                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    return x


def fig_training_curves():
    mae = pd.read_csv("models/mae_ecg_checkpoint_history.csv")
    base = pd.read_csv("models/resnet1d_baseline_history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.plot(mae.epoch, mae.train_loss, color=C_MAE, marker="o", markersize=4, label="Toplam (0.4·MSE + 0.6·Focal)")
    ax.plot(mae.epoch, 0.4 * mae.train_rec, color=C_3, marker="o", markersize=4, label="0.4·MSE (kopan kanallar)")
    ax.plot(mae.epoch, 0.6 * (mae.train_cls + mae.train_wall), color=C_BASE, marker="o", markersize=4,
            label="0.6·Focal (MI + kriz tipi)")
    ax.set_title("Eğitim kaybı (çift başlıklı MAE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kayıp")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(mae.epoch, mae.val_auroc, color=C_MAE, marker="o", markersize=4, label="Çift başlıklı MAE")
    ax.plot(base.epoch, base.val_auroc, color=C_BASE, marker="o", markersize=4, label="1D-ResNet")
    best = mae.loc[mae.val_auroc.idxmax()]
    ax.annotate(f"seçilen model\n(epoch {int(best.epoch)}, {best.val_auroc:.3f})", (best.epoch, best.val_auroc),
                xytext=(-70, -40), textcoords="offset points", fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_title("Doğrulama AUROC (kopmalı)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    ax.plot(mae.epoch, mae.val_r2, color=C_MAE, marker="o", markersize=4)
    ax.set_title("Doğrulama rekonstrüksiyon R²")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R² (kopan kanallar, gerçekçi, mV)")
    fig.text(0.5, -0.04, "Doğrulama: strat_fold 9, dengeli MI/Normal, batch başına 1–4 kanal kopması (%30 kritik duvar). "
             "Genlik farkındalıklı model; R² kopan kanalın genliği bilinmeden (gerçekçi koşulda) ölçülmüştür.",
             ha="center", fontsize=8, color=INK_2)
    fig.tight_layout()
    _save(fig, "1_egitim_egrileri.png")


def _load_test():
    with open("results/test_results.json") as f:
        return json.load(f)


def fig_dropout_f1(d):
    labels = [s[1] for s in SCENARIOS]
    mae = [d["mae"][k]["f1"] for k, _ in SCENARIOS]
    base = [d["resnet1d"][k]["f1"] for k, _ in SCENARIOS]
    fig, ax = plt.subplots(figsize=(11, 4))
    _grouped_bars(ax, labels, [("Çift başlıklı MAE", C_MAE, mae), ("1D-ResNet", C_BASE, base)], fmt="{:.3f}")
    ax.axhline(0.88, color=TARGET, linestyle="--", linewidth=1, label="Hedef F1 = 0.88")
    ax.axvline(4.5, color=GRID, linewidth=1)
    ax.text(2, 0.985, "Rastgele kopma", ha="center", fontsize=9, color=INK_2)
    ax.text(6, 0.985, "Kritik duvar kopması", ha="center", fontsize=9, color=INK_2)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("F1 (MI)")
    ax.set_title("Kanal kopmasına göre teşhis başarımı (test seti, 550 MI + 550 Normal)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    _save(fig, "2_kopma_f1.png")


def fig_false_negatives(d):
    drop = SCENARIOS[1:]
    labels = [s[1] for s in drop] + ["Havuzlanmış\n(hipotez)"]
    pooled = {m: float(np.mean([d[m][k]["fnr"] for k, _ in drop])) for m in ("mae", "resnet1d")}  # toplam FN / toplam MI
    mae = [d["mae"][k]["fnr"] for k, _ in drop] + [pooled["mae"]]
    base = [d["resnet1d"][k]["fnr"] for k, _ in drop] + [pooled["resnet1d"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    _grouped_bars(ax, labels, [("Çift başlıklı MAE", C_MAE, mae), ("1D-ResNet", C_BASE, base)], fmt="{:.3f}")
    ax.axvline(len(drop) - 0.5, color=GRID, linewidth=1)
    spec = d["resnet1d"]["kritik Anterior (V1, V2, V3, V4)"]["spec"]
    ax.annotate(f"1D-ResNet burada neredeyse her\nkayda MI diyor (özgüllük {spec:.2f})", (4.19, 0.052),
                xytext=(4.0, 0.205), textcoords="data", ha="center", fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    red = (pooled["resnet1d"] - pooled["mae"]) / pooled["resnet1d"]
    ax.set_title(f"Yanlış negatif oranı — 1–4 kanal kopmasında havuzlanmış azalma %{100 * red:.1f} (hipotez ≥ %25)")
    ax.set_ylabel("Yanlış negatif oranı (FNR)")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, "3_yanlis_negatif.png")


def fig_reconstruction(d):
    """Kopan kanalların R²'si, gerçekçi koşulda: önceki model (v1) ve genlik farkındalıklı ana model."""
    drop = SCENARIOS[1:]
    labels = [s[1] for s in drop]
    r = [d["mae"][k] for k, _ in drop]
    series = []
    if os.path.exists(V1_RESULTS):
        with open(V1_RESULTS) as f:
            v1 = json.load(f)["mae"]
        series.append(("Önceki model (v1, z-score çıkışlı)", C_BASE, [v1[k]["r2"] for k, _ in drop]))
    series += [
        ("Genlik farkındalıklı MAE", C_MAE, [x["r2"] for x in r]),
        ("Genlik farkındalıklı MAE + Einthoven (Risk 1 B planı)", C_3, [x["ein_r2"] for x in r]),
    ]
    fig, ax = plt.subplots(figsize=(11, 4))
    _grouped_bars(ax, labels, series, width=0.27)
    ax.axhline(0.80, color=TARGET, linestyle="--", linewidth=1, label="Hedef R² = 0.80")
    ax.axvline(3.5, color=GRID, linewidth=1)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("R² (kopan kanallar, mV)")
    ax.set_title("Kopan kanalların rekonstrüksiyonu — gerçekçi koşul (kopan kanalın genliği bilinmeden)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=2)
    _save(fig, "4_rekonstruksiyon_r2.png")


def fig_noise():
    with open("results/noise/noise_results.json") as f:
        d = json.load(f)
    names = {"bw": "Taban kayması", "ma": "Kas artefaktı", "em": "Elektrot hareketi"}
    snrs = sorted({r["snr_db"] for r in d["results"]})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, names.items()):
        for model, color, label in [("MAE", C_MAE, "Çift başlıklı MAE"), ("1D-ResNet", C_BASE, "1D-ResNet")]:
            vals = [np.mean([r[model]["f1"] for r in d["results"] if r["noise"] == key and r["snr_db"] == s]) for s in snrs]
            ax.plot(snrs, vals, color=color, marker="o", markersize=5, label=label)
            ax.axhline(d["clean"][model]["f1"], color=color, linestyle=":", linewidth=1)
        ax.axhline(0.88, color=TARGET, linestyle="--", linewidth=1, label="Hedef F1 = 0.88")
        ax.set_title(title)
        ax.set_xlabel("SNR (dB, filtre öncesi) — sağa doğru gürültü azalır")
        ax.set_xticks(snrs)
    axes[0].set_ylabel("F1 (1–4 parazitli kanal ortalaması)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.text(0.5, -0.1, "Gerçek gürültü: MIT-BIH Noise Stress Test Database. Noktalı çizgiler: gürültüsüz F1.",
             ha="center", fontsize=8, color=INK_2)
    fig.tight_layout()
    _save(fig, "5_gurultu_f1.png")


def main():
    _style()
    print("İP6 şekilleri:")
    fig_training_curves()
    d = _load_test()
    fig_dropout_f1(d)
    fig_false_negatives(d)
    fig_reconstruction(d)
    fig_noise()


if __name__ == "__main__":
    main()
