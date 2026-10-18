"""
Sinyal ön işleme modülü — preprocessing.py (TÜBİTAK Raporu Md. 2.1, İP1 başarı ölçütü).

- 0.5–45 Hz, 4. derece, çift yönlü (sıfır faz) Butterworth bant geçiren filtre
- Derivasyon bazında Z-score standardizasyonu
"""
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 45.0
FILTER_ORDER = 4
EPS = 1e-6


def butter_bandpass_filter(data, lowcut=LOWCUT_HZ, highcut=HIGHCUT_HZ, fs=100, order=FILTER_ORDER, axis=-1):
    """
    EKG sinyaline 4. derece çift yönlü Butterworth bant geçiren filtre uygular.
    0.5 Hz altı (taban çizgisi kayması / solunum) ve 45 Hz üstü (şebeke 50 Hz, kas tremoru)
    bileşenleri bastırır. Sayısal kararlılık için SOS (second-order sections) biçimi kullanılır.

    data : np.ndarray, filtre `axis` ekseni boyunca uygulanır (varsayılan: son eksen = zaman).
    """
    nyquist = 0.5 * fs
    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(f"Geçersiz kesim frekansları: {lowcut}-{highcut} Hz (Nyquist = {nyquist} Hz)")
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=axis)


def zscore_normalize(signal, axis=-1):
    """
    Her derivasyonu kendi ortalaması ve standart sapmasıyla standartlaştırır: z = (x - mu) / sigma.
    Düz (sabit) derivasyonlarda sigma ~ 0 olduğundan bu kanallar sıfıra çekilir.

    Döndürür: (normalize_sinyal, mu, sigma)  — mu/sigma, sinyali mV'a geri çevirmek için saklanır.
    """
    mu = signal.mean(axis=axis, keepdims=True)
    sigma = signal.std(axis=axis, keepdims=True)
    flat = sigma < EPS
    z = np.where(flat, 0.0, (signal - mu) / np.where(flat, 1.0, sigma))
    return z, np.squeeze(mu, axis=axis), np.squeeze(sigma, axis=axis)


def preprocess_ecg(signal, fs=100):
    """
    Tek bir kaydın tam ön işleme hattı.
    signal : (12, T) mV cinsinden ham EKG
    Döndürür: (z: (12, T) float32, mu: (12,), sigma: (12,))
    """
    filtered = butter_bandpass_filter(signal, fs=fs, axis=-1)
    z, mu, sigma = zscore_normalize(filtered, axis=-1)
    return z.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def test_filtering(record="data/ptbxl/records100/00000/00001_lr", lead=0, out_path="results/filtered_ecg_comparison.png"):
    """
    Ham ve 0.5–45 Hz filtrelenmiş EKG'yi karşılaştıran grafik (İP1 / İP6 raporu için).
    Çalıştırma: python -m src.preprocessing
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wfdb

    if not os.path.exists(record + ".hea"):
        print(f"Kayıt bulunamadı: {record}.hea — PTB-XL data/ptbxl/ altına çıkarılmalı.")
        return

    rec = wfdb.rdrecord(record)
    raw = rec.p_signal[:, lead]
    filtered = butter_bandpass_filter(raw, fs=rec.fs)
    t = np.arange(len(raw)) / rec.fs
    name = rec.sig_name[lead]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, raw, color="gray", linewidth=0.9)
    axes[0].set_title(f"Ham EKG Sinyali ({name}, {os.path.basename(record)})")
    axes[1].plot(t, filtered, color="green", linewidth=0.9)
    axes[1].set_title(f"Filtrelenmiş Sinyal (Butterworth {LOWCUT_HZ}–{HIGHCUT_HZ:g} Hz, {FILTER_ORDER}. derece, çift yönlü)")
    axes[1].set_xlabel("Zaman (sn)")
    for ax in axes:
        ax.set_ylabel("mV")
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Filtreleme karşılaştırma grafiği kaydedildi: {out_path}")


if __name__ == "__main__":
    test_filtering()
