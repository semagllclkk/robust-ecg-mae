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


def test_filtering(data_dir='data/raw'):
    import matplotlib.pyplot as plt
    import wfdb

    record_path = os.path.join(data_dir, "00001_lr")
    if not os.path.exists(record_path + ".hea"):
        print("Sinyal dosyası bulunamadı. Önce data_loader.py çalıştırılmalı!")
        return

    record = wfdb.rdrecord(record_path)
    raw_signal = record.p_signal[:1000, 0]  # İlk 1000 örnek, Lead I

    filtered_signal = butter_bandpass_filter(raw_signal, fs=record.fs)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(raw_signal, color='gray', label='Ham Sinyal (Raw)')
    plt.title("Ham EKG Sinyali")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal, color='green', label=f'Filtrelenmiş Sinyal (Bandpass {LOWCUT_HZ}-{HIGHCUT_HZ:g} Hz)')
    plt.title("Filtrelenmiş Temiz EKG Sinyali")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "filtered_ecg_comparison.png"))
    print("Filtreleme karşılaştırma grafiği 'data/raw/filtered_ecg_comparison.png' olarak kaydedildi!")
    plt.show()


if __name__ == '__main__':
    test_filtering()
