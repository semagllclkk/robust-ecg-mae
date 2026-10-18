"""Ön işleme hattı (rapor 2.1, İP1).

- 0.5–45 Hz, 4. derece, çift yönlü (filtfilt) Butterworth bant geçiren süzgeç
- Derivasyon bazında Z-score standardizasyonu
- Kopuk (maskelenmiş) derivasyonlar sıfırda tutulur (LeadDropout ile aynı semantik)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .leads import SAMPLE_RATE

LOW_HZ = 0.5
HIGH_HZ = 45.0
ORDER = 4

_SOS = butter(ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=SAMPLE_RATE, output="sos")


def bandpass(signal: np.ndarray) -> np.ndarray:
    """(12, T) sinyale sıfır fazlı bant geçiren süzgeç uygular."""
    return sosfiltfilt(_SOS, signal, axis=-1)


def zscore(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / (std + eps)


def lead_mask(disconnected: list[int], n_leads: int = 12) -> np.ndarray:
    """1 = sağlam, 0 = kopuk derivasyon."""
    mask = np.ones(n_leads, dtype=np.float32)
    mask[list(disconnected)] = 0.0
    return mask


def preprocess(raw: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ham (12, T) sinyalden (filtrelenmiş mV, model girdisi) döndürür.

    Filtrelenmiş sinyal ST ölçümü için mV ölçeğinde kalır; model girdisi
    Z-score uygulanmış ve kopuk kanalları sıfırlanmış tensördür.
    """
    filtered = bandpass(raw) * mask[:, None]
    model_input = zscore(filtered) * mask[:, None]
    return filtered, model_input.astype(np.float32)
