"""
ST segmenti sapması ölçümü (Click-to-Explain kartı, TÜBİTAK Raporu Md. 2.5).

Yöntem (100 Hz için yaklaşık):
  - R tepeleri referans derivasyonda (II, yoksa V5, I, V4 ...) bulunur
  - İzoelektrik taban: R'den 80–40 ms önce (PR segmenti) ortalaması
  - ST noktası: R + 100 ms (≈ J noktası + 60 ms)
  - Sapma = ST - taban, vuruşlar üzerinden medyan.  1 mm = 0.1 mV (standart EKG kağıdı)
Not: 0.5 Hz yüksek geçiren süzgeç ST seviyesini bir miktar etkileyebilir; değer klinik tanı değil,
açıklama amaçlı bir göstergedir.
"""
import numpy as np
from scipy.signal import find_peaks

from src.ecg_constants import FS, LEAD_NAMES

REFERENCE_ORDER = ["II", "V5", "I", "V4", "V6", "aVF", "III", "V3", "V2", "aVL", "V1", "aVR"]
MV_PER_MM = 0.1


def detect_r_peaks(signal_mv, fs=FS):
    x = signal_mv - np.median(signal_mv)
    if np.abs(x.min()) > np.abs(x.max()):  # baskın dalga negatifse (ör. ters QRS) çevir
        x = -x
    height = 0.5 * np.percentile(x, 99)
    peaks, _ = find_peaks(x, height=height, distance=int(0.3 * fs))
    return peaks


def st_deviation_mv(signals_mv, dropped=(), fs=FS):
    """
    signals_mv : (12, L) mV (kopan derivasyonlar rekonstrükte edilmiş olabilir)
    dropped    : kopan derivasyon indeksleri — R tepesi tespiti için referans seçilmez
    Döndürür   : (12,) mV ST sapması (vuruş bulunamazsa NaN), kullanılan referans derivasyon adı
    """
    ref = next(LEAD_NAMES.index(n) for n in REFERENCE_ORDER if LEAD_NAMES.index(n) not in set(dropped))
    peaks = detect_r_peaks(signals_mv[ref], fs)
    pr_start, pr_end, st_off = int(-0.08 * fs), int(-0.04 * fs), int(0.10 * fs)
    L = signals_mv.shape[1]
    peaks = peaks[(peaks + pr_start >= 0) & (peaks + st_off < L)]
    if len(peaks) == 0:
        return np.full(signals_mv.shape[0], np.nan), LEAD_NAMES[ref]
    devs = []
    for r in peaks:
        baseline = signals_mv[:, r + pr_start:r + pr_end].mean(axis=1)
        devs.append(signals_mv[:, r + st_off] - baseline)
    return np.median(np.stack(devs), axis=0), LEAD_NAMES[ref]
