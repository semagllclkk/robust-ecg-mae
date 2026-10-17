"""Sentetik 12 derivasyonlu EKG kaynağı (100 Hz).

PTB-XL entegrasyonu tamamlanana kadar canlı akış için kullanılır. Frontend'deki
Mock-API (src/services/ecgApi.js) ile aynı dalga morfolojisini üretir; böylece
backend açık/kapalı olsa da arayüz aynı sinyali gösterir.

Ekstremite derivasyonları fizyolojik olarak tutarlıdır: yalnızca I ve II
üretilir, diğerleri Einthoven/Goldberger bağıntılarıyla türetilir. ST
yükselmesi frontal düzlemde bir "hasar vektörü" olarak modellendiği için
karşı duvarlarda resiprokal ST çökmesi de kendiliğinden oluşur.
"""

import numpy as np

from .leads import SAMPLE_RATE, SCENARIO_WALL

HEART_RATE_INTERVAL = 60 / 72

# P, Q, R, S, T genlikleri (mV)
LEAD_I = np.array([0.12, -0.08, 0.7, -0.1, 0.25])
LEAD_II = np.array([0.15, -0.1, 1.0, -0.15, 0.3])
PRECORDIAL = np.array([
    [0.05, 0.0, 0.2, -0.7, 0.1],  # V1
    [0.06, 0.0, 0.4, -1.0, 0.3],  # V2
    [0.08, -0.05, 0.6, -0.6, 0.35],  # V3
    [0.1, -0.08, 1.0, -0.3, 0.4],  # V4
    [0.1, -0.1, 1.1, -0.15, 0.35],  # V5
    [0.1, -0.1, 0.9, -0.1, 0.3],  # V6
])

# Dalga merkezleri ve genişlikleri (s): P, Q, R, S, T
CENTERS = np.array([0.06, 0.155, 0.18, 0.205, 0.38])
SIGMAS = np.array([0.02, 0.008, 0.012, 0.009, 0.055])
ST_CENTER, ST_SIGMA = 0.27, 0.04
ST_ELEVATION_MV = 0.25

# Senaryo → (frontal hasar vektörü açısı [derece] veya None, ST yükselen V derivasyonları)
INJURY = {
    None: (None, []),
    "anterior": (None, [0, 1, 2, 3]),
    "inferior": (100.0, []),
    "lateral": (-10.0, [4, 5]),
}


def _gauss(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def _noise(sample_numbers: np.ndarray, lead_index: int) -> np.ndarray:
    seed = np.sin((sample_numbers * 13 + lead_index) * 12.9898) * 43758.5453
    return seed - np.floor(seed)


def limb_from_i_ii(lead_i: np.ndarray, lead_ii: np.ndarray) -> np.ndarray:
    """I ve II'den 6 ekstremite derivasyonunu (I, II, III, aVR, aVL, aVF) türetir."""
    return np.stack([
        lead_i,
        lead_ii,
        lead_ii - lead_i,
        -(lead_i + lead_ii) / 2,
        lead_i - lead_ii / 2,
        lead_ii - lead_i / 2,
    ])


def generate(start: int, count: int, condition: str = "normal") -> np.ndarray:
    """[start, start+count) örnek aralığı için (12, count) sinyal (mV) üretir."""
    if condition not in SCENARIO_WALL:
        raise ValueError(f"Bilinmeyen EKG senaryosu: {condition}")
    angle, st_precordial = INJURY[SCENARIO_WALL[condition]]

    n = np.arange(start, start + count, dtype=np.float64)
    t = n / SAMPLE_RATE
    beat = np.floor(t / HEART_RATE_INTERVAL)

    waves = np.zeros((5, count))
    st = np.zeros(count)
    for offset in (0, 1):
        d = t - (beat - offset) * HEART_RATE_INTERVAL
        for wave in range(5):
            waves[wave] += _gauss(d, CENTERS[wave], SIGMAS[wave])
        st += _gauss(d, ST_CENTER, ST_SIGMA)

    lead_i = LEAD_I @ waves
    lead_ii = LEAD_II @ waves
    if angle is not None:
        theta = np.deg2rad(angle)
        lead_i += ST_ELEVATION_MV * np.cos(theta) * st
        lead_ii += ST_ELEVATION_MV * np.cos(np.deg2rad(60) - theta) * st

    out = np.empty((12, count))
    out[:6] = limb_from_i_ii(lead_i, lead_ii)
    out[6:] = PRECORDIAL @ waves
    for v in st_precordial:
        out[6 + v] += ST_ELEVATION_MV * st

    for lead in range(12):
        out[lead] += 0.03 * np.sin(2 * np.pi * 0.25 * t + lead)
        out[lead] += 0.03 * (_noise(n, lead) - 0.5)
    return out
