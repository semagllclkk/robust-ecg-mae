"""12 derivasyon tanımları ve rapordaki (2.5) duvar eşlemesi."""

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
N_LEADS = len(LEAD_NAMES)
SAMPLE_RATE = 100  # Hz (rapor 2.1)
WINDOW_SAMPLES = 1000  # 10 s

# Rapor 2.5: V1–V4 → Anterior/Septal, II-III-aVF → Inferior, I-aVL-V5-V6 → Lateral
WALL_LEADS = {
    "anterior": [6, 7, 8, 9],
    "inferior": [1, 2, 5],
    "lateral": [0, 4, 10, 11],
}

WALL_LABELS = {
    "anterior": "Anterior MI",
    "inferior": "Inferior MI",
    "lateral": "Lateral MI",
}

WALL_ARTERY = {
    "anterior": "LAD (sol ön inen arter)",
    "inferior": "RCA (sağ koroner arter)",
    "lateral": "LCx (sirkumfleks arter)",
}

SCENARIO_WALL = {"normal": None, "ant": "anterior", "inf": "inferior", "lat": "lateral"}
