"""Proje genelinde ortak EKG sabitleri (TÜBİTAK Raporu Md. 2.5 duvar eşlemesi)."""

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100
SEQ_LEN = 1000

# Kalp duvarı -> derivasyonlar (Guo ve ark. [10] gruplaması)
WALL_GROUPS = {
    "Anterior": ["V1", "V2", "V3", "V4"],
    "Inferior": ["II", "III", "aVF"],
    "Lateral": ["I", "aVL", "V5", "V6"],
}
WALL_NAMES = list(WALL_GROUPS)
WALL_LEAD_INDICES = {w: [LEAD_NAMES.index(l) for l in leads] for w, leads in WALL_GROUPS.items()}

# Kalp duvarı -> PTB-XL MI alt sınıfı (kriz tipi etiketi)
WALL_TO_SUBCLASS = {"Anterior": "AMI", "Inferior": "IMI", "Lateral": "LMI"}

# Kalp duvarı -> en olası tıkanan koroner arter (Click-to-Explain kartı için)
WALL_TO_ARTERY = {
    "Anterior": "LAD (Sol Ön İnen Arter)",
    "Inferior": "RCA (Sağ Koroner Arter) / LCx",
    "Lateral": "LCx (Sirkumfleks Arter) / LAD diyagonal dalı",
}
