"""
Dinamik Uzamsal Maskeleme Modülü — LeadDropout Operatörü (TÜBİTAK Raporu Md. 2.2).

Zhang ve ark. [4] MLAE (Masked Lead Autoencoder) prensibi; PyTorch DataLoader seviyesinde çalışır.
Eğitim esnasında her bir grup (batch) için k ∈ [1, 4] aralığında rastgele sayıda derivasyon seçilerek
değerleri sıfırlanır (asistoli / kablo kopması simülasyonu). Maskelenen kanalları belirten
M ∈ {0,1}^12 ikili maske vektörü girdi tensörüne uygulanır:

    X_masked = X ⊙ M · 1ᵀ₁₀₀₀        (M_i = 1: derivasyon sağlam, M_i = 0: derivasyon koptu)

Rastgele kopmalara ek olarak "kritik" kopmalar (Rapor Hedef 2) simüle edilir: kopan derivasyonlar
tek bir kalp duvarının (Anterior V1–V4, Inferior II/III/aVF, Lateral I/aVL/V5/V6) kanallarından seçilir.

Kullanım:
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=LeadDropoutCollate())
    for x_masked, x, M, target in loader: ...
"""
import torch
from torch.utils.data import default_collate

from src.ecg_constants import WALL_LEAD_INDICES, WALL_NAMES

NUM_LEADS = 12

# (3, 12) bool: duvar -> derivasyon üyeliği
_WALL_MEMBERSHIP = torch.zeros(len(WALL_NAMES), NUM_LEADS, dtype=torch.bool)
for _i, _w in enumerate(WALL_NAMES):
    _WALL_MEMBERSHIP[_i, WALL_LEAD_INDICES[_w]] = True


def random_keep_mask(batch_size, min_drop=1, max_drop=4, n_drop=None, critical_prob=0.0, generator=None):
    """
    (B, 12) float M maskesi üretir; 1 = sağlam, 0 = koptu.
    - n_drop verilirse her örnekte tam o kadar kanal kopar; verilmezse batch için tek bir
      k ∈ [min_drop, max_drop] seçilir (Rapor Md. 2.2: "her bir grup (batch) için").
    - critical_prob: örneğin "kritik" kopma yaşama olasılığı; kopan kanallar öncelikle rastgele
      seçilen tek bir kalp duvarının derivasyonlarından seçilir.
    """
    if n_drop is None:
        n_drop = int(torch.randint(min_drop, max_drop + 1, (1,), generator=generator))
    scores = torch.rand(batch_size, NUM_LEADS, generator=generator)
    if critical_prob > 0:
        critical = torch.rand(batch_size, generator=generator) < critical_prob
        walls = torch.randint(0, len(WALL_NAMES), (batch_size,), generator=generator)
        scores = scores - (_WALL_MEMBERSHIP[walls] & critical[:, None]).float()
    ranks = scores.argsort(dim=1).argsort(dim=1)
    return (ranks >= n_drop).float()


def wall_keep_mask(batch_size, wall):
    """Belirli bir kalp duvarının tüm derivasyonlarının koptuğu M maskesi (kritik stres testi)."""
    M = torch.ones(batch_size, NUM_LEADS)
    M[:, WALL_LEAD_INDICES[wall]] = 0.0
    return M


def full_keep_mask(batch_size):
    return torch.ones(batch_size, NUM_LEADS)


def apply_lead_mask(x, M):
    """X_masked = X ⊙ M · 1ᵀ  —  x: (B, 12, L), M: (B, 12)"""
    return x * M.to(x.dtype).unsqueeze(-1)


class LeadDropoutCollate:
    """
    DataLoader collate_fn'i olarak çalışan LeadDropout operatörü.
    Döndürür: (X_masked, X, M, hedef)  — X temiz sinyaldir (rekonstrüksiyon hedefi).
    """

    def __init__(self, min_drop=1, max_drop=4, critical_prob=0.3):
        if not (1 <= min_drop <= max_drop <= 4):
            raise ValueError("Rapor Md. 2.2: 1 ≤ min_drop ≤ max_drop ≤ 4 olmalı")
        self.min_drop = min_drop
        self.max_drop = max_drop
        self.critical_prob = critical_prob

    def __call__(self, batch):
        x, target = default_collate(batch)
        M = random_keep_mask(x.shape[0], self.min_drop, self.max_drop, critical_prob=self.critical_prob)
        return apply_lead_mask(x, M), x, M, target
