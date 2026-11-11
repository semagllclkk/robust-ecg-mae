"""
Uzamsal komşuluk ağırlık matrisi — TÜBİTAK Raporu Risk 1 B Planı.

"3 veya 4 kanal aynı anda koptuğunda ... Spatial Attention katmanına ilave olarak, Einthoven Üçgeni fizyolojik
kısıtlarını zorunlu kılan ... önceden tanımlı uzamsal komşuluk ağırlık matrisleri regülarizasyon olarak eklenecektir."

Matris elektrotların fiziksel yerleşiminden kurulur:
  - Ekstremite derivasyonları (frontal düzlem, heksaksiyel sistem): açısal yakınlık, max(0, cos Δθ).
    Açılar: I 0°, II 60°, III 120°, aVR −150°, aVL −30°, aVF 90°.
  - Göğüs derivasyonları (yatay düzlem, V1→V6 göğüs üzerinde sırayla): exp(−|i−j| / ölçek).
  - İki düzlem arasında sabit, küçük bir taban değer (birbirinden bağımsız sayılmazlar ama yakın da değiller).

Regülarizasyon: modelin son Spatial Attention katmanındaki dikkat dağılımı (sorgu başına, sağlam anahtarlar üzerinde)
bu önsel dağılımdan uzaklaştıkça ceza alır: KL(önsel ‖ dikkat). Ağırlık 0 ise eğitim hiç değişmez.
"""
import numpy as np
import torch

LIMB_ANGLES_DEG = np.array([0.0, 60.0, 120.0, -150.0, -30.0, 90.0])  # I, II, III, aVR, aVL, aVF
PRECORDIAL_SCALE = 1.0
CROSS_PLANE = 0.05
EPS = 1e-8


def neighborhood_matrix():
    """(12, 12) negatif olmayan komşuluk ağırlıkları; LEAD_NAMES sırasıyla (I…aVF, V1…V6)."""
    W = np.full((12, 12), CROSS_PLANE)
    theta = np.deg2rad(LIMB_ANGLES_DEG)
    W[:6, :6] = np.clip(np.cos(theta[:, None] - theta[None, :]), 0.0, None)
    idx = np.arange(6)
    W[6:, 6:] = np.exp(-np.abs(idx[:, None] - idx[None, :]) / PRECORDIAL_SCALE)
    return W.astype(np.float32)


def neighborhood_kl(attention, M, prior=None):
    """
    attention : (B, 12, 12) sorgu × anahtar dikkat ağırlıkları (satır toplamı 1, kopan anahtarlar 0)
    M         : (B, 12) 1 = sağlam, 0 = koptu
    Önsel dağılım her sorgu için yalnızca sağlam anahtarlar üzerinde normalize edilir; KL(önsel ‖ dikkat) ortalaması.
    """
    if prior is None:
        prior = torch.as_tensor(neighborhood_matrix(), device=attention.device, dtype=attention.dtype)
    keep = (M > 0.5).to(attention.dtype).unsqueeze(1)                    # (B, 1, 12) sağlam anahtarlar
    p = prior.unsqueeze(0) * keep
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=EPS)
    a = attention.clamp(min=EPS)
    kl = (p * (torch.log(p.clamp(min=EPS)) - torch.log(a))).sum(dim=-1)  # (B, 12)
    return kl.mean()
