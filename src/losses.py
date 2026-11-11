"""
Birleşik kayıp fonksiyonu (TÜBİTAK Raporu Md. 2.4):
    L_toplam = α · L_MSE + β · L_Focal        (α = 0.4, β = 0.6)
    İsteğe bağlı (Risk 1 B planı): + λ · KL(uzamsal komşuluk önseli ‖ Spatial Attention), src/spatial_prior.py

- L_MSE   : yalnızca kasti olarak koparılan (maskelenen) derivasyonlarda ortalama karesel hata
- L_Focal : MI / Normal Focal Loss + kriz tipi (Anterior/Inferior/Lateral) Focal Loss;
            kriz tipi terimi yalnızca MI etiketli örneklerde hesaplanır
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spatial_prior import neighborhood_kl


def masked_mse(recon, target, dropped):
    """
    Yalnızca kopan (dropped=True, yani M=0) derivasyonlar üzerinde ortalama karesel hata.
    recon, target : (B, C, L);  dropped : (B, C) bool
    Batch'te hiç kanal düşmemişse 0 döner (grafik bağlantısı korunur).
    """
    m = dropped.unsqueeze(-1).to(recon.dtype)
    n = m.sum() * recon.shape[-1]
    if n == 0:
        return recon.sum() * 0.0
    return (((recon - target) ** 2) * m).sum() / n


def binary_focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    """
    Lin ve ark. [12] Focal Loss, ikili / çoklu-etiket sınıflandırma için.
    alpha: pozitif sınıf ağırlığı; None ise sınıf ağırlıklandırması yapılmaz.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    loss = (1.0 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * targets + (1.0 - alpha) * (1.0 - targets)) * loss
    return loss.mean() if reduction == "mean" else loss


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha_mse=0.4, beta_focal=0.6, gamma=2.0, focal_alpha=None, spatial_prior=0.0):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.spatial_prior = spatial_prior  # Risk 1 B planı: uzamsal komşuluk regülarizasyonu ağırlığı (0 = kapalı)
        self.beta_focal = beta_focal
        self.gamma = gamma
        self.focal_alpha = focal_alpha

    def forward(self, out, signal, target):
        mi = target["mi"]
        cls = binary_focal_loss(out["mi_logit"], mi, self.gamma, self.focal_alpha)

        is_mi = mi > 0.5
        if is_mi.any():
            wall = binary_focal_loss(out["wall_logits"][is_mi], target["walls"][is_mi], self.gamma)
        else:
            wall = out["wall_logits"].sum() * 0.0
        focal = cls + wall

        if out["recon"] is not None:
            rec = masked_mse(out["recon"], signal, out["M"] < 0.5)  # yalnızca kopan (M=0) kanallar
        else:
            rec = torch.zeros((), device=signal.device)

        total = self.alpha_mse * rec + self.beta_focal * focal
        parts = {"rec": rec.item(), "cls": cls.item(), "wall": wall.item(), "prior": 0.0}
        if self.spatial_prior > 0 and out.get("spatial_attention") is not None:
            prior = neighborhood_kl(out["spatial_attention"], out["M"])
            total = total + self.spatial_prior * prior
            parts["prior"] = prior.item()
        return total, parts
