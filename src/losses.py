"""
Birleşik kayıp fonksiyonu (TÜBİTAK Raporu Md. 2.4):
    L_toplam = α · L_MSE + β · L_Focal        (α = 0.4, β = 0.6)

- L_MSE   : yalnızca kasti olarak koparılan (maskelenen) derivasyonlarda ortalama karesel hata
- L_Focal : MI / Normal Focal Loss + kriz tipi (Anterior/Inferior/Lateral) Focal Loss;
            kriz tipi terimi yalnızca MI etiketli örneklerde hesaplanır
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse(recon, target, lead_mask):
    """
    Yalnızca düşürülen (lead_mask=True) derivasyonlar üzerinde ortalama karesel hata.
    recon, target : (B, C, L);  lead_mask : (B, C) bool
    Batch'te hiç kanal düşmemişse 0 döner (grafik bağlantısı korunur).
    """
    m = lead_mask.unsqueeze(-1).to(recon.dtype)
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
    def __init__(self, alpha_mse=0.4, beta_focal=0.6, gamma=2.0, focal_alpha=None):
        super().__init__()
        self.alpha_mse = alpha_mse
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
            rec = masked_mse(out["recon"], signal, out["lead_mask"])
        else:
            rec = torch.zeros((), device=signal.device)

        total = self.alpha_mse * rec + self.beta_focal * focal
        return total, {"rec": rec.item(), "cls": cls.item(), "wall": wall.item()}
