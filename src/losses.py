"""
Çok görevli kayıp (TÜBİTAK Raporu Md. 2.3):
    L = lambda_rec * MSE(yalnızca maskelenen kanallar) + lambda_cls * FocalLoss(MI / Normal)
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


def binary_focal_loss(logits, targets, gamma=2.0, alpha=None):
    """
    Lin vd. (2017) Focal Loss, ikili sınıflandırma için.
    alpha: pozitif (MI) sınıf ağırlığı; None ise sınıf ağırlıklandırması yapılmaz.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    loss = (1.0 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * targets + (1.0 - alpha) * (1.0 - targets)) * loss
    return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_cls=1.0, gamma=2.0, alpha=None):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, recon, logits, target_signal, labels, lead_mask):
        rec = masked_mse(recon, target_signal, lead_mask)
        cls = binary_focal_loss(logits.squeeze(-1), labels, self.gamma, self.alpha)
        total = self.lambda_rec * rec + self.lambda_cls * cls
        return total, {"rec": rec.item(), "cls": cls.item()}
