"""
Değerlendirme metrikleri (TÜBİTAK Raporu Hedef 4 ve Hipotez):
- Teşhis: F1 (hedef > 0.88), AUROC, duyarlılık, özgüllük, yanlış negatif oranı (FNR)
- Kriz tipi: MI kayıtlarında duvar bazlı AUROC
- Rekonstrüksiyon (yalnızca kopan kanallar): RMSE (hedef < 0.15), R² (hedef > 0.80), Pearson r
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.ecg_constants import WALL_NAMES


def best_f1_threshold(labels, probs):
    """Doğrulama setinde F1'i en büyükleyen karar eşiği (test setine bu eşik uygulanır)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _safe_auroc(labels, probs):
    return float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")


class ClassificationMeter:
    def __init__(self):
        self.mi_probs, self.mi_labels, self.wall_probs, self.wall_labels = [], [], [], []

    def update(self, out, target):
        self.mi_probs.append(torch.sigmoid(out["mi_logit"]).detach().cpu().numpy())
        self.mi_labels.append(target["mi"].cpu().numpy())
        self.wall_probs.append(torch.sigmoid(out["wall_logits"]).detach().cpu().numpy())
        self.wall_labels.append(target["walls"].cpu().numpy())

    def arrays(self):
        return np.concatenate(self.mi_labels), np.concatenate(self.mi_probs)

    def result(self, threshold=0.5):
        labels, probs = self.arrays()
        preds = (probs >= threshold).astype(int)
        pos, neg = labels == 1, labels == 0
        sens = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
        res = {
            "auroc": _safe_auroc(labels, probs),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "acc": float((preds == labels).mean()),
            "sens": sens,
            "spec": float((preds[neg] == 0).mean()) if neg.any() else float("nan"),
            "fnr": 1.0 - sens,
            "threshold": threshold,
        }
        wl, wp = np.concatenate(self.wall_labels)[pos], np.concatenate(self.wall_probs)[pos]
        wall_aurocs = [_safe_auroc(wl[:, i], wp[:, i]) for i in range(len(WALL_NAMES))]
        for name, v in zip(WALL_NAMES, wall_aurocs):
            res[f"auroc_{name}"] = v
        res["wall_auroc_macro"] = float(np.nanmean(wall_aurocs)) if not all(np.isnan(wall_aurocs)) else float("nan")
        return res


class ReconstructionMeter:
    """Kopan her (kayıt, derivasyon) çifti için rekonstrüksiyon kalitesi."""

    def __init__(self):
        self.sq_z = self.sq_mv = 0.0
        self.n = 0
        self.r2, self.pearson = [], []

    @torch.no_grad()
    def update(self, recon, signal, dropped, sigma):
        if recon is None or not dropped.any():
            return
        r, t = recon[dropped].float(), signal[dropped].float()   # (N, L)
        s = sigma.to(recon.device)[dropped].float()                # (N,)
        keep = s > 1e-6                                              # düz (sabit) derivasyonları atla
        r, t, s = r[keep], t[keep], s[keep]
        if r.numel() == 0:
            return
        err = r - t
        self.sq_z += float((err ** 2).sum())
        self.sq_mv += float(((err * s[:, None]) ** 2).sum())
        self.n += err.numel()

        tc, rc = t - t.mean(1, keepdim=True), r - r.mean(1, keepdim=True)
        sst = (tc ** 2).sum(1).clamp(min=1e-8)
        self.r2.append((1 - (err ** 2).sum(1) / sst).cpu().numpy())
        self.pearson.append(((tc * rc).sum(1) / (sst.sqrt() * (rc ** 2).sum(1).sqrt().clamp(min=1e-8))).cpu().numpy())

    def result(self):
        if self.n == 0:
            return {"rmse_z": float("nan"), "rmse_mv": float("nan"), "r2": float("nan"), "pearson_r": float("nan")}
        return {
            "rmse_z": (self.sq_z / self.n) ** 0.5,
            "rmse_mv": (self.sq_mv / self.n) ** 0.5,
            "r2": float(np.concatenate(self.r2).mean()),
            "pearson_r": float(np.concatenate(self.pearson).mean()),
        }
