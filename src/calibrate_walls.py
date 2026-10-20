"""
Kriz tipi (kalp duvarı) karar eşiklerini doğrulama setinde (strat_fold 9) belirler.

Sorun: API her duvar için P(duvar | MI) ≥ 0.5 eşiğini kullanıyordu. Lateral MI eğitimde ve testte
çok seyrek (test setinde 20 kayıt) olduğundan model lateral duvara sistematik olarak düşük olasılık
veriyor (Lateral MI kayıtlarında ~0.3–0.4) ve bu duvarı neredeyse hiç işaretlemiyor.

Çözüm: MI karar eşiğiyle (`best_f1_threshold`, src/train.py) aynı yöntem — her duvar için F1'i
en büyükleyen eşik doğrulama setinde seçilir; test setine dokunulmaz. Kopmaya dayanıklılık hedefiyle
uyumlu olması için olasılıklar hem 12 kanalda hem rastgele 1–4 kanal kopmasında toplanır.
Sonuç models/wall_thresholds.json dosyasına yazılır; API bu dosyayı bulursa kullanır.

Çalıştırma (proje kök klasöründen, data/processed gerekir):
    python -m src.calibrate_walls
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.dataset import PTBXLProcessedDataset
from src.ecg_constants import WALL_NAMES
from src.lead_dropout import full_keep_mask, random_keep_mask
from src.metrics import best_f1_threshold
from src.train import get_device, load_checkpoint, model_forward


def checkpoint_sha256(path):
    """Checkpoint dosyasının parmak izi: eşiklerin hangi model için hesaplandığını API'nin doğrulaması için."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wall_thresholds(path, checkpoint_path):
    """
    Kalibre eşikleri yalnızca hangi checkpoint için hesaplandıysa o model yüklüyken kullanır (SHA-256 eşleşmesi).
    Model değiştiyse (ör. yeniden eğitim) eski eşikler sessizce uygulanmaz: uyarı verilir, 0.5 kullanılır.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        cal = json.load(fh)
    if cal.get("checkpoint_sha256") != checkpoint_sha256(checkpoint_path):
        print(f"⚠ Uyarı: {path} başka bir model için kalibre edilmiş; duvar eşikleri 0.5 kullanılıyor. "
              "`python -m src.calibrate_walls` ile yeniden kalibre edin.")
        return {}
    return cal["thresholds"]


def wall_thresholds(wall_labels, wall_probs):
    """wall_labels, wall_probs: (N, 3) — yalnızca MI kayıtları. Duvar başına F1-optimum eşik."""
    return {w: best_f1_threshold(wall_labels[:, i], wall_probs[:, i]) for i, w in enumerate(WALL_NAMES)}


@torch.no_grad()
def collect(model, dataset, device, seed=0, batch_size=64):
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    for x, target in loader:
        mi = target["mi"] > 0.5
        if not mi.any():
            continue
        x, walls = x[mi], target["walls"][mi]
        tgt = {k: v[mi].to(device) for k, v in target.items()}
        for M in (full_keep_mask(len(x)), random_keep_mask(len(x), generator=generator)):
            M = M.to(device)
            out = model_forward(model, x.to(device) * M.unsqueeze(-1), M, tgt)  # genlik farkındalıklı model σ ister
            probs.append(torch.sigmoid(out["wall_logits"]).float().cpu().numpy())
            labels.append(walls.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser(description="Duvar başına karar eşiği kalibrasyonu (doğrulama seti)")
    parser.add_argument("--checkpoint", default="models/mae_ecg_checkpoint.pth")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="models/wall_thresholds.json")
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model.eval()
    labels, probs = collect(model, PTBXLProcessedDataset("val", args.processed_dir), device)
    thresholds = {w: round(t, 2) for w, t in wall_thresholds(labels, probs).items()}

    report = {}
    for i, w in enumerate(WALL_NAMES):
        before = f1_score(labels[:, i], probs[:, i] >= 0.5, zero_division=0)
        after = f1_score(labels[:, i], probs[:, i] >= thresholds[w], zero_division=0)
        report[w] = {"threshold": thresholds[w], "positives": int(labels[:, i].sum()),
                     "val_f1_at_0.5": round(float(before), 4), "val_f1_calibrated": round(float(after), 4)}
        print(f"{w:9s} eşik {thresholds[w]:.2f} | pozitif {report[w]['positives']:4d} | "
              f"val F1 {before:.3f} -> {after:.3f}")

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
                   "epoch": ckpt.get("epoch"), "split": "val (strat_fold 9)",
                   "thresholds": thresholds, "details": report}, fh, ensure_ascii=False, indent=2)
    print(f"Yazıldı: {args.output}")


if __name__ == "__main__":
    main()
