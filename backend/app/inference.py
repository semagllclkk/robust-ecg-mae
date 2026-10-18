"""Çıkarım katmanı.

İki arka uç vardır:

* ``ExportedModel``  — İP3'te eğitilen çift başlıklı MAE'nin ONNX (rapor Risk 2
  B planı) veya TorchScript çıktısı. ``MODEL_PATH`` ortam değişkeniyle yüklenir.
* ``PhysiologicalModel`` — model ağırlıkları hazır olana kadar kullanılan,
  öğrenme içermeyen yedek. Ekstremite derivasyonlarını Einthoven/Goldberger
  bağıntılarıyla, prekordiyal derivasyonları komşu derivasyon
  enterpolasyonuyla tamamlar; teşhisi ST yükselmesi ölçütüyle koyar.

Her iki arka uç da aynı ``Prediction`` sözleşmesini döndürür; frontend hangi
arka ucun çalıştığını ``modelSource`` alanından görür.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .leads import N_LEADS, SAMPLE_RATE, WALL_ARTERY, WALL_LABELS, WALL_LEADS
from .preprocessing import preprocess

WALLS = list(WALL_LEADS)
CLASS_ORDER = ["normal", *WALLS]  # model logits sırası
ST_THRESHOLD_MM = 1.0
# Rekonstrükte kanalların teşhise katkı ağırlığı: Einthoven tam (doğrusal bağıntı),
# enterpolasyon yaklaşık olduğundan kanıt değeri düşürülür (rapor: Presacan ve ark. [6]).
EVIDENCE_WEIGHT = {"einthoven": 1.0, "interpolation": 0.35, "unavailable": 0.0}

LIMB_BASIS = np.array([
    [1.0, 0.0],  # I
    [0.0, 1.0],  # II
    [-1.0, 1.0],  # III
    [-0.5, -0.5],  # aVR
    [1.0, -0.5],  # aVL
    [-0.5, 1.0],  # aVF
])


# ---------------------------------------------------------------- rekonstrüksiyon
def reconstruct_physiological(raw: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """Kopuk derivasyonları sağlam derivasyonlardan tamamlar.

    Dönen sinyalde sağlam kanallar değişmez; ``methods`` her kopuk kanal için
    kullanılan yöntemi verir.
    """
    out = raw.copy()
    methods: dict[int, str] = {}

    limb_ok = [i for i in range(6) if mask[i] > 0]
    limb_missing = [i for i in range(6) if mask[i] == 0]
    if limb_missing and np.linalg.matrix_rank(LIMB_BASIS[limb_ok]) == 2:
        coeffs, *_ = np.linalg.lstsq(LIMB_BASIS[limb_ok], raw[limb_ok], rcond=None)
        for i in limb_missing:
            out[i] = LIMB_BASIS[i] @ coeffs
            methods[i] = "einthoven"
    for i in limb_missing:
        if i not in methods:
            out[i] = 0.0
            methods[i] = "unavailable"

    v_ok = [v for v in range(6) if mask[6 + v] > 0]
    for v in range(6):
        if mask[6 + v] > 0:
            continue
        if not v_ok:
            out[6 + v] = 0.0
            methods[6 + v] = "unavailable"
            continue
        left = max((k for k in v_ok if k < v), default=None)
        right = min((k for k in v_ok if k > v), default=None)
        if left is not None and right is not None:
            w = (v - left) / (right - left)
            out[6 + v] = (1 - w) * raw[6 + left] + w * raw[6 + right]
        else:
            out[6 + v] = raw[6 + (left if left is not None else right)]
        methods[6 + v] = "interpolation"
    return out, methods


# ---------------------------------------------------------------- ST ölçümü
def detect_r_peaks(filtered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sağlam kanalların QRS enerjisinden R tepe indekslerini bulur."""
    energy = np.abs(np.diff(filtered[mask > 0], axis=-1)).sum(axis=0)
    if energy.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(energy, height=0.45 * energy.max(), distance=int(0.3 * SAMPLE_RATE))
    return peaks


def measure_st(filtered: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Derivasyon başına ST sapması (mm; 1 mm = 0.1 mV).

    Taban çizgisi PR segmentinden (R−70…R−40 ms), ST noktası J+20 ms (R+70 ms)
    civarından ölçülür; atımlar arası medyan alınır.
    """
    T = filtered.shape[-1]
    pre0, pre1, st_off = int(0.07 * SAMPLE_RATE), int(0.04 * SAMPLE_RATE), int(0.07 * SAMPLE_RATE)
    margin = int(0.5 * SAMPLE_RATE)  # filtfilt kenar etkisi
    values = []
    for r in peaks:
        if r - pre0 < margin or r + st_off >= T - margin:
            continue
        baseline = filtered[:, r - pre0 : r - pre1 + 1].mean(axis=-1)
        values.append(filtered[:, r + st_off] - baseline)
    if not values:
        return np.zeros(filtered.shape[0])
    return np.median(np.array(values), axis=0) * 10.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- sözleşme
@dataclass
class Prediction:
    label: str
    region: str | None
    confidence: float
    wall_probabilities: dict[str, float]
    st_by_lead: list[float]
    attention: list[float]
    reconstruction: dict[int, str] = field(default_factory=dict)
    model_source: str = "physiological"

    def to_json(self) -> dict:
        st_max = max(self.st_by_lead[i] for i in WALL_LEADS[self.region]) if self.region else 0.0
        return {
            "label": self.label,
            "region": self.region,
            "artery": WALL_ARTERY.get(self.region) if self.region else None,
            "confidence": round(self.confidence, 4),
            "stDeviation": round(float(st_max), 2),
            "stByLead": [round(float(v), 2) for v in self.st_by_lead],
            "attention": [round(float(v), 4) for v in self.attention],
            "wallProbabilities": {k: round(float(v), 4) for k, v in self.wall_probabilities.items()},
            "reconstruction": {str(k): v for k, v in self.reconstruction.items()},
            "modelSource": self.model_source,
        }


def _decide(probs: dict[str, float], missing: int, wall_missing: dict[str, int]) -> tuple[str | None, float]:
    best = max(WALLS, key=lambda w: probs[w])
    if probs[best] < 0.5:
        region, confidence = None, probs["normal"]
    else:
        region, confidence = best, probs[best]
        confidence *= 1 - 0.06 * wall_missing[best]
    confidence *= 1 - 0.02 * missing
    return region, float(np.clip(confidence, 0.5, 0.99))


# ---------------------------------------------------------------- arka uçlar
class PhysiologicalModel:
    source = "physiological"

    def reconstruct(self, raw: np.ndarray, mask: np.ndarray):
        return reconstruct_physiological(raw, mask)

    def predict(self, raw: np.ndarray, mask: np.ndarray) -> Prediction:
        completed, methods = self.reconstruct(raw, mask)
        filtered_obs, _ = preprocess(raw, mask)
        peaks = detect_r_peaks(filtered_obs, mask)
        full_mask = np.array([0.0 if methods.get(i) == "unavailable" else 1.0 for i in range(N_LEADS)])
        filtered_full, _ = preprocess(completed, full_mask)
        st = measure_st(filtered_full, peaks)
        evidence = st * np.array([EVIDENCE_WEIGHT.get(methods.get(i), 1.0) for i in range(N_LEADS)])

        probs: dict[str, float] = {}
        for wall, leads in WALL_LEADS.items():
            elevations = sorted((evidence[i] for i in leads), reverse=True)
            # ≥2 komşu derivasyonda ≥1 mm ST yükselmesi ölçütü
            probs[wall] = float(_sigmoid((elevations[1] - ST_THRESHOLD_MM) / 0.2))
        probs["normal"] = float(np.prod([1 - probs[w] for w in WALLS]))

        attention = np.where(mask > 0, np.abs(st) + 0.1, 0.0)
        attention = attention / attention.sum() if attention.sum() > 0 else attention

        missing = int((mask == 0).sum())
        wall_missing = {w: int(sum(mask[i] == 0 for i in leads)) for w, leads in WALL_LEADS.items()}
        region, confidence = _decide(probs, missing, wall_missing)
        return Prediction(
            label=WALL_LABELS[region] if region else "Normal Sinüs Ritmi",
            region=region,
            confidence=confidence,
            wall_probabilities={w: probs[w] for w in WALLS},
            st_by_lead=st.tolist(),
            attention=attention.tolist(),
            reconstruction=methods,
            model_source=self.source,
        )


class ExportedModel(PhysiologicalModel):
    """İP3 modelinin dışa aktarılmış hâli.

    Girdi : x [1, 12, 1000] (Z-score, kopuk kanallar 0), mask [1, 12]
    Çıktı : recon [1, 12, 1000] (Z-score uzayı), logits [1, 4]
            (sıra: normal, anterior, inferior, lateral), attention [1, 12] (opsiyonel)

    Z-score uzayındaki rekonstrüksiyon, modelin yanındaki ``<ad>.stats.json``
    dosyasındaki eğitim kümesi derivasyon std değerleriyle mV'a çevrilir.
    """

    def __init__(self, path: Path):
        self.path = path
        self.source = f"{path.suffix.lstrip('.')}:{path.name}"
        stats_path = path.with_suffix(".stats.json")
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        self.lead_std = np.asarray(stats.get("std", [0.2] * N_LEADS), dtype=np.float64)

        if path.suffix == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._run = lambda x, m: session.run(None, {"x": x, "mask": m})
        else:
            import torch

            module = torch.jit.load(str(path), map_location="cpu").eval()

            def run(x, m):
                with torch.no_grad():
                    outputs = module(torch.from_numpy(x), torch.from_numpy(m))
                return [o.numpy() for o in outputs]

            self._run = run

    def _infer(self, raw: np.ndarray, mask: np.ndarray):
        _, model_input = preprocess(raw, mask)
        outputs = self._run(model_input[None], mask[None].astype(np.float32))
        recon, logits = outputs[0][0], outputs[1][0]
        attention = outputs[2][0] if len(outputs) > 2 else None
        return recon, logits, attention

    def reconstruct(self, raw: np.ndarray, mask: np.ndarray):
        recon, _, _ = self._infer(raw, mask)
        out = raw.copy()
        methods = {}
        for i in np.flatnonzero(mask == 0):
            out[i] = recon[i] * self.lead_std[i]
            methods[int(i)] = "mae"
        return out, methods

    def predict(self, raw: np.ndarray, mask: np.ndarray) -> Prediction:
        recon, logits, attention = self._infer(raw, mask)
        completed = raw.copy()
        methods = {}
        for i in np.flatnonzero(mask == 0):
            completed[i] = recon[i] * self.lead_std[i]
            methods[int(i)] = "mae"

        filtered_obs, _ = preprocess(raw, mask)
        filtered_full, _ = preprocess(completed, np.ones(N_LEADS))
        st = measure_st(filtered_full, detect_r_peaks(filtered_obs, mask))

        exp = np.exp(logits - logits.max())
        softmax = exp / exp.sum()
        probs = dict(zip(CLASS_ORDER, softmax.tolist()))
        if attention is None:
            attention = np.where(mask > 0, 1.0 / max(1, mask.sum()), 0.0)

        best = max(WALLS, key=lambda w: probs[w])
        region = best if probs[best] > probs["normal"] else None
        confidence = probs[best] if region else probs["normal"]
        return Prediction(
            label=WALL_LABELS[region] if region else "Normal Sinüs Ritmi",
            region=region,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            wall_probabilities={w: probs[w] for w in WALLS},
            st_by_lead=st.tolist(),
            attention=np.asarray(attention, dtype=float).tolist(),
            reconstruction=methods,
            model_source=self.source,
        )


def load_backend():
    path = os.environ.get("MODEL_PATH")
    if path and Path(path).exists():
        return ExportedModel(Path(path))
    return PhysiologicalModel()
