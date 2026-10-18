"""Duvar eşiği kalibrasyonu ve ONNX dışa aktarımındaki STFT eşdeğerliği (veri seti gerektirmez)."""
import numpy as np
import torch

from src.calibrate_walls import wall_thresholds
from src.ecg_constants import WALL_NAMES
from src.export_onnx import _ConvSTFT
from src.model import STFTEncoder


def test_rare_wall_gets_lower_threshold():
    rng = np.random.default_rng(0)
    n = 400
    labels = np.zeros((n, 3))
    probs = rng.uniform(0.0, 0.25, size=(n, 3))
    labels[:150, 0], probs[:150, 0] = 1, rng.uniform(0.6, 0.95, 150)   # Anterior: belirgin
    labels[150:300, 1], probs[150:300, 1] = 1, rng.uniform(0.6, 0.95, 150)  # Inferior: belirgin
    labels[300:330, 2], probs[300:330, 2] = 1, rng.uniform(0.3, 0.45, 30)   # Lateral: seyrek, düşük olasılık
    thresholds = wall_thresholds(labels, probs)
    assert set(thresholds) == set(WALL_NAMES)
    assert thresholds["Lateral"] < 0.5          # 0.5 eşiği lateral MI'ın hiçbirini yakalamazdı
    lateral = WALL_NAMES.index("Lateral")
    assert ((probs[:, lateral] >= thresholds["Lateral"]) == labels[:, lateral].astype(bool)).mean() > 0.95


def test_conv_stft_matches_original():
    torch.manual_seed(0)
    stft = STFTEncoder(dim=32).eval()
    x = torch.randn(6, 1000)
    with torch.no_grad():
        assert torch.allclose(stft(x), _ConvSTFT(stft)(x), atol=1e-4)


def test_wall_thresholds_bound_to_checkpoint(tmp_path):
    """Kalibre eşikler yalnızca hesaplandıkları checkpoint ile kullanılır; model değişirse 0.5'e dönülür."""
    import json
    from src.calibrate_walls import checkpoint_sha256, load_wall_thresholds

    ckpt, other = tmp_path / "model.pth", tmp_path / "other.pth"
    ckpt.write_bytes(b"model-a")
    other.write_bytes(b"model-b")
    cal = tmp_path / "wall_thresholds.json"
    cal.write_text(json.dumps({"checkpoint_sha256": checkpoint_sha256(ckpt),
                               "thresholds": {"Anterior": 0.5, "Inferior": 0.45, "Lateral": 0.3}}))
    assert load_wall_thresholds(cal, ckpt)["Lateral"] == 0.3
    assert load_wall_thresholds(cal, other) == {}             # başka model -> kalibrasyon yok sayılır
    assert load_wall_thresholds(tmp_path / "yok.json", ckpt) == {}
