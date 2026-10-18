"""
Çekirdek bileşenlerin birim testleri (veri veya eğitilmiş model gerektirmez, CPU'da saniyeler sürer).
Çalıştırma (proje kök klasöründen): python -m pytest -q
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.ecg_constants import WALL_LEAD_INDICES
from src.einthoven import einthoven_complete
from src.lead_dropout import LeadDropoutCollate, apply_lead_mask, random_keep_mask, wall_keep_mask
from src.losses import binary_focal_loss, masked_mse
from src.metrics import ReconstructionMeter, best_f1_threshold
from src.model import ECGMaskedAutoencoder
from src.preprocessing import butter_bandpass_filter, preprocess_ecg, zscore_normalize

FS = 100
T = np.arange(1000) / FS


# --- İP1: ön işleme (Rapor Md. 2.1) -----------------------------------------------------------------

def _gain(freq_hz):
    x = np.sin(2 * np.pi * freq_hz * T)
    y = butter_bandpass_filter(x, fs=FS)
    return y[200:800].std() / x[200:800].std()


def test_bandpass_passes_ecg_band_and_rejects_outside():
    assert _gain(10.0) > 0.95          # QRS/T bandı geçer
    assert _gain(0.05) < 0.05          # taban kayması bastırılır
    assert _gain(48.0) < 0.3           # 45 Hz üstü bastırılır


def test_bandpass_is_zero_phase():
    x = np.sin(2 * np.pi * 5.0 * T)
    y = butter_bandpass_filter(x, fs=FS)
    lag = np.argmax(np.correlate(y[200:800], x[200:800], mode="full")) - 599
    assert lag == 0                    # çift yönlü: faz kayması yok


def test_zscore_per_lead_and_flat_lead():
    x = np.stack([3 * np.sin(2 * np.pi * T) + 2, np.zeros_like(T)])
    z, mu, sigma = zscore_normalize(x)
    assert abs(z[0].mean()) < 1e-6 and abs(z[0].std() - 1) < 1e-6
    assert np.all(z[1] == 0) and sigma[1] == 0   # düz derivasyon sıfıra çekilir


def test_preprocess_output_shape_and_dtype():
    z, mu, sigma = preprocess_ecg(np.random.randn(12, 1000))
    assert z.shape == (12, 1000) and z.dtype == np.float32 and mu.shape == sigma.shape == (12,)


# --- İP3: LeadDropout (Rapor Md. 2.2) ---------------------------------------------------------------

def test_random_keep_mask_k_per_batch_in_range():
    g = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(200):
        M = random_keep_mask(32, generator=g)
        k = 12 - M.sum(1)
        assert torch.all(k == k[0])                  # batch içinde tek k
        seen.add(int(k[0]))
    assert seen == {1, 2, 3, 4}                      # k ∈ [1, 4]


def test_critical_mask_drops_one_wall():
    M = random_keep_mask(500, n_drop=3, critical_prob=1.0, generator=torch.Generator().manual_seed(1))
    walls = [set(v) for v in WALL_LEAD_INDICES.values()]
    for row in M:
        dropped = set(torch.nonzero(row < 0.5).flatten().tolist())
        assert any(dropped <= w for w in walls)


def test_collate_applies_x_times_m():
    batch = [(torch.randn(12, 1000), {"mi": torch.tensor(1.0)}) for _ in range(8)]
    x_masked, x, M, _ = LeadDropoutCollate()(batch)
    assert torch.equal(x_masked, x * M.unsqueeze(-1))
    assert torch.equal(apply_lead_mask(x, M), x_masked)


def test_collate_rejects_out_of_report_range():
    with pytest.raises(ValueError):
        LeadDropoutCollate(min_drop=1, max_drop=5)


# --- İP3: model (Rapor Md. 2.3) ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ECGMaskedAutoencoder().eval()


def test_model_output_shapes(model):
    out = model(torch.randn(2, 12, 1000), torch.ones(2, 12))
    assert out["recon"].shape == (2, 12, 1000)
    assert out["mi_logit"].shape == (2,) and out["wall_logits"].shape == (2, 3)
    assert out["spatial_attention"].shape == (2, 12, 12) and out["lead_importance"].shape == (2, 12)


def test_no_leakage_from_dropped_leads(model):
    x = torch.randn(3, 12, 1000)
    M = wall_keep_mask(3, "Anterior")
    x2 = x.clone()
    x2[M < 0.5] = 50.0                                # kopan kanalın içeriği ne olursa olsun
    with torch.no_grad():
        a, b = model(x, M), model(x2, M)
    for k in ["recon", "mi_logit", "wall_logits", "spatial_attention", "lead_importance"]:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_spatial_attention_ignores_dropped_keys(model):
    M = wall_keep_mask(2, "Inferior")
    with torch.no_grad():
        out = model(torch.randn(2, 12, 1000), M)
    att = out["spatial_attention"]
    assert torch.allclose(att.sum(-1), torch.ones(2, 12), atol=1e-5)
    assert torch.all(att.transpose(1, 2)[M < 0.5] == 0)          # kopan kanallara dikkat yok
    assert torch.all(out["lead_importance"][M < 0.5] == 0)
    assert torch.allclose(out["lead_importance"].sum(1), torch.ones(2), atol=1e-5)


# --- Risk 1 B planı: Einthoven ----------------------------------------------------------------------

def _limb_from(I, II):
    return torch.stack([I, II, II - I, -(I + II) / 2, I - II / 2, II - I / 2])


@pytest.mark.parametrize("dropped", [[1, 2, 5], [0, 1, 2, 3], [2, 3, 4, 5], [0, 4]])
def test_einthoven_recovers_dropped_limb_leads(dropped):
    I, II = torch.randn(1000), torch.randn(1000)
    sig = torch.cat([_limb_from(I, II), torch.randn(6, 1000)]).unsqueeze(0)
    M = torch.ones(1, 12)
    M[0, dropped] = 0
    corrupted = sig.clone()
    corrupted[0, dropped] = 99.0                      # kopan içerik kullanılmamalı
    out = einthoven_complete(corrupted, M)
    assert torch.allclose(out, sig[:, :6], atol=1e-4)


# --- Kayıp (Rapor Md. 2.4) ---------------------------------------------------------------------------

def test_masked_mse_only_counts_dropped_leads():
    target = torch.zeros(2, 12, 100)
    recon = torch.zeros(2, 12, 100)
    recon[:, 0] = 2.0                                 # yalnızca kopan kanalda hata
    recon[:, 5] = 100.0                               # sağlam kanaldaki hata sayılmamalı
    dropped = torch.zeros(2, 12, dtype=torch.bool)
    dropped[:, 0] = True
    assert masked_mse(recon, target, dropped).item() == pytest.approx(4.0)
    assert masked_mse(recon, target, torch.zeros_like(dropped)).item() == 0.0


def test_focal_reduces_to_bce_and_downweights_easy():
    logits, y = torch.randn(64), (torch.rand(64) > 0.5).float()
    assert torch.allclose(binary_focal_loss(logits, y, gamma=0.0), F.binary_cross_entropy_with_logits(logits, y))
    easy = binary_focal_loss(torch.tensor([5.0]), torch.tensor([1.0]), gamma=2.0)
    hard = binary_focal_loss(torch.tensor([-1.0]), torch.tensor([1.0]), gamma=2.0)
    assert easy < 1e-4 < hard


# --- Metrikler ---------------------------------------------------------------------------------------

def test_reconstruction_meter_perfect_and_scaled():
    true = torch.randn(4, 12, 200)
    dropped = torch.zeros(4, 12, dtype=torch.bool)
    dropped[:, 3] = True
    m = ReconstructionMeter()
    m.update(true.clone(), true, dropped)
    r = m.result()
    assert r["rmse_mv"] == pytest.approx(0, abs=1e-6) and r["r2"] == pytest.approx(1) and r["pearson_r"] == pytest.approx(1)
    m2 = ReconstructionMeter()
    m2.update(true * 0.5, true, dropped)              # şekil doğru, genlik yanlış
    r2 = m2.result()
    assert r2["pearson_r"] == pytest.approx(1, abs=1e-5) and r2["r2"] < 0.8


def test_best_f1_threshold_separable():
    labels = np.array([0] * 50 + [1] * 50)
    probs = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.6, 1.0, 50)])
    assert 0.3 <= best_f1_threshold(labels, probs) <= 0.6


# --- Genlik farkındalıklı model ----------------------------------------------------------------------

def test_amplitude_aware_no_leakage_of_dropped_amplitude():
    torch.manual_seed(0)
    m = ECGMaskedAutoencoder(amplitude_aware=True, pop_sigma=[0.1] * 12).eval()
    x, sigma = torch.randn(2, 12, 1000), torch.rand(2, 12) * 0.3 + 0.05
    M = wall_keep_mask(2, "Lateral")
    x2, sigma2 = x.clone(), sigma.clone()
    x2[M < 0.5], sigma2[M < 0.5] = 50.0, 9.0         # kopan kanalın içeriği ve genliği ne olursa olsun
    with torch.no_grad():
        a, b = m(x, M, sigma=sigma), m(x2, M, sigma=sigma2)
    for k in ["recon", "mi_logit", "wall_logits", "lead_importance"]:
        assert torch.allclose(a[k], b[k], atol=1e-5), k
    with torch.no_grad():                            # sağlam kanalın genliği ise çıktıyı etkilemeli
        sigma3 = sigma.clone()
        sigma3[:, 1] *= 3
        c = m(x, M, sigma=sigma3)
    assert not torch.allclose(a["recon"], c["recon"])


def test_amplitude_aware_requires_sigma():
    m = ECGMaskedAutoencoder(amplitude_aware=True, pop_sigma=[0.1] * 12)
    with pytest.raises(ValueError):
        m(torch.randn(1, 12, 1000), torch.ones(1, 12))
