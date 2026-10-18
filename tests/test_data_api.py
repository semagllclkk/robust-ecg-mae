"""
Veri hattı ve API testleri. data/processed ve eğitilmiş model gerektirir; yoksa atlanır.
Çalıştırma (proje kök klasöründen): python -m pytest -q
"""
import os

import numpy as np
import pandas as pd
import pytest

PROCESSED = "data/processed"
CHECKPOINT = "models/mae_ecg_checkpoint.pth"

needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(PROCESSED, "meta.csv")),
                                reason="data/processed yok (python -m src.prepare_data)")
needs_model = pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="eğitilmiş model yok")


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(os.path.join(PROCESSED, "meta.csv"))


@needs_data
def test_no_patient_in_multiple_splits(meta):
    assert (meta.groupby("patient_id")["split"].nunique() == 1).all()


@needs_data
def test_split_follows_strat_fold(meta):
    expected = np.select([meta.strat_fold <= 8, meta.strat_fold == 9], ["train", "val"], "test")
    assert (meta.split == expected).all()


@needs_data
def test_balanced_subsets(meta):
    bal = meta[meta.balanced == 1]
    for split, g in bal.groupby("split"):
        assert (g.label == 1).sum() == (g.label == 0).sum(), split
    assert (meta[meta.balanced == 1].label >= 0).all()


@needs_data
def test_label_rule(meta):
    sup = meta[["NORM", "MI", "STTC", "CD", "HYP"]]
    assert (meta.loc[meta.label == 1, "MI"] == 1).all()
    pure_norm = (meta.NORM == 1) & (sup.sum(axis=1) == 1)
    assert (meta.loc[meta.label == 0].index.isin(meta[pure_norm].index)).all()


@needs_data
def test_cache_is_zscored():
    X = np.load(os.path.join(PROCESSED, "X.npy"), mmap_mode="r")
    sample = np.asarray(X[:50])
    std = sample.std(axis=-1)
    assert np.allclose(sample.mean(axis=-1), 0, atol=1e-3)
    assert np.all((np.abs(std - 1) < 1e-3) | (std == 0))


@needs_data
@needs_model
def test_api_end_to_end():
    from fastapi.testclient import TestClient
    import src.main as m

    with TestClient(m.app) as c:
        a = c.get("/api/sample-ecg", params={"ecg_id": 16038, "dropped": "V1,V2"}).json()
        b = c.get("/api/sample-ecg", params={"ecg_id": 16038, "dropped": "V1,V2"}).json()
        assert a["diagnosis"] == b["diagnosis"]                         # deterministik
        assert a["dropped_lead_names"] == ["V1", "V2"]
        assert np.array(a["reconstructed_signals"]).shape == (12, 1000)
        raw, rec = np.array(a["raw_signals"]), np.array(a["reconstructed_signals"])
        assert np.allclose(np.delete(raw, [6, 7], 0), np.delete(rec, [6, 7], 0), atol=1e-4)  # sağlam kanallar aynen
        assert c.get("/api/sample-ecg", params={"dropped": "I,II,III,aVR,aVL"}).status_code == 400
        assert c.get("/api/sample-ecg", params={"condition": "xyz"}).status_code == 400
        assert c.get("/api/sample-ecg", params={"ecg_id": 99999999}).status_code == 404
        inf = c.get("/api/sample-ecg", params={"ecg_id": 16038, "dropped": "II,III,aVF"}).json()
        assert inf["einthoven_completed_leads"] == ["II", "III", "aVF"]
        raw = np.array(inf["raw_signals"])
        assert np.abs(raw[[1, 2, 5]] - np.array(inf["reconstructed_signals"])[[1, 2, 5]]).max() < 0.01
        pdf = c.get("/api/generate-report", params={"ecg_id": 16038, "dropped": "V1"})
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
