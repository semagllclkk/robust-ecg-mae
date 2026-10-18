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


@needs_data
@needs_model
def test_ui_contract_for_dashboard():
    """Sema'nın arayüzünün (src/services/ecgApi.js) beklediği sözleşme."""
    from fastapi.testclient import TestClient
    import src.main as m

    with TestClient(m.app) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "ok" and h["sampleRate"] == 100 and len(h["leads"]) == 12 and "MAE" in h["modelSource"]

        ch = c.post("/api/chunk", json={"start": 950, "count": 200, "condition": "ant", "disconnectedLeads": [6, 7]}).json()
        data = np.array(ch["data"])
        assert data.shape == (12, 200) and np.all(data[[6, 7]] == 0)          # kopuk kanal sıfır
        assert set(ch["reconstructed"]) == {"6", "7"} and len(ch["reconstructed"]["6"]) == 200
        assert np.abs(np.array(ch["reconstructed"]["6"])).max() > 0           # rekonstrüksiyon var
        again = c.post("/api/chunk", json={"start": 1950, "count": 200, "condition": "ant", "disconnectedLeads": [6, 7]}).json()
        assert again["data"] == ch["data"]                                    # 10 s'lik kayıt döngüyle akar

        p = c.post("/api/predict", json={"condition": "ant", "disconnectedLeads": [6, 7]}).json()
        for key in ["label", "region", "artery", "confidence", "stDeviation", "stByLead", "attention",
                    "wallProbabilities", "reconstruction", "modelSource"]:
            assert key in p, key
        assert len(p["stByLead"]) == 12 and all(isinstance(v, float) for v in p["stByLead"])
        assert abs(sum(p["attention"]) - 1) < 1e-3 and p["attention"][6] == 0 and p["attention"][7] == 0
        assert set(p["wallProbabilities"]) == {"anterior", "inferior", "lateral"}
        assert p["region"] in (None, "anterior", "inferior", "lateral")
        assert p["reconstruction"] == {"6": "mae", "7": "mae"}
        rep = p["report"]                                                     # rapor metinleri modelden
        assert rep["summary"] and rep["patient"] and rep["clinician"] and rep["confidenceNote"]
        assert any("V1" in t and "V2" in t for t in rep["clinician"])          # kopuk kanallar ve kaynakları
        assert (rep["warning"] is not None) == (p["confidence"] < 0.70)
        lim = c.post("/api/predict", json={"condition": "inf", "disconnectedLeads": [1, 2, 5]}).json()
        assert lim["reconstruction"] == {"1": "einthoven", "2": "einthoven", "5": "einthoven"}

        assert c.post("/api/predict", json={"condition": "xyz"}).status_code == 422
        assert c.post("/api/predict", json={"condition": "normal", "disconnectedLeads": [0, 1, 2, 3, 4]}).status_code == 422
        assert c.post("/api/chunk", json={"start": 0, "count": 10, "condition": "normal", "disconnectedLeads": [12]}).status_code == 422


@needs_data
@needs_model
def test_concurrent_requests_do_not_crash():
    """Arayüz chunk ve predict'i aynı anda gönderir; GPU'da eşzamanlı çıkarım çökmemeli (Metal hatası)."""
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    import src.main as m

    with TestClient(m.app) as c:
        def call(i):
            leads = [i % 12, (i + 5) % 12]
            if i % 2:
                return c.post("/api/predict", json={"condition": "inf", "disconnectedLeads": leads}).status_code
            return c.post("/api/chunk", json={"start": i * 100, "count": 200, "condition": "inf",
                                              "disconnectedLeads": leads}).status_code
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert set(pool.map(call, range(40))) == {200}


@needs_model
@pytest.mark.skipif(not os.path.exists("data/demo/meta.csv"), reason="data/demo yok")
def test_dashboard_works_with_demo_data_only():
    """Tam önbellek (data/processed, ~1 GB) olmayan bir bilgisayarda arayüz repodaki demo setiyle çalışmalı."""
    from fastapi.testclient import TestClient
    import src.main as m

    old, m.PROCESSED_DIR = m.PROCESSED_DIR, "data/__yok__"
    m.store.clear(), m.ui_records.clear(), m._ui_cache.clear()
    try:
        with TestClient(m.app) as c:
            assert "demo" in c.get("/api/health").json()["dataSource"]
            for cond in ["normal", "ant", "inf", "lat", "mi", "sttc"]:
                p = c.post("/api/predict", json={"condition": cond, "disconnectedLeads": [6, 7]})
                assert p.status_code == 200, cond
                ch = c.post("/api/chunk", json={"start": 0, "count": 100, "condition": cond, "disconnectedLeads": [6, 7]})
                assert ch.status_code == 200 and set(ch.json()["reconstructed"]) == {"6", "7"}
            assert c.get("/api/generate-report", params={"ecg_id": 16038, "dropped": "V1"}).status_code == 200
    finally:
        m.PROCESSED_DIR = old
        m.store.clear(), m.ui_records.clear(), m._ui_cache.clear()
