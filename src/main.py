from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from collections import OrderedDict
from scipy.signal import resample
import torch
import numpy as np
import pandas as pd
import os
import io
import random
import threading
import time

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.ecg_constants import (FS, LEAD_NAMES, SEQ_LEN, WALL_GROUPS, WALL_LEAD_INDICES, WALL_NAMES,
                               WALL_TO_ARTERY, WALL_TO_SUBCLASS)
from src.model import ECGMaskedAutoencoder
from src.clinical_report import build_clinical_pdf
from src.einthoven import einthoven_complete
from src.preprocessing import preprocess_ecg
from src.st_analysis import MV_PER_MM, st_deviation_mv
from src.train import load_checkpoint

# --------------------------------------------------------------------------
# 1. Uygulama ve CORS Yapılandırması
# --------------------------------------------------------------------------
app = FastAPI(
    title="12-Lead ECG MAE Anomaly Detection & XAI API",
    description="Klinik Karar Destek Sistemi - Eksik Kanal Toleransı ve Anatomik XAI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# 2. Donanım ve Model Kurulumu
# --------------------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_PATH = "models/mae_ecg_checkpoint.pth"
PROCESSED_DIR = "data/processed"
WALL_THRESHOLD = 0.5  # kriz tipi: P(duvar | MI) bu eşiği geçen duvarlar etkilenmiş sayılır

model, MODEL_LOADED, DECISION_THRESHOLD = ECGMaskedAutoencoder().to(device).eval(), False, 0.5
MODEL_SOURCE = "MAE (eğitilmemiş)"
if os.path.exists(MODEL_PATH):
    try:
        model, _ckpt = load_checkpoint(MODEL_PATH, device)
        MODEL_LOADED, DECISION_THRESHOLD = True, float(_ckpt.get("threshold", 0.5))
        MODEL_SOURCE = f"MAE (PTB-XL, epoch {_ckpt['epoch']})"
        print(f"✓ EKG MAE Modeli yüklendi (epoch {_ckpt['epoch']}, karar eşiği {DECISION_THRESHOLD:.2f})")
    except Exception as e:
        print(f"⚠ Uyarı: Checkpoint yeni model mimarisiyle uyumsuz ({type(e).__name__}). "
              "`python -m src.train` ile yeniden eğitin. Eğitilmemiş ağırlıklarla (mock) çalışılıyor.")
else:
    print("⚠ Uyarı: Model ağırlık dosyası bulunamadı. Eğitilmemiş ağırlıklarla (mock) çalışılıyor.")

# Anatomik Duvar ve Lead Eşleşme Sözlüğü (TÜBİTAK Raporu Md 2.5)
ANATOMICAL_MAPPING = {
    "V1": "Septal / Anterior (Ön Duvar)",
    "V2": "Septal / Anterior (Ön Duvar)",
    "V3": "Anterior (Ön Duvar)",
    "V4": "Anterior (Ön Duvar)",
    "I": "Lateral (Yan Duvar)",
    "aVL": "High Lateral (Yüksek Yan Duvar)",
    "V5": "Lateral (Yan Duvar)",
    "V6": "Lateral (Yan Duvar)",
    "II": "Inferior (Alt Duvar)",
    "III": "Inferior (Alt Duvar)",
    "aVF": "Inferior (Alt Duvar)",
    "aVR": "Kavite / Sağ Ventrikül"
}

# --------------------------------------------------------------------------
# 3. Veri: önişlenmiş PTB-XL test seti (strat_fold 10 — model eğitimde hiç görmedi)
# --------------------------------------------------------------------------
store = {}

# Senaryo -> meta.csv filtresi
CONDITIONS = {
    "normal": lambda m: m["label"] == 0,
    "mi": lambda m: m["label"] == 1,
    "ant": lambda m: (m["label"] == 1) & (m["AMI"] == 1),
    "inf": lambda m: (m["label"] == 1) & (m["IMI"] == 1),
    "lat": lambda m: (m["label"] == 1) & (m["LMI"] == 1),
    "sttc": lambda m: (m["label"] == -1) & (m["STTC"] == 1),
}


@app.on_event("startup")
def load_dataset():
    # İlk çağrıdaki GPU (MPS/CUDA) çekirdek derleme gecikmesini kullanıcı isteğinden önce öde
    with torch.no_grad():
        model(torch.zeros(1, 12, SEQ_LEN, device=device), torch.ones(1, 12, device=device),
              **({"sigma": torch.ones(1, 12, device=device)} if getattr(model, "amplitude_aware", False) else {}))
    try:
        meta = pd.read_csv(os.path.join(PROCESSED_DIR, "meta.csv"), index_col="ecg_id")
        sigma = np.load(os.path.join(PROCESSED_DIR, "sigma.npy"))
        store["meta"] = meta.reset_index()
        store["X"] = np.load(os.path.join(PROCESSED_DIR, "X.npy"), mmap_mode="r")
        store["mu"] = np.load(os.path.join(PROCESSED_DIR, "mu.npy"))
        store["sigma"] = sigma
        # Kopan kanalın gerçek genliği bilinemez: mV'a çevirirken eğitim popülasyonunun medyan std'si kullanılır
        store["pop_sigma"] = np.median(sigma[(meta["split"] == "train").to_numpy()], axis=0)
        test = store["meta"]["split"] == "test"
        store["pools"] = {c: np.flatnonzero((test & f(store["meta"])).to_numpy()) for c, f in CONDITIONS.items()}
        print(f"✓ Önişlenmiş PTB-XL test seti yüklendi: { {c: len(v) for c, v in store['pools'].items()} }")
    except FileNotFoundError:
        print("⚠ data/processed bulunamadı. `python -m src.prepare_data` çalıştırın. "
              "/api/sample-ecg kullanılamaz, /api/predict çalışır.")


# --------------------------------------------------------------------------
# 4. Çıkarım, Dinamik Maskeleme ve Click-to-Explain
# --------------------------------------------------------------------------
def resolve_dropped(active_lead_count: int, dropped_leads):
    # TÜBİTAK Proje Kısıtı: En fazla 4 kanal kopması tolere edilir (En az 8 aktif kanal)
    if dropped_leads:
        unknown = [l for l in dropped_leads if l not in LEAD_NAMES]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Bilinmeyen derivasyon: {unknown}")
        disabled = sorted({LEAD_NAMES.index(l) for l in dropped_leads})
    elif active_lead_count < 12:
        disabled = sorted(random.sample(range(12), 12 - active_lead_count))
    else:
        disabled = []
    if len(disabled) > 4:
        raise HTTPException(
            status_code=400,
            detail=f"Sistem güvenlik sınırı: En az 8 aktif kanal gereklidir. Aktif kalan kanal sayısı: {12 - len(disabled)}"
        )
    return disabled


# FastAPI eşzamanlı istekleri ayrı iş parçacıklarında çalıştırır; Apple GPU'da (MPS/Metal) iki iş parçacığının
# modeli aynı anda çalıştırması Metal'i çökertir. Tüm GPU çalışması bu kilitle sıraya alınır (~20–30 ms/çağrı).
_gpu_lock = threading.RLock()


def run_inference(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray, disabled: list):
    with _gpu_lock:
        return _run_inference(z, mu, sigma, disabled)


def _run_inference(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray, disabled: list):
    """
    z      : (12, 1000) filtrelenmiş + z-score sinyal (kopan kanalların içeriği kullanılmaz)
    mu/sigma: (12,) mV'a dönüş istatistikleri (kopan kanallar için popülasyon değeri verilmeli)
    """
    # Rapor Md. 2.2: M ∈ {0,1}^12 (1 = sağlam, 0 = koptu), X_masked = X ⊙ M
    M = torch.ones(1, 12, device=device)
    M[0, disabled] = 0.0
    x = torch.tensor(np.asarray(z, dtype=np.float32), device=device).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        extra = {"sigma": torch.tensor(np.asarray(sigma, dtype=np.float32), device=device).unsqueeze(0)} \
            if getattr(model, "amplitude_aware", False) else {}
        out = model(x * M.unsqueeze(-1), M, **extra)  # kopan kanalların σ'sı modelde maskelenir
    if device.type == "mps":
        torch.mps.synchronize()
    inference_ms = (time.perf_counter() - t0) * 1000

    recon_z = out["recon"][0].float().cpu().numpy()
    if getattr(model, "physical_recon", False):
        # Fiziksel ölçekli çıktı (mV / σ_pop) -> bu fonksiyonun z uzayına (kopanlar için σ = σ_pop, μ = 0)
        pop = model.pop_sigma.cpu().numpy()[:, None]
        recon_z = (recon_z * pop - mu[:, None]) / np.maximum(sigma[:, None], 1e-6)
    mi_prob = float(torch.sigmoid(out["mi_logit"]).item())
    wall_prob = torch.sigmoid(out["wall_logits"][0]).float().cpu().numpy()
    # Md. 2.5: XAI, Spatial Attention katmanının dikkat ağırlıklarından
    lead_attn = out["lead_importance"][0].float().cpu().numpy()
    spatial = out["spatial_attention"][0].float().cpu().numpy()  # (sorgu, anahtar)
    reconstruction_sources = {
        LEAD_NAMES[i]: {LEAD_NAMES[j]: float(spatial[i, j]) for j in np.argsort(-spatial[i])[:3]}
        for i in disabled
    }

    # Ekranda gösterilecek 12 kanal (mV): sağlam kanallar orijinal, kopanlar MAE rekonstrüksiyonu
    z_filled = np.array(z, dtype=np.float32, copy=True)
    z_filled[disabled] = recon_z[disabled]
    filled_mv = z_filled * sigma[:, None] + mu[:, None]
    # Risk 1 B planı: kopan ekstremite derivasyonları sağlam olanlardan Einthoven bağıntılarıyla kesin hesaplanır
    limb_dropped = [i for i in disabled if i < 6]
    if limb_dropped:
        limb = einthoven_complete(torch.from_numpy(filled_mv).float().unsqueeze(0), M.cpu())[0].numpy()
        filled_mv[limb_dropped] = limb[limb_dropped]

    is_mi = mi_prob >= DECISION_THRESHOLD
    confidence = mi_prob if is_mi else 1.0 - mi_prob
    damage = {w: mi_prob * float(wall_prob[i]) for i, w in enumerate(WALL_NAMES)}
    top_wall = max(damage, key=damage.get)
    # Birden fazla duvar etkilenmiş olabilir: P(duvar | MI) ≥ 0.5 olan tüm duvarlar (yoksa en olası duvar)
    affected_walls = [w for i, w in enumerate(WALL_NAMES) if wall_prob[i] >= WALL_THRESHOLD] or [top_wall]
    if not is_mi:
        affected_walls = []

    st_mv, st_ref = st_deviation_mv(filled_mv, disabled)
    most_idx = int(np.argmax(lead_attn))
    most_lead = LEAD_NAMES[most_idx]

    click_to_explain = {}
    for i, w in enumerate(WALL_NAMES):
        idxs = WALL_LEAD_INDICES[w]
        missing = [LEAD_NAMES[j] for j in idxs if j in disabled]
        st_leads = {LEAD_NAMES[j]: (None if np.isnan(st_mv[j]) else round(float(st_mv[j]) / MV_PER_MM, 2)) for j in idxs}
        valid = [v for v in st_leads.values() if v is not None]
        max_st = max(valid, key=abs) if valid else None
        if w in affected_walls:
            st_text = f"En belirgin ST sapması {max_st:+.1f} mm. " if max_st is not None else ""
            text = (f"{w} duvarında MI şüphesi (olasılık %{100 * damage[w]:.0f}). "
                    f"{st_text}Şüpheli arter: {WALL_TO_ARTERY[w]}.")
        else:
            text = f"{w} duvarında belirgin MI bulgusu saptanmadı (hasar skoru %{100 * damage[w]:.0f})."
        if missing:
            text += f" {', '.join(missing)} kopuk; bu kanallar sağlam kanallardan rekonstrükte edildi."
        click_to_explain[w] = {
            "leads": WALL_GROUPS[w],
            "affected": w in affected_walls,
            "damage_score": damage[w],
            "wall_probability_given_mi": float(wall_prob[i]),
            "attention_share": float(lead_attn[idxs].sum()),
            "st_deviation_mm": st_leads,
            "max_st_deviation_mm": max_st,
            "suspected_artery": WALL_TO_ARTERY[w],
            "missing_leads": missing,
            "decision_confidence": confidence,
            "explanation": text,
        }

    mse_per_lead = np.mean((recon_z - np.asarray(z)) ** 2, axis=1)
    rec_err = float(mse_per_lead[disabled].mean()) if disabled else float(mse_per_lead.mean())

    if is_mi:
        walls_text = " + ".join(affected_walls)
        predicted = f"{walls_text} Myocardial Infarction ({walls_text} MI)"
    else:
        predicted = "Normal (MI bulgusu yok)"

    return {
        "model_loaded": MODEL_LOADED,
        "units": "mV",
        "einthoven_completed_leads": [LEAD_NAMES[i] for i in limb_dropped],
        "lead_names": LEAD_NAMES,
        "active_lead_count": 12 - len(disabled),
        "dropped_lead_names": [LEAD_NAMES[i] for i in disabled],
        "reconstructed_signals": filled_mv.tolist(),
        "anomaly_score": rec_err,
        "reconstruction_error_dropped": rec_err if disabled else None,
        "mse_per_lead": dict(zip(LEAD_NAMES, mse_per_lead.tolist())),
        "mi_probability": mi_prob,
        "wall_damage_scores": damage,
        "lead_attention": dict(zip(LEAD_NAMES, lead_attn.tolist())),
        "spatial_attention": spatial.tolist(),
        "reconstruction_sources": reconstruction_sources,
        "st_deviation_mm": {LEAD_NAMES[j]: (None if np.isnan(v) else round(float(v) / MV_PER_MM, 2))
                            for j, v in enumerate(st_mv)},
        "st_reference_lead": st_ref,
        "click_to_explain": click_to_explain,
        "diagnosis": {
            "predicted_class": predicted,
            "confidence": confidence,
            "is_mi": is_mi,
            "affected_wall": top_wall if is_mi else None,
            "affected_walls": affected_walls,
        },
        "xai_metrics": {
            "most_affected_lead": most_lead,
            "anatomical_region": ANATOMICAL_MAPPING.get(most_lead, "Genel Kardiyak"),
            "max_lead_attention": float(lead_attn[most_idx]),
            "threshold": DECISION_THRESHOLD,
            "is_anomaly": is_mi,
        },
        "inference_ms": round(inference_ms, 1),
    }


# --------------------------------------------------------------------------
# 5. API Endpoints
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "12-Lead ECG MAE & XAI Backend Servisi Aktif", "model_loaded": MODEL_LOADED}


def analyze_row(row: int, disabled: list):
    """Önişlenmiş veri setindeki bir kaydı verilen kopma maskesiyle analiz eder (deterministik)."""
    rec = store["meta"].iloc[row]
    z = np.asarray(store["X"][row], dtype=np.float32)
    mu, sigma = store["mu"][row], store["sigma"][row]

    # Kopan kanalın gerçek genliği/ofseti bilinmiyormuş gibi davran (gerçek saha koşulu)
    mu_inf, sigma_inf = mu.copy(), sigma.copy()
    mu_inf[disabled], sigma_inf[disabled] = 0.0, store["pop_sigma"][disabled]

    result = run_inference(z, mu_inf, sigma_inf, disabled)
    result["raw_signals"] = (z * sigma[:, None] + mu[:, None]).tolist()  # orijinal (filtreli) sinyal, kıyas için
    result["source"] = "PTB-XL Test Seti (strat_fold 10)" if rec["split"] == "test" else f"PTB-XL ({rec['split']})"
    result["patient_index"] = int(rec["ecg_id"])
    result["ecg_id"] = int(rec["ecg_id"])
    walls_true = [w for w in WALL_NAMES if rec[WALL_TO_SUBCLASS[w]] == 1]
    result["ground_truth"] = {
        "label": {1: "MI", 0: "Normal", -1: "Diğer (MI/Normal dışı)"}[int(rec["label"])],
        "mi_walls": walls_true,
    }
    return result


def find_row(ecg_id: int):
    rows = np.flatnonzero(store["meta"]["ecg_id"].to_numpy() == ecg_id)
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail=f"ecg_id {ecg_id} bulunamadı")
    return int(rows[0])


def parse_dropped(dropped):
    return [d.strip() for d in dropped.split(",") if d.strip()] if dropped else None


@app.get("/api/sample-ecg")
def get_sample_ecg(
    active_leads: int = Query(12, ge=8, le=12, description="Aktif kanal sayısı (8-12)"),
    dropped: str = Query(None, description="Virgülle ayrılmış kapatılacak kanallar (Örn: V1,V2)"),
    condition: str = Query("normal", description="Klinik senaryo: normal, mi, ant, inf, lat, sttc"),
    ecg_id: int = Query(None, description="Belirli bir kaydı analiz et (verilirse condition yok sayılır)"),
):
    if "pools" not in store:
        raise HTTPException(status_code=500, detail="Önişlenmiş veri seti yok. `python -m src.prepare_data` çalıştırın.")
    target_condition = condition.lower()
    if ecg_id is not None:
        row = find_row(ecg_id)
    else:
        if target_condition not in CONDITIONS:
            raise HTTPException(status_code=400, detail=f"Geçersiz senaryo '{condition}'. Seçenekler: {list(CONDITIONS)}")
        row = int(np.random.choice(store["pools"][target_condition]))

    disabled = resolve_dropped(active_leads, parse_dropped(dropped))
    result = analyze_row(row, disabled)
    result["selected_condition"] = target_condition
    return result


# --------------------------------------------------------------------------
# 6. Arayüz sözleşmesi (Sema'nın dashboard'u, İP5): /api/health, /api/chunk, /api/predict
#    Sinyal: senaryoya göre seçilen gerçek PTB-XL test kaydı (10 s, döngüyle akar).
#    Teşhis/XAI: eğitilmiş çift başlıklı MAE (yukarıdaki run_inference ile aynı çıktı).
# --------------------------------------------------------------------------
UI_WALL = {w: w.lower() for w in WALL_NAMES}            # "Anterior" -> "anterior"
MAX_DISCONNECTED = 4
ui_records = {}                                          # senaryo -> seçili kayıt satırı
_ui_cache = OrderedDict()                                # (satır, kopuk kanallar) -> analiz


class UILeadState(BaseModel):
    condition: str = "normal"
    disconnectedLeads: list[int] = Field(default_factory=list)

    @field_validator("condition")
    @classmethod
    def known_condition(cls, value):
        if value not in CONDITIONS:
            raise ValueError(f"Bilinmeyen EKG senaryosu: {value}")
        return value

    @field_validator("disconnectedLeads")
    @classmethod
    def valid_leads(cls, value):
        value = sorted(set(value))
        if any(i < 0 or i >= 12 for i in value):
            raise ValueError("Derivasyon indeksi 0–11 aralığında olmalı.")
        if len(value) > MAX_DISCONNECTED:
            raise ValueError(f"En fazla {MAX_DISCONNECTED} derivasyon koparılabilir.")
        return value


class UIChunkRequest(UILeadState):
    start: int = Field(ge=0)
    count: int = Field(gt=0, le=2000)


def ui_record_row(condition: str, new: bool = False) -> int:
    """Senaryo için gösterilen kayıt; sunucu açık kaldıkça aynı kalır (akış ve teşhis tutarlı olsun)."""
    if "pools" not in store:
        raise HTTPException(status_code=500, detail="Önişlenmiş veri seti yok. `python -m src.prepare_data` çalıştırın.")
    if new or condition not in ui_records:
        ui_records[condition] = int(np.random.choice(store["pools"][condition]))
    return ui_records[condition]


def ui_analysis(condition: str, disconnected: list[int]) -> dict:
    with _gpu_lock:  # önbellek de aynı kilitle: aynı analiz iki kez hesaplanmasın
        return _ui_analysis(condition, disconnected)


def _ui_analysis(condition: str, disconnected: list[int]) -> dict:
    key = (ui_record_row(condition), tuple(disconnected))
    if key not in _ui_cache:
        _ui_cache[key] = analyze_row(key[0], list(disconnected))
        if len(_ui_cache) > 64:
            _ui_cache.popitem(last=False)
    return _ui_cache[key]


@app.get("/api/health")
def health():
    return {"status": "ok", "modelSource": MODEL_SOURCE, "modelLoaded": MODEL_LOADED,
            "sampleRate": FS, "leads": LEAD_NAMES}


@app.post("/api/chunk")
def ui_chunk(req: UIChunkRequest):
    """Canlı akış: kaydın [start, start+count) aralığı (10 s'lik kayıt döngüyle tekrarlanır) + kopuk kanalların rekonstrüksiyonu."""
    r = ui_analysis(req.condition, req.disconnectedLeads)
    idx = (req.start + np.arange(req.count)) % SEQ_LEN
    data = np.asarray(r["raw_signals"])[:, idx]
    data[req.disconnectedLeads] = 0.0                    # kopuk kablodan sinyal gelmez
    rec = np.asarray(r["reconstructed_signals"])[:, idx]
    return {
        "start": req.start,
        "data": np.round(data, 4).tolist(),
        "reconstructed": {str(i): np.round(rec[i], 4).tolist() for i in req.disconnectedLeads},
    }


@app.post("/api/predict")
def ui_predict(req: UILeadState):
    """Teşhis, duvar olasılıkları, derivasyon başına ST (mm), Spatial Attention ağırlıkları."""
    started = time.perf_counter()
    r = ui_analysis(req.condition, req.disconnectedLeads)
    diag = r["diagnosis"]
    walls = diag.get("affected_walls") or []
    region = UI_WALL[diag["affected_wall"]] if diag["is_mi"] else None
    st = [float(r["st_deviation_mm"][name] or 0.0) for name in LEAD_NAMES]
    st_region = max(st[i] for i in WALL_LEAD_INDICES[diag["affected_wall"]]) if region else 0.0
    return {
        "label": f"{' + '.join(walls)} MI" if diag["is_mi"] else "Normal (MI bulgusu yok)",
        "region": region,
        "regions": [UI_WALL[w] for w in walls],
        "artery": WALL_TO_ARTERY[diag["affected_wall"]] if region else None,
        "confidence": round(float(diag["confidence"]), 4),
        "miProbability": round(float(r["mi_probability"]), 4),
        "stDeviation": round(st_region, 2),
        "stByLead": [round(v, 2) for v in st],
        "attention": [round(float(r["lead_attention"][name]), 4) for name in LEAD_NAMES],
        "wallProbabilities": {UI_WALL[w]: round(float(v), 4) for w, v in r["wall_damage_scores"].items()},
        "reconstruction": {str(i): ("einthoven" if LEAD_NAMES[i] in r["einthoven_completed_leads"] else "mae")
                           for i in req.disconnectedLeads},
        "reconstructionSources": r["reconstruction_sources"],
        "modelSource": MODEL_SOURCE,
        "ecgId": r["ecg_id"],
        "groundTruth": r["ground_truth"],
        "disconnectedLeads": req.disconnectedLeads,
        "latencyMs": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/api/new-record")
def ui_new_record(req: UILeadState):
    """Aynı senaryodan başka bir hasta kaydına geç."""
    row = ui_record_row(req.condition, new=True)
    return {"condition": req.condition, "ecgId": int(store["meta"].iloc[row]["ecg_id"])}


class PredictRequest(BaseModel):
    signal: list            # 12 x N (mV) — N x 12 de kabul edilir
    fs: int = FS            # örnekleme frekansı (Hz)
    missing_leads: list = []  # gerçekte kopuk derivasyon adları, örn. ["V1", "V2"]


@app.post("/api/predict-signal")
def predict_signal(req: PredictRequest):
    sig = np.asarray(req.signal, dtype=np.float64)
    if sig.ndim != 2 or 12 not in sig.shape:
        raise HTTPException(status_code=400, detail="signal 12 x N (veya N x 12) boyutunda olmalı")
    if sig.shape[0] != 12:
        sig = sig.T
    if req.fs != FS:
        sig = resample(sig, int(round(sig.shape[1] * FS / req.fs)), axis=1)
    if sig.shape[1] < SEQ_LEN:
        sig = np.pad(sig, ((0, 0), (0, SEQ_LEN - sig.shape[1])))
    sig = sig[:, :SEQ_LEN]
    if not np.all(np.isfinite(sig)):
        raise HTTPException(status_code=400, detail="signal NaN/Inf içeremez")

    disabled = resolve_dropped(12, req.missing_leads)
    sig[disabled] = 0.0
    z, mu, sigma = preprocess_ecg(sig, fs=FS)
    if "pop_sigma" in store:
        sigma[disabled] = store["pop_sigma"][disabled]
    result = run_inference(z, mu, sigma, disabled)
    result["raw_signals"] = (z * sigma[:, None] + mu[:, None]).tolist()
    result["source"] = "Kullanıcı yüklemesi"
    return result


@app.get("/api/generate-report")
def generate_pdf_report(
    patient_id: str = "PTB-XL-0842",
    anomaly_score: float = 0.025,
    diagnosis: str = "Anterior Myocardial Infarction",
    confidence: float = 0.96,
    active_leads: int = 12,
    dropped_leads: str = "",
    most_affected_lead: str = "V2",
    anatomical_region: str = "On Duvar (Anterior)",
    ecg_id: int = Query(None, description="Rapor bu kaydın gerçek model analizinden üretilir (önerilen)"),
    dropped: str = Query("", description="Analizdeki kopuk kanallar, örn. V1,V2 (ecg_id ile birlikte)"),
):
    # Önerilen yol: ekrandaki analizle aynı kayıt ve aynı kopmalar yeniden analiz edilir;
    # PDF tamamen model çıktısından üretilir (sabit metin veya arayüzden gelen değer kullanılmaz).
    if ecg_id is not None:
        if "pools" not in store:
            raise HTTPException(status_code=500, detail="Önişlenmiş veri seti yok.")
        result = analyze_row(find_row(ecg_id), resolve_dropped(12, parse_dropped(dropped)))
        return StreamingResponse(build_clinical_pdf(result), media_type="application/pdf",
                                 headers={"Content-Disposition": f"inline; filename=ecg_report_{ecg_id}.pdf"})

    # Eski yol (geriye dönük uyumluluk): arayüzden gelen parametrelerle şablon rapor.
    # Senaryo / Tanıya göre klinik parametreleri ve hekim yorumlarını dinamik belirle
    diag_lower = diagnosis.lower()
    
    if "inferior" in diag_lower or "alt" in diag_lower:
        diagnosis_title = "Inferior Myocardial Infarction"
        affected_lead = "II, III, aVF"
        anat_region = "Alt Duvar (Inferior Miyokard)"
        clinical_note = "Lead II, III ve aVF uzerindeki ST elevasyonu ve rekonstruksiyon sapmalari,"
        clinical_subnote = "iskemik surecin oncelikli olarak kalbin Alt (Inferior) Duvarina odaklandigini gostermektedir."
        patient_note = "Inceleme sonucunda, kalbinizin 'Alt Duvar' bolgesine ait elektriksel iletide hekim kontrolu gerektirebilecek olasi bir bulgu izlenmistir."
    elif "anterior" in diag_lower or "on" in diag_lower:
        diagnosis_title = "Anterior Myocardial Infarction"
        affected_lead = "V2, V3 (V1-V4)"
        anat_region = "On Duvar (Anterior Miyokard)"
        clinical_note = "Lead V1-V4 uzerindeki rekonstruksiyon hatasi ve dikkat agirliklari,"
        clinical_subnote = "iskemik surecin oncelikli olarak kalbin On (Anterior) Duvarina odaklandigini gostermektedir."
        patient_note = "Inceleme sonucunda, kalbinizin 'On Duvar' bolgesine ait elektriksel dalgalarda hekim kontrolu gerektirebilecek olasi bir duzensizlik izlenmistir."
    else:
        diagnosis_title = "Normal Sinus Ritmi (Fizyolojik Bulgular)"
        affected_lead = "Yok (Fizyolojik Sinirlar)"
        anat_region = "Saglikli Miyokard Doku"
        anomaly_score = 0.00420
        clinical_note = "Tum derivasyonlar normal fizyolojik sinirlar icerisindedir."
        clinical_subnote = "MAE rekonstruksiyon sapmasi tespit edilmemis olup akut iskemik patoloji izlenmemistir."
        patient_note = "Inceleme sonucunda, EKG ritminiz normal sinirlarda degerlendirilmistir. Belirgin bir ritim bozuklugu veya iskemik bulguya rastlanmamistir."

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # 1. Ust Kurumsal Baslik Seridi
    p.setFillColorRGB(0.08, 0.22, 0.38)
    p.rect(0, 720, 612, 80, fill=True, stroke=False)
    
    p.setFillColorRGB(1, 1, 1)
    p.setFont("Helvetica-Bold", 17)
    p.drawString(45, 765, "12-Lead ECG MAE Clinical Decision & XAI Report")
    p.setFont("Helvetica", 9)
    p.drawString(45, 745, "Yapay Zeka Destekli Kardiyak Karar Destek ve Eksik Kanal Tolerans Sistemi")
    p.drawRightString(567, 745, f"Protokol No: {patient_id}")

    # 2. Hasta & Sistem Guvenlik Karti (Gri Kutu)
    p.setFillColorRGB(0.95, 0.96, 0.98)
    p.roundRect(45, 630, 522, 70, 6, fill=True, stroke=False)
    
    p.setFillColorRGB(0.1, 0.1, 0.1)
    p.setFont("Helvetica-Bold", 10)
    p.drawString(60, 680, "Sistem Guvenilirlik ve Donanim Durumu:")
    p.setFont("Helvetica", 9)
    p.drawString(60, 662, f"- Aktif Kanal Sayisi: {active_leads} / 12 Derivasyon")
    p.drawString(60, 646, f"- Telafisi Saglanan Kopuk Kanallar: {dropped_leads if dropped_leads else 'Kopma Yok (Tam Veri)'}")
    
    conf_val = confidence * 100 if confidence <= 1.0 else confidence
    p.drawString(320, 662, f"- Model Guven Skoru: %{conf_val:.1f}")
    p.drawString(320, 646, "- Rekonstruksiyon: MAE Derin Ogrenme Tabanli")

    # 3. Klinik Teshis ve Hekim Bulgulari
    p.setFillColorRGB(0.08, 0.22, 0.38)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(45, 600, "1. Klinik Teshis ve XAI (Aciklanabilir Yapay Zeka) Bulgulari")
    
    p.setStrokeColorRGB(0.8, 0.8, 0.8)
    p.line(45, 592, 567, 592)
    
    p.setFillColorRGB(0.15, 0.15, 0.15)
    p.setFont("Helvetica", 10)
    p.drawString(60, 572, f"- Tani Tahmini: {diagnosis_title}")
    p.drawString(60, 552, f"- Global Hata Skoru (MSE): {anomaly_score:.5f}")
    p.drawString(60, 532, f"- En Yuksek Sapma Gosteren Derivasyon: Lead {affected_lead}")
    p.drawString(60, 512, f"- Eslenen Anatomik Bolge: {anat_region}")
    
    # 4. Hekim Icin XAI Yorumu
    p.setFont("Helvetica-Oblique", 9)
    p.drawString(60, 485, f"Klinik Cikarim: {clinical_note}")
    p.drawString(60, 472, f"{clinical_subnote}")

    # 5. Hasta Bilgilendirme Notu (Sade Aciklama Kutusu)
    p.setFillColorRGB(0.93, 0.97, 0.94)
    p.roundRect(45, 345, 522, 105, 6, fill=True, stroke=False)
    
    p.setFillColorRGB(0.1, 0.45, 0.2)
    p.setFont("Helvetica-Bold", 11)
    p.drawString(60, 430, "2. Hasta ve Hasta Yakini Bilgilendirme Notu (Ozet)")
    
    p.setFillColorRGB(0.15, 0.2, 0.15)
    p.setFont("Helvetica", 9)
    p.drawString(60, 408, "Bu rapor, EKG sinyallerinizin yapay zeka tarafindan incelenmesiyle olusturulmustur.")
    p.drawString(60, 392, f"{patient_note}")
    p.drawString(60, 376, "Lutfen bu on degerlendirme sonucunu kardiyoloji uzman hekiminizle paylasiniz.")

    p.showPage()
    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "inline; filename=ecg_report.pdf"})