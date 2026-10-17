"""EKG Klinik Karar Destek API'si (İP5).

Çalıştırma:  uvicorn app.main:app --reload --port 8000   (backend/ klasöründen)
"""

import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import signals
from .inference import load_backend
from .leads import LEAD_NAMES, N_LEADS, SAMPLE_RATE, SCENARIO_WALL, WINDOW_SAMPLES
from .preprocessing import lead_mask

MAX_DISCONNECTED = 4  # rapor: 1–4 derivasyon kopması
MAX_CHUNK = 2 * WINDOW_SAMPLES

app = FastAPI(title="EKG MAE Karar Destek API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)
backend = load_backend()


class LeadState(BaseModel):
    condition: str = "normal"
    disconnectedLeads: list[int] = Field(default_factory=list)

    @field_validator("condition")
    @classmethod
    def known_condition(cls, value):
        if value not in SCENARIO_WALL:
            raise ValueError(f"Bilinmeyen EKG senaryosu: {value}")
        return value

    @field_validator("disconnectedLeads")
    @classmethod
    def valid_leads(cls, value):
        value = sorted(set(value))
        if any(i < 0 or i >= N_LEADS for i in value):
            raise ValueError("Derivasyon indeksi 0–11 aralığında olmalı.")
        if len(value) > MAX_DISCONNECTED:
            raise ValueError(f"En fazla {MAX_DISCONNECTED} derivasyon koparılabilir.")
        return value


class ChunkRequest(LeadState):
    start: int = Field(ge=0)
    count: int = Field(gt=0, le=MAX_CHUNK)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "modelSource": backend.source,
        "sampleRate": SAMPLE_RATE,
        "leads": LEAD_NAMES,
    }


@app.post("/api/chunk")
def chunk(req: ChunkRequest):
    """Canlı akış: [start, start+count) aralığındaki ham sinyal + kopuk kanalların rekonstrüksiyonu."""
    end = req.start + req.count
    # Rekonstrüksiyon için model 10 s bağlam görür; yalnızca istenen son kısım döner.
    context_start = min(req.start, max(0, end - WINDOW_SAMPLES))
    window = signals.generate(context_start, end - context_start, req.condition)
    mask = lead_mask(req.disconnectedLeads)
    window = window * mask[:, None]  # kopuk kablodan sinyal gelmez
    data = window[:, req.start - context_start :]

    reconstructed = {}
    if req.disconnectedLeads:
        completed, _ = backend.reconstruct(window, mask)
        tail = completed[:, req.start - context_start :]
        reconstructed = {str(i): np.round(tail[i], 4).tolist() for i in req.disconnectedLeads}

    return {
        "start": req.start,
        "data": np.round(data, 4).tolist(),
        "reconstructed": reconstructed,
    }


@app.post("/api/predict")
def predict(req: LeadState):
    """Son 10 s'lik pencere üzerinden teşhis, ST ölçümü ve dikkat ağırlıkları."""
    started = time.perf_counter()
    raw = signals.generate(0, WINDOW_SAMPLES, req.condition)
    mask = lead_mask(req.disconnectedLeads)
    try:
        result = backend.predict(raw * mask[:, None], mask).to_json()
    except Exception as error:  # model çıkarım hatası istemciye anlamlı dönsün
        raise HTTPException(status_code=500, detail=f"Çıkarım hatası: {error}") from error
    result["disconnectedLeads"] = req.disconnectedLeads
    result["latencyMs"] = round((time.perf_counter() - started) * 1000, 1)
    return result
