from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from scipy.signal import resample

from model import ECGMaskedAutoencoder
from preprocess import butter_bandpass_filter

app = FastAPI(
    title="Robust ECG Masked Autoencoder API (PTB-XL Multi-Lead Supported)",
    description="Esnek uzunlukta ve multi-lead EKG sinyal işleme servisi",
    version="2.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/mae_ecg_checkpoint.pth"

# 12-lead destekli model yükleme
model = None
if os.path.exists(model_path):
    model = ECGMaskedAutoencoder(in_channels=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

class ECGRequest(BaseModel):
    # signal: List[float] (tek kanallı) veya List[List[float]] (çok kanallı / 12-lead)
    signal: list

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Robust ECG MAE (12-Lead PTB-XL Destekli) Servis Çalışıyor",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_ecg(data: ECGRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model ağırlıkları bulunamadı!")
        
    raw_signal = np.array(data.signal, dtype=np.float32)
    
    # 1. Boyut Şekillendirme (Single lead -> Multi lead dönüşümü)
    if raw_signal.ndim == 1:
        # Tek kanal geldiyse 12 kanala kopyala (fallback)
        raw_signal = np.tile(raw_signal, (12, 1))
    elif raw_signal.ndim == 2 and raw_signal.shape[0] != 12:
        # Kanal sayısı 12 değilse ilk kanalı 12 türeve çoğalt
        raw_signal = np.tile(raw_signal[0], (12, 1))
        
    num_leads, original_length = raw_signal.shape
    
    if original_length < 10:
        raise HTTPException(status_code=400, detail="Sinyal çok kısa. En az 10 örnek gönderilmelidir.")

    # 2. Resampling (1000 uzunluğuna getir)
    if original_length != 1000:
        resampled_signal = resample(raw_signal, 1000, axis=1)
    else:
        resampled_signal = raw_signal

    # 3. Butterworth Filtreleme (Her kanal için)
    filtered_signal = np.array([butter_bandpass_filter(resampled_signal[i], fs=100) for i in range(12)])
    
    # 4. PyTorch Tensor Dönüşümü
    input_tensor = torch.tensor(filtered_signal.copy(), dtype=torch.float32).unsqueeze(0).to(device)
    
    # 5. Model Inference
    with torch.no_grad():
        decoded, mask = model(input_tensor)
        
    output_signal = decoded.cpu().numpy().squeeze()
    
    return {
        "original_shape": [num_leads, original_length],
        "processed_shape": [12, 1000],
        "status": "success",
        "message": "12-Lead EKG sinyali başarıyla rekonstrükte edildi."
    }