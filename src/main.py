from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from scipy.signal import resample

from model import ECGMaskedAutoencoder
from preprocess import butter_bandpass_filter

app = FastAPI(
    title="Robust ECG Masked Autoencoder API",
    description="Esnek uzunlukta EKG sinyal temizleme ve reconstruction servisi",
    version="1.1.0"
)

# Model yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/mae_ecg_checkpoint.pth"

model = None
if os.path.exists(model_path):
    model = ECGMaskedAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

class ECGRequest(BaseModel):
    signal: list[float]  # Artık her uzunlukta liste kabul ediliyor!

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Robust ECG MAE Esnek Backend Servisi Çalışıyor",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_ecg(data: ECGRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model ağırlıkları bulunamadı!")
        
    original_length = len(data.signal)
    if original_length < 10:
        raise HTTPException(status_code=400, detail="Sinyal çok kısa. En az 10 örnek gönderilmelidir.")
        
    raw_signal = np.array(data.signal, dtype=np.float32)
    
    # 1. Otomatik Boyutlandırma (Esneklik): Her boyutu 1000 örneğe getir
    if original_length != 1000:
        resampled_signal = resample(raw_signal, 1000)
    else:
        resampled_signal = raw_signal

    # 2. Butterworth Bandpass Filtreleme
    filtered_signal = butter_bandpass_filter(resampled_signal, fs=100)
    
    # 3. PyTorch Tensor Dönüşümü (.copy() ile stride hatası önlendi)
    input_tensor = torch.tensor(filtered_signal.copy(), dtype=torch.float32).unsqueeze(0).to(device)
    
    # 4. Model Tahmini (Inference)
    with torch.no_grad():
        decoded, mask = model(input_tensor)
        
    # 5. İsteğe bağlı: Çıktıyı tekrar orijinal uzunluğuna geri ölçekle
    output_signal = decoded.cpu().numpy().squeeze().flatten()
    if original_length != 1000:
        output_signal = resample(output_signal, original_length)

    return {
        "original_length": original_length,
        "processed_length": 1000,
        "filtered_signal": filtered_signal.tolist(),
        "reconstructed_signal": output_signal.tolist(),
        "mask_pattern": mask.cpu().numpy().tolist(),
        "status": "success"
    }