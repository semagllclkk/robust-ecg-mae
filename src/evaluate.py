import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from model import ECGMaskedAutoencoder
from preprocess import butter_bandpass_filter
from dataset import create_dummy_ptbxl_data

def evaluate_reconstruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Kaydedilen Modeli Yükle (12-lead uyumlu)
    model_path = "models/mae_ecg_checkpoint.pth"
    if not os.path.exists(model_path):
        print("Hata: Kayıtlı model bulunamadı! Önce train.py çalıştırılmalı.")
        return

    model = ECGMaskedAutoencoder(in_channels=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model başarıyla yüklendi!")

    # 2. Test İçin Sentetik 12-Lead EKG Sinyali Oluştur ve Filtrele
    raw_signals = create_dummy_ptbxl_data(num_samples=1, num_leads=12, length=1000)[0]
    
    # Her kanala Butterworth Bandpass filtresi uygula
    filtered_signals = np.array([butter_bandpass_filter(raw_signals[i], fs=100) for i in range(12)])
    
    # Negative stride hatasını önlemek için .copy()
    input_tensor = torch.tensor(filtered_signals.copy(), dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Model Tahmini
    with torch.no_grad():
        decoded, mask = model(input_tensor)
        
    original = input_tensor.cpu().numpy().squeeze()[0] # Lead I görselleştirme için
    
    # 4. Görselleştirme
    plt.figure(figsize=(12, 5))
    plt.plot(original, label='Orijinal Filtrelenmiş EKG Sinyali (Lead I)', color='blue', alpha=0.7)
    plt.title("Robust ECG Masked Autoencoder - Sinyal Reconstruction Testi (12-Lead)")
    plt.xlabel("Örnek (Sample)")
    plt.ylabel("Genlik")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs("data/raw", exist_ok=True)
    output_img = "data/raw/reconstruction_test.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Test sonucu grafiği '{output_img}' konumuna başarıyla kaydedildi!")

if __name__ == '__main__':
    evaluate_reconstruction()