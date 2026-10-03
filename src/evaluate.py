import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from model import ECGMaskedAutoencoder
from preprocess import butter_bandpass_filter

def evaluate_reconstruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Kaydedilen Modeli Yükle
    model_path = "models/mae_ecg_checkpoint.pth"
    if not os.path.exists(model_path):
        print("Hata: Kayıtlı model bulunamadı! Önce train.py çalıştırılmalı.")
        return

    model = ECGMaskedAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model başarıyla yüklendi!")

    # 2. Test İçin Sentetik Örnek EKG Sinyali Oluştur
    t = np.linspace(0, 10, 1000)
    raw_signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.05, 1000)
    filtered_signal = butter_bandpass_filter(raw_signal, fs=100)
    
    # Negative stride hatasını önlemek için .copy() eklendi
    input_tensor = torch.tensor(filtered_signal.copy(), dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Model Tahmini
    with torch.no_grad():
        decoded, mask = model(input_tensor)
        
    original = input_tensor.cpu().numpy().squeeze()
    
    # 4. Görselleştirme
    plt.figure(figsize=(12, 5))
    plt.plot(original, label='Orijinal Temiz EKG Sinyali', color='blue', alpha=0.7)
    plt.title("Robust ECG Masked Autoencoder - Sinyal Reconstruction Testi")
    plt.xlabel("Örnek (Sample)")
    plt.ylabel("Genlik")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    output_img = "data/raw/reconstruction_test.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Test sonucu grafiği '{output_img}' konumuna başarıyla kaydedildi!")
    plt.show()

if __name__ == '__main__':
    evaluate_reconstruction()