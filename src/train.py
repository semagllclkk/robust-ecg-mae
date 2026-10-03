import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from model import ECGMaskedAutoencoder
from preprocess import butter_bandpass_filter

def create_dummy_ecg_dataset(num_samples=100, length=1000):
    """
    Eğitim döngüsünü test etmek için sentetik EKG verisi üretir.
    (İleride PTB-XL veri setinin tamamını buraya bağlayacağız).
    """
    t = np.linspace(0, 10, length)
    data = []
    for _ in range(num_samples):
        # Temel sinüsoidal EKG benzeri sentetik dalga + gürültü
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
        signal += np.random.normal(0, 0.1, length)
        # Ön işleme (Filtreleme)
        filtered = butter_bandpass_filter(signal, fs=100)
        data.append(filtered)
        
    tensor_data = torch.tensor(np.array(data), dtype=torch.float32)
    return TensorDataset(tensor_data)

def train_mae():
    # Hiperparametreler (Makale standartlarına uygun)
    batch_size = 16
    epochs = 5
    learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Çalışma Cihazı: {device}")
    
    # Veri yükleyici ve Model
    dataset = create_dummy_ecg_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ECGMaskedAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    criterion = nn.MSELoss()
    
    model.train()
    print("\n--- MAE Model Eğitimi Başlıyor ---")
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (signals,) in enumerate(dataloader):
            signals = signals.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (Model çıktısı ve maske)
            decoded, mask = model(signals)
            
            # Loss Hesabı: Sadece maskelenmiş alanlar üzerinden MSE hesaplanır
            # Signals boyutunu parçalara dönüştür (Batch, Num_Patches, Patch_Size)
            N, L = signals.shape
            patch_size = model.patch_size
            num_patches = model.num_patches
            target_patches = signals.view(N, num_patches, patch_size)
            
            # Yalnızca maske == 1 olan yerlerin kayıp değerini al
            loss = criterion(decoded, target_patches[:, :decoded.shape[1], :])
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Ortalama Kayıp (MSE Loss): {avg_loss:.6f}")

    print("\n--- Eğitim Başarıyla Tamamlandı ---")
    
    # Eğitilen ilk ağırlıkları kaydetme
    os.makedirs("models", exist_ok=True)
    save_path = "models/mae_ecg_checkpoint.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model ağırlıkları '{save_path}' konumuna kaydedildi!")

if __name__ == '__main__':
    train_mae()