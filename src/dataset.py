import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PTBXLECGDataset(Dataset):
    """
    PTB-XL EKG Veri Seti için PyTorch Dataset Sınıfı.
    Girdi: (N, num_leads, signal_length) -> Örn: (N, 12, 1000) veya (N, 1, 1000)
    """
    def __init__(self, signals: np.ndarray, transform=None):
        """
        signals: numpy array biçiminde EKG sinyalleri [Örnek Sayısı, Türev/Kanal, Uzunluk]
        """
        # Sinyal boyutunu kontrol et (Tek kanallı gelirse 3D yap)
        if signals.ndim == 2:
            signals = np.expand_dims(signals, axis=1)
            
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        if self.transform:
            signal = self.transform(signal)
            
        return signal

def create_dummy_ptbxl_data(num_samples=100, num_leads=12, length=1000):
    """
    Test ve geliştirme amaçlı PTB-XL formatında sentetik EKG verisi üretir.
    """
    t = np.linspace(0, 10, length)
    data = []
    for _ in range(num_samples):
        leads = []
        for l in range(num_leads):
            # Farklı türevleri simüle etmek için faz kaydırmalı sinüs ve gürültü
            lead_signal = np.sin(2 * np.pi * 1.2 * t + l * 0.2) + 0.1 * np.random.randn(length)
            leads.append(lead_signal)
        data.append(leads)
    return np.array(data, dtype=np.float32)

if __name__ == "__main__":
    # Hızlı Doğrulama Testi
    print("PTB-XL Dataset Test Ediliyor...")
    dummy_data = create_dummy_ptbxl_data(num_samples=50, num_leads=12, length=1000)
    
    dataset = PTBXLECGDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    sample_batch = next(iter(dataloader))
    print(f"Batch Şekli (Batch Size, Lead Sayısı, Sinyal Uzunluğu): {sample_batch.shape}")
    print("Dataset modülü başarıyla doğrulandı! 🚀")