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

class PTBXLProcessedDataset(Dataset):
    """
    `python -m src.prepare_data` ile üretilen önbellekten (data/processed) okuyan MI/Normal Dataset.

    split : "train" | "val" | "test"  (strat_fold 1–8 / 9 / 10)
    Döndürür: (sinyal (12, 1000) float32, hedef sözlüğü)
      hedef["mi"]    : skaler, 1 = MI, 0 = Normal
      hedef["walls"] : (3,) Anterior/Inferior/Lateral MI (AMI/IMI/LMI) çoklu etiket — kriz tipi
      hedef["mu"], hedef["sigma"] : (12,) z-score istatistikleri (mV cinsinden RMSE için)
    Yalnızca label >= 0 olan kayıtlar alınır. X.npy bellek eşlemeli (mmap) açılır;
    ~1 GB'lık dosya RAM'e tümüyle yüklenmez.
    """
    def __init__(self, split="train", processed_dir="data/processed", transform=None):
        import os
        import pandas as pd
        from src.ecg_constants import WALL_NAMES, WALL_TO_SUBCLASS

        meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"), index_col="ecg_id")
        mask = ((meta["split"] == split) & (meta["label"] >= 0)).to_numpy()

        self.indices = np.flatnonzero(mask)
        self.X = np.load(os.path.join(processed_dir, "X.npy"), mmap_mode="r")
        self.mu = np.load(os.path.join(processed_dir, "mu.npy"))[self.indices]
        self.sigma = np.load(os.path.join(processed_dir, "sigma.npy"))[self.indices]
        self.meta = meta.iloc[self.indices]
        self.labels = self.meta["label"].to_numpy(dtype=np.float32)
        self.walls = self.meta[[WALL_TO_SUBCLASS[w] for w in WALL_NAMES]].to_numpy(dtype=np.float32)
        self.ecg_ids = self.meta.index.to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        signal = torch.from_numpy(np.array(self.X[self.indices[idx]], dtype=np.float32))
        if self.transform:
            signal = self.transform(signal)
        target = {
            "mi": torch.tensor(self.labels[idx]),
            "walls": torch.from_numpy(self.walls[idx]),
            "mu": torch.from_numpy(self.mu[idx]),
            "sigma": torch.from_numpy(self.sigma[idx]),
        }
        return signal, target

    def positive_fraction(self):
        return float(self.labels.mean()) if len(self.labels) else 0.0

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