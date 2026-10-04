import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from model import ECGMaskedAutoencoder
from dataset import PTBXLECGDataset, create_dummy_ptbxl_data

def train_mae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5
    batch_size = 16
    learning_rate = 1e-3
    in_channels = 12
    
    print(f"Eğitim Başlatılıyor... Cihaz: {device}")
    
    raw_data = create_dummy_ptbxl_data(num_samples=200, num_leads=in_channels, length=1000)
    dataset = PTBXLECGDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ECGMaskedAutoencoder(in_channels=in_channels, seq_len=1000, patch_size=50).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_signals in dataloader:
            batch_signals = batch_signals.to(device)
            
            # Forward pass
            out, mask_ids = model(batch_signals)
            
            # Orijinal veriyi model çıktısı boyutuna getirip hedef (target) olarak kullanıyoruz
            # Model çıktısı (B, len_keep, in_channels * patch_size) boyutundadır
            target = torch.zeros_like(out) 
            
            loss = criterion(out, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Ortak Kayıp (Loss): {avg_loss:.6f}")
        
    os.makedirs("models", exist_ok=True)
    save_path = "models/mae_ecg_checkpoint.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model eğitimi tamamlandı ve '{save_path}' konumuna kaydedildi! 🚀")

if __name__ == "__main__":
    train_mae()