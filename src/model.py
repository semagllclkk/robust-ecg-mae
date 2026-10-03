import torch
import torch.nn as nn

class ECGMaskedAutoencoder(nn.Module):
    def __init__(self, signal_length=1000, patch_size=50, in_channels=1, embed_dim=128, mask_ratio=0.7):
        super(ECGMaskedAutoencoder, self).__init__()
        
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.mask_ratio = mask_ratio
        
        # Patch Embedding: Her EKG parçasını vektör temsiline dönüştürür
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        
        # Position Embedding: Sinyal parçalarının sırasını öğrenmesi için
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder (Basit Transformer / MLP Blok Yapısı)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decoder: Maskelenmiş parçaları yeniden oluşturmak için
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Output Projection: Tekrar orijinal sinyal boyutuna dönüştürme
        self.head = nn.Linear(embed_dim, patch_size)

    def random_masking(self, x):
        """
        Sinyal parçalarının (patches) rastgele belirlenen oranını (%70) maskeler.
        """
        N, L, D = x.shape  # Batch, Num_Patches, Embed_Dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # Rastgele gürültü indeksi
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Saklanacak (%30) parçaları seç
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Maske matrisi (1: maskelendi, 0: korundu)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x):
        # x boyutu: (Batch, Signal_Length) -> (Batch, Num_Patches, Patch_Size)
        N, L = x.shape
        x_patches = x.view(N, self.num_patches, self.patch_size)
        
        # 1. Patch Embedding + Position Embedding
        x_embed = self.patch_embed(x_patches) + self.pos_embed
        
        # 2. Random Masking
        x_masked, mask, ids_restore = self.random_masking(x_embed)
        
        # 3. Encoder
        latent = self.encoder(x_masked)
        
        # 4. Reconstruct (Yeniden Oluşturma)
        decoded = self.head(latent)
        
        return decoded, mask

if __name__ == '__main__':
    # Test Etme: 2 Örnek EKG Sinyali (1000 uzunluğunda)
    dummy_input = torch.randn(2, 1000)
    model = ECGMaskedAutoencoder()
    output, mask = model(dummy_input)
    
    print("--- MAE Model Test Başarılı ---")
    print(f"Girdi Boyutu: {dummy_input.shape}")
    print(f"Maske Boyutu: {mask.shape}")
    print(f"Encoder Çıktı Parça Sayısı (Saklanan %30): {output.shape[1]}")