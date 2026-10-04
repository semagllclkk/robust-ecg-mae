import torch
import torch.nn as nn

class ECGMaskedAutoencoder(nn.Module):
    """
    1D Masked Autoencoder (MAE) - Esnek Kanal Desteği (1-lead veya 12-lead EKG)
    """
    def __init__(self, in_channels=12, seq_len=1000, patch_size=50, embed_dim=128, mask_ratio=0.7):
        super(ECGMaskedAutoencoder, self).__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.mask_ratio = mask_ratio
        
        # Patch Embedding: (Batch, in_channels, seq_len) -> (Batch, num_patches, embed_dim)
        self.patch_embed = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder (Transformer Blocks)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Decoder (Reconstruction)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Projection Back to Signal Space: embed_dim -> (in_channels * patch_size)
        self.head = nn.Linear(embed_dim, in_channels * patch_size)

    def random_masking(self, x):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Maskelenmemiş patch'leri seç
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, ids_keep

    def forward(self, x):
        # x shape: (Batch, in_channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (Batch, 1, seq_len) yap
            
        B, C, L = x.shape
        
        # Patching & Embedding
        x_patches = self.patch_embed(x).transpose(1, 2)  # (B, num_patches, embed_dim)
        x_patches = x_patches + self.pos_embed
        
        # Masking
        x_masked, mask_ids = self.random_masking(x_patches)
        
        # Encoder
        encoded = self.encoder(x_masked)
        
        # Decoder
        decoded_patches = self.decoder(encoded, encoded)
        
        # Project back to signal
        out = self.head(decoded_patches)  # (B, len_keep, in_channels * patch_size)
        
        return out, mask_ids

if __name__ == "__main__":
    print("Esnek MAE Modeli Test Ediliyor...")
    # 12-lead PTB-XL testi
    model_12lead = ECGMaskedAutoencoder(in_channels=12)
    dummy_input_12 = torch.randn(4, 12, 1000)
    out_12, mask_12 = model_12lead(dummy_input_12)
    print(f"12-Lead Çıktı Şekli: {out_12.shape}")
    
    # 1-lead Klasik EKG testi
    model_1lead = ECGMaskedAutoencoder(in_channels=1)
    dummy_input_1 = torch.randn(4, 1, 1000)
    out_1, mask_1 = model_1lead(dummy_input_1)
    print(f"1-Lead Çıktı Şekli: {out_1.shape}")
    print("Model modülü başarıyla doğrulandı! 🚀")