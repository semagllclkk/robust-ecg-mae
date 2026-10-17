"""
Kanal bazlı maskeli otokodlayıcı + MI sınıflandırıcı (TÜBİTAK Raporu Md. 2.2–2.3).

- LeadDropout   : eğitimde her örnekten rastgele 0..max_drop derivasyonu tümüyle düşürür
                  (proje kısıtı: en fazla 4 kanal kopması -> en az 8 aktif kanal).
- Tokenizasyon  : her (derivasyon, zaman-patch) çifti ayrı bir token'dır (12 x 20 = 240 token).
                  Düşürülen derivasyonların token'ları öğrenilebilir mask token ile değiştirilir,
                  böylece silinen sinyal modele hiçbir şekilde sızmaz.
- Çıktılar      : 12 derivasyonun tamamının rekonstrüksiyonu + MI/Normal sınıflandırma logit'i.
"""
import torch
import torch.nn as nn


def random_lead_mask(batch_size, num_leads=12, max_drop=4, n_drop=None, generator=None):
    """
    (B, num_leads) bool maske üretir; True = derivasyon düşürüldü.
    n_drop verilirse her örnekten tam o kadar, verilmezse örnek başına U{0..max_drop} kadar kanal düşer.
    CPU'da üretilir (deterministik değerlendirme için generator desteklenir); çağıran cihaza taşır.
    """
    if n_drop is None:
        k = torch.randint(0, max_drop + 1, (batch_size, 1), generator=generator)
    else:
        k = torch.full((batch_size, 1), int(n_drop))
    ranks = torch.rand(batch_size, num_leads, generator=generator).argsort(dim=1).argsort(dim=1)
    return ranks < k


class LeadDropout(nn.Module):
    """Eğitim modunda rastgele derivasyonları sıfırlar; değerlendirmede yalnızca verilen maskeyi uygular."""

    def __init__(self, num_leads=12, max_drop=4):
        super().__init__()
        self.num_leads = num_leads
        self.max_drop = max_drop

    def forward(self, x, lead_mask=None):
        B, C, _ = x.shape
        if lead_mask is None:
            if self.training and self.max_drop > 0:
                lead_mask = random_lead_mask(B, C, self.max_drop).to(x.device)
            else:
                lead_mask = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        return x.masked_fill(lead_mask.unsqueeze(-1), 0.0), lead_mask


class ECGMaskedAutoencoder(nn.Module):
    """
    forward(x, lead_mask=None) -> (recon (B, C, L), logits (B, num_classes), lead_mask (B, C))

    lead_mask: gerçekte kopmuş kanallar biliniyorsa (çıkarım sırasında) True olarak verilir.
    Verilmezse eğitimde LeadDropout rastgele maske üretir, değerlendirmede tüm kanallar aktiftir.
    """

    def __init__(self, in_channels=12, seq_len=1000, patch_size=50, embed_dim=128,
                 depth=4, decoder_depth=2, num_heads=4, num_classes=1, max_drop=4, dropout=0.1):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError(f"seq_len ({seq_len}) patch_size'a ({patch_size}) tam bölünmeli")

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.lead_dropout = LeadDropout(in_channels, max_drop)

        # Patch Embedding (tüm derivasyonlar için ortak): (B, C, P, patch_size) -> (B, C, P, D)
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.lead_embed = nn.Parameter(torch.zeros(1, in_channels, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        def transformer(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

        self.encoder = transformer(depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Rekonstrüksiyon dalı: token -> patch_size örnek
        self.decoder = transformer(decoder_depth)
        self.recon_head = nn.Linear(embed_dim, patch_size)

        # Sınıflandırma dalı: görünür token'ların ortalaması -> logit
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        for p in (self.lead_embed, self.pos_embed, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x, lead_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        P, D = self.num_patches, self.pos_embed.shape[-1]

        x, lead_mask = self.lead_dropout(x, lead_mask)

        tokens = self.patch_embed(x.reshape(B, C, P, self.patch_size))  # (B, C, P, D)
        drop = lead_mask.view(B, C, 1, 1)
        tokens = torch.where(drop, self.mask_token.expand(B, C, P, D), tokens)
        tokens = (tokens + self.lead_embed + self.pos_embed).reshape(B, C * P, D)

        encoded = self.encoder_norm(self.encoder(tokens))  # (B, C*P, D)

        recon = self.recon_head(self.decoder(encoded)).reshape(B, C, L)

        visible = (~lead_mask).repeat_interleave(P, dim=1).unsqueeze(-1).float()  # (B, C*P, 1)
        pooled = (encoded * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1.0)
        logits = self.cls_head(pooled)

        return recon, logits, lead_mask


if __name__ == "__main__":
    print("Kanal bazlı MAE modeli test ediliyor...")
    model = ECGMaskedAutoencoder()
    x = torch.randn(4, 12, 1000)
    recon, logits, mask = model.train()(x)
    print(f"Eğitim  -> recon {tuple(recon.shape)}, logits {tuple(logits.shape)}, düşen kanal/örnek {mask.sum(1).tolist()}")
    mask = random_lead_mask(4, n_drop=4)
    recon, logits, mask = model.eval()(x, lead_mask=mask)
    print(f"Çıkarım -> 4 kanal kopuk, recon {tuple(recon.shape)}, logits {tuple(logits.shape)}")
    print(f"Parametre sayısı: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
