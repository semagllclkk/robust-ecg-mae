"""
Çift Başlıklı Maskeli Öz-Kodlayıcı (Multi-Task MAE) — TÜBİTAK Raporu Md. 2.2–2.3.

Girdi (B, 12, 1000) z-score EKG
 ├─ LeadDropout (Md. 2.2): eğitimde batch başına k ∈ [1, 4] derivasyon sıfırlanır (rastgele veya
 │   aynı kalp duvarından "kritik" kopma). X_masked = X ⊙ M
 ├─ Ortak Omurga (Md. 2.3.1)
 │   ├─ Derivasyon başına 1D-CNN rezidüel bloklar (P, QRS, ST, T morfolojisi): 1000 → 500 → 250 → 125
 │   ├─ STFT frekans dalı (Wei ve ark. [5] bimodal yaklaşımı): log-genlik spektrogramı darboğaza eklenir
 │   └─ Uzamsal-zamansal Transformer (Na ve ark. [7]): derivasyonlar arası (spatial) + zaman (temporal) dikkat
 ├─ Başlık A — Rekonstrüksiyon (Md. 2.3.2): 1D Transpose Conv kod çözücü; IRC ara katman bağlantıları
 │   (Wei ve ark. [5]) uzamsal dikkatle karıştırılarak kopan kanala sağlam kanallardan bilgi taşır
 └─ Başlık B — Teşhis (Md. 2.3.3): Global Average Pooling + derivasyon dikkati + Dense
     → MI logit'i, kriz tipi (Anterior/Inferior/Lateral) logit'leri, XAI için derivasyon dikkat ağırlıkları

Kopan derivasyonların öznitelikleri her ölçekte öğrenilebilir mask token ile değiştirilir ve dikkat
katmanlarında anahtar olarak maskelenir; böylece silinen sinyal modele sızmaz.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ecg_constants import WALL_LEAD_INDICES, WALL_NAMES

# (3, 12) bool: duvar -> derivasyon üyeliği
_WALL_MEMBERSHIP = torch.zeros(len(WALL_NAMES), 12, dtype=torch.bool)
for _i, _w in enumerate(WALL_NAMES):
    _WALL_MEMBERSHIP[_i, WALL_LEAD_INDICES[_w]] = True


def random_lead_mask(batch_size, num_leads=12, min_drop=1, max_drop=4, n_drop=None,
                     critical_prob=0.0, generator=None):
    """
    (B, num_leads) bool maske üretir; True = derivasyon koptu.
    - n_drop verilirse her örnekte tam o kadar kanal kopar; verilmezse batch için tek bir
      k ∈ [min_drop, max_drop] seçilir (Rapor Md. 2.2).
    - critical_prob: örneğin "kritik" kopma yaşama olasılığı; bu durumda kopan kanallar öncelikle
      rastgele seçilen tek bir kalp duvarının derivasyonlarından seçilir (Rapor Hedef 2).
    CPU'da üretilir (deterministik değerlendirme için generator desteklenir); çağıran cihaza taşır.
    """
    if n_drop is None:
        n_drop = int(torch.randint(min_drop, max_drop + 1, (1,), generator=generator))
    scores = torch.rand(batch_size, num_leads, generator=generator)
    if critical_prob > 0:
        critical = torch.rand(batch_size, generator=generator) < critical_prob
        walls = torch.randint(0, len(WALL_NAMES), (batch_size,), generator=generator)
        scores = scores - (_WALL_MEMBERSHIP[walls] & critical[:, None]).float()
    ranks = scores.argsort(dim=1).argsort(dim=1)
    return ranks < n_drop


def wall_lead_mask(batch_size, wall, num_leads=12):
    """Belirli bir kalp duvarının tüm derivasyonlarının koptuğu maske (stres testi için)."""
    mask = torch.zeros(batch_size, num_leads, dtype=torch.bool)
    mask[:, WALL_LEAD_INDICES[wall]] = True
    return mask


class LeadDropout(nn.Module):
    """Eğitim modunda rastgele/kritik derivasyon kopması simüle eder; değerlendirmede yalnızca verilen maskeyi uygular."""

    def __init__(self, num_leads=12, min_drop=1, max_drop=4, critical_prob=0.3):
        super().__init__()
        self.num_leads = num_leads
        self.min_drop = min_drop
        self.max_drop = max_drop
        self.critical_prob = critical_prob

    def forward(self, x, lead_mask=None):
        B, C, _ = x.shape
        if lead_mask is None:
            if self.training and self.max_drop > 0:
                lead_mask = random_lead_mask(B, C, self.min_drop, self.max_drop,
                                             critical_prob=self.critical_prob).to(x.device)
            else:
                lead_mask = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        return x.masked_fill(lead_mask.unsqueeze(-1), 0.0), lead_mask


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel_size, stride, pad, bias=False)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel_size, 1, pad, bias=False)
        self.norm2 = nn.GroupNorm(8, cout)
        self.shortcut = (nn.Identity() if cin == cout and stride == 1
                         else nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout)))

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.shortcut(x))


class SpatialMixer(nn.Module):
    """
    Her zaman adımında 12 derivasyon arasında çok başlıklı uzamsal dikkat.
    Kopan derivasyonlar anahtar olarak maskelenir; sorgu olarak kalırlar ve sağlam kanallardan beslenirler.
    """

    def __init__(self, dim, num_leads=12, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.lead_embed = nn.Parameter(torch.zeros(1, num_leads, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim, 1))
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.lead_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, feats, lead_mask):
        # feats: (B, C, D, T)
        B, C, D, T = feats.shape
        feats = torch.where(lead_mask.view(B, C, 1, 1), self.mask_token.expand(B, C, D, T), feats)
        h = feats.permute(0, 3, 1, 2).reshape(B * T, C, D)          # (B*T, C, D)
        q, k, v = (self.qkv(self.norm(h + self.lead_embed.reshape(1, C, D)))
                   .reshape(B * T, C, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4))
        # Kopan kanallar anahtar olarak dışlanır: sağlam kanallardan beslenirler
        key_mask = (~lead_mask).repeat_interleave(T, dim=0)[:, None, None, :]  # (B*T, 1, 1, C), True = katıl
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        h = h + self.proj(out.transpose(1, 2).reshape(B * T, C, D))
        return h.reshape(B, T, C, D).permute(0, 2, 3, 1)             # (B, C, D, T)


class TransformerLayer(nn.Module):
    """
    Pre-norm Transformer katmanı: Attention(Q, K, V) = softmax(QKᵀ / √d_k) V  (Rapor Md. 2.3.1).
    nn.TransformerEncoderLayer yerine doğrudan SDPA kullanılır: Apple MPS'te ~%40 daha hızlı.
    """

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        N, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(N, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, L, D)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.ffn(self.norm2(x)))


class SpatioTemporalBlock(nn.Module):
    """
    Faktörize uzamsal-zamansal Transformer bloğu: önce derivasyonlar arası, sonra zaman ekseni boyunca dikkat.
    Kopan derivasyonların token'ları zaten mask token olduğundan anahtar maskesine gerek yoktur (sızıntı oluşmaz).
    """

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.spatial = TransformerLayer(dim, num_heads, dropout)
        self.temporal = TransformerLayer(dim, num_heads, dropout)

    def forward(self, tokens):
        # tokens: (B, C, T, D)
        B, C, T, D = tokens.shape
        h = self.spatial(tokens.permute(0, 2, 1, 3).reshape(B * T, C, D))
        h = h.reshape(B, T, C, D).permute(0, 2, 1, 3).reshape(B * C, T, D)
        h = self.temporal(h)
        return h.reshape(B, C, T, D)


class STFTEncoder(nn.Module):
    """Derivasyon başına kısa zamanlı Fourier dönüşümü (log-genlik) -> darboğaz zaman çözünürlüğünde öznitelik."""

    def __init__(self, dim, n_fft=64, hop=8, seq_len=1000):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.num_frames = seq_len // hop  # 1000 / 8 = 125 (darboğaz uzunluğu)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.proj = nn.Sequential(nn.Conv1d(n_fft // 2 + 1, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1))

    def forward(self, x):
        # x: (N, L) -> (N, D, num_frames)
        pad = (self.n_fft - self.hop) // 2
        frames = F.pad(x, (pad, pad)).unfold(-1, self.n_fft, self.hop)[:, :self.num_frames]  # (N, F, n_fft)
        spec = torch.fft.rfft(frames * self.window, dim=-1).abs()
        return self.proj(torch.log1p(spec).transpose(1, 2))


class ECGMaskedAutoencoder(nn.Module):
    """
    forward(x, lead_mask=None) -> dict
      recon       (B, 12, 1000) tüm derivasyonların rekonstrüksiyonu (z-score uzayında)
      mi_logit    (B,)          MI / Normal
      wall_logits (B, 3)        Anterior / Inferior / Lateral MI (kriz tipi)
      lead_attn   (B, 12)       Teşhis başlığının derivasyon dikkat ağırlıkları (XAI), kopan kanallar = 0
      lead_mask   (B, 12)       kullanılan kopma maskesi
    lead_mask gerçekte kopmuş kanallar biliniyorsa (çıkarımda) True olarak verilir.
    """

    def __init__(self, in_channels=12, seq_len=1000, embed_dim=128, depth=2, num_heads=4,
                 min_drop=1, max_drop=4, critical_prob=0.3, dropout=0.1, use_stft=True):
        super().__init__()
        if seq_len % 8 != 0:
            raise ValueError("seq_len 8'e tam bölünmeli (3 adet 2x örnek azaltma)")
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_tokens = seq_len // 8
        D = embed_dim

        self.lead_dropout = LeadDropout(in_channels, min_drop, max_drop, critical_prob)

        # --- Ortak omurga: 1D-CNN zamansal bloklar (derivasyonlar arasında ağırlık paylaşımlı) ---
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, 2, 3, bias=False), nn.GroupNorm(8, 32), nn.GELU())  # 1000 -> 500
        self.enc1 = ResBlock1d(32, 64, stride=2)   # 500 -> 250
        self.enc2 = ResBlock1d(64, D, stride=2)    # 250 -> 125
        self.stft = STFTEncoder(D, seq_len=seq_len) if use_stft else None

        # --- Uzamsal-zamansal Transformer (darboğaz) ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, D))
        self.lead_embed = nn.Parameter(torch.zeros(1, in_channels, 1, D))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, self.num_tokens, D))
        self.blocks = nn.ModuleList([SpatioTemporalBlock(D, num_heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)

        # --- IRC: ara katman bağlantıları. 250 çözünürlükteki bağlantı uzamsal dikkatle derivasyonlar
        #     arası karıştırılır (kopan kanal sağlam kanallardan beslenir); 500'deki bağlantıda kopan
        #     kanalların öznitelikleri mask token ile değiştirilir (bellek sınırı nedeniyle karıştırılmaz).
        self.irc = SpatialMixer(64, in_channels, num_heads)
        self.stem_mask_token = nn.Parameter(torch.zeros(1, 32, 1))

        # --- Başlık A: 1D Transpose Convolution kod çözücü ---
        self.up3 = nn.ConvTranspose1d(D, 64, 4, 2, 1)    # 125 -> 250
        self.dec3 = ResBlock1d(128, 64)
        self.up2 = nn.ConvTranspose1d(64, 32, 4, 2, 1)   # 250 -> 500
        self.dec2 = ResBlock1d(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 32, 4, 2, 1)   # 500 -> 1000
        self.dec1 = ResBlock1d(32, 32)
        self.recon_out = nn.Conv1d(32, 1, 1)

        # --- Başlık B: Global Average Pooling + derivasyon dikkati + Dense ---
        self.lead_attn = nn.Sequential(nn.Linear(D, D // 2), nn.Tanh(), nn.Linear(D // 2, 1))
        self.cls_mlp = nn.Sequential(nn.LayerNorm(D), nn.Dropout(dropout), nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout))
        self.mi_head = nn.Linear(D, 1)
        self.wall_head = nn.Linear(D, len(WALL_NAMES))

        for p in (self.mask_token, self.lead_embed, self.time_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x, lead_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        x, lead_mask = self.lead_dropout(x, lead_mask)

        # Omurga (derivasyon başına)
        h = x.reshape(B * C, 1, L)
        s0 = self.stem(h)                  # (B*C, 32, 500)
        s1 = self.enc1(s0)                 # (B*C, 64, 250)
        z = self.enc2(s1)                  # (B*C, D, 125)
        if self.stft is not None:
            z = z + self.stft(x.reshape(B * C, L))
        D, T = z.shape[1], z.shape[2]

        tokens = z.reshape(B, C, D, T).permute(0, 1, 3, 2)  # (B, C, T, D)
        tokens = torch.where(lead_mask.view(B, C, 1, 1), self.mask_token.expand(B, C, T, D), tokens)
        tokens = tokens + self.lead_embed + self.time_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)                           # (B, C, T, D)

        # Başlık A: rekonstrüksiyon
        s1 = self.irc(s1.reshape(B, C, 64, -1), lead_mask).reshape(B * C, 64, -1)
        drop_rows = lead_mask.reshape(B * C, 1, 1)
        s0 = torch.where(drop_rows, self.stem_mask_token.expand_as(s0), s0)
        d = tokens.reshape(B * C, T, D).transpose(1, 2)
        d = self.dec3(torch.cat([self.up3(d), s1], dim=1))
        d = self.dec2(torch.cat([self.up2(d), s0], dim=1))
        d = self.dec1(self.up1(d))
        recon = self.recon_out(d).reshape(B, C, L)

        # Başlık B: teşhis
        lead_feats = tokens.mean(dim=2)                                   # GAP (zaman) -> (B, C, D)
        scores = self.lead_attn(lead_feats).squeeze(-1).masked_fill(lead_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)                               # (B, C)
        pooled = self.cls_mlp((attn.unsqueeze(-1) * lead_feats).sum(dim=1))

        return {
            "recon": recon,
            "mi_logit": self.mi_head(pooled).squeeze(-1),
            "wall_logits": self.wall_head(pooled),
            "lead_attn": attn,
            "lead_mask": lead_mask,
        }


class ResNet1DBaseline(nn.Module):
    """
    Karşılaştırma için klasik 1D-ResNet sınıflandırıcı (Ribeiro ve ark. [1] benzeri, küçültülmüş).
    LeadDropout ile eğitilmez; kopan kanallar yalnızca sıfırlanmış girdi olarak görülür. Rekonstrüksiyon yapmaz.
    """

    def __init__(self, in_channels=12, width=64, dropout=0.1, **_):
        super().__init__()
        self.lead_dropout = LeadDropout(in_channels, 0, 0, 0.0)  # eğitimde kopma yok
        self.stem = nn.Sequential(nn.Conv1d(in_channels, width, 15, 1, 7, bias=False), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(
            ResBlock1d(width, width, 2, 15), ResBlock1d(width, 2 * width, 2, 15),
            ResBlock1d(2 * width, 2 * width, 2, 15), ResBlock1d(2 * width, 4 * width, 2, 15),
        )
        self.drop = nn.Dropout(dropout)
        self.mi_head = nn.Linear(4 * width, 1)
        self.wall_head = nn.Linear(4 * width, len(WALL_NAMES))

    def forward(self, x, lead_mask=None):
        x, lead_mask = self.lead_dropout(x, lead_mask)
        h = self.drop(self.blocks(self.stem(x)).mean(dim=-1))
        return {"recon": None, "mi_logit": self.mi_head(h).squeeze(-1), "wall_logits": self.wall_head(h),
                "lead_attn": None, "lead_mask": lead_mask}


ARCHITECTURES = {"mae": ECGMaskedAutoencoder, "resnet1d": ResNet1DBaseline}


def build_model(arch, config):
    return ARCHITECTURES[arch](**config)


if __name__ == "__main__":
    print("Çift başlıklı MAE test ediliyor...")
    model = ECGMaskedAutoencoder()
    x = torch.randn(4, 12, 1000)
    out = model.train()(x)
    print(f"Eğitim  -> recon {tuple(out['recon'].shape)}, mi {tuple(out['mi_logit'].shape)}, "
          f"duvar {tuple(out['wall_logits'].shape)}, kopan kanal/örnek {out['lead_mask'].sum(1).tolist()}")
    out = model.eval()(x, lead_mask=wall_lead_mask(4, "Anterior"))
    print(f"Çıkarım -> Anterior (V1–V4) kopuk, dikkat toplamı {out['lead_attn'].sum(1).tolist()}")
    print(f"MAE parametre: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M   "
          f"1D-ResNet parametre: {sum(p.numel() for p in ResNet1DBaseline().parameters()) / 1e6:.2f} M")
