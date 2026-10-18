"""
Çift Başlıklı Maskeli Öz-Kodlayıcı (Multi-Task MAE) — TÜBİTAK Raporu Md. 2.3.

Girdi: X_masked (B, 12, 1000) z-score EKG ve M (B, 12) ikili maske (1 = sağlam, 0 = koptu).
Maskeleme DataLoader seviyesinde LeadDropout ile yapılır (src/lead_dropout.py, Md. 2.2).

 1. Ortak Omurga (Backbone)
    - 1D-CNN blokları: derivasyon başına zamansal yerel dalga morfolojisi (P, QRS, ST, T)
      1000 → 500 → 250 → 125 (derivasyonlar arasında ağırlık paylaşımlı)
    - STFT frekans dalı: zaman serisi ile frekans spektrumunu birlikte işleyen bimodal yaklaşım (Wei ve ark. [5], Md. 2.1)
    - Spatio-Temporal Transformer (Na ve ark. [7]): her blokta çok başlıklı uzamsal dikkat (Spatial Attention,
      derivasyonlar arası) + zamansal dikkat. Attention(Q, K, V) = Softmax(QKᵀ / √d_k) · V
      Uzamsal dikkatte anahtarlar yalnızca sağlam kanallardır; böylece dikkat ağırlıkları "kalan sağlam
      kanalların kopan kanalları ne derece temsil edebileceğini" doğrudan verir.
 2. Başlık A — Sinyal Rekonstrüksiyonu: MCMA (Chen ve ark. [8]) benzeri, omurganın simetriği olan
    1D Transpose Convolution kod çözücü. IRC ara katman bağlantıları (Wei ve ark. [5]) kodlayıcıdan
    kod çözücüye taşınır; 250 çözünürlükteki bağlantı Lead Fusing (Liu ve ark. [9]) ile derivasyonlar
    arası birleştirilir.
 3. Başlık B — Teşhis: Global Average Pooling + Dense katmanlar → MI var/yok + kriz tipi
    (Anterior / Inferior / Lateral MI).

Kopan derivasyonların öznitelikleri her ölçekte öğrenilebilir mask token ile değiştirilir;
silinen sinyal modele sızmaz.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ecg_constants import WALL_NAMES


class ResBlock1d(nn.Module):
    """1D-CNN rezidüel blok (zamansal yerel morfoloji)."""

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


class STFTEncoder(nn.Module):
    """Derivasyon başına kısa zamanlı Fourier dönüşümü (log-genlik) → darboğaz zaman çözünürlüğünde öznitelik."""

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


def _mha(qkv_proj, x, num_heads, key_keep=None, return_weights=False):
    """
    Çok başlıklı dikkat: Softmax(QKᵀ / √d_k) · V.
    key_keep: (N, L) bool, True = anahtar olarak kullanılabilir (kopan kanallar False).
    return_weights=True ise ağırlıklar açıkça hesaplanır ve başlıklar üzerinden ortalaması döner.
    """
    N, L, D = x.shape
    q, k, v = qkv_proj(x).reshape(N, L, 3, num_heads, D // num_heads).permute(2, 0, 3, 1, 4)  # (N, H, L, d)
    if return_weights:
        scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        if key_keep is not None:
            scores = scores.masked_fill(~key_keep[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v
        weights = attn.mean(dim=1)  # (N, L, L)
    else:
        mask = key_keep[:, None, None, :] if key_keep is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        weights = None
    return out.transpose(1, 2).reshape(N, L, D), weights


class TransformerLayer(nn.Module):
    """Pre-norm Transformer katmanı (çok başlıklı dikkat + ileri beslemeli ağ)."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_keep=None, return_weights=False):
        a, w = _mha(self.qkv, self.norm1(x), self.num_heads, key_keep, return_weights)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, w


class SpatioTemporalBlock(nn.Module):
    """
    Spatio-Temporal Transformer bloğu (Na ve ark. [7]):
      - Spatial Attention: her zaman adımında 12 derivasyon arasında dikkat (anahtarlar = sağlam kanallar)
      - Temporal Attention: her derivasyonda zaman ekseni boyunca dikkat
    """

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.spatial = TransformerLayer(dim, num_heads, dropout)
        self.temporal = TransformerLayer(dim, num_heads, dropout)

    def forward(self, tokens, M, return_weights=False):
        # tokens: (B, C, T, D), M: (B, C)
        B, C, T, D = tokens.shape
        h = tokens.permute(0, 2, 1, 3).reshape(B * T, C, D)
        key_keep = (M > 0.5).repeat_interleave(T, dim=0)                       # (B*T, C)
        h, w = self.spatial(h, key_keep, return_weights)
        h = h.reshape(B, T, C, D).permute(0, 2, 1, 3).reshape(B * C, T, D)
        h, _ = self.temporal(h)
        if w is not None:
            w = w.reshape(B, T, C, C).mean(dim=1)                                # zaman ortalaması (B, C, C)
        return h.reshape(B, C, T, D), w


class LeadFusing(nn.Module):
    """
    Lead Fusing (Liu ve ark. [9]): ara katman (IRC) özniteliklerini her zaman adımında derivasyonlar arası
    dikkatle birleştirir. Kopan kanallar sorgu olarak kalır, anahtar olarak dışlanır; sağlam kanallardan beslenir.
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

    def forward(self, feats, M):
        # feats: (B, C, D, T)
        B, C, D, T = feats.shape
        dropped = (M < 0.5).view(B, C, 1, 1)
        feats = torch.where(dropped, self.mask_token.expand(B, C, D, T), feats)
        h = feats.permute(0, 3, 1, 2).reshape(B * T, C, D)
        q = self.norm(h + self.lead_embed.reshape(1, C, D))
        a, _ = _mha(self.qkv, q, self.num_heads, (M > 0.5).repeat_interleave(T, dim=0))
        h = h + self.proj(a)
        return h.reshape(B, T, C, D).permute(0, 2, 3, 1)


class ECGMaskedAutoencoder(nn.Module):
    """
    forward(x_masked, M, sigma=None) -> dict
      recon            (B, 12, 1000) tüm derivasyonların rekonstrüksiyonu — Başlık A
                                     z-score uzayında; amplitude_aware=True ise mV / σ_pop ölçeğinde
      mi_logit         (B,)          MI var/yok                                                — Başlık B
      wall_logits      (B, 3)        kriz tipi: Anterior / Inferior / Lateral MI               — Başlık B
      spatial_attention (B, 12, 12)  son Spatial Attention katmanının ağırlıkları (sorgu × anahtar,
                                     başlık ve zaman ortalaması). Satır i: derivasyon i'nin hangi sağlam
                                     kanallardan beslendiği (kopan kanallar için temsil ağırlıkları)
      lead_importance  (B, 12)       her sağlam derivasyonun aldığı ortalama uzamsal dikkat (toplam 1,
                                     kopanlarda 0) — 3D duvar haritalaması için (Md. 2.5)
      M                (B, 12)       kullanılan maske
    """

    def __init__(self, in_channels=12, seq_len=1000, embed_dim=128, depth=2, num_heads=4, dropout=0.1, use_stft=True,
                 amplitude_aware=False, pop_sigma=None):
        super().__init__()
        if seq_len % 8 != 0:
            raise ValueError("seq_len 8'e tam bölünmeli (3 adet 2x örnek azaltma)")
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_tokens = seq_len // 8
        D = embed_dim

        # --- Ortak omurga: 1D-CNN zamansal bloklar ---
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, 2, 3, bias=False), nn.GroupNorm(8, 32), nn.GELU())  # 1000 -> 500
        self.enc1 = ResBlock1d(32, 64, stride=2)   # 500 -> 250
        self.enc2 = ResBlock1d(64, D, stride=2)    # 250 -> 125
        self.stft = STFTEncoder(D, seq_len=seq_len) if use_stft else None

        # --- Spatio-Temporal Transformer (darboğaz) ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, D))
        self.lead_embed = nn.Parameter(torch.zeros(1, in_channels, 1, D))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, self.num_tokens, D))
        self.blocks = nn.ModuleList([SpatioTemporalBlock(D, num_heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)

        # --- Genlik farkındalığı (isteğe bağlı) ---
        # Derivasyon bazlı z-score (Md. 2.1) her kanalın genliğini siler; kopan kanalın genliği ise bilinemez.
        # amplitude_aware=True: sağlam kanalların log-genliği (σ) lead token'larına eklenir ve rekonstrüksiyon
        # fiziksel ölçekte üretilir: çıktı = mV / σ_pop (σ_pop: eğitim popülasyonunun derivasyon başına medyan σ'sı).
        self.amplitude_aware = amplitude_aware
        self.physical_recon = amplitude_aware
        if amplitude_aware:
            if pop_sigma is None:
                raise ValueError("amplitude_aware=True için pop_sigma (12 değer) gerekli")
            self.register_buffer("pop_sigma", torch.as_tensor(pop_sigma, dtype=torch.float32))
            self.amp_embed = nn.Sequential(nn.Linear(2, D), nn.GELU(), nn.Linear(D, D))

        # --- IRC ara katman bağlantıları ---
        self.irc_fusing = LeadFusing(64, in_channels, num_heads)   # 250 çözünürlük, derivasyonlar arası birleştirme
        self.stem_mask_token = nn.Parameter(torch.zeros(1, 32, 1))  # 500 çözünürlük, kopan kanallar maskelenir

        # --- Başlık A: simetrik 1D Transpose Convolution kod çözücü ---
        self.up3 = nn.ConvTranspose1d(D, 64, 4, 2, 1)    # 125 -> 250   (enc2'nin simetriği)
        self.dec3 = ResBlock1d(128, 64)
        self.up2 = nn.ConvTranspose1d(64, 32, 4, 2, 1)   # 250 -> 500   (enc1'in simetriği)
        self.dec2 = ResBlock1d(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 32, 4, 2, 1)   # 500 -> 1000  (stem'in simetriği)
        self.dec1 = ResBlock1d(32, 32)
        self.recon_out = nn.Conv1d(32, 1, 1)

        # --- Başlık B: Global Average Pooling + Dense ---
        self.dense = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout))
        self.mi_head = nn.Linear(D, 1)
        self.wall_head = nn.Linear(D, len(WALL_NAMES))

        for p in (self.mask_token, self.lead_embed, self.time_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x, M=None, sigma=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        if M is None:
            M = torch.ones(B, C, device=x.device)
        M = M.to(x.dtype)
        x = x * M.unsqueeze(-1)                       # X_masked = X ⊙ M (çağıran uygulamışsa etkisiz)
        dropped = M < 0.5

        # 1. Ortak omurga (derivasyon başına)
        h = x.reshape(B * C, 1, L)
        s0 = self.stem(h)                  # (B*C, 32, 500)
        s1 = self.enc1(s0)                 # (B*C, 64, 250)
        z = self.enc2(s1)                  # (B*C, D, 125)
        if self.stft is not None:
            z = z + self.stft(x.reshape(B * C, L))
        D, T = z.shape[1], z.shape[2]

        tokens = z.reshape(B, C, D, T).permute(0, 1, 3, 2)  # (B, C, T, D)
        tokens = torch.where(dropped.view(B, C, 1, 1), self.mask_token.expand(B, C, T, D), tokens)
        tokens = tokens + self.lead_embed + self.time_embed
        if self.amplitude_aware:
            if sigma is None:
                raise ValueError("amplitude_aware model sağlam kanalların σ'sını ister")
            log_amp = torch.log(sigma.to(x.dtype).clamp(min=1e-3) / self.pop_sigma.to(x.dtype)) * M  # kopanlar 0
            amp = self.amp_embed(torch.stack([log_amp, M], dim=-1))                                # (B, C, D)
            tokens = tokens + amp.unsqueeze(2)
        spatial_attention = None
        for i, blk in enumerate(self.blocks):
            tokens, w = blk(tokens, M, return_weights=(i == len(self.blocks) - 1))
            if w is not None:
                spatial_attention = w
        tokens = self.norm(tokens)                           # (B, C, T, D)

        # 2. Başlık A: rekonstrüksiyon
        s1 = self.irc_fusing(s1.reshape(B, C, 64, -1), M).reshape(B * C, 64, -1)
        s0 = torch.where(dropped.reshape(B * C, 1, 1), self.stem_mask_token.expand_as(s0), s0)
        d = tokens.reshape(B * C, T, D).transpose(1, 2)
        d = self.dec3(torch.cat([self.up3(d), s1], dim=1))
        d = self.dec2(torch.cat([self.up2(d), s0], dim=1))
        d = self.dec1(self.up1(d))
        recon = self.recon_out(d).reshape(B, C, L)

        # 3. Başlık B: Global Average Pooling (zaman + sağlam derivasyonlar) + Dense
        keep = M.view(B, C, 1, 1)
        pooled = (tokens * keep).sum(dim=(1, 2)) / (keep.sum(dim=1).squeeze(-1) * T)  # (B, D)
        h = self.dense(pooled)

        # XAI: her sağlam derivasyonun sağlam sorgulardan aldığı ortalama dikkat
        q_keep = M.unsqueeze(-1)                                                        # (B, C, 1)
        received = (spatial_attention * q_keep).sum(dim=1) / q_keep.sum(dim=1)          # (B, C)
        lead_importance = received / received.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return {
            "recon": recon,
            "mi_logit": self.mi_head(h).squeeze(-1),
            "wall_logits": self.wall_head(h),
            "spatial_attention": spatial_attention,
            "lead_importance": lead_importance,
            "M": M,
        }


class ResNet1DBaseline(nn.Module):
    """
    Karşılaştırma için klasik 1D-ResNet sınıflandırıcı (Ribeiro ve ark. [1] benzeri, küçültülmüş).
    LeadDropout ile eğitilmez; kopan kanallar yalnızca sıfırlanmış girdi olarak görülür. Rekonstrüksiyon yapmaz.
    """

    def __init__(self, in_channels=12, width=64, dropout=0.1, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, width, 15, 1, 7, bias=False), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(
            ResBlock1d(width, width, 2, 15), ResBlock1d(width, 2 * width, 2, 15),
            ResBlock1d(2 * width, 2 * width, 2, 15), ResBlock1d(2 * width, 4 * width, 2, 15),
        )
        self.drop = nn.Dropout(dropout)
        self.mi_head = nn.Linear(4 * width, 1)
        self.wall_head = nn.Linear(4 * width, len(WALL_NAMES))

    def forward(self, x, M=None):
        if M is None:
            M = torch.ones(x.shape[0], x.shape[1], device=x.device)
        x = x * M.to(x.dtype).unsqueeze(-1)
        h = self.drop(self.blocks(self.stem(x)).mean(dim=-1))
        return {"recon": None, "mi_logit": self.mi_head(h).squeeze(-1), "wall_logits": self.wall_head(h),
                "spatial_attention": None, "lead_importance": None, "M": M}


ARCHITECTURES = {"mae": ECGMaskedAutoencoder, "resnet1d": ResNet1DBaseline}


def build_model(arch, config):
    return ARCHITECTURES[arch](**config)


if __name__ == "__main__":
    from src.lead_dropout import random_keep_mask, wall_keep_mask

    print("Çift başlıklı MAE test ediliyor...")
    model = ECGMaskedAutoencoder()
    x = torch.randn(4, 12, 1000)
    M = random_keep_mask(4, critical_prob=0.5)
    out = model.train()(x, M)
    print(f"Eğitim  -> recon {tuple(out['recon'].shape)}, mi {tuple(out['mi_logit'].shape)}, "
          f"duvar {tuple(out['wall_logits'].shape)}, kopan kanal/örnek {(12 - M.sum(1)).tolist()}")
    out = model.eval()(x, wall_keep_mask(4, "Anterior"))
    print(f"Çıkarım -> Anterior (V1–V4) kopuk, uzamsal dikkat {tuple(out['spatial_attention'].shape)}, "
          f"önem toplamı {out['lead_importance'].sum(1).tolist()}")
    print(f"MAE parametre: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M   "
          f"1D-ResNet parametre: {sum(p.numel() for p in ResNet1DBaseline().parameters()) / 1e6:.2f} M")
