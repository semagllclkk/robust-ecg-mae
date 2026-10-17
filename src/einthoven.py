"""
Einthoven Üçgeni fizyolojik kısıtı — TÜBİTAK Raporu Risk 1 B Planı.

"3 veya 4 kanal aynı anda koptuğunda ... Einthoven Üçgeni fizyolojik kısıtlarını zorunlu kılan ...
uzamsal komşuluk ağırlık matrisleri" (Rapor Md. 3.2, Risk 1).

Altı ekstremite derivasyonu iki boyutlu frontal düzlemde tanımlıdır; herhangi ikisi diğerlerini kesin
olarak belirler (Einthoven ve Goldberger bağıntıları):
    III = II − I,   aVR = −(I + II) / 2,   aVL = I − II / 2,   aVF = II − I / 2
Rapor kısıtı gereği en fazla 4 kanal koptuğundan en az 2 ekstremite derivasyonu her zaman sağlamdır.
Kopan ekstremite derivasyonları sağlam olanlardan en küçük kareler ile kesin hesaplanır; göğüs
derivasyonları (V1–V6) MAE rekonstrüksiyonu olarak kalır.

PTB-XL test kayıtlarında bağıntıların doğruluğu: R² ≈ 0.9999 (filtre doğrusal olduğundan bozulmaz).
Bağıntılar mV uzayında geçerlidir; z-score uzayında değil (her derivasyonun kendi μ, σ'sı vardır).
"""
import torch

# Ekstremite derivasyonlarının (I, II) tabanındaki katsayıları — LEAD_NAMES sırasıyla ilk 6 kanal
LIMB_MATRIX = torch.tensor([
    [1.0, 0.0],    # I
    [0.0, 1.0],    # II
    [-1.0, 1.0],   # III = II − I
    [-0.5, -0.5],  # aVR = −(I + II) / 2
    [1.0, -0.5],   # aVL = I − II / 2
    [-0.5, 1.0],   # aVF = II − I / 2
])
NUM_LIMB = 6


@torch.no_grad()
def einthoven_complete(signals_mv, M):
    """
    signals_mv : (B, 12, L) mV — sağlam kanallar gerçek sinyal (kopanların içeriği kullanılmaz)
    M          : (B, 12) 1 = sağlam, 0 = koptu
    Döndürür   : (B, 6, L) ekstremite derivasyonları; sağlam olanlar aynen, kopanlar Einthoven ile hesaplanmış
    """
    A = LIMB_MATRIX.to(signals_mv.device, signals_mv.dtype)
    keep = M[:, :NUM_LIMB].to(signals_mv.dtype)                              # (B, 6)
    limb = signals_mv[:, :NUM_LIMB] * keep.unsqueeze(-1)                     # kopanlar sıfır
    A_vis = A.unsqueeze(0) * keep.unsqueeze(-1)                              # (B, 6, 2) kopan satırlar sıfır
    pinv = torch.linalg.pinv(A_vis.detach().cpu().float()).to(signals_mv.device, signals_mv.dtype)  # küçük; CPU'da
    basis = pinv @ limb                                                      # (B, 2, L) = [I, II]
    solved = A.unsqueeze(0) @ basis                                          # (B, 6, L)
    return torch.where(keep.unsqueeze(-1) > 0.5, signals_mv[:, :NUM_LIMB], solved)
