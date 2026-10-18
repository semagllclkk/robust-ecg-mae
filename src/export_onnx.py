"""
Eğitilmiş MAE'yi ONNX Runtime formatına dışa aktarır (TÜBİTAK Raporu Risk 2 B planı:
"Eğitilen PyTorch modeli ONNX Runtime formatına derlenerek çıkarım süresi 50 ms'nin altına indirilecek").

ONNX, STFT dalındaki `unfold` + `torch.fft.rfft` işlemlerini dışa aktaramaz. Dışa aktarım sırasında aynı
hesap, sabit ağırlıklı bir Conv1d olarak yazılır: pencerelenmiş gerçek DFT (cos/sin çekirdekleri, adım = hop).
Bu matematiksel olarak özdeştir; betik PyTorch ve ONNX çıktılarını karşılaştırıp farkı doğrular.
Eğitim kodu ve checkpoint değişmez.

Çalıştırma (proje kök klasöründen):
    python -m src.export_onnx                      # models/mae_ecg_checkpoint.pth -> models/mae_ecg.onnx
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.model import STFTEncoder
from src.train import load_checkpoint

OUTPUT_KEYS = ["recon", "mi_logit", "wall_logits", "spatial_attention", "lead_importance"]


class _ConvSTFT(torch.nn.Module):
    """STFTEncoder ile aynı çıktıyı ONNX'e aktarılabilir işlemlerle üretir."""

    def __init__(self, stft: STFTEncoder):
        super().__init__()
        n_fft, hop = stft.n_fft, stft.hop
        self.hop, self.num_frames, self.proj = hop, stft.num_frames, stft.proj
        self.pad = (n_fft - hop) // 2
        k = torch.arange(n_fft, dtype=torch.float64)
        j = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        window = stft.window.to(torch.float64)
        angle = 2 * math.pi * j * k / n_fft
        self.register_buffer("cos_kernel", (torch.cos(angle) * window).float().unsqueeze(1))
        self.register_buffer("sin_kernel", (-torch.sin(angle) * window).float().unsqueeze(1))

    def forward(self, x):
        h = F.pad(x, (self.pad, self.pad)).unsqueeze(1)                       # (N, 1, L + 2·pad)
        real = F.conv1d(h, self.cos_kernel, stride=self.hop)[..., : self.num_frames]
        imag = F.conv1d(h, self.sin_kernel, stride=self.hop)[..., : self.num_frames]
        return self.proj(torch.log1p(torch.sqrt(real * real + imag * imag)))  # (N, D, num_frames)


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, sigma):
        extra = {"sigma": sigma} if getattr(self.model, "amplitude_aware", False) else {}
        out = self.model(x, mask, **extra)
        return tuple(out[k] for k in OUTPUT_KEYS)


def export(checkpoint: str, output: str, opset: int = 17):
    model, ckpt = load_checkpoint(checkpoint, torch.device("cpu"))
    model.eval()
    wrapper = _ExportWrapper(model)
    sigma = torch.full((1, 12), 0.2)

    # Referans çıktılar dışa aktarımdan ÖNCE alınır: torch.onnx.export modülü izlerken durumunu değiştirir
    generator = torch.Generator().manual_seed(0)
    cases = []
    for dropped in ([], [7], [1, 2, 5], [6, 7, 8, 9]):
        m = torch.ones(1, 12)
        m[0, dropped] = 0
        xx = torch.randn(1, 12, 1000, generator=generator) * m.unsqueeze(-1)
        with torch.no_grad():
            cases.append((xx, m, [t.clone() for t in wrapper(xx, m, sigma)]))

    original_stft = model.stft
    if original_stft is not None:
        model.stft = _ConvSTFT(original_stft)
    x = torch.randn(1, 12, 1000, generator=generator)
    mask = torch.ones(1, 12)
    mask[0, [2, 7]] = 0
    torch.onnx.export(
        wrapper, (x * mask.unsqueeze(-1), mask, sigma), output,
        input_names=["x", "mask", "sigma"], output_names=OUTPUT_KEYS,
        opset_version=opset, dynamo=False,
    )
    model.stft = original_stft

    import onnxruntime as ort

    session = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
    inputs = {i.name for i in session.get_inputs()}  # genlik farkındasız modelde "sigma" grafikten düşer

    def feed(x_, m_, s_):
        return {k: v for k, v in {"x": x_, "mask": m_, "sigma": s_}.items() if k in inputs}

    worst = 0.0
    for xx, m, ref in cases:
        got = session.run(None, feed(xx.numpy(), m.numpy(), sigma.numpy()))
        worst = max(worst, max(float(np.abs(a.numpy() - b).max()) for a, b in zip(ref, got)))

    timings = []
    sample = feed(x.numpy(), mask.numpy(), sigma.numpy())
    for _ in range(30):
        t0 = time.perf_counter()
        session.run(None, sample)
        timings.append((time.perf_counter() - t0) * 1000)

    print(f"ONNX modeli yazıldı: {output} (epoch {ckpt.get('epoch')})")
    print(f"PyTorch ile en büyük mutlak fark: {worst:.2e}")
    print(f"ONNX Runtime çıkarım süresi (CPU): medyan {np.median(timings[5:]):.1f} ms")
    if worst > 1e-3:
        raise SystemExit("ONNX çıktısı PyTorch ile uyuşmuyor; dışa aktarım reddedildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE -> ONNX dışa aktarımı (Risk 2 B planı)")
    parser.add_argument("--checkpoint", default="models/mae_ecg_checkpoint.pth")
    parser.add_argument("--output", default="models/mae_ecg.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
