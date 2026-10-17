# EKG MAE Karar Destek API'si (İP5)

FastAPI servisi; arayüze canlı 12 derivasyonlu sinyal, kopuk kanal rekonstrüksiyonu
ve teşhis/XAI çıktısı sağlar.

## Çalıştırma

```bash
pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --port 8000
```

Frontend (`npm run dev`) `/api` isteklerini Vite proxy ile `:8000`'e iletir. Backend
kapalıysa arayüz otomatik olarak tarayıcı içi Mock-API'ye düşer ve başlıkta
`Mock-API (backend kapalı)` yazar.

Testler: `cd backend && python -m unittest discover -s tests -t .`

## Uç noktalar

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/api/health` | Durum ve aktif model kaynağı |
| POST | `/api/chunk` | `{start, count, condition, disconnectedLeads}` → ham sinyal + kopuk kanalların rekonstrüksiyonu |
| POST | `/api/predict` | `{condition, disconnectedLeads}` → tanı, güven, duvar olasılıkları, derivasyon başına ST (mm), dikkat ağırlıkları |

`disconnectedLeads` en fazla 4 derivasyon içerebilir (rapor: 1–4 kanal kopması).

## Hat

1. **Ön işleme** (`preprocessing.py`, rapor 2.1): 0.5–45 Hz, 4. derece, çift yönlü
   Butterworth; derivasyon bazında Z-score; kopuk kanallar 0.
2. **Çıkarım** (`inference.py`):
   - `MODEL_PATH` bir `.onnx` ya da TorchScript `.pt` dosyasını gösteriyorsa İP3 modeli kullanılır.
   - Aksi hâlde öğrenme içermeyen **fizyolojik yedek** çalışır: ekstremite
     derivasyonları Einthoven/Goldberger bağıntılarıyla, prekordiyal derivasyonlar
     komşu enterpolasyonla tamamlanır; teşhis "≥2 komşu derivasyonda ≥1 mm ST
     yükselmesi" ölçütüyle konur. Enterpolasyonla tamamlanan kanalların kanıt
     ağırlığı düşürülür. V1–V4'ün tamamı koptuğunda bu yedek ön duvar MI'ı
     yakalayamaz; MAE modelinin kapatması beklenen boşluk budur.
3. **Sinyal kaynağı** (`signals.py`): PTB-XL entegrasyonuna kadar sentetik 100 Hz EKG.
   Ekstremite derivasyonları Einthoven'a uygundur; MI senaryolarında hasar vektörü
   karşı duvarlarda resiprokal ST çökmesi de üretir.

## İP3 model sözleşmesi

Zilan'ın eğittiği model şu imzayla dışa aktarılmalıdır:

- Girdi: `x` float32 `[1, 12, 1000]` (Z-score, kopuk kanallar 0), `mask` float32 `[1, 12]` (1 = sağlam)
- Çıktı: `recon` `[1, 12, 1000]` (Z-score uzayı), `logits` `[1, 4]`
  (sıra: normal, anterior, inferior, lateral), opsiyonel `attention` `[1, 12]`

ONNX için giriş adları `x` ve `mask` olmalıdır. Rekonstrüksiyonu mV'a çevirmek için
modelin yanına eğitim kümesi derivasyon standart sapmalarıyla `<ad>.stats.json`
(`{"std": [12 değer]}`) konur.

```bash
MODEL_PATH=backend/models/mae.onnx python -m uvicorn app.main:app --app-dir backend --port 8000
```

## 3D model atfı

`public/heart.glb`, NIH HuBMAP **Human Reference Atlas** (Visible Human Male, `VH_M_`)
kalp modelidir; CC BY 4.0 lisanslıdır. Tez ve raporlarda kaynak gösterilmelidir:
HuBMAP Consortium, *Human Reference Atlas 3D Reference Objects*, https://humanatlas.io
