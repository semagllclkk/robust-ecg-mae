# Backend ↔ Arayüz Birleştirme Notları

Bu belge, Zilan'ın backend tarafını (İP1, İP3, İP5-backend) Sema'nın arayüzüyle (İP2, İP4) birleştirmek için hazırlandı.

## Çalıştırma (proje kök klasöründen)

```bash
python -m src.prepare_data          # bir kez: data/processed/ önbelleği (~2 dk, 1 GB)
python -m src.train                 # MAE eğitimi -> models/mae_ecg_checkpoint.pth
python -m src.train --arch resnet1d # karşılaştırma modeli -> models/resnet1d_baseline.pth
python -m src.evaluate --baseline models/resnet1d_baseline.pth   # stres testleri -> results/
uvicorn src.main:app --port 8000    # API
```

Eğitim ve API aynı anda çalışırsa ikisi de çok yavaşlar, çünkü Apple GPU'yu paylaşırlar.

## API sözleşmesi

### `GET /api/sample-ecg`

Model, PTB-XL test setinden (strat_fold 10) gerçek bir kayıt seçer; bu kayıtları eğitimde hiç görmemiştir.

| Parametre | Değerler |
|---|---|
| `condition` | `normal`, `mi`, `ant`, `inf`, `lat`, `sttc` |
| `dropped` | Virgülle ayrılmış derivasyon adları, örn. `V1,V2` (en fazla 4) |
| `active_leads` | 8–12; `dropped` boşsa bu sayıya göre rastgele kanal koparılır |

### `POST /api/predict`

Kullanıcının kendi EKG'si için. Gövde: `{"signal": 12×N mV, "fs": 100, "missing_leads": ["V1"]}`. Sinyal 100 Hz'e yeniden örneklenir ve 10 saniyeye kırpılır.

### Yanıt alanları

**Eski alanlar.** Arayüzün okuduğu bu alanların adları değişmedi:

| Alan | Açıklama |
|---|---|
| `raw_signals` | 12×1000, mV, filtreli orijinal sinyal |
| `reconstructed_signals` | 12×1000, mV. Sağlam kanallar orijinal, kopan kanallar MAE rekonstrüksiyonu |
| `diagnosis.predicted_class`, `diagnosis.confidence`, `diagnosis.is_mi` | Artık gerçek model çıktısı |
| `dropped_lead_names`, `active_lead_count`, `lead_names` | |
| `wall_damage_scores` | `{Anterior, Inferior, Lateral}`, 0–1. 3D kalp renklendirmesi için: P(MI) × P(duvar ∣ MI) |
| `xai_metrics.most_affected_lead`, `xai_metrics.anatomical_region` | Teşhis başlığının en çok dikkat verdiği derivasyon |
| `mse_per_lead`, `anomaly_score` | z-score biriminde rekonstrüksiyon hatası. `anomaly_score` kopan kanalların ortalamasıdır |

**Yeni alanlar:**

| Alan | Açıklama |
|---|---|
| `mi_probability` | 0–1 |
| `diagnosis.affected_wall` | MI ise en olası duvar |
| `lead_attention` | `{I: …, V6: …}`, toplamı 1, kopan kanallarda 0. XAI ısı haritası için |
| `click_to_explain.{Anterior,Inferior,Lateral}` | 3D duvara tıklanınca gösterilecek kart: `explanation` (hazır Türkçe metin), `damage_score`, `st_deviation_mm` (derivasyon başına), `max_st_deviation_mm`, `suspected_artery`, `missing_leads`, `decision_confidence`, `attention_share` |
| `st_deviation_mm`, `st_reference_lead` | Tüm derivasyonlar için ST sapması (mm) |
| `ground_truth` | `{label, mi_walls}`. Demo kaydının gerçek etiketi (yalnızca `sample-ecg`'de) |
| `model_loaded` | `false` ise eğitilmiş ağırlık yok, çıktılar anlamsızdır |
| `inference_ms` | Model çıkarım süresi (Rapor Risk 2 hedefi < 50 ms) |
| `ecg_id`, `source`, `selected_condition`, `units` | |

## Arayüzde birlikte düzeltilmesi gerekenler (Sema)

1. **Teşhis sabit yazılıyor.** `src/services/ecgApi.js`, `condition` `normal`, `ant` veya `inf` iken backend'in teşhisini yok sayıyor. Etiketi ve güveni sabit yazıp her kopan kanal için %3.5 düşürüyor. Bu, rapordaki "kopmaya rağmen teşhis" iddiasını sınamayı engelliyor. `data.diagnosis` her senaryoda doğrudan kullanılmalı.
2. **`smoothAndNormalize` sinyali bozuyor.** Her derivasyonu ±0.75'e ölçekliyor. Bu, derivasyonlar arası genlik farkını ve ST seviyesini (mV) siliyor. Klinik gösterim için mV ölçeği korunmalı.
3. **Senaryo adları farklı.** Arayüz `mi` yerine `ant`/`inf` gönderiyor; backend artık `normal, mi, ant, inf, lat, sttc` değerlerinin hepsini kabul ediyor.
4. **PDF parametreleri sabit.** `App.jsx` PDF raporuna `anomaly_score: '0.02500'` gibi sabit değerler gönderiyor. Yanıttaki gerçek `anomaly_score`, `xai_metrics.most_affected_lead` ve `click_to_explain` değerleri kullanılmalı. Ayrıca `confidence / 100` hatalı: güven zaten 0–1 aralığında.
5. **3D kalp backend verisini okumuyor.** `AnatomicHeart.jsx` şu an yalnızca `disconnectedLeads` alıyor. `wall_damage_scores` (renk gradyanı) ve `click_to_explain` (tıklama kartı) bağlanmalı (İP4).

## Rapor ↔ kod durumu (Zilan'ın iş paketleri)

| Rapor maddesi | Durum | Dosya |
|---|---|---|
| İP1: PTB-XL, MI (Anterior/Inferior/Lateral) ve Normal etiketleri | ✓ | `src/prepare_data.py` |
| İP1: 0.5–45 Hz, 4. derece, çift yönlü Butterworth | ✓ | `src/preprocess.py` |
| İP1: Z-score, [12, 1000] tensör, 100 Hz | ✓ derivasyon ve kayıt bazında | `src/preprocess.py` |
| Md. 2.1: FFT/STFT bimodal | ✓ STFT dalı darboğaza ekleniyor | `src/model.py` (`STFTEncoder`) |
| Md. 2.2: LeadDropout, k ∈ [1,4], X ⊙ M | ✓ ve "kritik" duvar kopması | `src/model.py` (`LeadDropout`) |
| Md. 2.3.1: 1D-CNN + Spatial Attention + Spatio-Temporal Transformer | ✓ | `src/model.py` |
| Md. 2.3.2: 1D Transpose Conv kod çözücü + IRC | ✓ | `src/model.py` |
| Md. 2.3.3: GAP + Dense, MI ve kriz tipi | ✓ | `src/model.py` |
| Md. 2.4: 0.4·MSE(maskeli) + 0.6·Focal | ✓ | `src/losses.py` |
| Hedef 4: F1 > 0.88, RMSE < 0.15, R² > 0.80 | Ölçülüyor; sonuçlar eğitim bitince | `src/evaluate.py` |
| Hipotez: 1D-ResNet'e göre FNR ≥ %25 azalma | Ölçülüyor | `src/evaluate.py --baseline` |
| İP5: FastAPI, gerçek model çıktısı, Click-to-Explain verisi | ✓ backend | `src/main.py` |
| Risk 2: ONNX, < 50 ms | Gerekirse; `inference_ms` ile izleniyor | |

### Rapor metninde düzeltilmesi önerilenler

- PTB-XL v1.0.3'te **21.799** kayıt var (raporda 21.837; bu sayı eski sürüme ait).
- Ham PTB-XL'de **18.869** hasta var (raporda 18.885).
- Lateral MI (LMI) test setinde yalnızca 20 kayıt. Lateral duvar sonuçları istatistiksel olarak zayıf kalacak.
- RMSE < 0.15 hedefinin birimi belirtilmemiş. Kod hem z-score hem mV cinsinden raporluyor; z-score biriminde 0.15, R² ≈ 0.98'e karşılık gelir ve çok iddialıdır. mV olarak tanımlanması önerilir.
