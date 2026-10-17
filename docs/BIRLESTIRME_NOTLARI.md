# Backend ↔ Arayüz Birleştirme Notları

Bu belge, Zilan'ın backend tarafını (İP1, İP3, İP5-backend) Sema'nın arayüzüyle (İP2, İP4) birleştirmek için hazırlandı.

## Çalıştırma (proje kök klasöründen)

```bash
python -m src.prepare_data          # bir kez: data/processed/ temiz tensör deposu (~2 dk, 1 GB)
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
| `reconstructed_signals` | 12×1000, mV. Sağlam kanallar orijinal; kopan göğüs kanalları MAE rekonstrüksiyonu, kopan ekstremite kanalları Einthoven kısıtıyla hesaplanmış |
| `diagnosis.predicted_class`, `diagnosis.confidence`, `diagnosis.is_mi` | Artık gerçek model çıktısı |
| `dropped_lead_names`, `active_lead_count`, `lead_names` | |
| `wall_damage_scores` | `{Anterior, Inferior, Lateral}`, 0–1. 3D kalp renklendirmesi için: P(MI) × P(duvar ∣ MI) |
| `xai_metrics.most_affected_lead`, `xai_metrics.anatomical_region` | Spatial Attention katmanında en çok dikkat alan derivasyon (Md. 2.5) |
| `mse_per_lead`, `anomaly_score` | z-score biriminde rekonstrüksiyon hatası. `anomaly_score` kopan kanalların ortalamasıdır |

**Yeni alanlar:**

| Alan | Açıklama |
|---|---|
| `mi_probability` | 0–1 |
| `einthoven_completed_leads` | Einthoven bağıntılarıyla kesin hesaplanan kopan ekstremite derivasyonları (Risk 1 B planı) |
| `diagnosis.affected_wall` | MI ise en olası duvar |
| `lead_attention` | `{I: …, V6: …}`, Spatial Attention katmanından derivasyon önemi; toplamı 1, kopan kanallarda 0. XAI ısı haritası için |
| `spatial_attention` | 12×12 matris (sorgu × anahtar). Satır i: derivasyon i'nin hangi sağlam kanallardan beslendiği |
| `reconstruction_sources` | Her kopan kanal için onu en çok temsil eden 3 sağlam kanal ve ağırlığı, örn. `{"V2": {"V3": 0.31, …}}` |
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

Rapor sabittir; kod rapora birebir uyar.

| Rapor maddesi | Kodda | Dosya |
|---|---|---|
| İP1: PTB-XL, SCP-ECG etiket matrisleri | `Y.npy` (NORM, MI, STTC, CD, HYP, AMI, IMI, LMI, PMI) + `meta.csv` | `src/prepare_data.py` |
| Md. 2.1: MI (Anterior/Inferior/Lateral) ve Normal, dengeli alt kümeler | Her bölümde tüm MI + aynı sayıda Normal (eğitim 4379/4379, doğrulama 540/540, test 550/550) | `src/prepare_data.py`, `src/dataset.py` |
| İP1: 0.5–45 Hz, 4. derece, çift yönlü Butterworth | ✓ | `src/preprocessing.py` |
| İP1: Z-score, [12, 1000] tensör, 100 Hz | ✓ derivasyon bazında | `src/preprocessing.py` |
| İP1 başarı ölçütü: temizlenmiş tensör deposu + `preprocessing.py` | `data/processed/` + `src/preprocessing.py` | |
| Md. 2.1: FFT/STFT bimodal | STFT dalı omurgaya ekleniyor | `src/model.py` (`STFTEncoder`) |
| Md. 2.2: LeadDropout, DataLoader seviyesinde, batch başına k ∈ [1,4], M ∈ {0,1}¹², X_masked = X ⊙ M·1ᵀ | ✓ ayrıca "kritik" duvar kopması (Hedef 2) | `src/lead_dropout.py` |
| Md. 2.3.1: 1D-CNN + Lead Fusing + Spatio-Temporal Transformer, Softmax(QKᵀ/√d_k)V | ✓ çok başlıklı uzamsal dikkat; anahtarlar sağlam kanallar | `src/model.py` |
| Md. 2.3.2: simetrik 1D Transpose Conv kod çözücü + IRC | ✓ | `src/model.py` |
| Md. 2.3.3: Global Average Pooling + Dense, MI ve kriz tipi | ✓ | `src/model.py` |
| Md. 2.4: 0.4·MSE(yalnızca maskeli kanallar) + 0.6·Focal | ✓ | `src/losses.py` |
| Md. 2.5: XAI, Spatial Attention ağırlıkları → 3D duvarlar | `lead_attention`, `spatial_attention`, `reconstruction_sources` | `src/model.py`, `src/main.py` |
| Hedef 4: 1–4 kanal kopmasında F1 > 0.88, RMSE < 0.15, R² > 0.80 | Test setinde ölçülüyor | `src/evaluate.py` |
| Risk 1 B planı: Einthoven üçgeni fizyolojik kısıtı | Kopan ekstremite derivasyonları (I, II, III, aVR, aVL, aVF) sağlam olanlardan kesin hesaplanır; saf MAE ve MAE + Einthoven metrikleri ayrı raporlanır | `src/einthoven.py` |
| Hipotez: 1D-ResNet'e göre yanlış negatif ≥ %25 azalma, Pearson r | Ölçülüyor | `src/evaluate.py --baseline` |
| İP3 başarı ölçütü: eğitilmiş `.pth` | `models/mae_ecg_checkpoint.pth` | `src/train.py` |
| İP5: FastAPI, Click-to-Explain verisi | ✓ backend | `src/main.py` |

### Veri ve ölçüm notları

- **Kayıt sayısı:** Rapordaki 21.837 kayıt (18.885 hasta), PTB-XL'in ilk sürümündeki sayıdır. PhysioNet v1.0.2'de yinelenen 38 kayıt çıkarılmıştır (`data/ptbxl/ptbxl_v102_changelog.txt`). Kullanılan v1.0.3 aynı kayıtların tekilleştirilmiş hâlidir: 21.799 kayıt, 18.869 hasta.
- **RMSE:** `src/evaluate.py` hem z-score hem mV biriminde raporlar. R² ve Pearson r de yanında verilir.
- **Lateral MI:** Test setinde yalnızca 20 kayıt var. Lateral duvar AUROC'u bu yüzden geniş güven aralıklı olacak.
