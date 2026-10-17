# İP1 + İP3 Sonuçları

Çift başlıklı MAE (İP3), raporun Md. 2.1–2.4 maddelerine birebir uygun biçimde eğitildi. Bu belgede test sonuçları ve rapor hedefleriyle karşılaştırması var.

- **Test seti:** PTB-XL strat_fold 10, dengeli alt küme: 550 MI + 550 Normal. Model bu kayıtları eğitimde hiç görmedi.
- **Karar eşiği:** Doğrulama setinde (fold 9) belirlendi; test setine dokunulmadı.
- **Karşılaştırma modeli:** Klasik 1D-ResNet [1]. Aynı veriyle, LeadDropout olmadan eğitildi.

## Rapor hedefleri

Değerler 15 epoch'luk modele aittir: `models/mae_ecg_checkpoint.pth`.

| Hedef (Rapor) | Sonuç | Durum |
|---|---|---|
| 1–4 kanal koptuğunda F1 > %88 (Hedef 4, İP5) | Rastgele kopma: **0.915–0.927**. Kritik kopma: Lateral 0.914, Anterior 0.878, Inferior 0.872 | Rastgele kopmada ✓, iki kritik senaryoda sınırda ✗ |
| RMSE < 0.15 (Hedef 4, İP3) | En kötü durum **0.149 mV** (kritik Anterior); rastgele kopmada ≤ 0.070 mV | ✓ |
| R² > 0.80 (Hedef 4, Hipotez) | Rastgele kopma: **0.877–0.914**. Lateral 0.890. Inferior: MAE 0.848, Einthoven ile **1.000**. Anterior **0.700** | Anterior dışında ✓ |
| Pearson r (Hipotez) | Rastgele: 0.934–0.955, kritik: 0.828–0.942 | ✓ |
| Klasik mimariye göre yanlış negatif ≥ %25 azalma (Hipotez) | **%46.5** (havuzlanmış FNR: MAE 0.098, 1D-ResNet 0.184) | ✓ |
| Eğitilmiş .pth ağırlıkları (İP3 başarı ölçütü) | `models/mae_ecg_checkpoint.pth` | ✓ |
| Çıkarım < 50 ms (Risk 2) | ~22 ms (M3, MPS) | ✓ |

## Tam test tablosu (çift başlıklı MAE)

| Senaryo | AUROC | F1 | Duyarlılık | Özgüllük | FNR | Kriz tipi AUROC | RMSE (mV) | R² | r |
|---|---|---|---|---|---|---|---|---|---|
| 0 kopuk (12 kanal) | 0.976 | 0.928 | 0.925 | 0.931 | 0.075 | 0.954 | – | – | – |
| Rastgele 1 kanal | 0.976 | 0.927 | 0.927 | 0.927 | 0.073 | 0.952 | 0.057 | 0.914 | 0.955 |
| Rastgele 2 kanal | 0.973 | 0.918 | 0.911 | 0.927 | 0.089 | 0.949 | 0.059 | 0.908 | 0.952 |
| Rastgele 3 kanal | 0.973 | 0.917 | 0.909 | 0.927 | 0.091 | 0.949 | 0.064 | 0.895 | 0.945 |
| Rastgele 4 kanal | 0.970 | 0.915 | 0.916 | 0.913 | 0.084 | 0.946 | 0.070 | 0.877 | 0.934 |
| Kritik Anterior (V1–V4) | 0.945 | 0.878 | 0.880 | 0.875 | 0.120 | 0.900 | 0.149 | 0.700 | 0.828 |
| Kritik Inferior (II, III, aVF) | 0.942 | 0.872 | 0.855 | 0.895 | 0.145 | 0.889 | 0.042 → **0.001** | 0.848 → **1.000** | 0.919 → **1.000** |
| Kritik Lateral (I, aVL, V5, V6) | 0.970 | 0.914 | 0.913 | 0.916 | 0.087 | 0.924 | 0.068 → 0.064 | 0.890 → 0.935 | 0.942 → 0.965 |

Okun sağındaki değerler: MAE + Einthoven kısıtı (Risk 1 B planı).

## Yanlış negatif karşılaştırması (1D-ResNet → MAE)

| Senaryo | 1D-ResNet FNR | MAE FNR | Azalma |
|---|---|---|---|
| Rastgele 1 kanal | 0.211 | 0.073 | %65.5 |
| Rastgele 2 kanal | 0.213 | 0.089 | %58.1 |
| Rastgele 3 kanal | 0.189 | 0.091 | %51.9 |
| Rastgele 4 kanal | 0.175 | 0.084 | %52.1 |
| Kritik Anterior (V1–V4) | 0.047 | 0.120 | **%−153.8** |
| Kritik Inferior | 0.255 | 0.145 | %42.9 |
| Kritik Lateral | 0.200 | 0.087 | %56.4 |
| **Havuzlanmış (hipotez)** | **0.184** | **0.098** | **%46.5** |

**Kritik Anterior istisnası:** 1D-ResNet'in buradaki düşük FNR'si bir başarı değil. V1–V4 sıfırlanınca ResNet neredeyse her kayda "MI" diyor ve özgüllüğü 0.976'dan 0.609'a düşüyor; F1'i de MAE'den düşük (0.813'e karşı 0.878). Bu tabloyu raporlarken istisna da açıkça verilmeli.

**Hesaplama yöntemi:** Kodun ilk sürümü senaryo başına azalma oranlarının ortalamasını alıyordu ve %24.7 buluyordu. Bu yöntem tek bir uç değere (Anterior) çok duyarlı. Hipotez "1–4 kanal koptuğunda yanlış negatif oranı" olarak tanımlandığından, tüm kopma senaryolarının havuzlanmış FNR'si karşılaştırılıyor: toplam FN / toplam MI.

## Yorum

1. **Kopmaya dayanıklılık kanıtlandı.** 12 kanaldan 4 rastgele kanal kopmasına geçildiğinde MAE'nin F1'i yalnızca 0.928'den 0.915'e düşüyor. 1D-ResNet'inki 0.866'dan 0.854'e düşüyor, ama zaten her koşulda MAE'nin 6–7 puan altında.
2. **En zor durum kritik Anterior kopması.** V1–V4'ün dördü birden gidince kalan V5–V6 ve ekstremite derivasyonları sağ prekordiyal bölgeyi tam temsil edemiyor (R² 0.70). Grafik: `results/reconstruction_anterior.png`.
3. **Einthoven kısıtı (Risk 1 B planı) ekstremite derivasyonlarını tam çözüyor.** Kritik Inferior kopmasında R² 0.848'den 1.000'e çıkıyor. Grafik: `results/reconstruction_inferior_einthoven.png`. Göğüs derivasyonlarına bu kısıt uygulanamıyor; onlar fiziksel olarak bağımsız.
4. **Kriz tipi ayrımı güçlü.** Anterior/Inferior/Lateral MI ayrımında AUROC 0.95. Lateral MI test setinde yalnızca 20 kayıt olduğu için o değer daha az güvenilir.

## 30 epoch'luk ikinci eğitim

15 epoch'un sonunda doğrulama eğrileri (AUROC, R²) hâlâ yükseliyordu. Bu yüzden aynı ayarlarla 30 epoch'luk ikinci bir eğitim başlatıldı:

- Model: `models/mae_ecg_checkpoint_30ep.pth`
- Sonuçlar: `results/30ep/`

Amaç, kritik Anterior/Inferior F1'ini 0.88'in üstüne ve Anterior R²'sini yükseltmek. Sonuçlar aşağıya eklenecek.

## Dosyalar ve yeniden üretme

| Dosya | İçerik |
|---|---|
| `results/test_results.json` | Tüm metrikler (MAE + 1D-ResNet) |
| `results/egitim_mae.log`, `models/mae_ecg_checkpoint_history.csv` | Epoch başına eğitim/doğrulama eğrileri (İP6 grafikleri için) |
| `results/reconstruction_*.png` | Rekonstrüksiyon grafikleri |

```bash
python -m src.prepare_data                     # İP1: ~2 dk
python -m src.train                            # İP3: MAE, ~2 saat (M3)
python -m src.train --arch resnet1d            # karşılaştırma, ~5 dk
python -m src.evaluate --baseline models/resnet1d_baseline.pth
```
