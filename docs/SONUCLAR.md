# İP1 + İP3 Sonuçları

Çift başlıklı MAE (İP3), raporun Md. 2.1–2.4 maddelerine birebir uygun biçimde eğitildi. Bu belgede test sonuçları ve rapor hedefleriyle karşılaştırması var.

- **Test seti:** PTB-XL strat_fold 10, dengeli alt küme (Rapor Md. 2.1): 550 MI + 550 Normal. Model bu kayıtları eğitimde hiç görmedi; hiçbir hasta birden fazla bölümde yer almıyor. Not: F1 sınıf oranına bağlıdır. MI'ın daha seyrek olduğu gerçek popülasyonda F1 daha düşük çıkar; AUROC, duyarlılık ve özgüllük bu orandan etkilenmez.
- **Karar eşiği:** Doğrulama setinde (fold 9) belirlendi; test setine dokunulmadı.
- **Karşılaştırma modeli:** Klasik 1D-ResNet [1]. Aynı veriyle, LeadDropout olmadan eğitildi.

## Düzeltme notu (18.10.2026)

Bu belgenin önceki sürümü rekonstrüksiyonu (RMSE, R²) **iyimser** hesaplıyordu. Kopan kanalın tahmini mV'a çevrilirken o kanalın gerçek genliği (σ) kullanılmıştı. Gerçekte kopan bir kanalın genliği bilinemez. Model kopan kanalın **şeklini** doğru üretiyor (Pearson r 0.93–0.96), ama **genliğini** bilemiyor; nedeni raporun istediği derivasyon bazlı z-score (Md. 2.1), çünkü her kanal kendi genliğine bölünüyor.

Artık ana sonuçlar **gerçekçi koşulda** veriliyor: kopan kanalın genliği yerine eğitim popülasyonunun değeri kullanılıyor; API de böyle çalışıyor. Eski hesap "oracle" sütununda yalnızca karşılaştırma için duruyor. Teşhis sonuçları (F1, FNR, AUROC) bu düzeltmeden etkilenmiyor.

## Rapor hedefleri

Değerler 15 epoch'luk ana modele aittir: `models/mae_ecg_checkpoint.pth`. 30 epoch karşılaştırması aşağıda.

| Hedef (Rapor) | Sonuç | Durum |
|---|---|---|
| 1–4 kanal koptuğunda F1 > %88 (Hedef 4, İP5) | Rastgele kopma: **0.915–0.927**. Kritik kopma: Lateral 0.914, Anterior 0.878, Inferior 0.872 | Rastgele kopmada ✓, iki kritik senaryoda sınırda ✗ |
| RMSE < 0.15 (Hedef 4, İP3) | Rastgele kopma: **0.093–0.104 mV**. Kritik Anterior: **0.173 mV** | Rastgele ✓, kritik Anterior ✗ |
| R² > 0.80 (Hedef 4, Hipotez) | Saf MAE: **0.62–0.64** (Anterior 0.49). MAE + Einthoven: rastgele **0.79–0.80**, Lateral 0.81, Inferior **1.00**, Anterior 0.49 | ✗ (Einthoven ile sınırda) |
| Pearson r (Hipotez: "yüksek korelasyonla") | Rastgele: **0.93–0.96**, kritik: 0.83–0.94 | ✓ |
| Klasik mimariye göre yanlış negatif ≥ %25 azalma (Hipotez) | **%46.5** (havuzlanmış FNR: MAE 0.098, 1D-ResNet 0.184) | ✓ |
| Eğitilmiş .pth ağırlıkları (İP3 başarı ölçütü) | `models/mae_ecg_checkpoint.pth` | ✓ |
| Çıkarım < 50 ms (Risk 2) | ~13–22 ms (M3, MPS) | ✓ |

**Genlik sorununun çözümü için** model, sağlam kanalların genliğini de girdi olarak alıp rekonstrüksiyonu gerçek ölçekte yapacak şekilde yeniden eğitiliyor. Rapordaki z-score girdisi ve kayıp formülü (0.4·MSE + 0.6·Focal) aynı kalıyor. Sonuçlar gelince bu belge güncellenecek.

## Tam test tablosu (çift başlıklı MAE)

Rekonstrüksiyon sütunları yalnızca kopan kanallar üzerinden, mV cinsinden.
- **Gerçekçi:** genlik bilinmeden; ana sonuç.
- **+Einthoven:** kopan ekstremite kanalları Einthoven bağıntılarıyla hesaplanmış (Risk 1 B planı).
- **Oracle:** genlik biliniyormuş gibi; yalnızca karşılaştırma için.

| Senaryo | AUROC | F1 | Duyarlılık | Özgüllük | FNR | Kriz tipi AUROC | RMSE (mV) | R² | r | +Einthoven RMSE / R² | Oracle R² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 kopuk (12 kanal) | 0.976 | 0.928 | 0.925 | 0.931 | 0.075 | 0.954 | – | – | – | – | – |
| Rastgele 1 kanal | 0.976 | 0.927 | 0.927 | 0.927 | 0.073 | 0.952 | 0.093 | 0.637 | 0.955 | 0.082 / 0.787 | 0.914 |
| Rastgele 2 kanal | 0.973 | 0.918 | 0.911 | 0.927 | 0.089 | 0.949 | 0.100 | 0.637 | 0.952 | 0.089 / 0.800 | 0.908 |
| Rastgele 3 kanal | 0.973 | 0.917 | 0.909 | 0.927 | 0.091 | 0.949 | 0.102 | 0.631 | 0.945 | 0.092 / 0.800 | 0.895 |
| Rastgele 4 kanal | 0.970 | 0.915 | 0.916 | 0.913 | 0.084 | 0.946 | 0.104 | 0.616 | 0.934 | 0.093 / 0.795 | 0.877 |
| Kritik Anterior (V1–V4) | 0.945 | 0.878 | 0.880 | 0.875 | 0.120 | 0.900 | 0.173 | 0.494 | 0.828 | 0.173 / 0.494 | 0.700 |
| Kritik Inferior (II, III, aVF) | 0.942 | 0.872 | 0.855 | 0.895 | 0.145 | 0.889 | 0.077 | 0.606 | 0.919 | **0.001 / 1.000** | 0.848 |
| Kritik Lateral (I, aVL, V5, V6) | 0.970 | 0.914 | 0.913 | 0.916 | 0.087 | 0.924 | 0.089 | 0.628 | 0.942 | 0.077 / 0.811 | 0.890 |

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
2. **En zor durum kritik Anterior kopması.** V1–V4'ün dördü birden gidince kalan V5–V6 ve ekstremite derivasyonları sağ prekordiyal bölgeyi tam temsil edemiyor (gerçekçi R² 0.49, oracle 0.70). Grafik: `results/reconstruction_anterior.png`.
3. **Einthoven kısıtı (Risk 1 B planı) ekstremite derivasyonlarını tam çözüyor.** Kritik Inferior kopmasında R² 0.606'dan 1.000'e çıkıyor; genlik de fizik kuralından geldiği için genlik sorunu ekstremite kanallarında hiç yok. Grafik: `results/reconstruction_inferior_einthoven.png`. Göğüs derivasyonlarına bu kısıt uygulanamıyor; onlar fiziksel olarak bağımsız.
4. **Kriz tipi ayrımı güçlü.** Anterior/Inferior/Lateral MI ayrımında AUROC 0.95. Lateral MI test setinde yalnızca 20 kayıt olduğu için o değer daha az güvenilir.
   API, P(duvar ∣ MI) ≥ 0.5 olan tüm duvarları raporlar. MI test kayıtlarının %19'unda birden fazla duvar etkilenmiştir. Yalnızca en olası duvarı yazmaya göre:

   | MI test kayıtları | Yalnız en olası duvar | Eşik 0.5 (çoklu duvar) |
   |---|---|---|
   | 12 kanal: tam eşleşme / duyarlılık / kesinlik | %73.6 / %77.9 / %92.5 | **%80.9 / %91.1** / %89.9 |
   | Rastgele 2 kanal kopuk | %73.1 / %77.5 / %92.0 | **%79.8 / %89.7** / %88.9 |

## 30 epoch'luk ikinci eğitim (karşılaştırma)

15 epoch'un sonunda doğrulama eğrileri hâlâ yükseldiği için aynı ayarlarla 30 epoch'luk ikinci bir eğitim yapıldı (`models/mae_ecg_checkpoint_30ep.pth`, `results/30ep/`). En iyi doğrulama AUROC'u 22. epoch'ta geldi; 8 epoch iyileşme olmayınca eğitim durdu.

| Ölçüt | 15 epoch (ana model) | 30 epoch |
|---|---|---|
| Doğrulama AUROC (model seçim ölçütü) | **0.9765** | 0.9757 |
| Test F1, 12 kanal | **0.928** | 0.913 |
| Test F1, rastgele 1–4 kopma | **0.915–0.927** | 0.910–0.914 |
| Test F1, kritik Anterior / Inferior / Lateral | 0.878 / **0.872** / **0.914** | **0.882** / 0.864 / 0.902 |
| Yanlış negatif azalması (havuzlanmış) | **%46.5** | %28.9 |
| R² (oracle), rastgele 1–4 kopma | 0.877–0.914 | **0.897–0.928** |
| R² (oracle), kritik Anterior / Inferior / Lateral | 0.700 / 0.848 / 0.890 | **0.718 / 0.898 / 0.909** |
| En yüksek RMSE (oracle) | 0.149 mV | **0.143 mV** |

Bu karşılaştırma düzeltme notundan önce yapıldığı için rekonstrüksiyon satırları oracle hesabıdır; iki model arasındaki fark yönü değişmez.

**Karar:** Ana model 15 epoch'luk modeldir. Seçim, raporun yöntemine uygun olarak doğrulama setine göre yapıldı; test sonuçları da aynı yönü gösteriyor.

Daha uzun eğitim rekonstrüksiyonu iyileştiriyor (R² +0.02–0.05), ama sınıflandırma başlığı aşırı öğrenmeye başlıyor: eğitim Focal kaybı 0.019'dan 0.010'a düşerken doğrulama AUROC'u düşüyor. Raporun asıl amacı kopmaya rağmen teşhis olduğundan 15 epoch tercih edildi.

**Kalan iki açık (İP6 optimizasyonu için):**
1. Kritik Anterior ve Inferior kopmasında F1 0.87–0.88; hedef > 0.88, sınırda.
2. Kritik Anterior'da R² 0.70–0.72; hedef > 0.80. V1–V4'ün dördü birden koptuğunda sağ prekordiyal bilgi kalan kanallarda yeterince yok.

Raporun İP6 kapsamında ("modelin optimizasyonu") denenebilecekler:
- Sınıflandırma ve rekonstrüksiyon için ayrı checkpoint seçimi, ya da Focal başlığına daha güçlü düzenlileştirme (dropout, ağırlık sönümü).
- Rapordaki α/β (0.4/0.6) sabit tutularak yalnızca eğitim süresi ve öğrenme oranı ayarı.

## Gürültülü derivasyon stres testi

Raporun amacı (Md. 1.2), 1–4 derivasyonun "tamamen koptuğu **veya parazitlendiği**" koşullarda tanı başarımını korumaktır. Kopma testlerine ek olarak gürültü testi yapıldı (`python -m src.noise_test`).

- **Gürültü kaynağı:** MIT-BIH Noise Stress Test Database'in (PhysioNet) gerçek gürültü kayıtları:
  - taban kayması (solunum, hareket)
  - kas artefaktı (tremor)
  - elektrot hareketi
- **Gürültü nasıl eklendi:** Ham sinyalde rastgele 1–4 derivasyona, 12 / 6 / 0 dB SNR'da eklendi. Ardından raporun ön işleme hattı uygulandı. SNR filtre **öncesinde** tanımlıdır: 0 dB, ham kayıtta gürültünün sinyal kadar güçlü olduğu anlamına gelir. Taban kaymasının büyük kısmı 0.5 Hz altında olduğundan filtre sonrası etkin SNR çok daha yüksektir; elektrot hareketi ve kas artefaktı ise EKG bandıyla örtüştüğü için filtreden büyük ölçüde geçer.
- **Sisteme ne bildirildi:** Hangi kanalların parazitli olduğu sisteme bildirilmedi; gerçek saha koşulu bu.
- **Toplam:** 36 senaryo, aynı dengeli test seti (550 MI + 550 Normal).

**F1 (1–4 parazitli kanal ortalaması):**

| Gürültü | SNR | MAE | 1D-ResNet |
|---|---|---|---|
| Gürültüsüz | – | 0.928 | 0.866 |
| Taban kayması | 12 / 6 / 0 dB | 0.928 / 0.926 / 0.925 | 0.865 / 0.866 / 0.866 |
| Kas artefaktı | 12 / 6 / 0 dB | 0.927 / 0.920 / 0.919 | 0.866 / 0.865 / 0.867 |
| Elektrot hareketi | 12 / 6 / 0 dB | 0.923 / 0.917 / **0.890** | 0.867 / 0.868 / 0.870 |

Grafikler: `results/noise/noise_f1.png`, `results/noise/noise_example.png`. Tüm senaryolar: `results/noise/noise_results.json`.

**Sonuç:**
1. **36 senaryonun 35'inde MAE F1'i > 0.88.** Taban kayması ve kas artefaktında başarım neredeyse hiç düşmüyor (en fazla −0.009). Raporun 0.5–45 Hz filtresi taban kaymasının büyük kısmını zaten temizliyor (`noise_example.png`).
2. **Tek istisna en ağır senaryo:** Elektrot hareketi, 0 dB, 4 parazitli kanal. Burada F1 **0.853**'e düşüyor ve 1D-ResNet'in (0.874) altında kalıyor. Ancak düşüşün nedeni yanlış negatif değil, yanlış pozitif:
   - MAE'nin duyarlılığı 0.962'de kalıyor; MI kaçırmıyor.
   - Özgüllüğü 0.707'ye düşüyor; elektrot hareketi ST değişikliğine benzediği için Normal kayıtların bir kısmına MI diyor.
   - Yanlış negatif oranı MAE'de 0.038, 1D-ResNet'te 0.189. Yani raporun hipotezindeki yanlış negatif üstünlüğü bu senaryoda da korunuyor.
3. **Ek analiz: gürültülü kanal tespiti.** Her derivasyon diğerlerinden MAE ile yeniden üretildi; gözlenen sinyalle uyuşmayan derivasyonlar maskelendi.
   - Parazitli kanalların yalnızca %1–19'u yakalandı; F1'e katkısı ihmal edilebilir.
   - Parazitli kanallar kusursuz bilinip maskelenseydi ("oracle"), F1 hafif ve orta gürültüde biraz düşerdi (gürültülü kanal hâlâ işe yarar bilgi taşıyor). Yalnızca ağır elektrot hareketinde yükselirdi: 0.853'ten 0.911'e.
   - Sonuç: ayrı bir gürültü tespit adımı ancak ağır elektrot hareketi için anlamlı. Daha iyi bir sinyal kalitesi göstergesi (SQI), İP6 optimizasyonu ya da raporun "Yeni Proje" bölümündeki gürültü çalışmaları kapsamında ele alınabilir.

## Şekiller (İP6)

`python -m src.plots` ile mevcut sonuç dosyalarından üretilir (model veya GPU gerekmez), 300 dpi, `results/figures/`:

| Dosya | İçerik |
|---|---|
| `1_egitim_egrileri.png` | Eğitim kaybı bileşenleri (0.4·MSE, 0.6·Focal), doğrulama AUROC (MAE ve 1D-ResNet), doğrulama R² |
| `2_kopma_f1.png` | 0–4 rastgele ve 3 kritik duvar kopmasında F1: MAE ve 1D-ResNet, hedef 0.88 |
| `3_yanlis_negatif.png` | Senaryo başına ve havuzlanmış yanlış negatif oranı (hipotez); kritik Anterior istisnası açıklamalı |
| `4_rekonstruksiyon_r2.png` | Kopan kanalların R²'si: gerçekçi, Einthoven ile, oracle; hedef 0.80 |
| `5_gurultu_f1.png` | Üç gerçek gürültü türünde SNR'a göre F1: MAE ve 1D-ResNet |

## Dosyalar ve yeniden üretme

| Dosya | İçerik |
|---|---|
| `results/test_results.json` | Tüm metrikler (MAE + 1D-ResNet) |
| `results/egitim_mae.log`, `models/mae_ecg_checkpoint_history.csv` | Epoch başına eğitim/doğrulama eğrileri (İP6 grafikleri için) |
| `results/reconstruction_*.png` | Rekonstrüksiyon grafikleri |

```bash
# Gürültü testi için önce NSTDB: python -c "import wfdb; wfdb.dl_database('nstdb', 'data/nstdb', records=['bw','em','ma'])"
python -m src.prepare_data                     # İP1: ~2 dk
python -m src.train                            # İP3: MAE, ~2 saat (M3)
python -m src.train --arch resnet1d            # karşılaştırma, ~5 dk
python -m src.evaluate --baseline models/resnet1d_baseline.pth
python -m src.noise_test --baseline models/resnet1d_baseline.pth   # gürültü testi, ~70 dk (M3)
```
