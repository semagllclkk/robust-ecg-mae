# İP1 + İP3 Sonuçları

Çift başlıklı MAE (İP3), raporun Md. 2.1–2.4 maddelerine birebir uygun biçimde eğitildi. Bu belgede test sonuçları ve rapor hedefleriyle karşılaştırması var.

- **Ana model:** genlik farkındalıklı çift başlıklı MAE, `models/mae_ecg_checkpoint.pth` (15 epoch, doğrulama AUROC 0.9778).
- **Test seti:** PTB-XL strat_fold 10, dengeli alt küme (Rapor Md. 2.1): 550 MI + 550 Normal. Model bu kayıtları eğitimde hiç görmedi; hiçbir hasta birden fazla bölümde yer almıyor. Not: F1 sınıf oranına bağlıdır. MI'ın daha seyrek olduğu gerçek popülasyonda F1 daha düşük çıkar; AUROC, duyarlılık ve özgüllük bu orandan etkilenmez.
- **Karar eşikleri:** MI eşiği (0.45) ve duvar eşikleri doğrulama setinde (fold 9) belirlendi; test setine dokunulmadı.
- **Karşılaştırma modeli:** Klasik 1D-ResNet [1]. Aynı veriyle, LeadDropout olmadan eğitildi.
- **Rekonstrüksiyon gerçekçi koşulda ölçülür:** Kopan kanalın gerçek genliği bilinemez; metrikler bu bilgi kullanılmadan, API ile aynı koşulda hesaplanır.

## Ana model: genlik farkındalıklı MAE

Raporun istediği derivasyon bazlı z-score (Md. 2.1) her kanalı kendi genliğine böler. Kopan bir kanalın genliği bilinemediği için ilk model (v1) kopan kanalın **şeklini** doğru üretse de (Pearson r 0.93–0.96) **genliğini** bilemiyordu; gerçekçi R² 0.62–0.64'te kalıyordu.

Ana modelde girdi yine z-score'dur ve kayıp formülü aynıdır (0.4·maskeli MSE + 0.6·Focal). Tek fark: sağlam kanalların log-genliği her derivasyonun token'ına eklenir ve rekonstrüksiyon fiziksel ölçekte (mV / popülasyon σ) üretilir. Kopan kanalın genliği modele hiçbir şekilde ulaşmaz; bu, testle doğrulanmıştır (`tests/test_core.py`).

## Rapor hedefleri

| Hedef (Rapor) | Sonuç | Durum |
|---|---|---|
| 1–4 kanal koptuğunda F1 > %88 (Hedef 4, İP5) | Rastgele kopma: **0.917–0.923**. Kritik kopma: Inferior 0.902, Lateral 0.910, Anterior **0.885** | ✓ tüm senaryolarda |
| RMSE < 0.15 (Hedef 4, İP3) | Rastgele kopma: **0.067–0.079 mV**. Kritik Inferior 0.035, Lateral 0.070, Anterior **0.166 mV** | Kritik Anterior dışında ✓ |
| R² > 0.80 (Hedef 4, Hipotez) | Rastgele kopma: **0.829–0.871** (Einthoven ile 0.885–0.900). Kritik Inferior 0.886 (Einthoven ile **1.000**), Lateral 0.863 (0.907), Anterior **0.544** | Kritik Anterior dışında ✓ |
| Pearson r (Hipotez: "yüksek korelasyonla") | Rastgele: **0.93–0.95**, kritik: 0.82–0.95 | ✓ |
| Klasik mimariye göre yanlış negatif ≥ %25 azalma (Hipotez) | **%49.2** (havuzlanmış FNR: MAE 0.094, 1D-ResNet 0.184) | ✓ |
| Eğitilmiş .pth ağırlıkları (İP3 başarı ölçütü) | `models/mae_ecg_checkpoint.pth` | ✓ |
| Çıkarım < 50 ms (Risk 2) | ~13–22 ms (M3, MPS) | ✓ |

**Tek açık:** V1–V4'ün dördü birden koptuğunda (kritik Anterior) R² 0.544 ve RMSE 0.166 mV. Sağ prekordiyal bölgenin bilgisi kalan V5–V6 ve ekstremite derivasyonlarında fiziksel olarak tam bulunmuyor; teşhis ise bu senaryoda da hedefin üzerinde (F1 0.885).

## Tam test tablosu (genlik farkındalıklı MAE)

Rekonstrüksiyon sütunları yalnızca kopan kanallar üzerinden, mV cinsinden, gerçekçi koşulda. **+Einthoven:** kopan ekstremite kanalları Einthoven bağıntılarıyla hesaplanmış (Risk 1 B planı).

| Senaryo | AUROC | F1 | Duyarlılık | Özgüllük | FNR | Kriz tipi AUROC | RMSE (mV) | R² | r | +Einthoven RMSE / R² |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 kopuk (12 kanal) | 0.979 | 0.926 | 0.918 | 0.935 | 0.082 | 0.964 | – | – | – | – |
| Rastgele 1 kanal | 0.979 | 0.921 | 0.915 | 0.929 | 0.085 | 0.963 | 0.067 | 0.871 | 0.953 | 0.065 / 0.899 |
| Rastgele 2 kanal | 0.977 | 0.917 | 0.909 | 0.927 | 0.091 | 0.960 | 0.069 | 0.863 | 0.951 | 0.066 / 0.900 |
| Rastgele 3 kanal | 0.977 | 0.923 | 0.918 | 0.929 | 0.082 | 0.959 | 0.073 | 0.849 | 0.944 | 0.070 / 0.893 |
| Rastgele 4 kanal | 0.976 | 0.919 | 0.913 | 0.925 | 0.087 | 0.956 | 0.079 | 0.829 | 0.933 | 0.075 / 0.885 |
| Kritik Anterior (V1–V4) | 0.949 | 0.885 | 0.885 | 0.884 | 0.115 | 0.916 | 0.166 | 0.544 | 0.819 | 0.166 / 0.544 |
| Kritik Inferior (II, III, aVF) | 0.965 | 0.902 | 0.900 | 0.904 | 0.100 | 0.947 | 0.035 | 0.886 | 0.947 | **0.001 / 1.000** |
| Kritik Lateral (I, aVL, V5, V6) | 0.973 | 0.910 | 0.905 | 0.915 | 0.095 | 0.935 | 0.070 | 0.863 | 0.943 | 0.066 / 0.907 |

## Yanlış negatif karşılaştırması (1D-ResNet → MAE)

| Senaryo | 1D-ResNet FNR | MAE FNR | Azalma |
|---|---|---|---|
| Rastgele 1 kanal | 0.211 | 0.085 | %59.5 |
| Rastgele 2 kanal | 0.213 | 0.091 | %57.3 |
| Rastgele 3 kanal | 0.189 | 0.082 | %56.7 |
| Rastgele 4 kanal | 0.175 | 0.087 | %50.0 |
| Kritik Anterior (V1–V4) | 0.047 | 0.115 | **%−142.3** |
| Kritik Inferior | 0.255 | 0.100 | %60.7 |
| Kritik Lateral | 0.200 | 0.095 | %52.7 |
| **Havuzlanmış (hipotez)** | **0.184** | **0.094** | **%49.2** |

**Kritik Anterior istisnası:** 1D-ResNet'in buradaki düşük FNR'si bir başarı değil. V1–V4 sıfırlanınca ResNet neredeyse her kayda "MI" diyor ve özgüllüğü 0.976'dan 0.609'a düşüyor; F1'i de MAE'den düşük (0.813'e karşı 0.885). Bu tabloyu raporlarken istisna da açıkça verilmeli.

**Hesaplama yöntemi:** Hipotez "1–4 kanal koptuğunda yanlış negatif oranı" olarak tanımlandığından tüm kopma senaryolarının havuzlanmış FNR'si karşılaştırılıyor: toplam FN / toplam MI. Senaryo başına oranların ortalaması tek bir uç değere (Anterior) çok duyarlı olduğu için kullanılmıyor.

## Yorum

1. **Kopmaya dayanıklılık:** 12 kanaldan 4 rastgele kanal kopmasına geçildiğinde MAE'nin F1'i 0.926'dan 0.919'a, yani neredeyse hiç düşmüyor. 1D-ResNet her koşulda 6–7 puan altında.
2. **Rekonstrüksiyon:** Genlik farkındalığıyla rastgele kopmalarda gerçekçi R² 0.62–0.64'ten 0.83–0.87'ye çıktı; Einthoven kısıtıyla 0.89–0.90. Grafik: `results/figures/4_rekonstruksiyon_r2.png`.
3. **En zor durum kritik Anterior kopması:** V1–V4'ün dördü birden gidince R² 0.544 (v1: 0.494). Grafik: `results/reconstruction_anterior.png`.
4. **Einthoven kısıtı (Risk 1 B planı)** ekstremite derivasyonlarını tam çözüyor: kritik Inferior'da R² 1.000. Göğüs derivasyonlarına uygulanamıyor; onlar fiziksel olarak bağımsız. Grafik: `results/reconstruction_inferior_einthoven.png`.
5. **Kriz tipi ayrımı:** Anterior/Inferior/Lateral ayrımında AUROC 0.96 (12 kanal). API birden fazla etkilenen duvarı raporlar; duvar eşikleri doğrulama setinde kalibre edilmiştir (aşağıda). Lateral MI test setinde yalnızca 20 kayıt olduğu için o duvarın sonuçları daha az güvenilir.

## Önceki model (v1) ile karşılaştırma

İlk model (v1, z-score çıkışlı) `models/mae_ecg_checkpoint_v1.pth`, sonuçları `results/v1/`. Model seçimi raporun yöntemine uygun olarak doğrulama setine göre yapıldı (AUROC: v1 0.9765, ana model 0.9778); test sonuçları da aynı yönü gösteriyor.

| Senaryo | F1 (v1 → ana) | R² gerçekçi (v1 → ana) | R² + Einthoven | RMSE mV |
|---|---|---|---|---|
| 12 kanal | 0.928 → 0.926 | – | – | – |
| Rastgele 1 kanal | 0.927 → 0.921 | 0.637 → **0.871** | 0.787 → **0.899** | 0.093 → 0.067 |
| Rastgele 2 kanal | 0.918 → 0.917 | 0.637 → **0.863** | 0.800 → **0.900** | 0.100 → 0.069 |
| Rastgele 3 kanal | 0.917 → 0.923 | 0.631 → **0.849** | 0.800 → **0.893** | 0.102 → 0.073 |
| Rastgele 4 kanal | 0.915 → 0.919 | 0.616 → **0.829** | 0.795 → **0.885** | 0.104 → 0.079 |
| Kritik Anterior | 0.878 → **0.885** | 0.494 → 0.544 | 0.494 → 0.544 | 0.173 → 0.166 |
| Kritik Inferior | 0.872 → **0.902** | 0.606 → **0.886** | 1.000 → 1.000 | 0.077 → 0.035 |
| Kritik Lateral | 0.914 → 0.910 | 0.628 → **0.863** | 0.811 → **0.907** | 0.089 → 0.070 |
| Yanlış negatif azalması | %46.5 → **%49.2** | | | |

**Ölçüm düzeltmesi notu (18.10.2026):** Bu belgenin ilk sürümü v1'in rekonstrüksiyonunu iyimser hesaplıyordu: kopan kanalın tahmini mV'a çevrilirken o kanalın gerçek genliği kullanılmıştı ("oracle", R² 0.88–0.91). Gerçekte bu genlik bilinemez; o sayılar yalnızca karşılaştırma için `results/v1/test_results.json` içinde (`oracle_*`) duruyor.

### v1 için 30 epoch denemesi

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

**O aşamadaki açıklar (genlik farkındalıklı modelle büyük ölçüde kapandı):**
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

**F1 (1–4 parazitli kanal ortalaması), ana model:**

| Gürültü | SNR | Genlik farkındalıklı MAE | 1D-ResNet |
|---|---|---|---|
| Gürültüsüz | – | 0.925 | 0.866 |
| Taban kayması | 12 / 6 / 0 dB | 0.926 / 0.926 / 0.925 | 0.865 / 0.866 / 0.866 |
| Kas artefaktı | 12 / 6 / 0 dB | 0.925 / 0.926 / 0.922 | 0.866 / 0.865 / 0.867 |
| Elektrot hareketi | 12 / 6 / 0 dB | 0.926 / 0.921 / **0.909** | 0.867 / 0.868 / 0.870 |

Grafikler: `results/noise/noise_f1.png`, `results/figures/5_gurultu_f1.png`, `results/noise/noise_example.png`. Tüm senaryolar: `results/noise/noise_results.json`. Önceki modelin (v1) sonuçları: `results/v1/noise/`.

**Sonuç:**
1. **36 senaryonun 36'sında F1 > 0.88** (en düşük 0.892). Taban kayması ve kas artefaktında başarım neredeyse hiç düşmüyor (en fazla −0.003). Raporun 0.5–45 Hz filtresi taban kaymasının büyük kısmını zaten temizliyor (`noise_example.png`).
2. **En ağır senaryo** (elektrot hareketi, 0 dB, 4 parazitli kanal):
   - F1 0.892; 1D-ResNet 0.874. Önceki model (v1) bu senaryoda 0.853'e düşüp 1D-ResNet'in altında kalıyordu.
   - Duyarlılık 0.949, özgüllük 0.822 (v1: 0.707). Elektrot hareketi ST değişikliğine benzediği için kalan hatalar daha çok yanlış pozitiftir.
   - Yanlış negatif oranı MAE'de 0.051, 1D-ResNet'te 0.189: raporun hipotezindeki yanlış negatif üstünlüğü bu senaryoda da korunuyor.
3. **Ek analiz: gürültülü kanal tespiti.** Her derivasyon diğerlerinden MAE ile yeniden üretildi; gözlenen sinyalle uyuşmayan derivasyonlar maskelendi.
   - Parazitli kanalların yalnızca %1–15'i yakalandı; F1'e katkısı yok.
   - Parazitli kanallar kusursuz bilinip maskelenseydi ("oracle"), F1 hafif ve orta gürültüde düşerdi (0.917–0.919; gürültülü kanal hâlâ işe yarar bilgi taşıyor), yalnızca ağır elektrot hareketinde biraz yükselirdi (0.909 → 0.917).
   - Sonuç: ana modelle ayrı bir gürültü tespit adımına gerek görünmüyor.

## Şekiller (İP6)

`python -m src.plots` ile mevcut sonuç dosyalarından üretilir (model veya GPU gerekmez), 300 dpi, `results/figures/`:

| Dosya | İçerik |
|---|---|
| `1_egitim_egrileri.png` | Eğitim kaybı bileşenleri (0.4·MSE, 0.6·Focal), doğrulama AUROC (MAE ve 1D-ResNet), doğrulama R² |
| `2_kopma_f1.png` | 0–4 rastgele ve 3 kritik duvar kopmasında F1: MAE ve 1D-ResNet, hedef 0.88 |
| `3_yanlis_negatif.png` | Senaryo başına ve havuzlanmış yanlış negatif oranı (hipotez); kritik Anterior istisnası açıklamalı |
| `4_rekonstruksiyon_r2.png` | Kopan kanalların gerçekçi R²'si: v1, genlik farkındalıklı model ve Einthoven ile; hedef 0.80 |
| `5_gurultu_f1.png` | Üç gerçek gürültü türünde SNR'a göre F1: MAE ve 1D-ResNet |

## Dosyalar ve yeniden üretme

| Dosya | İçerik |
|---|---|
| `results/test_results.json` | Tüm metrikler (ana model + 1D-ResNet) |
| `results/v1/`, `models/mae_ecg_checkpoint_v1.pth` | Önceki model (v1) ve sonuçları |
| `results/egitim_mae.log`, `models/mae_ecg_checkpoint_history.csv` | Epoch başına eğitim/doğrulama eğrileri (İP6 grafikleri için) |
| `results/reconstruction_*.png` | Rekonstrüksiyon grafikleri |

```bash
# Gürültü testi için önce NSTDB: python -c "import wfdb; wfdb.dl_database('nstdb', 'data/nstdb', records=['bw','em','ma'])"
python -m src.prepare_data                     # İP1: ~2 dk
python -m src.train --amplitude-aware          # İP3: ana model, ~2 saat (M3)
python -m src.calibrate_walls                  # duvar eşikleri (doğrulama seti), ~30 sn
python -m src.train --arch resnet1d            # karşılaştırma, ~5 dk
python -m src.evaluate --baseline models/resnet1d_baseline.pth
python -m src.noise_test --baseline models/resnet1d_baseline.pth   # gürültü testi, ~70 dk (M3)
```

## Duvar eşiği ve çıkarım süresi notları (18.10.2026)

### Lateral duvarın kaçırılması
API'de her duvar için aynı eşik kullanılıyordu: P(duvar ∣ MI) ≥ 0.5. Demo setinin 213 test kaydında arayüzün kullandığı kod yolu (`analyze_row`) ile ölçüm (v1 modeliyle):

| | 12 kanal | Rastgele 2 kanal kopuk |
|---|---|---|
| MI tespiti (172 MI) | 158 doğru, 14 kaçırıldı (%8) | 156 doğru, 16 kaçırıldı |
| Duvar: tam eşleşme / kısmi / tamamen yanlış | 86 / 25 / 8 | 83 / 25 / 9 |

MI tespiti yukarıdaki FNR (0.075) ile tutarlı. Duvar hatalarının kaynağı Lateral:
- Yalnız Lateral MI olan 9 kaydın hiçbirinde Lateral işaretlenmedi; 6'sı Anterior olarak verildi.
- Anterior+Lateral olan 7 kaydın hepsi yalnız Anterior olarak verildi.
- Bu kayıtlarda P(Lateral ∣ MI) 0.30–0.41 aralığında; 0.5 eşiğinin altında kalıyor.

Kod hatası değil: duvar sırası eğitim ve API'de aynı (`WALL_NAMES`). Lateral MI seyrek olduğundan model bu duvara düşük olasılık veriyor. Tek bir 0.5 eşiği seyrek sınıf için fazla yüksek.

**Düzeltme:** `python -m src.calibrate_walls` her duvar için F1'i en büyükleyen eşiği **doğrulama setinde** (fold 9) seçer. Bu, MI karar eşiğiyle aynı yöntemdir; test setine dokunulmaz. Sonuç `models/wall_thresholds.json` dosyasına yazılır ve API bu dosyayı bulunca kullanır. Dosya yoksa davranış değişmez (0.5). Kalibrasyon için `data/processed` gerekir; demo setinde doğrulama kaydı yok.

**Kalibre eşikler** (doğrulama seti; MI kayıtlarında 12 kanal + rastgele 1–4 kopma):

| Duvar | v1 eşiği | Ana model eşiği | Ana model doğrulama F1 (0.5 → kalibre) |
|---|---|---|---|
| Anterior | 0.47 | 0.51 | 0.902 → 0.905 |
| Inferior | 0.38 | 0.47 | 0.912 → 0.918 |
| Lateral | 0.43 | 0.48 | 0.467 → 0.554 |

Lateral için doğrulama setinde yalnızca 40 pozitif örnek var (20 kayıt × 2 kopma koşulu); bu iyileşme istatistiksel olarak zayıftır. Eşik dosyası checkpoint'in SHA-256 parmak izini taşır: API eşikleri yalnızca hesaplandıkları model yüklüyken kullanır, model değişirse uyarı verip 0.5'e döner. Yeniden eğitimden sonra `python -m src.calibrate_walls` tekrar çalıştırılmalıdır.

### Çıkarım süresi (Risk 2)
- **M3 (MPS):** 13–22 ms, hedef < 50 ms ✓.
- **Intel dizüstü CPU (10 çekirdek):**
  - PyTorch: en iyi 70–90 ms; ölçümler iş parçacığı sayısına göre dalgalı.
  - ONNX Runtime (`python -m src.export_onnx`): benzer süre (~80–95 ms). Çıktılar PyTorch ile özdeş (en büyük fark 5·10⁻⁶).
  - Dinamik int8 nicemleme: daha yavaş (~216 ms) ve 173 kaydın 10'unda kararı değiştiriyor; kullanılmadı.
- **Sonuç:** 50 ms hedefi GPU'lu geliştirme donanımında sağlanıyor; yalnız CPU'lu bilgisayarda bu model ~80 ms'de çalışıyor. API aynı kayıt ve kopma için sonucu önbellekten verdiği için arayüzde tekrarlanan isteklerde gecikme hissedilmiyor.
