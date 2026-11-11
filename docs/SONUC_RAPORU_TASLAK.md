# Proje Sonuç Raporu — Taslak

**Proje adı:** Eksik ve Gürültülü Derivasyon Koşullarında Maskeli Öz-Kodlayıcılar ile Dayanıklı Miyokard Enfarktüsü Tespiti ve Anatomik Haritalama
**Program:** TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı
**Kurum:** Pamukkale Üniversitesi, Bilgisayar Mühendisliği Bölümü
**Proje yürütücüsü:** Zilan Aykal · **Araştırmacı:** Sema Gül Çelik · **Danışman:** Doç. Dr. Meriç Çetin

> Taslak notu: `[...]` ile işaretli yerler bilgi bekliyor. Sayıların tamamı `docs/SONUCLAR.md` ve `results/` altındaki dosyalardan alınmıştır. Kaynak numaraları Araştırma Önerisi Formu'ndaki kaynakçayla aynıdır; [13]–[15] bu raporda eklenmiştir.

## 1. Özet

Bu projede, 12 derivasyonlu elektrokardiyogramda (EKG) 1–4 derivasyonun koptuğu veya parazitlendiği koşullarda miyokard enfarktüsünü (MI) tespit eden ve kopan derivasyonları yeniden üreten çift başlıklı bir Maskeli Öz-Kodlayıcı (MAE) geliştirilmiştir. Model PTB-XL veri kümesinde [11], DataLoader seviyesinde uygulanan kanal bazlı LeadDropout ile eğitilmiştir. Ortak omurga 1D-CNN blokları, STFT tabanlı spektral kol, Lead Fusing ve Spatio-Temporal Transformer katmanlarından oluşur. Başlık A simetrik 1D Transpose Conv kod çözücüyle kopan derivasyonları yeniden üretir, Başlık B ise Global Average Pooling ve Dense katmanlarıyla MI varlığını ve etkilenen duvarı (Anterior, Inferior, Lateral) sınıflandırır. Kayıp fonksiyonu 0.4·MSE (yalnızca kopan derivasyonlarda) + 0.6·Focal Loss'tur.

Eğitimde görülmemiş test kümesinde (550 MI + 550 Normal) model, 1–4 derivasyon kopmasının tüm senaryolarında F1 > 0.88 hedefini sağlamıştır (0.885–0.923). Kopan derivasyonların rekonstrüksiyonunda rastgele kopmalarda RMSE 0.067–0.079 mV ve R² 0.829–0.871 elde edilmiş, Einthoven kısıtıyla R² 0.885–0.900'e yükselmiştir. Klasik 1D-ResNet [1] mimarisine göre yanlış negatif oranı %49.2 azalmıştır (hipotez: ≥ %25). Gerçek gürültü kayıtlarıyla (MIT-BIH NSTDB) yapılan 36 senaryonun tamamında F1 > 0.88 korunmuştur. Spatial Attention ağırlıkları ve sınıflandırma çıktısı, FastAPI üzerinden 3D kalp modeline aktarılmış; tıklandığında ST sapmasını, şüpheli koroner arteri ve karar güvenini gösteren Click-to-Explain arayüzü geliştirilmiştir. Çıkarım süresi 13–22 ms'dir (hedef < 50 ms).

**Anahtar kelimeler:** Elektrokardiyografi, miyokard enfarktüsü, maskeli öz-kodlayıcı, derivasyon kopması, açıklanabilir yapay zekâ

## 2. Amaç ve Hedefler

Projenin amacı, saha ve acil koşullarında sık görülen elektrot kopması ve parazit durumunda MI tanı başarımını korumak ve kararı klinisyene anatomik olarak açıklamaktır. Başvuruda tanımlanan ölçülebilir hedefler ve gerçekleşme durumları:

| Hedef | Ölçüt | Gerçekleşen | Durum |
|---|---|---|---|
| Kopmaya dayanıklı teşhis | 1–4 derivasyon koptuğunda F1 > 0.88 | 0.885–0.923 (7 kopma senaryosu) | ✓ |
| Rekonstrüksiyon hatası | RMSE < 0.15 | Rastgele kopma 0.067–0.079 mV; kritik Anterior 0.166 mV | Kritik Anterior dışında ✓ |
| Rekonstrüksiyon uyumu | R² > 0.80, yüksek Pearson r | Rastgele kopma R² 0.829–0.871, r 0.93–0.95; kritik Anterior R² 0.544 | Kritik Anterior dışında ✓ |
| Hipotez | 1D-ResNet'e göre yanlış negatifte ≥ %25 azalma | %49.2 | ✓ |
| Gecikme (Risk 2) | Çıkarım < 50 ms | 13–22 ms | ✓ |

## 3. Yöntem

### 3.1 Veri ve ön işleme (İP1)

- **Veri kümesi:** PTB-XL v1.0.3 [11]. Başvuruda belirtilen 21.837 kayıt / 18.885 hasta ilk sürüme aittir; v1.0.2'de yinelenen 38 kayıt çıkarıldığından güncel sürümde 21.799 kayıt / 18.869 hasta bulunmaktadır.
- **Etiketler:** SCP-ECG kodlarından MI (herhangi bir MI alt sınıfı) ve Normal (yalnız NORM) etiketleri çıkarılmış; MI kayıtları için duvar etiketleri AMI → Anterior, IMI → Inferior, LMI → Lateral olarak eşlenmiştir. Bir kayıt birden fazla duvar taşıyabilir.
- **Bölümleme:** PTB-XL'in hasta bazlı `strat_fold` alanı kullanılmıştır: 1–8 eğitim, 9 doğrulama, 10 test. Hiçbir hasta birden fazla bölümde yer almaz.
- **Dengeli alt kümeler:** Eğitim 4.379 MI + 4.379 Normal, doğrulama 540 + 540, test 550 + 550.
- **Filtre:** 0.5–45 Hz, 4. derece, çift yönlü (sıfır fazlı) Butterworth bant geçiren filtre; taban kayması, 50 Hz şebeke gürültüsü ve kas tremoru bastırılmıştır.
- **Normalizasyon:** 100 Hz örneklemede her derivasyon için z-score, x = (x − μ) / σ. Model girdisi [12, 1000] tensördür.
- **Çıktı:** `src/preprocessing.py` modülü ve `data/processed` altındaki .npy tensör deposu (İP1 başarı ölçütü).

### 3.2 LeadDropout (İP3)

Maskeleme MLAE [4] prensibine uygun olarak PyTorch DataLoader seviyesinde uygulanmıştır. Her batch için k ∈ [1, 4] derivasyon sıfırlanır; M ∈ {0,1}¹² ikili maskesi (1 = sağlam) ile X_masked = X ⊙ M·1ᵀ₁₀₀₀ hesaplanır. Rastgele kopmaya ek olarak batch'lerin %30'unda bir duvarın tüm derivasyonları birlikte koparılır (kritik kopma: Anterior V1–V4, Inferior II/III/aVF, Lateral I/aVL/V5/V6).

### 3.3 Çift başlıklı MAE mimarisi (İP3)

- **Ortak omurga:** Her derivasyon için paylaşılan 1D-CNN blokları (P, QRS, ST, T morfolojisi); Wei ve ark. [5] bimodal yaklaşımına uygun STFT spektral kolu; Lead Fusing [9] ara katman bağlantıları; Spatio-Temporal Transformer [7] blokları. Uzamsal dikkat yalnızca sağlam derivasyonları anahtar olarak kullanır; böylece kopan derivasyonun ne ölçüde hangi sağlam derivasyonlardan temsil edildiği dikkat ağırlıklarıyla hesaplanır: Attention(Q, K, V) = Softmax(QKᵀ / √d_k) V.
- **Başlık A (rekonstrüksiyon):** MCMA [8] benzeri, 1D Transpose Conv katmanlarından oluşan simetrik kod çözücü ve IRC ara katman bağlantıları [5].
- **Başlık B (teşhis):** Sağlam derivasyonlar üzerinde Global Average Pooling + Dense; Zhou ve ark. [3] ile Ribeiro ve ark. [1] çalışmalarındaki gibi MI var/yok ve kriz tipi (Anterior/Inferior/Lateral) çıktıları.
- **Genlik farkındalığı:** Derivasyon bazlı z-score her kanalı kendi genliğine böldüğünden, kopan kanalın genliği bilinemez. Sağlam derivasyonların log-genliği derivasyon token'larına eklenmiş ve rekonstrüksiyon fiziksel ölçekte üretilmiştir. Girdi ve kayıp formülü değişmemiştir; kopan kanalın genliğinin modele ulaşmadığı birim testlerle doğrulanmıştır.
- **Boyut:** 1.29 M parametre.

### 3.4 Kayıp fonksiyonu ve eğitim

L_toplam = 0.4 · L_MSE + 0.6 · L_Focal. L_MSE yalnızca kopan derivasyonlarda hesaplanır. L_Focal [12] (γ = 2) MI/Normal teriminden ve yalnız MI örneklerinde hesaplanan kriz tipi teriminden oluşur. Eğitim AdamW, 5·10⁻⁴ öğrenme oranı, ısınma + kosinüs azalma, batch 32, bfloat16 karma hassasiyetle 15 epoch sürmüştür. Model ve karar eşikleri doğrulama kümesinde seçilmiştir; test kümesi yalnız son değerlendirmede kullanılmıştır.

Eğitim, Araştırma Olanakları tablosunda yer alan Apple M3 geliştirme bilgisayarında (MPS) yaklaşık 2 saatte tamamlanmıştır.

### 3.5 Değerlendirme

- **Kopma senaryoları:** 0 kopuk; rastgele 1, 2, 3, 4 derivasyon; kritik Anterior, Inferior, Lateral.
- **Metrikler:** F1, AUROC, duyarlılık, özgüllük, yanlış negatif oranı (FNR); kopan derivasyonlarda RMSE (mV), R², Pearson r; kriz tipi AUROC.
- **Gerçekçi rekonstrüksiyon ölçümü:** Kopan derivasyonun gerçek genliği kullanılmadan, popülasyon σ değeriyle mV'a çevrilerek ölçülmüştür (sistemin sahada çalıştığı koşul).
- **Karşılaştırma:** Aynı veriyle, LeadDropout olmadan eğitilen 1D-ResNet [1]. Hipotez havuzlanmış FNR ile (toplam FN / toplam MI, tüm kopma senaryoları) sınanmıştır.
- **Gürültü testi:** MIT-BIH Noise Stress Test Database'in [13, 14] taban kayması, kas artefaktı ve elektrot hareketi kayıtları, ham sinyalde rastgele 1–4 derivasyona 12/6/0 dB SNR ile eklenmiş, ardından ön işleme hattı uygulanmıştır (36 senaryo). Parazitli derivasyonlar sisteme bildirilmemiştir.

### 3.6 Arayüz ve açıklanabilirlik (İP2, İP4, İP5)

- **İP2 (Sema Gül Çelik):** React tabanlı gösterge paneli; 12 derivasyonun tek tek açılıp kapatılabildiği EKG kâğıdı görünümü ve Mock-API. Arayüzün kendi kare sayacıyla (EKG çizim döngüsü) dört koşulda 10'ar saniye ölçüm yapılmıştır: 12 derivasyon akış, 4 derivasyon kopması (V1–V4) ve rekonstrüksiyon çizimi, 3D kalp modelinin döndürülmesi, art arda yeni kayıt yüklenmesi. Tüm koşullarda en düşük değer 60 FPS'tir (Apple M3, Chrome 157, 60 Hz ekran; gerçek FastAPI arka ucu ve model ile).
- **İP4 (Sema Gül Çelik):** Guo ve ark. [10] derivasyon gruplamasına uygun olarak 3D kalp modeli (HuBMAP Human Reference Atlas [15]) üzerinde etkilenen duvarın kırmızı gradyanla işaretlenmesi; duvara tıklandığında açılan Click-to-Explain kartı: ST sapması (mm), şüpheli koroner arter (Anterior → LAD, Inferior → RCA, Lateral → LCx), eksik derivasyonlara rağmen karar güveni.
- **İP5 (ortak):** FastAPI arka ucu, eğitilmiş modeli Mock-API ile aynı sözleşmeyle sunar. Çıktılar: MI olasılığı, etkilenen duvarlar (doğrulama kümesinde kalibre edilmiş duvar eşikleri), Spatial Attention ağırlıkları, derivasyon önem skorları, kopan derivasyonların rekonstrüksiyonu ve kaynakları, dinamik klinisyen/hasta metinleri ve PDF klinik rapor.

## 4. Bulgular

### 4.1 Kopma senaryoları (test kümesi, 550 MI + 550 Normal)

| Senaryo | AUROC | F1 | Duyarlılık | Özgüllük | FNR | RMSE (mV) | R² | r |
|---|---|---|---|---|---|---|---|---|
| 0 kopuk (12 derivasyon) | 0.979 | 0.926 | 0.918 | 0.935 | 0.082 | – | – | – |
| Rastgele 1 | 0.979 | 0.921 | 0.915 | 0.929 | 0.085 | 0.067 | 0.871 | 0.953 |
| Rastgele 2 | 0.977 | 0.917 | 0.909 | 0.927 | 0.091 | 0.069 | 0.863 | 0.951 |
| Rastgele 3 | 0.977 | 0.923 | 0.918 | 0.929 | 0.082 | 0.073 | 0.849 | 0.944 |
| Rastgele 4 | 0.976 | 0.919 | 0.913 | 0.925 | 0.087 | 0.079 | 0.829 | 0.933 |
| Kritik Anterior (V1–V4) | 0.949 | 0.885 | 0.885 | 0.884 | 0.115 | 0.166 | 0.544 | 0.819 |
| Kritik Inferior (II, III, aVF) | 0.965 | 0.902 | 0.900 | 0.904 | 0.100 | 0.035 | 0.886 | 0.947 |
| Kritik Lateral (I, aVL, V5, V6) | 0.973 | 0.910 | 0.905 | 0.915 | 0.095 | 0.070 | 0.863 | 0.943 |

12 derivasyondan 4 rastgele derivasyon kopmasına geçildiğinde F1 0.926'dan 0.919'a düşmektedir; model kopmaya karşı neredeyse duyarsızdır. Kriz tipi ayrımında 12 derivasyonla AUROC 0.964'tür.

### 4.2 Yanlış negatif karşılaştırması (hipotez)

| Senaryo | 1D-ResNet FNR | MAE FNR | Azalma |
|---|---|---|---|
| Rastgele 1 | 0.211 | 0.085 | %59.5 |
| Rastgele 2 | 0.213 | 0.091 | %57.3 |
| Rastgele 3 | 0.189 | 0.082 | %56.7 |
| Rastgele 4 | 0.175 | 0.087 | %50.0 |
| Kritik Anterior | 0.047 | 0.115 | %−142.3 |
| Kritik Inferior | 0.255 | 0.100 | %60.7 |
| Kritik Lateral | 0.200 | 0.095 | %52.7 |
| **Havuzlanmış** | **0.184** | **0.094** | **%49.2** |

Hipotez doğrulanmıştır. Kritik Anterior senaryosundaki istisna açıkça belirtilmelidir: V1–V4 sıfırlandığında 1D-ResNet kayıtların çoğunu MI olarak sınıflandırmakta, özgüllüğü 0.976'dan 0.609'a düşmektedir. Bu senaryoda ResNet'in F1'i 0.813, MAE'nin 0.885'tir.

### 4.3 Rekonstrüksiyon ve Einthoven kısıtı

Kopan ekstremite derivasyonları Einthoven ve Goldberger bağıntılarıyla sağlam ekstremite derivasyonlarından hesaplandığında (Risk 1 B planı) rastgele kopmalarda R² 0.885–0.900'e, kritik Lateral'de 0.907'ye, kritik Inferior'da 1.000'e (RMSE 0.001 mV) yükselmektedir. Göğüs derivasyonları fiziksel olarak birbirinden bağımsız olduğundan bu kısıt V1–V6'ya uygulanamaz.

### 4.4 Gürültülü derivasyonlar

| Gürültü | SNR 12 / 6 / 0 dB — MAE F1 | 1D-ResNet F1 |
|---|---|---|
| Gürültüsüz | 0.925 | 0.866 |
| Taban kayması | 0.926 / 0.926 / 0.925 | 0.865 / 0.866 / 0.866 |
| Kas artefaktı | 0.925 / 0.926 / 0.922 | 0.866 / 0.865 / 0.867 |
| Elektrot hareketi | 0.926 / 0.921 / 0.909 | 0.867 / 0.868 / 0.870 |

36 senaryonun tamamında F1 > 0.88'dir. En ağır senaryoda (elektrot hareketi, 0 dB, 4 parazitli derivasyon) F1 0.892, FNR 0.051'dir (1D-ResNet: F1 0.874, FNR 0.189).

### 4.5 Çıkarım süresi

Apple M3 (MPS) üzerinde tek kayıt için 13–22 ms ölçülmüştür (hedef < 50 ms). Yalnız CPU'lu bir dizüstü bilgisayarda PyTorch ve ONNX Runtime ile ~80 ms ölçülmüştür. Arka uç aynı kayıt ve kopma için sonucu önbellekten verdiğinden arayüzde tekrarlanan isteklerde gecikme hissedilmemektedir.

### 4.6 Şekiller

`results/figures/` altında 300 dpi: eğitim eğrileri, kopma senaryolarında F1, yanlış negatif karşılaştırması, rekonstrüksiyon R², gürültü testi F1.

## 5. İş Paketlerinin Gerçekleşme Durumu

| İP | Takvim | Faaliyet | Sorumlu | Başarı ölçütü (katkı) | Gerçekleşme |
|---|---|---|---|---|---|
| İP1 | 1. ay | PTB-XL temini, etiket matrisleri, 0.5–45 Hz Butterworth, z-score hattı | Zilan Aykal | Temizlenmiş tensör veri deposunun ve `preprocessing.py` modülünün sorunsuz çalışması (%15) | Sağlandı |
| İP2 | 2. ay | Dashboard iskeleti, 12 kanal aç/kapa, Mock-API | Sema Gül Çelik | Modelden bağımsız çalışan, sahte verilerle akıcı 60 FPS sinyal çizen arayüz prototipi (%15) | Sağlandı: Mock-API ile modelden bağımsız çalışıyor; gerçek arka uçla ölçülen en düşük değer 60 FPS |
| İP3 | 3. ay | LeadDropout, 1D-CNN + Spatial Attention tabanlı çift başlıklı MAE eğitimi | Zilan Aykal | Sinyal rekonstrüksiyonunda düşük hata (RMSE < 0.15) ve eğitilmiş .pth ağırlıkları (%25) | Sağlandı, bir istisnayla: rastgele 1–4 kopmada RMSE 0.067–0.079 mV; kritik Anterior kopmasında 0.166 mV |
| İP4 | 4. ay | 3D kalp anatomisi, dikkat ağırlıklarının renklendirilmesi, Click-to-Explain | Sema Gül Çelik | 3D kalp duvarı üzerinde tıklanabilir ve dinamik açıklama üreten etkileşimli XAI modülü (%20) | Sağlandı |
| İP5 | 5.–6. ay | Modelin FastAPI ile arayüze bağlanması, 1–4 kanal stres testlerinin canlı doğrulanması | Zilan Aykal, Sema Gül Çelik | Eksik kanal koşullarında %88 üzeri F1 sağlayan, entegre çalışan canlı karar destek sistemi (%15) | Sağlandı: tüm kopma senaryolarında F1 0.885–0.923 |
| İP6 | 7.–9. ay | Başarım metrikleri, model optimizasyonu, grafikler | Zilan Aykal, Sema Gül Çelik | Proje sonuç raporunun ve teslim metinlerinin eksiksiz tamamlanması (%10) | Metrikler, optimizasyon denemeleri ve grafikler tamamlandı; sonuç raporu bu belgeyle hazırlanıyor |

## 6. Risk Yönetimi

- **Risk 1 (3–4 derivasyon kopmasında başarım düşüşü):** Teşhis tüm senaryolarda hedefin üzerinde kalmıştır. Rekonstrüksiyonda kritik Anterior kopması (V1–V4) hedefin altında kalmıştır (R² 0.544). B planı kapsamında (a) Einthoven üçgeni kısıtı ekstremite derivasyonlarına uygulanmış ve kritik Inferior'u tam çözmüştür; (b) Spatial Attention katmanına elektrot yerleşiminden kurulan uzamsal komşuluk ağırlık matrisi regülarizasyon olarak eklenmiştir.
  Komşuluk regülarizasyonuyla eğitilen model, kritik Anterior rekonstrüksiyonunda ana modelle aynı sonucu vermiştir (R² 0.544, RMSE 0.163 mV). Dikkat ağırlıkları önsel matrise yaklaşmış, ancak V1–V4'ün dördü birlikte koptuğunda eksik bilgi kalan derivasyonlarda bulunmadığından rekonstrüksiyon değişmemiştir. Doğrulama başarımı daha yüksek olan ana model korunmuştur.
- **Risk 2 (gecikme):** ONNX dışa aktarımı yapılmış, çıktıların PyTorch ile özdeş olduğu doğrulanmıştır. GPU'lu donanımda hedef sağlanmıştır; önbellekleme uygulanmıştır.

## 7. Tartışma ve Sınırlılıklar

1. **Kritik Anterior rekonstrüksiyonu:** V1–V4'ün dördü birlikte koptuğunda sağ prekordiyal bölgenin bilgisi kalan V5, V6 ve ekstremite derivasyonlarında fiziksel olarak tam bulunmamaktadır. Teşhis başarımı ise bu senaryoda da hedefin üzerindedir (F1 0.885).
2. **Sınıf oranı:** Test kümesi dengelidir. MI'ın daha seyrek olduğu gerçek popülasyonda F1 daha düşük çıkar; AUROC, duyarlılık ve özgüllük bu orandan etkilenmez.
3. **Lateral MI:** Test kümesinde yalnız 20 Lateral MI kaydı bulunduğundan bu duvarın sonuçları istatistiksel olarak daha zayıftır.
4. **Gürültülü derivasyon tespiti:** MAE tutarlılığıyla parazitli derivasyon tespiti denenmiş, katkı sağlamamıştır. Model parazitli derivasyonları ayrı bir tespit adımı olmadan tolere etmektedir.
5. **Klinik doğrulama:** Sonuçlar tek bir açık veri kümesine dayanmaktadır; klinik kullanım öncesinde dış veri kümelerinde ve prospektif olarak doğrulanması gerekir.

## 8. Sonuç

Proje kapsamında, eksik ve gürültülü derivasyon koşullarında MI tespitini koruyan ve kararını 3D anatomik model üzerinde açıklayan uçtan uca bir sistem geliştirilmiştir. Başvuruda tanımlanan teşhis, yanlış negatif ve gecikme hedeflerinin tamamı, rekonstrüksiyon hedefleri ise kritik Anterior kopması dışında sağlanmıştır.

## 9. Bütçe Kullanımı

| Kalem | Öneride | Tutar | İP | Gerçekleşme |
|---|---|---|---|---|
| Bulut GPU (Google Colab Pro vb.) | 6 ay | 3.500 TL | İP3 | [harcama durumu] Eğitim yerel Apple M3 bilgisayarda yapılabildiğinden bulut GPU'ya ihtiyaç [duyuldu / duyulmadı]. |
| 512 GB harici NVMe SSD | 1 adet | 3.500 TL | İP1, İP3, İP5 | [alındı mı] Tensör deposu (~1 GB), PTB-XL (~3 GB) ve model kontrol noktaları |
| 512 GB harici NVMe SSD | 1 adet | 3.500 TL | İP2, İP4, İP5 | [alındı mı] Arayüz bileşenleri, Mock-API verileri, entegrasyon testleri |
| **Toplam** | | **10.500 TL** | | [gerçekleşen toplam] |

## 10. Yaygın Etki

- **Bilimsel/akademik:** Rekonstrüksiyon metrikleri (R², RMSE) ve eksik kanal stres testi sonuçları IEEE TIPTEKNO veya Ulusal Biyomedikal Mühendisliği Kongresi'nde bildiri olarak sunulacaktır. [Başvuru tarihi ve durum eklenecek.]
- **Prototip:** Derivasyon kopması senaryolarını simüle eden, eksik EKG sinyalini sağlam derivasyonlardan tamamlayan ve 3D kalp modeli üzerinde karar gerekçesi sunan (Click-to-Explain) karar destek yazılım prototipi geliştirilmiştir.
- **Yeni proje:** Öneride öngörülen ikinci dönem çalışmaları için altyapı hazırdır: gerçek gürültü kayıtlarıyla stres testi hattı (`src/noise_test.py`) kurulmuş, arayüzün FPS ve çıkarım gecikmesi ölçülmüştür; modüler veri hattı ve model, EEG gibi farklı fizyolojik zaman serilerine uyarlanabilecek yapıdadır.
- Kaynak kodu ve eğitilmiş model: [GitHub deposu bağlantısı, erişim durumu]

## 11. Kaynaklar

[1] Ribeiro, A. H., Ribeiro, M. H., Paixão, G. M., Oliveira, D. M., Gomes, P. R., Canazart, J. A., ... & Antonio, L. R. (2020). Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*, 11(1), 1760. https://doi.org/10.1038/s41467-020-15432-4

[2] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16000–16009. https://doi.org/10.1109/CVPR52688.2022.01553

[3] Zhou, Y., Diao, X., Huo, Y., Liu, Y., Sun, Z., Fan, X., & Zhao, W. (2025). Enhancing automatic multilabel diagnosis of electrocardiogram signals: A masked transformer approach. *Computers in Biology and Medicine*, 196, 110674. https://doi.org/10.1016/j.compbiomed.2025.110674

[4] Zhang, H., Liu, W., Shi, J., Chang, S., Wang, H., He, J., & Huang, Q. (2022). MaeFE: Masked autoencoders family of electrocardiogram for self-supervised pretraining and transfer learning. *IEEE Transactions on Instrumentation and Measurement*, 72, 1–15. https://doi.org/10.1109/TIM.2022.3228267

[5] Wei, Y., Lian, C., Xu, B., Zhao, P., Yang, H., & Zeng, Z. (2025). Bimodal Masked Autoencoders with internal representation connections for electrocardiogram classification. *Pattern Recognition*, 161, 111311. https://doi.org/10.1016/j.patcog.2024.111311

[6] Presacan, O., Dorobanţiu, A., Isaksen, J. L., Willi, T., Graff, C., Riegler, M. A., ... & Thambawita, V. (2025). Evaluating the feasibility of 12-lead electrocardiogram reconstruction from limited leads using deep learning. *Communications Medicine*, 5(1), 139. https://doi.org/10.1038/s43856-025-00814-w

[7] Na, Y., Park, M., Tae, Y., & Joo, S. (2024). Guiding masked representation learning to capture spatio-temporal relationship of electrocardiogram. *International Conference on Learning Representations (ICLR 2024)*, pp. 15012–15035. https://doi.org/10.48550/arXiv.2310.11899

[8] Chen, J., Wu, W., Liu, T., & Hong, S. (2024). Multi-channel masked autoencoder and comprehensive evaluations for reconstructing 12-lead ECG from arbitrary single-lead ECG. *npj Cardiovascular Health*, 1(1), 34. https://doi.org/10.1038/s44325-024-00036-4

[9] Liu, X., Xu, Y., Xu, H., He, L., Long, S., Huang, Y., ... & Liu, X. (2025). Advancing interpretable cardiac disease diagnosis via a transformer-convolutional hybrid network on electrocardiograms. *Engineering Applications of Artificial Intelligence*, 152, 110675. https://doi.org/10.1016/j.engappai.2025.110675

[10] Guo, L., Zhan, Q., Yang, J., An, Y., Long, J., & Ma, N. (2025). Lead-grouped multi-stage learning for myocardial infarction localization. *Methods*, 234, 315–323. https://doi.org/10.1016/j.ymeth.2025.01.015

[11] Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154. https://doi.org/10.1038/s41597-020-0495-6

[12] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2020). Focal loss for dense object detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(2), 318–327. https://doi.org/10.1109/TPAMI.2018.2858826

[13] Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. *Computers in Cardiology*, 11, 381–384.

[14] Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

[15] HuBMAP Consortium. Human Reference Atlas 3D Reference Objects (Visible Human Male heart). https://humanatlas.io, CC BY 4.0.
