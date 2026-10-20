# Robust ECG-MAE

**Eksik ve Gürültülü Derivasyon Koşullarında Maskeli Öz-Kodlayıcılar ile Dayanıklı Miyokard Enfarktüsü Tespiti ve Anatomik Haritalama**

TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projesi · Pamukkale Üniversitesi, Bilgisayar Mühendisliği Bölümü

- **Proje yürütücüsü:** Zilan Aykal (model ve sinyal işleme)
- **Araştırmacı:** Sema Gül Çelik (arayüz ve 3D anatomik XAI)
- **Danışman:** Doç. Dr. Meriç Çetin

12 derivasyonlu EKG'de 1–4 derivasyon koptuğunda ya da parazitlendiğinde bile miyokard enfarktüsünü (MI) tespit eden çift başlıklı bir Maskeli Öz-Kodlayıcı (MAE). Model kopan derivasyonları sağlam olanlardan yeniden üretir. Bulgular, tıklandığında gerekçe sunan (Click-to-Explain) 3D kalp modeli üzerinde gösterilir.

## Sonuçlar

PTB-XL test kümesinde (strat_fold 10, 550 MI + 550 Normal, eğitimde görülmemiş):

| Rapor hedefi | Sonuç |
|---|---|
| 1–4 derivasyon koptuğunda F1 > 0.88 | Tüm senaryolarda 0.885–0.923 |
| Kopan derivasyonların rekonstrüksiyonu R² > 0.80 | Rastgele 1–4 kopmada 0.83–0.87 (Einthoven kısıtıyla 0.89–0.90) |
| RMSE < 0.15 | Rastgele kopmada 0.067–0.079 mV |
| Klasik 1D-ResNet'e göre yanlış negatif ≥ %25 azalma | %49.2 |
| Gürültülü derivasyonlarda (gerçek NSTDB gürültüsü, 36 senaryo) F1 > 0.88 | 36/36 senaryo |
| Çıkarım < 50 ms | 13–22 ms (Apple M3) |

Açık konu: V1–V4'ün dördü birlikte koptuğunda R² 0.544 (teşhis F1 0.885 ile hedefin üstünde).

Ayrıntılar ve grafikler: [docs/SONUCLAR.md](docs/SONUCLAR.md), [results/figures/](results/figures/). Canlı stres testi: [docs/STRES_TESTI.md](docs/STRES_TESTI.md).

## Hızlı başlangıç (arayüz ve eğitilmiş model)

Veri indirmek gerekmez: eğitilmiş model (`models/`) ve arayüz için demo veri seti (`data/demo/`, PTB-XL test kümesinden 213 gerçek kayıt) repoda.

Gereksinimler: Python 3.10+ (3.14 ile test edildi), Node.js 18+ (22 ile test edildi).

```bash
pip install -r requirements.txt
npm install

python -m uvicorn src.main:app --port 8000     # 1. terminal: backend (FastAPI)
npm run dev                                    # 2. terminal: arayüz → http://localhost:5173
```

Arayüzün başlığında `FastAPI · MAE (PTB-XL, epoch 15, genlik farkındalıklı)` görünmelidir. Backend kapalıysa arayüz tarayıcı içi Mock-API'ye geçer.

Arayüzde:
- Senaryo seçilir: Normal, Anterior, Inferior ya da Lateral MI.
- Derivasyonlar koparılır (en fazla 4); kopan derivasyonun rekonstrüksiyonu mor kesikli çizgiyle gösterilir.
- 3D kalpte etkilenen duvara tıklanınca Click-to-Explain kartı açılır.
- "Yeni hasta" ile aynı senaryodan başka bir kayda geçilir; "Rapor (PDF)" ile hastaya özel klinik rapor alınır.

## Tam veriyle çalışma (eğitim ve değerlendirme)

```bash
# PTB-XL v1.0.3 (~1.8 GB)
curl -L -o data/ptbxl.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
unzip -q data/ptbxl.zip -d data/ && mv data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 data/ptbxl

python -m src.prepare_data                     # İP1: filtre + z-score + strat_fold → data/processed (~2 dk)
python -m src.train --amplitude-aware          # İP3: ana model (~2 saat, Apple M3)
python -m src.train --arch resnet1d            # karşılaştırma modeli (~5 dk)
python -m src.calibrate_walls                  # duvar karar eşikleri (doğrulama kümesi)
python -m src.evaluate --baseline models/resnet1d_baseline.pth     # test stres testleri → results/
python -m src.noise_test --baseline models/resnet1d_baseline.pth   # gürültü testi (~70 dk; önce NSTDB, aşağıda)
python -m src.plots                            # İP6 grafikleri → results/figures/
```

Gürültü testi için MIT-BIH Noise Stress Test Database:
```bash
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/nstdb', records=['bw', 'em', 'ma'])"
```

Backend, `data/processed` varsa tam test kümesini, yoksa `data/demo`'yu kullanır.

## Yöntem (Rapor Md. 2.1–2.5)

| Bileşen | Dosya |
|---|---|
| 0.5–45 Hz, 4. derece çift yönlü Butterworth + derivasyon bazlı Z-score | `src/preprocessing.py` |
| PTB-XL etiketleri (MI, Normal, kriz tipi), dengeli alt kümeler, strat_fold bölümleme | `src/prepare_data.py`, `src/dataset.py` |
| LeadDropout: DataLoader seviyesinde, batch başına k ∈ [1, 4] derivasyon; rastgele ve kritik duvar kopması | `src/lead_dropout.py` |
| Çift başlıklı MAE: 1D-CNN omurga, STFT (bimodal), Spatio-Temporal Transformer (Spatial Attention), Lead Fusing IRC, 1D Transpose Conv kod çözücü, GAP + Dense teşhis başlığı | `src/model.py` |
| Kayıp: 0.4·MSE (yalnızca kopan derivasyonlar) + 0.6·Focal | `src/losses.py` |
| Einthoven üçgeni kısıtı (Risk 1 B planı) | `src/einthoven.py` |
| Metrikler (F1, AUROC, FNR, RMSE, R², Pearson r) | `src/metrics.py` |
| FastAPI backend, Click-to-Explain verisi, PDF raporu | `src/main.py`, `src/report_text.py`, `src/clinical_report.py`, `src/st_analysis.py` |
| ONNX dışa aktarımı (Risk 2 B planı) | `src/export_onnx.py` |

Ana model, sağlam derivasyonların genlik bilgisini de kullanan "genlik farkındalıklı" sürümdür. Girdi ve kayıp formülü rapordakiyle aynıdır. İlk sürüm (`models/mae_ecg_checkpoint_v1.pth`, `results/v1/`) karşılaştırma için saklanmıştır.

## Klasör yapısı

```
src/              Python: veri hattı, model, eğitim, değerlendirme, FastAPI backend
src/components/   React arayüzü (EKG kâğıdı, 3D kalp, Click-to-Explain kartı)
src/services/     Arayüzün API istemcisi (FastAPI / Mock-API), PDF raporu
backend/          Kural tabanlı fizyolojik yedek motor ve testleri (karşılaştırma için)
models/           Eğitilmiş modeller, eğitim geçmişleri, duvar eşikleri
data/demo/        Arayüz için demo veri seti (PTB-XL test kümesinden 213 kayıt)
results/          Test sonuçları, gürültü testi, grafikler (figures/), önceki model (v1/)
docs/             Sonuçlar, stres testi, birleştirme notları
tests/            pytest testleri
```

## Testler

```bash
python -m pytest                                           # backend, model, veri, API (tests/)
cd backend && python -m unittest discover -s tests -t .    # kural tabanlı motor
```

Veri gerektiren testler `data/processed` yoksa atlanır.

## Veri ve atıflar

- **PTB-XL** v1.0.3 — Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. PhysioNet, CC BY 4.0. Raporda geçen 21.837 kayıt ilk sürüme aittir; v1.0.2'de yinelenen 38 kayıt çıkarılmıştır (21.799 kayıt).
- **MIT-BIH Noise Stress Test Database** — Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. *Computers in Cardiology*, 11, 381–384. PhysioNet.
- **3D kalp modeli** (`public/heart.glb`) — HuBMAP Consortium, *Human Reference Atlas 3D Reference Objects* (Visible Human Male), https://humanatlas.io, CC BY 4.0. Tez ve raporlarda kaynak gösterilmelidir.
- PhysioNet: Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
