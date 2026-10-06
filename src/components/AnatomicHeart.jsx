import { Suspense, lazy, useMemo, useState } from 'react';
import { LEADS } from '../data/ecg.js';
import HeartEcgMonitor from './heart/HeartEcgMonitor.jsx';

const HeartScene = lazy(() => import('./heart/HeartScene.jsx'));

const REGIONS = [
  {
    id: 'anterior',
    label: 'V3–V4 · Ön duvar',
    leads: [8, 9],
  },
  {
    id: 'inferior',
    label: 'II, III, aVF · Alt duvar',
    leads: [1, 2, 5],
  },
  {
    id: 'lateral',
    label: 'I, aVL, V5–V6 · Yan duvar',
    leads: [0, 4, 10, 11],
  },
  {
    id: 'septal',
    label: 'V1–V2 · Septum / Sağ ventrikül',
    leads: [6, 7],
  },
];

const VIEWS = [
  { id: 'anterior', label: 'Ön (Anterior)' },
  { id: 'posterior', label: 'Arka (Posterior)' },
  { id: 'left-lateral', label: 'Sol Yan' },
  { id: 'right-lateral', label: 'Sağ Yan' },
  { id: 'superior', label: 'Üst (Superior)' },
  { id: 'inferior', label: 'Alt (Inferior)' },
];

const buttonClass =
  'rounded-md border px-2 py-2 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent';

export default function AnatomicHeart({ disconnectedLeads = [] }) {
  const [bpm, setBpm] = useState(72);
  const [highlightedRegion, setHighlightedRegion] = useState(null);
  const [cameraView, setCameraView] = useState('anterior');
  const [beatActive, setBeatActive] = useState(true);
  const [wallOpacityPercent, setWallOpacityPercent] = useState(0); // 0% saydam = 100% opak
  const [explodeValue, setExplodeValue] = useState(0);
  const [selectedInfo, setSelectedInfo] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);

  const [layers, setLayers] = useState({
    vessels: true,
    coronary: true,
    atria: true,
    ventricles: true,
    inner: true,
  });

  const toggleLayer = (layerKey) => {
    setLayers((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
  };

  const affectedRegions = useMemo(
    () =>
      REGIONS.filter((region) =>
        region.leads.some((leadIndex) =>
          disconnectedLeads.includes(leadIndex),
        ),
      ).map((region) => region.id),
    [disconnectedLeads],
  );

  return (
    <section className="flex flex-col gap-4">
      {/* 3D Model Ana Görüntüleme Kartı */}
      <div className="flex flex-col rounded-2xl border border-line bg-panel p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-accent">
            3D Gerçekçi Anatomik Kalp (HRA)
          </h2>
          <span className="rounded bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent">
            12-Derivasyon XAI Eşlemesi
          </span>
        </div>

        <div className="relative h-[420px] min-h-[360px] overflow-hidden rounded-xl border border-line bg-[#0a0e14]">
          {/* Yükleniyor Göstergesi */}
          {!modelLoaded && (
            <div
              className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-[#0a0e14]/95 text-xs text-muted"
              role="status"
            >
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
              <span className="animate-pulse tracking-wide text-text">
                Anatomik 3D kalp modeli yükleniyor…
              </span>
            </div>
          )}

          <Suspense
            fallback={
              <div className="absolute inset-0 grid place-items-center text-xs text-muted">
                3D bileşen başlatılıyor…
              </div>
            }
          >
            <HeartScene
              affectedRegions={affectedRegions}
              beatActive={beatActive}
              bpm={bpm}
              cameraView={cameraView}
              explode={explodeValue}
              highlightedRegion={highlightedRegion}
              layers={layers}
              onLoaded={() => setModelLoaded(true)}
              onSelectInfo={(info) => setSelectedInfo(info)}
              wallOpacity={1 - wallOpacityPercent / 100}
            />
          </Suspense>

          {/* Tıkla-Açıkla Kartı (Click-to-Explain Info Overlay) */}
          {selectedInfo && (
            <div
              className="absolute left-3 top-3 z-30 max-w-[320px] rounded-xl border border-line bg-[#161b22]/95 p-3.5 shadow-2xl backdrop-blur-md"
              role="dialog"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h4 className="text-sm font-semibold text-text">
                    {selectedInfo.title}
                  </h4>
                  <span className="text-[11px] font-medium text-accent">
                    {selectedInfo.category}
                  </span>
                </div>
                <button
                  aria-label="Bilgi kartını kapat"
                  className="rounded p-1 text-muted transition hover:bg-control hover:text-text"
                  onClick={() => setSelectedInfo(null)}
                  type="button"
                >
                  ✕
                </button>
              </div>

              {selectedInfo.desc && (
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  {selectedInfo.desc}
                </p>
              )}

              {selectedInfo.leads && (
                <div className="mt-2.5 rounded-md border border-line bg-control/60 p-2 text-[11px]">
                  <div className="text-text">
                    <span className="font-semibold text-accent">İlişkili Derivasyonlar:</span>{' '}
                    {selectedInfo.leads}
                  </div>
                  {selectedInfo.artery && (
                    <div className="mt-1 text-muted">
                      <span className="font-semibold text-text">Besleyen Arter:</span>{' '}
                      {selectedInfo.artery}
                    </div>
                  )}
                </div>
              )}

              {selectedInfo.extra && (
                <p className="mt-2 text-[10px] italic text-muted">
                  {selectedInfo.extra}
                </p>
              )}
            </div>
          )}

          {/* Kontrol İpuçları */}
          <p className="pointer-events-none absolute bottom-2 left-2 right-2 rounded-md border border-line bg-panel/85 px-2.5 py-1.5 text-[10px] leading-snug text-muted backdrop-blur-sm">
            🖱️ <b>Sol Tık + Sürükle:</b> 360° Çevir | <b>Tekerlek:</b> Yakınlaştır | <b>Sağ Tık:</b> Kaydır | <b>Tıkla:</b> Yapıyı Açıkla
          </p>
        </div>

        <p className="mb-0 mt-2.5 text-xs text-muted">
          {affectedRegions.length > 0
            ? '⚡ Kopan derivasyon(lar) nedeniyle etkilenen kalp bölgesi pembe-beyaz yanıp sönmektedir.'
            : 'Fizyolojik kasılma, koroner damarlar ve kapak hareketleri dinamik olarak modellenmiştir.'}
        </p>
      </div>

      {/* Standart Anatomik Bakış Açıları */}
      <div className="rounded-2xl border border-line bg-panel p-4">
        <h3 className="mb-2.5 text-xs font-semibold text-text">
          Standart Anatomik Bakış Açıları
        </h3>
        <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-6">
          {VIEWS.map((view) => (
            <button
              aria-pressed={cameraView === view.id}
              className={`${buttonClass} ${
                cameraView === view.id
                  ? 'border-accent bg-accent/10 text-accent font-semibold'
                  : 'border-line bg-control text-muted hover:border-accent hover:text-accent'
              }`}
              key={view.id}
              onClick={() => setCameraView(view.id)}
              type="button"
            >
              {view.label}
            </button>
          ))}
        </div>
      </div>

      {/* XAI Segment Eşlemesi (Derivasyon) */}
      <div className="rounded-2xl border border-line bg-panel p-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="m-0 text-xs font-semibold text-text">
            XAI Segment Eşlemesi (Derivasyon)
          </h3>
          <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[10px] font-medium text-accent">
            {REGIONS.find((region) => region.id === highlightedRegion)?.label ??
              'Seçilmedi'}
          </span>
        </div>
        <p className="mb-2.5 mt-0 text-xs leading-relaxed text-muted">
          Model dikkatinin yoğunlaştığı anatomik bölgeyi 3D kalp üzerinde vurgular:
        </p>
        <div className="grid grid-cols-2 gap-1.5">
          {REGIONS.map((region) => (
            <button
              aria-pressed={highlightedRegion === region.id}
              className={`${buttonClass} ${
                highlightedRegion === region.id
                  ? 'border-accent bg-accent/10 text-accent font-semibold'
                  : 'border-line bg-control text-muted hover:border-accent hover:text-accent'
              }`}
              key={region.id}
              onClick={() =>
                setHighlightedRegion((current) =>
                  current === region.id ? null : region.id,
                )
              }
              type="button"
            >
              {region.label}
            </button>
          ))}
        </div>
        <button
          className={`${buttonClass} mt-2 w-full border-line bg-control text-muted hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50`}
          disabled={!highlightedRegion}
          onClick={() => setHighlightedRegion(null)}
          type="button"
        >
          Vurguyu Temizle
        </button>
        {affectedRegions.length > 0 && (
          <div className="mb-0 mt-3 rounded-lg border border-pink-500/30 bg-pink-500/10 p-2.5 text-xs text-pink-300" role="status">
            <span className="font-semibold">Kopan Kanallar:</span>{' '}
            {disconnectedLeads.map((index) => LEADS[index]?.name ?? index).join(', ')}
            <div className="mt-1 text-[11px] text-pink-200/80">
              İlişkili anatomik bölge 3D kalp üzerinde pembe-beyaz yanıp sönüyor.
            </div>
          </div>
        )}
      </div>

      {/* Katmanlar ve Parçalama (Layers & Dissection) */}
      <div className="rounded-2xl border border-line bg-panel p-4">
        <h3 className="mb-3 text-xs font-semibold text-text">
          Katmanlar ve Parçalama
        </h3>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={layers.vessels}
              className="accent-accent"
              onChange={() => toggleLayer('vessels')}
              type="checkbox"
            />
            <span>Büyük Damarlar</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={layers.coronary}
              className="accent-accent"
              onChange={() => toggleLayer('coronary')}
              type="checkbox"
            />
            <span>Koroner Arterler</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={layers.atria}
              className="accent-accent"
              onChange={() => toggleLayer('atria')}
              type="checkbox"
            />
            <span>Kulakçıklar (Atria)</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={layers.ventricles}
              className="accent-accent"
              onChange={() => toggleLayer('ventricles')}
              type="checkbox"
            />
            <span>Karıncıklar (Ventricles)</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={layers.inner}
              className="accent-accent"
              onChange={() => toggleLayer('inner')}
              type="checkbox"
            />
            <span>İç Yapılar / Kapaklar</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={beatActive}
              className="accent-accent"
              onChange={(e) => setBeatActive(e.target.checked)}
              type="checkbox"
            />
            <span>Kalp Atımı</span>
          </label>
        </div>

        {/* Dış Duvar Saydamlığı */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-muted">
            <span>Dış Duvar Saydamlığı</span>
            <span className="font-medium text-accent">%{wallOpacityPercent}</span>
          </div>
          <input
            aria-label="Dış duvar saydamlığı"
            className="mt-1.5 w-full accent-accent"
            max="100"
            min="0"
            onChange={(e) => setWallOpacityPercent(Number(e.target.value))}
            type="range"
            value={wallOpacityPercent}
          />
        </div>

        {/* Parçalama (Exploded view) */}
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-muted">
            <span>Parçalama Görünümü (Exploded)</span>
            <span className="font-medium text-accent">{explodeValue}</span>
          </div>
          <input
            aria-label="Parçalama görünümü mesafesi"
            className="mt-1.5 w-full accent-accent"
            max="20"
            min="0"
            onChange={(e) => setExplodeValue(Number(e.target.value))}
            step="0.5"
            type="range"
            value={explodeValue}
          />
        </div>
      </div>

      {/* Canlı EKG Monitörü & Kalp Atım Hızı */}
      <div className="rounded-2xl border border-line bg-panel p-4">
        <HeartEcgMonitor bpm={bpm} />
        <label className="mt-3 flex items-center justify-between gap-3 text-xs text-muted">
          Kalp atım hızı (BPM)
          <span className="font-semibold text-accent">{bpm} BPM</span>
        </label>
        <input
          aria-label="Kalp atım hızı"
          className="mt-2 w-full accent-accent"
          max="160"
          min="40"
          onChange={(event) => setBpm(Number(event.target.value))}
          type="range"
          value={bpm}
        />
        <div className="mt-1 flex justify-between text-[10px] text-muted">
          <span>40 BPM</span>
          <span>160 BPM</span>
        </div>
      </div>
    </section>
  );
}
