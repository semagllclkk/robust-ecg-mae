import { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { LEADS, WALLS, WALL_LEADS } from '../data/ecg.js';
import ExplainCard from './heart/ExplainCard.jsx';
import HeartEcgMonitor from './heart/HeartEcgMonitor.jsx';

const HeartScene = lazy(() => import('./heart/HeartScene.jsx'));

const VIEWS = [
  { id: 'anterior', label: 'Ön' },
  { id: 'posterior', label: 'Arka' },
  { id: 'left-lateral', label: 'Sol Yan' },
  { id: 'right-lateral', label: 'Sağ Yan' },
  { id: 'superior', label: 'Üst' },
  { id: 'inferior', label: 'Alt' },
];

const LAYERS = [
  ['vessels', 'Büyük Damarlar'],
  ['coronary', 'Koroner Arterler'],
  ['atria', 'Kulakçıklar'],
  ['ventricles', 'Karıncıklar'],
  ['inner', 'Kapaklar / İç Yapılar'],
];

const buttonClass =
  'rounded-md border px-2 py-2 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent';

export default function AnatomicHeart({ disconnectedLeads = [], prediction = null, scenario }) {
  const [bpm, setBpm] = useState(72);
  const [highlightedRegion, setHighlightedRegion] = useState(null);
  const [userOverride, setUserOverride] = useState(false);
  const [cameraView, setCameraView] = useState('anterior');
  const [beatActive, setBeatActive] = useState(true);
  const [wallOpacityPercent, setWallOpacityPercent] = useState(0);
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

  useEffect(() => {
    setUserOverride(false);
  }, [scenario]);

  // Model hasar tespit ettiğinde ilgili duvar otomatik vurgulanır (rapor 2.5)
  useEffect(() => {
    if (!userOverride) {
      setHighlightedRegion(prediction?.region ?? null);
    }
  }, [prediction?.region, userOverride]);

  // Kablosu kopan derivasyonların ait olduğu duvarlar (sinyali eksik bölgeler)
  const affectedRegions = useMemo(
    () =>
      Object.entries(WALL_LEADS)
        .filter(([, leads]) => leads.some((lead) => disconnectedLeads.includes(lead)))
        .map(([wall]) => wall),
    [disconnectedLeads],
  );

  const toggleLayer = (key) => setLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <section className="flex flex-col gap-4">
      <div className="flex flex-col rounded-2xl border border-line bg-panel p-4">
        <div className="mb-3 flex items-center justify-between gap-2">
          <h2 className="m-0 text-xs font-semibold uppercase tracking-wider text-accent">
            3D Anatomik Kalp
          </h2>
          <span className="rounded bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent">
            Click-to-Explain
          </span>
        </div>

        <div className="heart-stage relative h-[440px] overflow-hidden rounded-xl border border-line">
          {!modelLoaded && (
            <div
              className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 text-xs text-slate-600"
              role="status"
            >
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-accent" />
              <span className="animate-pulse tracking-wide">Anatomik 3D kalp modeli yükleniyor…</span>
            </div>
          )}

          <Suspense
            fallback={
              <div className="absolute inset-0 grid place-items-center text-xs text-slate-600">
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
              highlightStrength={
                highlightedRegion && prediction?.wallProbabilities && !userOverride
                  ? Math.max(0.55, prediction.wallProbabilities[highlightedRegion] ?? 1)
                  : 1
              }
              layers={layers}
              onLoaded={() => setModelLoaded(true)}
              onSelectInfo={setSelectedInfo}
              wallOpacity={1 - wallOpacityPercent / 100}
            />
          </Suspense>

          {selectedInfo && (
            <ExplainCard
              disconnectedLeads={disconnectedLeads}
              info={selectedInfo}
              onClose={() => setSelectedInfo(null)}
              prediction={prediction}
            />
          )}

          <p className="pointer-events-none absolute bottom-2 left-2 right-2 m-0 rounded-md bg-white/70 px-2.5 py-1.5 text-[10px] leading-snug text-slate-700 backdrop-blur-sm">
            <b>Sürükle:</b> çevir · <b>Tekerlek:</b> yakınlaştır · <b>Sağ tık:</b> kaydır ·{' '}
            <b>Tıkla:</b> yapıyı ve model gerekçesini göster
          </p>
        </div>

        <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
          <span>
            <i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full bg-red-600 align-middle" />
            Modelin hasar tespit ettiği duvar
          </span>
          <span>
            <i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full bg-pink-300 align-middle" />
            Kopuk derivasyonun duvarı (yanıp söner)
          </span>
        </div>
      </div>

      <div className="rounded-2xl border border-line bg-panel p-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="m-0 text-xs font-semibold text-text">XAI · Duvar Olasılıkları</h3>
          <span className="text-[10px] text-muted">
            {prediction ? `Kaynak: ${prediction.modelSource}` : 'Bekleniyor…'}
          </span>
        </div>
        <div className="flex flex-col gap-1.5">
          {Object.entries(WALLS).map(([wall, meta]) => {
            const probability = prediction?.wallProbabilities?.[wall] ?? 0;
            const active = highlightedRegion === wall;
            return (
              <button
                aria-pressed={active}
                className={`${buttonClass} text-left ${
                  active
                    ? 'border-accent bg-accent/10 text-text'
                    : 'border-line bg-control text-muted hover:border-accent hover:text-text'
                }`}
                key={wall}
                onClick={() => {
                  setUserOverride(true);
                  setHighlightedRegion((current) => (current === wall ? null : wall));
                }}
                type="button"
              >
                <div className="flex items-center justify-between gap-2">
                  <span>
                    <b className="font-semibold">{meta.label}</b>{' '}
                    <span className="text-[10px] text-muted">
                      {meta.leads} · {meta.artery}
                    </span>
                  </span>
                  <span className={probability >= 0.5 ? 'font-semibold text-red-400' : ''}>
                    %{Math.round(probability * 100)}
                  </span>
                </div>
                <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-line">
                  <div
                    className={`h-full rounded-full transition-all ${
                      probability >= 0.5 ? 'bg-red-500' : 'bg-accent/60'
                    }`}
                    style={{ width: `${Math.round(probability * 100)}%` }}
                  />
                </div>
              </button>
            );
          })}
        </div>
        <button
          className={`${buttonClass} mt-2 w-full border-line bg-control text-muted hover:border-accent hover:text-accent`}
          onClick={() => {
            setUserOverride(false);
            setHighlightedRegion(prediction?.region ?? null);
          }}
          type="button"
        >
          Model tahminine dön
        </button>
        {disconnectedLeads.length > 0 && (
          <div
            className="mt-3 rounded-lg border border-pink-500/30 bg-pink-500/10 p-2.5 text-xs text-pink-200"
            role="status"
          >
            <span className="font-semibold">Kopuk kanallar:</span>{' '}
            {disconnectedLeads.map((index) => LEADS[index]?.name ?? index).join(', ')}
            <div className="mt-1 text-[11px] text-pink-200/80">
              Sinyalleri sağlam kanallardan rekonstrükte ediliyor; ait oldukları duvar yanıp söner.
            </div>
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-line bg-panel p-4">
        <h3 className="mb-2.5 mt-0 text-xs font-semibold text-text">Bakış Açısı</h3>
        <div className="grid grid-cols-6 gap-1.5">
          {VIEWS.map((view) => (
            <button
              aria-pressed={cameraView === view.id}
              className={`${buttonClass} px-1 ${
                cameraView === view.id
                  ? 'border-accent bg-accent/10 font-semibold text-accent'
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

      <div className="rounded-2xl border border-line bg-panel p-4">
        <h3 className="mb-3 mt-0 text-xs font-semibold text-text">Katmanlar ve Parçalama</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {LAYERS.map(([key, label]) => (
            <label
              className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text"
              key={key}
            >
              <input
                checked={layers[key]}
                className="accent-accent"
                onChange={() => toggleLayer(key)}
                type="checkbox"
              />
              <span>{label}</span>
            </label>
          ))}
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-control p-2 text-muted hover:text-text">
            <input
              checked={beatActive}
              className="accent-accent"
              onChange={(event) => setBeatActive(event.target.checked)}
              type="checkbox"
            />
            <span>Kalp Atımı</span>
          </label>
        </div>

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
            onChange={(event) => setWallOpacityPercent(Number(event.target.value))}
            type="range"
            value={wallOpacityPercent}
          />
        </div>

        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-muted">
            <span>Parçalama Görünümü</span>
            <span className="font-medium text-accent">{explodeValue}</span>
          </div>
          <input
            aria-label="Parçalama görünümü mesafesi"
            className="mt-1.5 w-full accent-accent"
            max="20"
            min="0"
            onChange={(event) => setExplodeValue(Number(event.target.value))}
            step="0.5"
            type="range"
            value={explodeValue}
          />
        </div>
      </div>

      <div className="rounded-2xl border border-line bg-panel p-4">
        <HeartEcgMonitor bpm={bpm} />
        <label className="mt-3 flex items-center justify-between gap-3 text-xs text-muted" htmlFor="bpm">
          Animasyon kalp hızı
          <span className="font-semibold text-accent">{bpm} BPM</span>
        </label>
        <input
          className="mt-2 w-full accent-accent"
          id="bpm"
          max="160"
          min="40"
          onChange={(event) => setBpm(Number(event.target.value))}
          type="range"
          value={bpm}
        />
      </div>
    </section>
  );
}
