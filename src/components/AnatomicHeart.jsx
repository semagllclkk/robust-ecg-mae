import { Suspense, lazy, useMemo, useState } from 'react';
import { LEADS } from '../data/ecg.js';
import HeartEcgMonitor from './heart/HeartEcgMonitor.jsx';

const HeartScene = lazy(() => import('./heart/HeartScene.jsx'));

const REGIONS = [
  {
    id: 'anterior',
    label: 'V3–V4 · Ön duvar',
    leads: [6, 7, 8, 9],
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
  { id: 'anterior', label: 'Ön' },
  { id: 'posterior', label: 'Arka' },
  { id: 'left-lateral', label: 'Sol yan' },
  { id: 'superior', label: 'Üst' },
];

const buttonClass =
  'rounded-md border px-2 py-2 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent';

export default function AnatomicHeart({ disconnectedLeads }) {
  const [bpm, setBpm] = useState(72);
  const [highlightedRegion, setHighlightedRegion] = useState(null);
  const [cameraView, setCameraView] = useState('anterior');

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
      <div className="flex flex-col rounded-2xl border border-line bg-panel p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-accent">
          3D Anatomik Kalp
        </h2>
        <div className="relative h-[360px] min-h-[300px] overflow-hidden rounded-lg border border-line bg-[#0a0e14]">
          <Suspense
            fallback={
              <div
                className="absolute inset-0 grid place-items-center text-xs text-muted"
                role="status"
              >
                3D kalp modeli yükleniyor…
              </div>
            }
          >
            <HeartScene
              affectedRegions={affectedRegions}
              bpm={bpm}
              cameraView={cameraView}
              highlightedRegion={highlightedRegion}
            />
          </Suspense>
          <p className="pointer-events-none absolute bottom-2 left-2 right-2 rounded-md border border-line bg-panel/85 px-2 py-1.5 text-[10px] leading-snug text-muted">
            Sol tık + sürükle: döndür · Tekerlek: yakınlaştır · Sağ tık: kaydır
          </p>
        </div>
        <p className="mb-0 mt-2 text-xs text-muted">
          {affectedRegions.length > 0
            ? 'Pembe-kırmızı vurgular kopuk derivasyonların ilişkili bölgelerini gösterir.'
            : 'Kalbi fareyle döndürüp yakınlaştırabilirsiniz.'}
        </p>
      </div>

      <div className="rounded-2xl border border-line bg-panel p-4">
        <h3 className="mb-2 text-xs font-semibold text-text">
          Standart anatomik bakış açıları
        </h3>
        <div className="grid grid-cols-4 gap-1.5">
          {VIEWS.map((view) => (
            <button
              aria-pressed={cameraView === view.id}
              className={`${buttonClass} ${
                cameraView === view.id
                  ? 'border-accent bg-accent/10 text-accent'
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
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="m-0 text-xs font-semibold text-text">
            XAI segment eşlemesi
          </h3>
          <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">
            {REGIONS.find((region) => region.id === highlightedRegion)?.label ??
              'Seçilmedi'}
          </span>
        </div>
        <p className="mb-2 mt-0 text-xs leading-relaxed text-muted">
          Model dikkatinin yoğunlaştığı bölgeyi vurgulamak için seçin.
        </p>
        <div className="grid grid-cols-2 gap-1.5">
          {REGIONS.map((region) => (
            <button
              aria-pressed={highlightedRegion === region.id}
              className={`${buttonClass} ${
                highlightedRegion === region.id
                  ? 'border-accent bg-accent/10 text-accent'
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
          <p className="mb-0 mt-3 text-xs text-bad" role="status">
            Kopuk kanallar:{' '}
            {disconnectedLeads.map((index) => LEADS[index].name).join(', ')}
          </p>
        )}
      </div>

      <div className="rounded-2xl border border-line bg-panel p-4">
        <HeartEcgMonitor bpm={bpm} />
        <label className="mt-3 flex items-center justify-between gap-3 text-xs text-muted">
          Kalp atım hızı
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
