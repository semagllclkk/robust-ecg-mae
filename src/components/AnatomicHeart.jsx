import { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { LEADS, WALLS, WALL_LEADS } from '../data/ecg.js';
import ExplainCard from './heart/ExplainCard.jsx';
import Panel from './ui/Panel.jsx';
import { Button, ProbabilityBar, SegmentedControl, Slider } from './ui/controls.jsx';

const HeartScene = lazy(() => import('./heart/HeartScene.jsx'));

const VIEWS = [
  { value: 'anterior', label: 'Ön' },
  { value: 'posterior', label: 'Arka' },
  { value: 'left-lateral', label: 'Sol' },
  { value: 'right-lateral', label: 'Sağ' },
  { value: 'superior', label: 'Üst' },
  { value: 'inferior', label: 'Alt' },
];

const LAYERS = [
  ['vessels', 'Büyük damarlar'],
  ['coronary', 'Koroner arterler'],
  ['atria', 'Kulakçıklar'],
  ['ventricles', 'Karıncıklar'],
  ['inner', 'Kapaklar'],
];

function WallProbabilities({ prediction, highlightedRegion, onSelect, onReset, disconnectedLeads }) {
  return (
    <Panel
      actions={
        <Button onClick={onReset} variant="ghost">
          Modele dön
        </Button>
      }
      title="Duvar olasılıkları"
    >
      <div className="flex flex-col gap-1">
        {Object.entries(WALLS).map(([wall, meta]) => {
          const probability = prediction?.wallProbabilities?.[wall] ?? 0;
          const active = highlightedRegion === wall;
          const missing = WALL_LEADS[wall].filter((lead) => disconnectedLeads.includes(lead));
          return (
            <button
              aria-pressed={active}
              className={`rounded-md border px-2.5 py-2 text-left transition-colors ${
                active ? 'border-accent/40 bg-accent/5' : 'border-transparent hover:bg-sunken'
              }`}
              key={wall}
              onClick={() => onSelect(wall)}
              type="button"
            >
              <div className="mb-1.5 flex items-baseline justify-between gap-2">
                <span className="text-xs font-medium text-text">{meta.label}</span>
                <span className={`num text-xs ${probability >= 0.5 ? 'font-semibold text-danger' : 'text-muted'}`}>
                  %{Math.round(probability * 100)}
                </span>
              </div>
              <ProbabilityBar tone={probability >= 0.5 ? 'danger' : 'subtle'} value={probability} />
              <div className="mt-1 flex justify-between text-[11px] text-subtle">
                <span>
                  {meta.leads} · {meta.artery}
                </span>
                {missing.length > 0 && (
                  <span className="text-recon">{missing.map((lead) => LEADS[lead].name).join(', ')} kopuk</span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </Panel>
  );
}

export default function AnatomicHeart({ disconnectedLeads = [], prediction = null, scenario }) {
  const [highlightedRegion, setHighlightedRegion] = useState(null);
  const [userOverride, setUserOverride] = useState(false);
  const [cameraView, setCameraView] = useState('anterior');
  const [beatActive, setBeatActive] = useState(true);
  const [bpm, setBpm] = useState(72);
  const [wallOpacityPercent, setWallOpacityPercent] = useState(0);
  const [explodeValue, setExplodeValue] = useState(0);
  const [selectedInfo, setSelectedInfo] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [layers, setLayers] = useState({ vessels: true, coronary: true, atria: true, ventricles: true, inner: true });

  useEffect(() => {
    setUserOverride(false);
  }, [scenario]);

  // Model hasar tespit ettiğinde ilgili duvar otomatik vurgulanır (rapor 2.5)
  useEffect(() => {
    if (!userOverride) setHighlightedRegion(prediction?.region ?? null);
  }, [prediction?.region, userOverride]);

  // Kablosu kopan derivasyonların ait olduğu duvarlar
  const affectedRegions = useMemo(
    () =>
      Object.entries(WALL_LEADS)
        .filter(([, leads]) => leads.some((lead) => disconnectedLeads.includes(lead)))
        .map(([wall]) => wall),
    [disconnectedLeads],
  );

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
      <Panel
        actions={
          <div className="hidden items-center gap-3 text-[11px] text-muted sm:flex">
            <span className="flex items-center gap-1.5">
              <i className="h-2 w-2 rounded-full bg-danger" />
              Hasar tespiti
            </span>
            <span className="flex items-center gap-1.5">
              <i className="h-2 w-2 rounded-full bg-pink-300" />
              Kopuk kanal duvarı
            </span>
          </div>
        }
        bodyClassName="flex flex-col gap-3 p-3"
        meta="Tıkla: yapı ve model gerekçesi"
        title="Anatomik kalp"
      >
        <div className="heart-stage relative h-[480px] overflow-hidden rounded-md">
          {!modelLoaded && (
            <div className="absolute inset-0 z-20 grid place-items-center text-xs text-muted" role="status">
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-line-strong border-t-accent" />
                3D model yükleniyor
              </span>
            </div>
          )}
          <Suspense fallback={null}>
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
          <p className="pointer-events-none absolute bottom-2 right-3 m-0 text-[11px] text-slate-500">
            Sürükle: çevir · Tekerlek: yakınlaştır · Sağ tık: kaydır
          </p>
        </div>
        <SegmentedControl label="Bakış açısı" onChange={setCameraView} options={VIEWS} value={cameraView} />
      </Panel>

      <div className="flex flex-col gap-4">
        <WallProbabilities
          disconnectedLeads={disconnectedLeads}
          highlightedRegion={highlightedRegion}
          onReset={() => {
            setUserOverride(false);
            setHighlightedRegion(prediction?.region ?? null);
          }}
          onSelect={(wall) => {
            setUserOverride(true);
            setHighlightedRegion((current) => (current === wall ? null : wall));
          }}
          prediction={prediction}
        />

        <Panel title="Görünüm">
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-x-3 gap-y-2">
              {LAYERS.map(([key, label]) => (
                <label className="flex cursor-pointer items-center gap-2 text-xs text-muted" key={key}>
                  <input
                    checked={layers[key]}
                    className="accent-[rgb(var(--accent))]"
                    onChange={() => setLayers((prev) => ({ ...prev, [key]: !prev[key] }))}
                    type="checkbox"
                  />
                  {label}
                </label>
              ))}
              <label className="flex cursor-pointer items-center gap-2 text-xs text-muted">
                <input
                  checked={beatActive}
                  className="accent-[rgb(var(--accent))]"
                  onChange={(event) => setBeatActive(event.target.checked)}
                  type="checkbox"
                />
                Kalp atımı
              </label>
            </div>
            <Slider
              display={`%${wallOpacityPercent}`}
              label="Dış duvar saydamlığı"
              max="100"
              min="0"
              onChange={(event) => setWallOpacityPercent(Number(event.target.value))}
              value={wallOpacityPercent}
            />
            <Slider
              label="Parçalama"
              max="20"
              min="0"
              onChange={(event) => setExplodeValue(Number(event.target.value))}
              step="0.5"
              value={explodeValue}
            />
            <Slider
              display={`${bpm} bpm`}
              label="Animasyon hızı"
              max="160"
              min="40"
              onChange={(event) => setBpm(Number(event.target.value))}
              value={bpm}
            />
          </div>
        </Panel>
      </div>
    </div>
  );
}
