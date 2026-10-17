import { LEADS, SCENARIOS, WALLS } from '../data/ecg.js';
import Panel from './ui/Panel.jsx';
import { Button, ProbabilityBar } from './ui/controls.jsx';

const WALL_BAR = { ant: 'bg-wall-ant', inf: 'bg-wall-inf', lat: 'bg-wall-lat', neu: 'bg-wall-neu' };
const GROUPS = [
  { title: 'Ekstremite', leads: [0, 1, 2, 3, 4, 5] },
  { title: 'Prekordiyal', leads: [6, 7, 8, 9, 10, 11] },
];

export function LeadPanel({ disconnectedLeads, maxDisconnected, notice, onToggleLead, onRandomDisconnect, onReconnectAll }) {
  return (
    <Panel
      meta={<span className="num">{disconnectedLeads.length}/{maxDisconnected} kopuk</span>}
      title="Derivasyonlar"
    >
      <div className="flex flex-col gap-3">
        {GROUPS.map((group) => (
          <div key={group.title}>
            <p className="mb-1.5 mt-0 text-[11px] font-medium text-subtle">{group.title}</p>
            <div className="grid grid-cols-6 gap-1">
              {group.leads.map((index) => {
                const lead = LEADS[index];
                const off = disconnectedLeads.includes(index);
                return (
                  <button
                    aria-label={`${lead.name} derivasyonu ${off ? 'kopuk' : 'bağlı'}`}
                    aria-pressed={!off}
                    className={`relative h-9 overflow-hidden rounded-md border text-xs font-semibold transition-colors ${
                      off
                        ? 'border-danger/40 bg-danger/5 text-danger line-through'
                        : 'border-line bg-surface text-text hover:border-line-strong hover:bg-sunken'
                    }`}
                    key={lead.name}
                    onClick={() => onToggleLead(index)}
                    title={off ? 'Kabloyu bağla' : 'Kabloyu kopar'}
                    type="button"
                  >
                    <span className={`absolute inset-x-0 top-0 h-[3px] ${off ? 'bg-danger/50' : WALL_BAR[lead.wall]}`} />
                    {lead.name}
                  </button>
                );
              })}
            </div>
          </div>
        ))}

        {notice && (
          <p className="m-0 rounded-md bg-danger/5 px-2.5 py-1.5 text-xs text-danger" role="alert">
            {notice}
          </p>
        )}

        <div className="grid grid-cols-2 gap-2">
          <Button onClick={onRandomDisconnect}>Rastgele kopar</Button>
          <Button disabled={!disconnectedLeads.length} onClick={onReconnectAll}>
            Tümünü bağla
          </Button>
        </div>

        <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-line pt-3 text-[11px] text-muted">
          {[
            ['ant', WALLS.anterior.label],
            ['inf', WALLS.inferior.label],
            ['lat', WALLS.lateral.label],
          ].map(([key, label]) => (
            <span className="flex items-center gap-1.5" key={key}>
              <i className={`h-2 w-2 rounded-sm ${WALL_BAR[key]}`} />
              {label}
            </span>
          ))}
        </div>
      </div>
    </Panel>
  );
}

export function ScenarioPanel({ scenario, onScenarioChange }) {
  return (
    <Panel meta="Sinyal kaynağı" title="Senaryo">
      <div className="flex flex-col gap-1" role="radiogroup">
        {SCENARIOS.map((option) => {
          const active = option.value === scenario;
          return (
            <button
              aria-checked={active}
              className={`flex h-8 items-center gap-2.5 rounded-md px-2.5 text-left text-xs transition-colors ${
                active ? 'bg-accent/10 font-medium text-accent' : 'text-muted hover:bg-sunken hover:text-text'
              }`}
              key={option.value}
              onClick={() => onScenarioChange(option.value)}
              role="radio"
              type="button"
            >
              <span
                className={`grid h-3.5 w-3.5 place-items-center rounded-full border ${
                  active ? 'border-accent' : 'border-line-strong'
                }`}
              >
                {active && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
              </span>
              {option.label}
            </button>
          );
        })}
      </div>
    </Panel>
  );
}

export function DiagnosisPanel({ prediction, disconnectedLeads }) {
  const isMi = Boolean(prediction?.region);
  return (
    <Panel meta={prediction?.latencyMs != null ? `${prediction.latencyMs} ms` : null} title="Model çıktısı">
      {!prediction ? (
        <p className="m-0 text-xs text-subtle">Tahmin bekleniyor…</p>
      ) : (
        <div className="flex flex-col gap-3">
          <div>
            <p className={`m-0 text-lg font-semibold leading-tight ${isMi ? 'text-danger' : 'text-text'}`}>
              {prediction.label}
            </p>
            <p className="m-0 mt-0.5 text-xs text-muted">
              {isMi ? `Şüpheli arter: ${prediction.artery}` : 'Akut iskemi bulgusu yok'}
            </p>
          </div>
          <div>
            <div className="mb-1 flex justify-between text-xs text-muted">
              Karar güveni
              <span className="num text-text">%{Math.round(prediction.confidence * 100)}</span>
            </div>
            <ProbabilityBar tone={isMi ? 'danger' : 'accent'} value={prediction.confidence} />
          </div>
          <dl className="m-0 grid grid-cols-2 gap-y-1.5 text-xs">
            <dt className="text-muted">Maks. ST</dt>
            <dd className="num m-0 text-right">
              {isMi ? `+${prediction.stDeviation.toFixed(1)} mm` : '—'}
            </dd>
            <dt className="text-muted">Kopuk kanal</dt>
            <dd className="num m-0 text-right">{disconnectedLeads.length}</dd>
            <dt className="text-muted">Model</dt>
            <dd className="m-0 truncate text-right">{prediction.modelSource}</dd>
          </dl>
        </div>
      )}
    </Panel>
  );
}
