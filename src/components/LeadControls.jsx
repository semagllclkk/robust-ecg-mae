import { LEADS, LEAD_COLORS, SCENARIOS } from '../data/ecg.js';

const baseButton =
  'rounded-lg border border-line bg-control px-3.5 py-2.5 text-left font-medium transition hover:border-accent hover:text-accent';

export default function LeadControls({
  disconnectedLeads,
  onToggleLead,
  onReconnectAll,
  onRandomDisconnect,
  scenario,
  onScenarioChange,
  prediction,
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex flex-col rounded-2xl border border-line bg-panel p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-accent">
          Derivasyon Kontrolü
        </h2>
        <div className="grid grid-cols-3 gap-2">
          {LEADS.map((lead, index) => {
            const isDisconnected = disconnectedLeads.includes(index);
            return (
              <button
                aria-pressed={!isDisconnected}
                className={`rounded-lg border px-2 py-2.5 font-semibold transition ${
                  isDisconnected
                    ? 'border-dashed border-bad bg-bad/5 text-bad opacity-70 line-through'
                    : 'hover:bg-white/10'
                }`}
                key={lead.name}
                onClick={() => onToggleLead(index)}
                style={
                  isDisconnected
                    ? undefined
                    : {
                        borderColor: LEAD_COLORS[lead.wall],
                        color: LEAD_COLORS[lead.wall],
                      }
                }
                type="button"
              >
                {lead.name}
              </button>
            );
          })}
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            className={baseButton}
            onClick={onRandomDisconnect}
            type="button"
          >
            Rastgele Kopar
          </button>
          <button
            className={`${baseButton} flex-1`}
            onClick={onReconnectAll}
            type="button"
          >
            Tümünü Bağla
          </button>
        </div>
      </div>

      <div className="flex flex-1 flex-col rounded-2xl border border-line bg-panel p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-accent">
          Senaryo (Mock-API)
        </h2>
        <label className="sr-only" htmlFor="scenario">
          EKG senaryosu
        </label>
        <select
          className={`${baseButton} w-full cursor-pointer`}
          id="scenario"
          onChange={(event) => onScenarioChange(event.target.value)}
          value={scenario}
        >
          {SCENARIOS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {prediction && (
          <div className="mt-4 rounded-lg border border-line bg-background/60 p-3">
            <p className="m-0 text-xs text-muted">Mock tahmin</p>
            <p className="mb-0 mt-1 font-semibold text-text">
              {prediction.label}
              <span className="ml-2 text-xs font-normal text-muted">
                %{Math.round(prediction.confidence * 100)}
              </span>
            </p>
          </div>
        )}

        <div className="mt-auto flex flex-col gap-2 pt-6 text-xs text-muted">
          <span>
            <i className="mr-2 inline-block h-3 w-3 rounded bg-lead-ant align-middle" />
            Ön Duvar (V1-V4)
          </span>
          <span>
            <i className="mr-2 inline-block h-3 w-3 rounded bg-lead-inf align-middle" />
            Alt Duvar (II, III, aVF)
          </span>
          <span>
            <i className="mr-2 inline-block h-3 w-3 rounded bg-lead-lat align-middle" />
            Yan Duvar (I, aVL, V5, V6)
          </span>
        </div>
      </div>
    </section>
  );
}
