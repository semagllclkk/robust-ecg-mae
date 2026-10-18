import { LEADS, RECONSTRUCTION_METHODS, WALLS, WALL_LEADS } from '../../data/ecg.js';

function formatSt(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)} mm`;
}

function Row({ label, children, tone = 'text-text' }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted">{label}</span>
      <span className={`num text-right ${tone}`}>{children}</span>
    </div>
  );
}

/** Click-to-Explain: tıklanan yapının anatomisi + ilgili duvar için model gerekçesi (rapor 2.5). */
export default function ExplainCard({ info, prediction, disconnectedLeads, onClose }) {
  const wall = info.wall;
  const wallLeads = wall ? WALL_LEADS[wall] : [];

  let reasoning = null;
  if (wall && prediction) {
    const peakLead = wallLeads.reduce((best, lead) =>
      prediction.stByLead[lead] > prediction.stByLead[best] ? lead : best,
    );
    const attentionShare = wallLeads.reduce((sum, lead) => sum + prediction.attention[lead], 0);
    const missing = wallLeads.filter((lead) => disconnectedLeads.includes(lead));
    const probability = prediction.wallProbabilities[wall] ?? 0;
    const isDiagnosis = prediction.region === wall;

    reasoning = (
      <div className="mt-3 space-y-1.5 border-t border-line pt-3 text-xs">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="font-semibold text-text">Model gerekçesi · {WALLS[wall].label}</span>
          {isDiagnosis && (
            <span className="rounded bg-danger/10 px-1.5 py-0.5 text-[10px] font-semibold text-danger">TANI</span>
          )}
        </div>
        <Row label="Duvar olasılığı" tone={probability >= 0.5 ? 'text-danger font-semibold' : 'text-text'}>
          %{Math.round(probability * 100)}
        </Row>
        <Row
          label={`Maks. ST sapması (${LEADS[peakLead].name})`}
          tone={prediction.stByLead[peakLead] >= 1 ? 'text-warning font-semibold' : 'text-text'}
        >
          {formatSt(prediction.stByLead[peakLead])}
        </Row>
        <Row label="Şüpheli koroner arter">{WALLS[wall].artery}</Row>
        <Row label="Model dikkat payı">%{Math.round(attentionShare * 100)}</Row>
        <Row label="Eksik kanallar">
          {missing.length
            ? missing
                .map(
                  (lead) =>
                    `${LEADS[lead].name} (${RECONSTRUCTION_METHODS[prediction.reconstruction?.[lead]] ?? '—'})`,
                )
                .join(', ')
            : 'Yok'}
        </Row>
        <Row label="Karar güveni">
          %{Math.round(prediction.confidence * 100)}
          {disconnectedLeads.length > 0 && (
            <span className="ml-1 text-muted">({disconnectedLeads.length} kopuk kanal)</span>
          )}
        </Row>
      </div>
    );
  }

  return (
    <div
      className="absolute left-3 top-3 z-30 max-h-[calc(100%-24px)] w-[300px] max-w-[calc(100%-24px)] overflow-y-auto rounded-panel border border-line bg-surface p-4 shadow-[0_8px_24px_rgb(15_23_42/0.12)]"
      role="dialog"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <span className="text-[11px] font-medium text-subtle">{info.category}</span>
          <h4 className="m-0 text-sm font-semibold leading-snug text-text">{info.title}</h4>
        </div>
        <button
          aria-label="Bilgi kartını kapat"
          className="-mr-1 -mt-1 grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-sunken hover:text-text"
          onClick={onClose}
          type="button"
        >
          <svg aria-hidden="true" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 16 16">
            <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {info.desc && <p className="mb-0 mt-2 text-xs leading-relaxed text-muted">{info.desc}</p>}

      {info.leads && (
        <p className="mb-0 mt-2 text-xs text-muted">
          <span className="font-medium text-text">{info.leads}</span>
          {info.artery && <> · {info.artery}</>}
        </p>
      )}

      {reasoning}
    </div>
  );
}
