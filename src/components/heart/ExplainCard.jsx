import { LEADS, RECONSTRUCTION_METHODS, WALLS, WALL_LEADS } from '../../data/ecg.js';

function formatSt(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)} mm`;
}

function Row({ label, children, tone = 'text-text' }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted">{label}</span>
      <span className={`text-right font-medium ${tone}`}>{children}</span>
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
      <div
        className={`mt-2.5 space-y-1 rounded-md border p-2 text-[11px] ${
          isDiagnosis ? 'border-red-500/40 bg-red-500/10' : 'border-line bg-control/60'
        }`}
      >
        <div className="mb-1 font-semibold text-text">
          XAI Gerekçesi · {WALLS[wall].label}
          {isDiagnosis && <span className="ml-1.5 text-red-400">(model tanısı)</span>}
        </div>
        <Row label="Duvar olasılığı" tone={probability >= 0.5 ? 'text-red-400' : 'text-text'}>
          %{Math.round(probability * 100)}
        </Row>
        <Row
          label={`Maks. ST sapması (${LEADS[peakLead].name})`}
          tone={prediction.stByLead[peakLead] >= 1 ? 'text-amber-300' : 'text-text'}
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
      className="absolute left-3 top-3 z-30 max-h-[calc(100%-56px)] w-[300px] max-w-[calc(100%-24px)] overflow-y-auto rounded-xl border border-line bg-panel/95 p-3.5 shadow-2xl backdrop-blur-md"
      role="dialog"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <h4 className="m-0 text-sm font-semibold text-text">{info.title}</h4>
          <span className="text-[11px] font-medium text-accent">{info.category}</span>
        </div>
        <button
          aria-label="Bilgi kartını kapat"
          className="rounded p-1 text-muted transition hover:bg-control hover:text-text"
          onClick={onClose}
          type="button"
        >
          ✕
        </button>
      </div>

      {info.desc && <p className="mb-0 mt-2 text-xs leading-relaxed text-muted">{info.desc}</p>}

      {info.leads && (
        <div className="mt-2 text-[11px] text-text">
          <span className="font-semibold text-accent">İlişkili derivasyonlar:</span> {info.leads}
          {info.artery && <span className="text-muted"> · {info.artery}</span>}
        </div>
      )}

      {reasoning}
    </div>
  );
}
