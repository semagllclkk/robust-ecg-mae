import { LEADS, RECONSTRUCTION_METHODS, SCENARIOS, WALLS } from '../data/ecg.js';

/** Yazdırmaya hazır HTML rapor açar; tarayıcının "PDF olarak kaydet" seçeneğiyle indirilir. */
export function openReport({ scenario, disconnectedLeads, prediction }) {
  const scenarioLabel =
    SCENARIOS.find((s) => s.value === scenario)?.label ?? scenario;
  const connectedLeads = LEADS.filter(
    (_, i) => !disconnectedLeads.includes(i),
  ).map((l) => l.name);
  const disconnectedNames = disconnectedLeads.map(
    (i) => LEADS[i]?.name ?? i,
  );
  const timestamp = new Date().toLocaleString('tr-TR');
  const predLabel = prediction?.label ?? '–';
  const predConf = prediction ? `%${Math.round(prediction.confidence * 100)}` : '–';
  const signed = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
  const predST = prediction?.region ? `${signed(prediction.stDeviation)} mm` : '–';
  const narrative = prediction?.report; // backend: model çıktısından üretilen hekim/hasta metinleri (Mock'ta yok)
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
  const list = (items) => `<ul>${items.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>`;
  const truth = prediction?.groundTruth;
  const truthText = truth
    ? `${truth.label}${truth.mi_walls?.length ? ` (${truth.mi_walls.join(', ')})` : ''}`
    : null;
  const wallRows = prediction
    ? Object.entries(WALLS)
        .map(
          ([wall, meta]) =>
            `<tr><th>${meta.label} (${meta.leads})</th><td>%${Math.round(
              (prediction.wallProbabilities?.[wall] ?? 0) * 100,
            )}</td></tr>`,
        )
        .join('')
    : '';
  const stRow = prediction
    ? LEADS.map((lead, i) => `<td>${lead.name}<br>${signed(prediction.stByLead[i])}</td>`).join('')
    : '';
  const reconText =
    disconnectedLeads
      .map(
        (i) =>
          `${LEADS[i].name}: ${RECONSTRUCTION_METHODS[prediction?.reconstruction?.[i]] ?? '–'}`,
      )
      .join(', ') || 'Yok';

  const html = `<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>EKG Klinik Karar Destek Raporu</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #111; font-size: 13px; }
h1 { font-size: 17px; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 13px; margin-top: 24px; color: #444; text-transform: uppercase; letter-spacing: .05em; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
td, th { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #f0f0f0; font-weight: 600; }
.badge { display: inline-block; background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; border-radius: 4px; padding: 2px 8px; font-weight: 600; }
.warn { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
.alert { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; border-radius: 6px; padding: 10px 14px; font-weight: 600; margin-top: 12px; }
.summary { font-size: 14px; font-weight: 600; margin: 12px 0 0; }
.note { font-size: 11px; color: #666; margin-top: 6px; }
.patient { background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 6px; padding: 4px 16px; }
ul { margin: 8px 0; padding-left: 20px; } li { margin: 4px 0; line-height: 1.45; }
footer { margin-top: 40px; font-size: 11px; color: #888; border-top: 1px solid #ddd; padding-top: 8px; }
@media print { button { display: none; } }
</style>
</head>
<body>
<h1>EKG Klinik Karar Destek Raporu</h1>
<p><strong>Tarih/Saat:</strong> ${timestamp}${prediction?.ecgId ? ` · <strong>Kayıt:</strong> PTB-XL #${prediction.ecgId}` : ''}</p>
${narrative ? `<p class="summary">${esc(narrative.summary)}</p>` : ''}
${narrative?.warning ? `<div class="alert">⚠ ${esc(narrative.warning)}</div>` : ''}

<h2>Model Tahmini</h2>
<table>
<tr><th>Tanı</th><td><span class="badge ${prediction?.region ? 'warn' : ''}">${predLabel}</span></td></tr>
<tr><th>Model Güveni</th><td>${predConf}</td></tr>
${prediction?.miProbability !== undefined ? `<tr><th>MI Olasılığı</th><td>%${Math.round(prediction.miProbability * 100)}</td></tr>` : ''}
<tr><th>ST Sapması</th><td>${predST}</td></tr>
<tr><th>Şüpheli Arter</th><td>${prediction?.artery ?? '–'}</td></tr>
<tr><th>Model Kaynağı</th><td>${prediction?.modelSource ?? '–'}</td></tr>
</table>

<h2>Duvar Olasılıkları</h2>
<table>${wallRows}</table>
${narrative ? `<p class="note">${esc(narrative.confidenceNote)}</p>` : ''}

${narrative ? `<h2>Hekim İçin Değerlendirme (Açıklanabilir Yapay Zekâ)</h2>${list(narrative.clinician)}` : ''}

<h2>Derivasyon Bazında ST Sapması (mm)</h2>
<table><tr>${stRow}</tr></table>

<h2>Derivasyon Durumu</h2>
<table>
<tr><th>Bağlı Kanallar (${connectedLeads.length}/12)</th><td>${connectedLeads.join(', ') || '–'}</td></tr>
<tr><th>Kopuk Kanallar (${disconnectedNames.length})</th><td>${disconnectedNames.join(', ') || 'Yok'}</td></tr>
<tr><th>Rekonstrüksiyon</th><td>${reconText}</td></tr>
</table>

${narrative ? `<h2>Hasta ve Hasta Yakını İçin Açıklama</h2><div class="patient">${list(narrative.patient)}</div>` : ''}

<h2>Senaryo</h2>
<p>${scenarioLabel}${truthText ? ` · Demo kaydının gerçek etiketi (PTB-XL): ${esc(truthText)}` : ''}</p>

<footer>
Bu rapor EKG İzleme Paneli tarafından otomatik üretilmiştir. Araştırma prototipidir;
klinik karar için kullanılamaz.
</footer>
<script>window.print();</script>
</body>
</html>`;

  const win = window.open('', '_blank');
  if (win) {
    win.document.write(html);
    win.document.close();
  }
}
