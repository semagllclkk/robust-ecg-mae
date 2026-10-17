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
  const predST = prediction?.region ? `+${prediction.stDeviation.toFixed(1)} mm` : '–';
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
    ? LEADS.map((lead, i) => `<td>${lead.name}<br>${prediction.stByLead[i].toFixed(1)}</td>`).join('')
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
footer { margin-top: 40px; font-size: 11px; color: #888; border-top: 1px solid #ddd; padding-top: 8px; }
@media print { button { display: none; } }
</style>
</head>
<body>
<h1>EKG Klinik Karar Destek Raporu</h1>
<p><strong>Tarih/Saat:</strong> ${timestamp}</p>

<h2>Model Tahmini</h2>
<table>
<tr><th>Tanı</th><td><span class="badge ${prediction?.region ? 'warn' : ''}">${predLabel}</span></td></tr>
<tr><th>Model Güveni</th><td>${predConf}</td></tr>
<tr><th>ST Sapması</th><td>${predST}</td></tr>
<tr><th>Şüpheli Arter</th><td>${prediction?.artery ?? '–'}</td></tr>
<tr><th>Model Kaynağı</th><td>${prediction?.modelSource ?? '–'}</td></tr>
</table>

<h2>Duvar Olasılıkları</h2>
<table>${wallRows}</table>

<h2>Derivasyon Bazında ST Sapması (mm)</h2>
<table><tr>${stRow}</tr></table>

<h2>Derivasyon Durumu</h2>
<table>
<tr><th>Bağlı Kanallar (${connectedLeads.length}/12)</th><td>${connectedLeads.join(', ') || '–'}</td></tr>
<tr><th>Kopuk Kanallar (${disconnectedNames.length})</th><td>${disconnectedNames.join(', ') || 'Yok'}</td></tr>
<tr><th>Rekonstrüksiyon</th><td>${reconText}</td></tr>
</table>

<h2>Senaryo</h2>
<p>${scenarioLabel}</p>

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
