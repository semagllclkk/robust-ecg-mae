import { useCallback, useEffect, useState } from 'react';
import AnatomicHeart from './components/AnatomicHeart.jsx';
import DashboardLayout from './components/DashboardLayout.jsx';
import ECGCanvas from './components/ECGCanvas.jsx';
import LeadControls from './components/LeadControls.jsx';
import { api } from './services/ecgApi.js';
import { LEADS, RECONSTRUCTION_METHODS, SCENARIOS, WALLS } from './data/ecg.js';

const MAX_DISCONNECTED = 4;

export default function App() {
  const [disconnectedLeads, setDisconnectedLeads] = useState([]);
  const [scenario, setScenario] = useState('normal');
  const [prediction, setPrediction] = useState(null);
  const [fps, setFps] = useState(null);
  const [apiStatus, setApiStatus] = useState('Bağlanıyor…');
  const [backend, setBackend] = useState({ mode: 'probing', modelSource: null });
  const [notice, setNotice] = useState(null);

  useEffect(() => api.subscribe(setBackend), []);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(null), 2500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const updateApiStatus = useCallback((status) => setApiStatus(status), []);
  const updateFps = useCallback((value) => setFps(value), []);

  useEffect(() => {
    let cancelled = false;
    api
      .predict({ condition: scenario, disconnectedLeads })
      .then((result) => {
        if (!cancelled) setPrediction(result);
      })
      .catch((error) => {
        if (cancelled) return;
        console.error('EKG tahmini alınamadı:', error);
        setApiStatus('Hata');
      });

    return () => {
      cancelled = true;
    };
  }, [scenario, disconnectedLeads]);

  function toggleLead(index) {
    setDisconnectedLeads((current) => {
      if (current.includes(index)) {
        return current.filter((leadIndex) => leadIndex !== index);
      }
      if (current.length >= MAX_DISCONNECTED) {
        setNotice(`En fazla ${MAX_DISCONNECTED} kanal koparılabilir.`);
        return current;
      }
      return [...current, index];
    });
  }

  function disconnectRandomLeads() {
    const count = 1 + Math.floor(Math.random() * MAX_DISCONNECTED);
    const leads = Array.from({ length: 12 }, (_, index) => index);
    for (let i = leads.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [leads[i], leads[j]] = [leads[j], leads[i]];
    }
    setDisconnectedLeads(leads.slice(0, count).sort((a, b) => a - b));
  }

  function downloadReport() {
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

  return (
    <DashboardLayout
      apiStatus={apiStatus}
      backend={backend}
      fps={fps}
      onDownloadReport={downloadReport}
    >
      <LeadControls
        disconnectedLeads={disconnectedLeads}
        onRandomDisconnect={disconnectRandomLeads}
        onReconnectAll={() => setDisconnectedLeads([])}
        onScenarioChange={setScenario}
        notice={notice}
        onToggleLead={toggleLead}
        prediction={prediction}
        scenario={scenario}
      />
      <ECGCanvas
        disconnectedLeads={disconnectedLeads}
        onApiStatusChange={updateApiStatus}
        onFpsChange={updateFps}
        scenario={scenario}
      />
      <section className="flex flex-col gap-4 min-[901px]:col-span-2 min-[1201px]:col-span-1">
        <AnatomicHeart
          disconnectedLeads={disconnectedLeads}
          prediction={prediction}
          scenario={scenario}
        />
      </section>
    </DashboardLayout>
  );
}
