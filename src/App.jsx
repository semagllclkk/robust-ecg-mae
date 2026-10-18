import { useCallback, useEffect, useState } from 'react';
import AnatomicHeart from './components/AnatomicHeart.jsx';
import AppShell from './components/AppShell.jsx';
import ECGCanvas from './components/ECGCanvas.jsx';
import { DiagnosisPanel, LeadPanel, ScenarioPanel } from './components/LeadControls.jsx';
import { api } from './services/ecgApi.js';
import { openReport } from './services/report.js';

const MAX_DISCONNECTED = 4;

export default function App() {
  const [disconnectedLeads, setDisconnectedLeads] = useState([]);
  const [scenario, setScenario] = useState('normal');
  const [prediction, setPrediction] = useState(null);
  const [streamStatus, setStreamStatus] = useState('Bağlanıyor…');
  const [backend, setBackend] = useState({ mode: 'probing', modelSource: null });
  const [notice, setNotice] = useState(null);
  const [recordKey, setRecordKey] = useState(0);
  const [changingPatient, setChangingPatient] = useState(false);

  const updateStreamStatus = useCallback((status) => setStreamStatus(status), []);

  useEffect(() => api.subscribe(setBackend), []);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(null), 2500);
    return () => window.clearTimeout(timer);
  }, [notice]);

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
        setStreamStatus('Hata');
      });
    return () => {
      cancelled = true;
    };
  }, [scenario, disconnectedLeads, recordKey]);

  async function showNewPatient() {
    setChangingPatient(true);
    try {
      await api.newRecord({ condition: scenario });
      setRecordKey((key) => key + 1); // EKG akışı ve tahmin yeni kayıtla yeniden alınır
    } catch (error) {
      console.error('Yeni hasta kaydı alınamadı:', error);
      setNotice('Yeni hasta kaydı alınamadı.');
    } finally {
      setChangingPatient(false);
    }
  }

  function toggleLead(index) {
    setDisconnectedLeads((current) => {
      if (current.includes(index)) {
        return current.filter((leadIndex) => leadIndex !== index);
      }
      if (current.length >= MAX_DISCONNECTED) {
        setNotice(`En fazla ${MAX_DISCONNECTED} kanal koparılabilir.`);
        return current;
      }
      return [...current, index].sort((a, b) => a - b);
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

  return (
    <AppShell
      backend={backend}
      onDownloadReport={() => openReport({ scenario, disconnectedLeads, prediction })}
      sidebar={
        <>
          <LeadPanel
            disconnectedLeads={disconnectedLeads}
            maxDisconnected={MAX_DISCONNECTED}
            notice={notice}
            onRandomDisconnect={disconnectRandomLeads}
            onReconnectAll={() => setDisconnectedLeads([])}
            onToggleLead={toggleLead}
          />
          <ScenarioPanel
            canChangePatient={backend.mode === 'fastapi'}
            changingPatient={changingPatient}
            ecgId={prediction?.ecgId}
            onNewPatient={showNewPatient}
            onScenarioChange={setScenario}
            scenario={scenario}
          />
          <DiagnosisPanel disconnectedLeads={disconnectedLeads} prediction={prediction} />
        </>
      }
      streamStatus={streamStatus}
    >
      <ECGCanvas
        disconnectedLeads={disconnectedLeads}
        onApiStatusChange={updateStreamStatus}
        recordKey={recordKey}
        scenario={scenario}
      />
      <AnatomicHeart disconnectedLeads={disconnectedLeads} prediction={prediction} scenario={scenario} />
    </AppShell>
  );
}
