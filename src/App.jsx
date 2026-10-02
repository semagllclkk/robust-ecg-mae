import { useCallback, useEffect, useState } from 'react';
import AnatomicHeart from './components/AnatomicHeart.jsx';
import DashboardLayout from './components/DashboardLayout.jsx';
import ECGCanvas from './components/ECGCanvas.jsx';
import LeadControls from './components/LeadControls.jsx';
import { api } from './services/ecgApi.js';

export default function App() {
  const [disconnectedLeads, setDisconnectedLeads] = useState([]);
  const [scenario, setScenario] = useState('normal');
  const [prediction, setPrediction] = useState(null);
  const [fps, setFps] = useState(null);
  const [apiStatus, setApiStatus] = useState('Bağlanıyor…');

  const updateApiStatus = useCallback((status) => setApiStatus(status), []);
  const updateFps = useCallback((value) => setFps(value), []);

  useEffect(() => {
    let cancelled = false;
    setPrediction(null);
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
      if (current.length >= 4) {
        window.alert('Maksimum 4 kanal koparılabilir.');
        return current;
      }
      return [...current, index];
    });
  }

  function disconnectRandomLeads() {
    const count = 1 + Math.floor(Math.random() * 4);
    const shuffled = Array.from({ length: 12 }, (_, index) => index).sort(
      () => Math.random() - 0.5,
    );
    setDisconnectedLeads(shuffled.slice(0, count));
  }

  return (
    <DashboardLayout apiStatus={apiStatus} fps={fps}>
      <LeadControls
        disconnectedLeads={disconnectedLeads}
        onRandomDisconnect={disconnectRandomLeads}
        onReconnectAll={() => setDisconnectedLeads([])}
        onScenarioChange={setScenario}
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
        <AnatomicHeart disconnectedLeads={disconnectedLeads} />
      </section>
    </DashboardLayout>
  );
}
