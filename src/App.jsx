import { useCallback, useEffect, useState } from 'react';
import AnatomicHeart from './components/AnatomicHeart.jsx';
import DashboardLayout from './components/DashboardLayout.jsx';
import ECGCanvas from './components/ECGCanvas.jsx';
import LeadControls from './components/LeadControls.jsx';
import { api } from './services/ecgApi.js';

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

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

  // PDF Rapor İndirme Fonksiyonu
  function handleDownloadReport() {
    const activeLeadsCount = 12 - disconnectedLeads.length;
    const droppedNames = disconnectedLeads.map((idx) => LEAD_NAMES[idx]).join(', ');
    const diagnosisText = prediction?.label || (scenario === 'mi' ? 'Anterior Myocardial Infarction' : 'Normal Sinus Ritmi');
    const confidenceVal = prediction?.confidence ? (prediction.confidence / 100) : 0.94;
    
    const params = new URLSearchParams({
      patient_id: 'PTB-XL-0842',
      anomaly_score: '0.02500',
      diagnosis: diagnosisText,
      confidence: confidenceVal.toString(),
      active_leads: activeLeadsCount.toString(),
      dropped_leads: droppedNames || 'Kopma Yok (Tam Veri)',
      most_affected_lead: droppedNames ? droppedNames.split(', ')[0] : 'V2',
      anatomical_region: scenario === 'mi' ? 'Septal / Anterior (On Duvar)' : 'Saglikli Miyokard Doku'
    });

    const reportUrl = `http://127.0.0.1:8000/api/generate-report?${params.toString()}`;
    window.open(reportUrl, '_blank');
  }

  return (
    <DashboardLayout apiStatus={apiStatus} fps={fps}>
      <div className="flex flex-col gap-4">
        <LeadControls
          disconnectedLeads={disconnectedLeads}
          onRandomDisconnect={disconnectRandomLeads}
          onReconnectAll={() => setDisconnectedLeads([])}
          onScenarioChange={setScenario}
          onToggleLead={toggleLead}
          prediction={prediction}
          scenario={scenario}
        />
        
        {/* Klinik Rapor Butonu */}
        <button
          onClick={handleDownloadReport}
          className="w-full py-2.5 px-4 rounded-xl font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-950/40 border border-emerald-500/30 active:scale-98"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Klinik PDF Raporu Oluştur
        </button>
      </div>

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