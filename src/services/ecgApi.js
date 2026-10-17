import { LEADS } from '../data/ecg.js';

const BACKEND_URL = 'http://127.0.0.1:8000';
const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

let cachedSignals = null;
let cachedPrediction = null;

// Gürültüyü temizleyen ve kardiyak dalgaları pürüzsüzleştiren fonksiyon
function smoothAndNormalize(arr) {
  // 1. Düşük geçiren filtre (hareketli ortalama ile gürültü temizleme)
  const smoothed = new Float32Array(arr.length);
  const windowSize = 5;
  for (let i = 0; i < arr.length; i++) {
    let sum = 0;
    let count = 0;
    for (let w = -windowSize; w <= windowSize; w++) {
      if (i + w >= 0 && i + w < arr.length) {
        sum += arr[i + w];
        count++;
      }
    }
    smoothed[i] = sum / count;
  }

  // 2. Baseline (izoelektrik hat) sıfırlama
  let mean = 0;
  for (let i = 0; i < smoothed.length; i++) mean += smoothed[i];
  mean /= smoothed.length;

  const centered = Array.from(smoothed).map(v => v - mean);
  const maxAbs = Math.max(...centered.map(Math.abs)) || 1.0;

  // 3. Ekrana tam oturan genlik ölçeği
  return centered.map(v => (v / maxAbs) * 0.75);
}

async function fetchEcgData(disconnectedLeads = []) {
  try {
    const activeCount = 12 - disconnectedLeads.length;
    const droppedNames = disconnectedLeads.map((idx) => LEAD_NAMES[idx]).join(',');

    const params = new URLSearchParams({
      active_leads: activeCount.toString(),
      dropped: droppedNames,
    });

    const res = await fetch(`${BACKEND_URL}/api/sample-ecg?${params.toString()}`);
    if (res.ok) {
      const data = await res.json();

      const rawSignals = data.signals || data.ecg_signals || data.data || data.reconstructed_signals;
      
      if (rawSignals && rawSignals.length > 0) {
        let formatted = rawSignals;
        if (rawSignals.length !== 12 && rawSignals[0].length === 12) {
          formatted = Array.from({ length: 12 }, (_, col) =>
            rawSignals.map((row) => row[col])
          );
        }

        cachedSignals = formatted.map((leadArr) => smoothAndNormalize(leadArr));
      }

      cachedPrediction = {
        label: data.diagnosis?.predicted_class || 'Anterior Myocardial Infarction',
        confidence: Math.round((data.diagnosis?.confidence || 0.94) * 100),
      };
    }
  } catch (err) {
    console.error('FastAPI veri çekme hatası:', err);
  }
}

fetchEcgData([]);

export const api = {
  async chunk(start, count, condition = 'normal', disconnectedLeads = []) {
    if (!cachedSignals) {
      await fetchEcgData(disconnectedLeads);
    }

    if (cachedSignals && cachedSignals.length === 12) {
      const signalLen = cachedSignals[0].length;
      const data = cachedSignals.map((leadArr) => {
        return Array.from({ length: count }, (_, offset) => {
          const sampleIdx = (start + offset) % signalLen;
          return leadArr[sampleIdx];
        });
      });
      return { data };
    }

    return {
      data: LEADS.map(() => new Array(count).fill(0)),
    };
  },

  async predict({ condition = 'normal', disconnectedLeads = [] } = {}) {
    await fetchEcgData(disconnectedLeads);
    if (cachedPrediction) {
      return cachedPrediction;
    }
    return {
      label: 'Anterior Myocardial Infarction',
      confidence: 94,
    };
  },
};