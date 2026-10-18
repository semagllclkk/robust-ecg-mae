import { LEADS, WALL_LEADS } from '../data/ecg.js';

/*
 * EKG servis katmanı.
 *
 * Önce FastAPI backend'ine (/api, Vite proxy → :8000) bağlanmayı dener; backend
 * kapalıysa aynı sözleşmeyi uygulayan tarayıcı içi Mock-API'ye düşer ve arka
 * planda backend'i yoklamaya devam eder.
 *
 * Sözleşme:
 *   chunk({ start, count, condition, disconnectedLeads })
 *     → { data: number[12][count], reconstructed: { [leadIndex]: number[count] } }
 *   predict({ condition, disconnectedLeads })
 *     → { label, region, regions, artery, confidence, stDeviation, stByLead[12],
 *         attention[12], wallProbabilities, reconstruction, modelSource }
 *     FastAPI ayrıca: reconstructionSources, report.warning, groundTruth, ecgId
 */

const API_BASE = import.meta.env.VITE_API_URL ?? '/api';
const SAMPLE_RATE = 100;
const WINDOW_SAMPLES = 1000;
const HEART_RATE_INTERVAL = 60 / 72;
const REPROBE_MS = 10000;

/* ------------------------------------------------------------------ Mock-API */
// backend/app/signals.py ile aynı morfoloji
const LEAD_I = [0.12, -0.08, 0.7, -0.1, 0.25];
const LEAD_II = [0.15, -0.1, 1, -0.15, 0.3];
const PRECORDIAL = [
  [0.05, 0, 0.2, -0.7, 0.1],
  [0.06, 0, 0.4, -1, 0.3],
  [0.08, -0.05, 0.6, -0.6, 0.35],
  [0.1, -0.08, 1, -0.3, 0.4],
  [0.1, -0.1, 1.1, -0.15, 0.35],
  [0.1, -0.1, 0.9, -0.1, 0.3],
];
const CENTERS = [0.06, 0.155, 0.18, 0.205, 0.38];
const SIGMAS = [0.02, 0.008, 0.012, 0.009, 0.055];
const ST_CENTER = 0.27;
const ST_SIGMA = 0.04;
const ST_ELEVATION_MV = 0.25;
const INJURY = {
  normal: { angle: null, precordial: [] },
  ant: { angle: null, precordial: [0, 1, 2, 3] },
  inf: { angle: 100, precordial: [] },
  lat: { angle: -10, precordial: [4, 5] },
};
const LIMB_BASIS = [
  [1, 0],
  [0, 1],
  [-1, 1],
  [-0.5, -0.5],
  [1, -0.5],
  [-0.5, 1],
];
const WALL_LABELS = { anterior: 'Anterior MI', inferior: 'Inferior MI', lateral: 'Lateral MI' };
const WALL_ARTERY = {
  anterior: 'LAD (sol ön inen arter)',
  inferior: 'RCA (sağ koroner arter)',
  lateral: 'LCx (sirkumfleks arter)',
};
const EVIDENCE_WEIGHT = { einthoven: 1, interpolation: 0.35, unavailable: 0 };

const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
const gauss = (x, c, s) => Math.exp(-((x - c) ** 2) / (2 * s * s));
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);

function noise(sampleNumber, leadIndex) {
  const seed = Math.sin((sampleNumber * 13 + leadIndex) * 12.9898) * 43758.5453;
  return seed - Math.floor(seed);
}

function sampleAll(n, condition) {
  const injury = INJURY[condition];
  if (!injury) throw new Error(`Bilinmeyen EKG senaryosu: ${condition}`);
  const t = n / SAMPLE_RATE;
  const beat = Math.floor(t / HEART_RATE_INTERVAL);
  const waves = [0, 0, 0, 0, 0];
  let st = 0;
  for (let offset = 0; offset <= 1; offset += 1) {
    const d = t - (beat - offset) * HEART_RATE_INTERVAL;
    for (let w = 0; w < 5; w += 1) waves[w] += gauss(d, CENTERS[w], SIGMAS[w]);
    st += gauss(d, ST_CENTER, ST_SIGMA);
  }
  let i = dot(LEAD_I, waves);
  let ii = dot(LEAD_II, waves);
  if (injury.angle !== null) {
    const theta = (injury.angle * Math.PI) / 180;
    i += ST_ELEVATION_MV * Math.cos(theta) * st;
    ii += ST_ELEVATION_MV * Math.cos(Math.PI / 3 - theta) * st;
  }
  const out = [i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2];
  PRECORDIAL.forEach((c, v) => {
    out.push(dot(c, waves) + (injury.precordial.includes(v) ? ST_ELEVATION_MV * st : 0));
  });
  return out.map(
    (value, lead) =>
      value +
      0.03 * Math.sin(2 * Math.PI * 0.25 * t + lead) +
      0.03 * (noise(n, lead) - 0.5),
  );
}

function generate(start, count, condition, disconnected) {
  const data = LEADS.map(() => new Array(count));
  for (let k = 0; k < count; k += 1) {
    const values = sampleAll(start + k, condition);
    for (let lead = 0; lead < 12; lead += 1) {
      data[lead][k] = disconnected.includes(lead) ? 0 : values[lead];
    }
  }
  return data;
}

/** Einthoven/Goldberger (ekstremite) + komşu enterpolasyonu (prekordiyal). */
function reconstruct(data, disconnected) {
  const count = data[0].length;
  const out = {};
  const methods = {};
  const limbOk = [0, 1, 2, 3, 4, 5].filter((i) => !disconnected.includes(i));
  const limbMissing = [0, 1, 2, 3, 4, 5].filter((i) => disconnected.includes(i));

  if (limbMissing.length) {
    // 2×2 normal denklemleriyle (I, II) en küçük kareler çözümü
    let a = 0, b = 0, c = 0;
    limbOk.forEach((i) => {
      const [p, q] = LIMB_BASIS[i];
      a += p * p; b += p * q; c += q * q;
    });
    const det = a * c - b * b;
    limbMissing.forEach((m) => {
      if (Math.abs(det) < 1e-9) {
        methods[m] = 'unavailable';
        return;
      }
      const series = new Array(count);
      for (let k = 0; k < count; k += 1) {
        let bp = 0, bq = 0;
        limbOk.forEach((i) => {
          bp += LIMB_BASIS[i][0] * data[i][k];
          bq += LIMB_BASIS[i][1] * data[i][k];
        });
        const leadI = (c * bp - b * bq) / det;
        const leadII = (a * bq - b * bp) / det;
        series[k] = LIMB_BASIS[m][0] * leadI + LIMB_BASIS[m][1] * leadII;
      }
      out[m] = series;
      methods[m] = 'einthoven';
    });
  }

  const vOk = [0, 1, 2, 3, 4, 5].filter((v) => !disconnected.includes(6 + v));
  for (let v = 0; v < 6; v += 1) {
    if (!disconnected.includes(6 + v)) continue;
    if (!vOk.length) {
      methods[6 + v] = 'unavailable';
      continue;
    }
    const left = vOk.filter((k) => k < v).pop();
    const right = vOk.find((k) => k > v);
    const series = new Array(count);
    for (let k = 0; k < count; k += 1) {
      if (left !== undefined && right !== undefined) {
        const w = (v - left) / (right - left);
        series[k] = (1 - w) * data[6 + left][k] + w * data[6 + right][k];
      } else {
        series[k] = data[6 + (left ?? right)][k];
      }
    }
    out[6 + v] = series;
    methods[6 + v] = 'interpolation';
  }
  return { reconstructed: out, methods };
}

/** Mock, atım zamanlamasını bildiği için ST'yi doğrudan PR → J+20 ms farkından ölçer. */
function measureSt(data) {
  const beats = [];
  for (let b = 1; b * HEART_RATE_INTERVAL + 0.3 < WINDOW_SAMPLES / SAMPLE_RATE; b += 1) {
    const r = Math.round((b * HEART_RATE_INTERVAL + CENTERS[2]) * SAMPLE_RATE);
    beats.push(r);
  }
  return data.map((series) => {
    const values = beats.map((r) => {
      const baseline = (series[r - 7] + series[r - 6] + series[r - 5] + series[r - 4]) / 4;
      return (series[r + 7] - baseline) * 10;
    });
    values.sort((x, y) => x - y);
    return values[Math.floor(values.length / 2)];
  });
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const mockApi = {
  source: 'mock',

  async chunk({ start, count, condition = 'normal', disconnectedLeads = [] }) {
    await wait(10);
    // Rekonstrüksiyon gerekmiyorsa 10 s bağlam üretilmez
    const contextStart = disconnectedLeads.length
      ? Math.min(start, Math.max(0, start + count - WINDOW_SAMPLES))
      : start;
    const window = generate(contextStart, start + count - contextStart, condition, disconnectedLeads);
    const offset = start - contextStart;
    const { reconstructed } = disconnectedLeads.length
      ? reconstruct(window, disconnectedLeads)
      : { reconstructed: {} };
    const slice = (series) => series.slice(offset);
    return {
      data: window.map(slice),
      reconstructed: Object.fromEntries(
        Object.entries(reconstructed).map(([lead, series]) => [lead, slice(series)]),
      ),
    };
  },

  async predict({ condition = 'normal', disconnectedLeads = [] }) {
    await wait(60);
    const data = generate(0, WINDOW_SAMPLES, condition, disconnectedLeads);
    const { reconstructed, methods } = reconstruct(data, disconnectedLeads);
    const completed = data.map((series, lead) => reconstructed[lead] ?? series);
    const st = measureSt(completed).map((v, lead) =>
      methods[lead] === 'unavailable' ? 0 : v,
    );
    const evidence = st.map((v, lead) => v * (EVIDENCE_WEIGHT[methods[lead]] ?? 1));

    const wallProbabilities = {};
    Object.entries(WALL_LEADS).forEach(([wall, leads]) => {
      const sorted = leads.map((i) => evidence[i]).sort((x, y) => y - x);
      wallProbabilities[wall] = sigmoid((sorted[1] - 1) / 0.2);
    });
    const normal = Object.values(wallProbabilities).reduce((p, v) => p * (1 - v), 1);
    const best = Object.keys(wallProbabilities).reduce((x, y) =>
      wallProbabilities[y] > wallProbabilities[x] ? y : x,
    );
    const region = wallProbabilities[best] >= 0.5 ? best : null;
    const wallMissing = region
      ? WALL_LEADS[region].filter((i) => disconnectedLeads.includes(i)).length
      : 0;
    let confidence = region ? wallProbabilities[region] * (1 - 0.06 * wallMissing) : normal;
    confidence *= 1 - 0.02 * disconnectedLeads.length;

    const rawAttention = st.map((v, lead) =>
      disconnectedLeads.includes(lead) ? 0 : Math.abs(v) + 0.1,
    );
    const total = rawAttention.reduce((s, v) => s + v, 0);

    const regions = region
      ? Object.keys(wallProbabilities).filter((wall) => wallProbabilities[wall] >= 0.5)
      : [];

    return {
      label: region ? regions.map((wall) => WALL_LABELS[wall].replace(' MI', '')).join(' + ') + ' MI' : 'Normal Sinüs Ritmi',
      region,
      regions,
      artery: region ? WALL_ARTERY[region] : null,
      confidence: Math.min(0.99, Math.max(0.5, confidence)),
      stDeviation: region ? Math.max(...WALL_LEADS[region].map((i) => st[i])) : 0,
      stByLead: st,
      attention: rawAttention.map((v) => v / total),
      wallProbabilities,
      reconstruction: methods,
      modelSource: 'mock',
    };
  },
};

/* --------------------------------------------------------------- FastAPI */
async function post(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(`API ${response.status}: ${JSON.stringify(detail.detail ?? detail)}`);
  }
  return response.json();
}

const httpApi = {
  chunk: (params) => post('/chunk', params),
  predict: (params) => post('/predict', params),
};

/* ------------------------------------------------------------- seçici */
const listeners = new Set();
const state = { mode: 'probing', modelSource: null, lastProbe: 0 };

function setMode(mode, modelSource) {
  if (state.mode === mode && state.modelSource === modelSource) return;
  state.mode = mode;
  state.modelSource = modelSource;
  listeners.forEach((listener) => listener({ ...state }));
}

async function probe() {
  state.lastProbe = Date.now();
  try {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 1500);
    const response = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    window.clearTimeout(timer);
    if (!response.ok) throw new Error(String(response.status));
    const health = await response.json();
    setMode('fastapi', health.modelSource);
  } catch {
    setMode('mock', 'mock');
  }
}

const ready = probe();

async function call(method, params) {
  await ready;
  if (state.mode === 'mock' && Date.now() - state.lastProbe > REPROBE_MS) {
    probe();
  }
  if (state.mode === 'fastapi') {
    try {
      return await httpApi[method](params);
    } catch (error) {
      // Backend kapandı ya da proxy hata döndü → Mock'a düş, sonra yeniden yokla
      console.warn('FastAPI çağrısı başarısız, Mock-API kullanılıyor:', error);
      setMode('mock', 'mock');
    }
  }
  return mockApi[method](params);
}

export const api = {
  chunk: (params) => call('chunk', params),
  predict: (params) => call('predict', params),
  subscribe(listener) {
    listeners.add(listener);
    listener({ ...state });
    return () => listeners.delete(listener);
  },
};
