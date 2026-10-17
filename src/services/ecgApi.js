import { LEADS } from '../data/ecg.js';

const SAMPLE_RATE = 100;
const HEART_RATE_INTERVAL = 60 / 72;
const wait = (milliseconds) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function randomNoise(value) {
  const seed = Math.sin(value * 12.9898) * 43758.5453;
  return seed - Math.floor(seed);
}

function gauss(x, center, sigma) {
  return Math.exp(-((x - center) ** 2) / (2 * sigma ** 2));
}

function sampleAt(sampleNumber, leadIndex, condition) {
  const lead = LEADS[leadIndex];
  const [pAmp, qAmp, rAmp, sAmp, tAmp] = lead.coefficients;
  const time = sampleNumber / SAMPLE_RATE;
  const beat = Math.floor(time / HEART_RATE_INTERVAL);
  let value = 0;

  for (let b = beat; b <= beat + 1; b += 1) {
    const d = time - b * HEART_RATE_INTERVAL;
    value += pAmp * gauss(d, 0.08, 0.025);
    value += qAmp * gauss(d, 0.15, 0.008);
    value += rAmp * gauss(d, 0.18, 0.012);
    value += sAmp * gauss(d, 0.21, 0.009);
    value += tAmp * gauss(d, 0.38, 0.055);

    const isAffected =
      (condition === 'ant' && lead.wall === 'ant') ||
      (condition === 'inf' && lead.wall === 'inf') ||
      (condition === 'lat' && lead.wall === 'lat');

    if (isAffected) {
      value += 0.25 * gauss(d, 0.28, 0.040);
    }
  }

  return (
    value +
    0.03 * Math.sin(2 * Math.PI * 0.25 * time + leadIndex) +
    0.03 * (randomNoise(sampleNumber * 13 + leadIndex) - 0.5)
  );
}

export const api = {
  async chunk(start, count, condition = 'normal') {
    await wait(10);
    const data = LEADS.map((_, leadIndex) =>
      Array.from({ length: count }, (_, offset) =>
        sampleAt(start + offset, leadIndex, condition),
      ),
    );
    return { data };
  },

  async predict({ condition = 'normal', disconnectedLeads = [] } = {}) {
    await wait(120);
    const predictions = {
      normal: { label: 'Normal Sinüs Ritmi', confidence: 0.98 },
      ant: { label: 'Anterior MI', confidence: 0.94 },
      inf: { label: 'Inferior MI', confidence: 0.93 },
    };
    const result = predictions[condition];

    if (!result) {
      throw new Error(`Bilinmeyen EKG senaryosu: ${condition}`);
    }

    return {
      ...result,
      confidence: disconnectedLeads.length
        ? Math.max(0.5, result.confidence - disconnectedLeads.length * 0.04)
        : result.confidence,
    };
  },
};
