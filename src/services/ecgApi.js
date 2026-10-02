import { LEADS } from '../data/ecg.js';

const SAMPLE_RATE = 100;
const HEART_RATE_INTERVAL = 60 / 72;
const wait = (milliseconds) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function randomNoise(value) {
  const seed = Math.sin(value * 12.9898) * 43758.5453;
  return seed - Math.floor(seed);
}

function sampleAt(sampleNumber, leadIndex, condition) {
  const lead = LEADS[leadIndex];
  const time = sampleNumber / SAMPLE_RATE;
  const beat = Math.floor(time / HEART_RATE_INTERVAL);
  let value = 0;

  for (let currentBeat = beat; currentBeat <= beat + 1; currentBeat += 1) {
    const delta = time - currentBeat * HEART_RATE_INTERVAL;
    value +=
      lead.coefficients[2] *
      Math.exp(-((delta - 0.02) ** 2) / (2 * 0.01 ** 2));
    if (
      (condition === 'ant' && lead.wall === 'ant') ||
      (condition === 'inf' && lead.wall === 'inf')
    ) {
      value += 0.3 * Math.exp(-((delta - 0.14) ** 2) / (2 * 0.05 ** 2));
    }
  }

  return (
    value +
    0.05 * Math.sin(2 * Math.PI * 0.25 * time + leadIndex) +
    0.04 * (randomNoise(sampleNumber * 13 + leadIndex) - 0.5)
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
