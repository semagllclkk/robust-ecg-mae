export const LEADS = [
  { name: 'I', wall: 'lat' },
  { name: 'II', wall: 'inf' },
  { name: 'III', wall: 'inf' },
  { name: 'aVR', wall: 'neu' },
  { name: 'aVL', wall: 'lat' },
  { name: 'aVF', wall: 'inf' },
  { name: 'V1', wall: 'ant' },
  { name: 'V2', wall: 'ant' },
  { name: 'V3', wall: 'ant' },
  { name: 'V4', wall: 'ant' },
  { name: 'V5', wall: 'lat' },
  { name: 'V6', wall: 'lat' },
];

// Rapor 2.5: kalp duvarı → derivasyon eşlemesi
export const WALL_LEADS = {
  anterior: [6, 7, 8, 9],
  inferior: [1, 2, 5],
  lateral: [0, 4, 10, 11],
};

export const WALLS = {
  anterior: { label: 'Ön Duvar / Septum', leads: 'V1–V4', artery: 'LAD' },
  inferior: { label: 'Alt Duvar', leads: 'II, III, aVF', artery: 'RCA' },
  lateral: { label: 'Yan Duvar', leads: 'I, aVL, V5, V6', artery: 'LCx' },
};

export const LEAD_COLORS = {
  ant: '#4cc9f0',
  inf: '#f6c453',
  lat: '#7bd88f',
  neu: '#9aa3b2',
};

export const RECONSTRUCTION_COLOR = '#ff6eb0';

export const RECONSTRUCTION_METHODS = {
  einthoven: 'Einthoven/Goldberger',
  interpolation: 'komşu enterpolasyon',
  mae: 'MAE modeli',
  unavailable: 'tamamlanamadı',
};

export const SCENARIOS = [
  { value: 'normal', label: 'Normal Sinüs Ritmi' },
  { value: 'ant', label: 'Anterior MI (Ön Duvar)' },
  { value: 'inf', label: 'Inferior MI (Alt Duvar)' },
  { value: 'lat', label: 'Lateral MI (Yan Duvar)' },
];
