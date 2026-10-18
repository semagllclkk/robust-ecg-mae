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

// index.css'deki --wall-* ve --recon belirteçleriyle aynı (canvas CSS değişkeni okuyamaz)
export const LEAD_COLORS = {
  ant: '#2563eb',
  inf: '#d97706',
  lat: '#059669',
  neu: '#8b93a1',
};

export const RECONSTRUCTION_COLOR = '#7c3aed';

export const RECONSTRUCTION_METHODS = {
  einthoven: 'Einthoven/Goldberger',
  interpolation: 'komşu enterpolasyon',
  mae: 'MAE modeli',
  unavailable: 'tamamlanamadı',
};

export const SCENARIOS = [
  { value: 'normal', label: 'Normal sinüs ritmi', short: 'Normal' },
  { value: 'ant', label: 'Anterior MI', short: 'Anterior' },
  { value: 'inf', label: 'Inferior MI', short: 'Inferior' },
  { value: 'lat', label: 'Lateral MI', short: 'Lateral' },
];
