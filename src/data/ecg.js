export const LEADS = [
  { name: 'I', wall: 'lat', coefficients: [0.12, -0.08, 0.7, -0.1, 0.25] },
  { name: 'II', wall: 'inf', coefficients: [0.15, -0.1, 1, -0.15, 0.3] },
  { name: 'III', wall: 'inf', coefficients: [0.08, -0.05, 0.4, -0.1, 0.12] },
  { name: 'aVR', wall: 'neu', coefficients: [-0.12, 0.05, -0.6, 0.05, -0.22] },
  { name: 'aVL', wall: 'lat', coefficients: [0.05, -0.05, 0.3, -0.1, 0.1] },
  { name: 'aVF', wall: 'inf', coefficients: [0.12, -0.08, 0.7, -0.12, 0.22] },
  { name: 'V1', wall: 'ant', coefficients: [0.05, 0, 0.2, -0.7, 0.1] },
  { name: 'V2', wall: 'ant', coefficients: [0.06, 0, 0.4, -1, 0.3] },
  { name: 'V3', wall: 'ant', coefficients: [0.08, -0.05, 0.6, -0.6, 0.35] },
  { name: 'V4', wall: 'ant', coefficients: [0.1, -0.08, 1, -0.3, 0.4] },
  { name: 'V5', wall: 'lat', coefficients: [0.1, -0.1, 1.1, -0.15, 0.35] },
  { name: 'V6', wall: 'lat', coefficients: [0.1, -0.1, 0.9, -0.1, 0.3] },
];

export const LEAD_COLORS = {
  ant: '#4cc9f0',
  inf: '#f6c453',
  lat: '#7bd88f',
  neu: '#9aa3b2',
};

export const SCENARIOS = [
  { value: 'normal', label: 'Normal Sinüs Ritmi' },
  { value: 'ant', label: 'Anterior MI (Ön Duvar)' },
  { value: 'inf', label: 'Inferior MI (Alt Duvar)' },
  { value: 'lat', label: 'Lateral MI (Yan Duvar)' },
];
