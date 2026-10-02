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

export const REGION_INFO = {
  ant: {
    title: 'Anterior (Ön Duvar)',
    description:
      'V1, V2, V3, V4 derivasyonlarını temsil eder. Kesinti durumunda ön yüzdeki iskemik bulguların tespiti zorlaşır.',
    leads: [6, 7, 8, 9],
    path:
      'M 50 20 C 35 5, 10 15, 10 35 C 10 50, 25 65, 50 95 L 60 85 C 45 65, 40 45, 50 20 Z',
    fillClass: 'fill-ischemic-anterior',
  },
  inf: {
    title: 'Inferior (Alt Duvar)',
    description:
      'II, III, aVF derivasyonlarını temsil eder. Bu bölgedeki elektrot kopmaları alt taban hasarının (Inferior MI) atlanmasına neden olabilir.',
    leads: [1, 2, 5],
    path: 'M 10 35 C 0 50, 15 75, 50 95 L 30 75 C 10 55, 10 40, 10 35 Z',
    fillClass: 'fill-ischemic-inferior',
  },
  lat: {
    title: 'Lateral (Yan Duvar)',
    description:
      'I, aVL, V5, V6 derivasyonlarını temsil eder. Model, sağlam kanallardaki Spatial Attention ile bu bölgeyi rekonstrükte eder.',
    leads: [0, 4, 10, 11],
    path: 'M 50 20 C 80 0, 100 30, 85 60 L 50 95 L 30 75 C 20 65, 30 40, 50 40 Z',
    fillClass: 'fill-ischemic-lateral',
  },
};

export const SCENARIOS = [
  { value: 'normal', label: 'Normal Sinüs Ritmi' },
  { value: 'ant', label: 'Anterior MI (Ön Duvar)' },
  { value: 'inf', label: 'Inferior MI (Alt Duvar)' },
];
