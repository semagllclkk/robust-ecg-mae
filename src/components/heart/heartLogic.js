import * as THREE from 'three';

const V3 = (x, y, z) => new THREE.Vector3(x, y, z);

/* ---------- Gürültü (doku benekliliği ve epikardiyal yağ için) ---------- */
function hash(i, j, k) {
  let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(k, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function vnoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  const l = (a, b, t) => a + (b - a) * t;
  return l(
    l(l(hash(xi, yi, zi), hash(xi + 1, yi, zi), u), l(hash(xi, yi + 1, zi), hash(xi + 1, yi + 1, zi), u), v),
    l(l(hash(xi, yi, zi + 1), hash(xi + 1, yi, zi + 1), u), l(hash(xi, yi + 1, zi + 1), hash(xi + 1, yi + 1, zi + 1), u), v),
    w
  );
}

export function fbm(x, y, z) {
  return (
    0.5 * vnoise(x, y, z) +
    0.25 * vnoise(2 * x, 2 * y, 2 * z) +
    0.125 * vnoise(4 * x, 4 * y, 4 * z) +
    0.0625 * vnoise(8 * x, 8 * y, 8 * z)
  ) / 0.9375;
}

export function smoothstep(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export function srgb(hex) {
  return new THREE.Color(hex).convertSRGBToLinear();
}

/* ---------- Değişken yarıçaplı tüp (damarlar için) ---------- */
export function buildTubeGeometry(points, radiusFn, opts = {}) {
  const segs = opts.segs || 96;
  const radial = opts.radial || 28;
  const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
  const frames = curve.computeFrenetFrames(segs, false);
  const pos = [];
  const nor = [];
  const idx = [];
  const row = radial + 1;

  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const P = curve.getPointAt(t);
    const N = frames.normals[i];
    const B = frames.binormals[i];
    const r = radiusFn(t);
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      const cs = Math.cos(a);
      const sn = Math.sin(a);
      const dx = N.x * cs + B.x * sn;
      const dy = N.y * cs + B.y * sn;
      const dz = N.z * cs + B.z * sn;
      pos.push(P.x + dx * r, P.y + dy * r, P.z + dz * r);
      nor.push(dx, dy, dz);
    }
  }

  for (let i = 0; i < segs; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * row + j;
      const b = (i + 1) * row + j;
      const c = (i + 1) * row + j + 1;
      const d = i * row + j + 1;
      idx.push(a, b, d, b, c, d);
    }
  }

  // Yüz normali dışa baksın
  const p = (n) => V3(pos[n * 3], pos[n * 3 + 1], pos[n * 3 + 2]);
  const fn = p(idx[1]).sub(p(idx[0])).cross(p(idx[2]).sub(p(idx[0])));
  if (fn.dot(V3(nor[idx[0] * 3], nor[idx[0] * 3 + 1], nor[idx[0] * 3 + 2])) < 0) {
    for (let k = 0; k < idx.length; k += 3) {
      const t = idx[k + 1];
      idx[k + 1] = idx[k + 2];
      idx[k + 2] = t;
    }
  }

  // Uç kapakları
  function cap(ringStart, tangentSign, tNorm) {
    const centre = curve.getPointAt(tNorm);
    const tan = curve.getTangentAt(tNorm).multiplyScalar(tangentSign);
    const ci = pos.length / 3;
    pos.push(centre.x, centre.y, centre.z);
    nor.push(tan.x, tan.y, tan.z);
    const ringBase = pos.length / 3;
    for (let j = 0; j <= radial; j++) {
      const s = ringStart + j;
      pos.push(pos[s * 3], pos[s * 3 + 1], pos[s * 3 + 2]);
      nor.push(tan.x, tan.y, tan.z);
    }
    for (let j = 0; j < radial; j++) {
      const A = p(ci);
      const Bv = p(ringBase + j);
      const C = p(ringBase + j + 1);
      const n = Bv.clone().sub(A).cross(C.clone().sub(A));
      if (n.dot(tan) >= 0) idx.push(ci, ringBase + j, ringBase + j + 1);
      else idx.push(ci, ringBase + j + 1, ringBase + j);
    }
  }

  if (opts.capStart) cap(0, -1, 0);
  if (opts.capEnd) cap(segs * row, 1, 1);

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/* ---------- Vertex renkleri: doku beneklenmesi ---------- */
export function paintGeometry(geo, baseLinear, amp, scale, seed = 0) {
  const pos = geo.attributes.position;
  const n = pos.count;
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const f = fbm(x * scale + seed, y * scale, z * scale - seed);
    const m = 1 + (f - 0.5) * 2 * amp;
    col[i * 3] = baseLinear.r * m;
    col[i * 3 + 1] = baseLinear.g * (1 + (f - 0.5) * 2 * amp * 0.8);
    col[i * 3 + 2] = baseLinear.b * (1 + (f - 0.5) * 2 * amp * 0.7);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return col;
}

/* ---------- Epikardiyal yağ ---------- */
export function addEpicardialFat(geo, colors, polylines, fatLinear, reach, seed = 0) {
  const pos = geo.attributes.position;
  const n = pos.count;
  const r2max = reach * reach;
  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    let best = 1e9;
    for (let p = 0; p < polylines.length; p++) {
      const dx = x - polylines[p].x, dy = y - polylines[p].y, dz = z - polylines[p].z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) best = d2;
    }
    if (best > r2max) continue;
    const d = Math.sqrt(best);
    const nz = fbm(x * 0.35 + seed, y * 0.35, z * 0.35);
    const w = smoothstep(reach, reach * 0.25, d) * (0.45 + 0.7 * nz);
    const k = Math.min(0.85, w);
    colors[i * 3] += (fatLinear.r - colors[i * 3]) * k;
    colors[i * 3 + 1] += (fatLinear.g - colors[i * 3 + 1]) * k;
    colors[i * 3 + 2] += (fatLinear.b - colors[i * 3 + 2]) * k;
  }
  geo.attributes.color.needsUpdate = true;
}

/* ---------- Damar tanımları ---------- */
export function vesselSpecs() {
  const S = [];
  const sm = smoothstep;
  S.push({
    id: 'aorta',
    owner: 'lv',
    kind: 'arter',
    hex: 0xb3463c,
    pts: [
      V3(8.9, 497, 28), V3(9, 509, 28), V3(6, 525, 27), V3(2, 542, 24),
      V3(3, 555, 17), V3(10, 564, 5), V3(22, 564, -8), V3(34, 556, -22),
      V3(42, 538, -36), V3(46, 515, -50), V3(47, 490, -58),
    ],
    r: (t) => 12.2 + 1.4 * Math.exp(-Math.pow((t - 0.03) / 0.035, 2)) - 1.4 * sm(0.4, 0.85, t),
    capEnd: true,
  });

  S.push({
    id: 'brakiyosefalik',
    owner: 'lv',
    kind: 'arter',
    hex: 0xb3463c,
    pts: [V3(6, 556, 15), V3(5, 570, 15), V3(2, 590, 14), V3(-2, 608, 13)],
    r: (t) => 5.6 - 1.2 * t,
    capEnd: true,
  });

  S.push({
    id: 'sol-karotis',
    owner: 'lv',
    kind: 'arter',
    hex: 0xb3463c,
    pts: [V3(14, 559, 3), V3(15, 575, 1), V3(16, 594, 0), V3(16, 610, 0)],
    r: (t) => 3.7 - 0.6 * t,
    capEnd: true,
  });

  S.push({
    id: 'sol-subklavyen',
    owner: 'lv',
    kind: 'arter',
    hex: 0xb3463c,
    pts: [V3(22, 558, -2), V3(28, 574, -5), V3(38, 590, -8), V3(48, 602, -10)],
    r: (t) => 4.8 - 0.8 * t,
    capEnd: true,
  });

  S.push({
    id: 'pulmoner-govde',
    owner: 'rv',
    kind: 'pa',
    hex: 0x7b86b4,
    pts: [V3(18, 509, 44), V3(19, 523, 41), V3(20, 537, 36), V3(22, 549, 29)],
    r: (t) => 12.8 - 1.6 * t,
  });

  S.push({
    id: 'sag-pulmoner-arter',
    owner: 'rv',
    kind: 'pa',
    hex: 0x7b86b4,
    pts: [V3(22, 549, 29), V3(17, 548, 14), V3(8, 547, 0), V3(-6, 547, -4), V3(-20, 546, -2), V3(-32, 544, 2)],
    r: (t) => 9 - 1.8 * t,
    capEnd: true,
  });

  S.push({
    id: 'sol-pulmoner-arter',
    owner: 'rv',
    kind: 'pa',
    hex: 0x7b86b4,
    pts: [V3(22, 549, 29), V3(32, 550, 22), V3(44, 547, 14), V3(55, 541, 6)],
    r: (t) => 8.6 - 1.6 * t,
    capEnd: true,
  });

  S.push({
    id: 'vena-kava-superior',
    owner: 'ra',
    kind: 'ven',
    hex: 0x66719f,
    pts: [V3(-22, 503, 22), V3(-22, 520, 21), V3(-23, 545, 19), V3(-25, 578, 15)],
    r: (t) => 9.0 - 0.6 * t,
    capEnd: true,
  });

  S.push({
    id: 'vena-kava-inferior',
    owner: 'ra',
    kind: 'ven',
    hex: 0x66719f,
    pts: [V3(-22, 451, 12), V3(-22, 447, 0), V3(-22, 438, -14), V3(-22, 426, -26), V3(-22, 414, -34)],
    r: (t) => 10.5 - 0.5 * t,
    capEnd: true,
  });

  const pvR = (t) => 6.6 - 0.8 * t;
  S.push({
    id: 'sol-pulmoner-ven', owner: 'la', kind: 'pv', hex: 0xa6525b,
    pts: [V3(28, 494, -4), V3(36, 494, -14), V3(44, 494, -23)], r: pvR,
  });
  S.push({
    id: 'sol-ust-pv', owner: 'la', kind: 'pv', hex: 0xa6525b,
    pts: [V3(44, 494, -23), V3(52, 502, -30), V3(60, 508, -35)], r: (t) => 4.8 - 0.6 * t, capEnd: true,
  });
  S.push({
    id: 'sol-alt-pv', owner: 'la', kind: 'pv', hex: 0xa6525b,
    pts: [V3(44, 494, -23), V3(52, 486, -30), V3(60, 480, -35)], r: (t) => 4.8 - 0.6 * t, capEnd: true,
  });
  S.push({
    id: 'sag-pulmoner-ven', owner: 'la', kind: 'pv', hex: 0xa6525b,
    pts: [V3(-18, 493, -5), V3(-27, 493, -13), V3(-36, 493, -22)], r: pvR,
  });
  S.push({
    id: 'sag-ust-pv', owner: 'la', kind: 'pv', hex: 0xa6525b,
    pts: [V3(-36, 493, -22), V3(-45, 501, -28), V3(-53, 507, -32)], r: (t) => 4.8 - 0.6 * t, capEnd: true,
  });
  S.push({
    id: 'sag-alt-pv', owner: 'la', kind: 'pv', hex: 0xa6525b,
    pts: [V3(-36, 493, -22), V3(-45, 485, -28), V3(-53, 479, -32)], r: (t) => 4.8 - 0.6 * t, capEnd: true,
  });

  return S;
}

export const LAD_XY = [
  [33, 501.5], [37.5, 500.1], [40.2, 494.7], [42, 488.2], [44.3, 483.4],
  [43.7, 477.3], [45.4, 471], [45.8, 464.7], [48.1, 459.2], [51.7, 449.1],
  [56.2, 439.3], [59.5, 428.5],
];

export const RCA_XY = [
  [-3.9, 492.9], [-10.3, 484.7], [-17.8, 477.4], [-22.7, 465.7],
  [-24.3, 457], [-24.2, 449.4], [-22.5, 445.1], [-18, 437.4], [-7.9, 433.8],
];

export function projectToSurface(x, y, meshes, offset) {
  const rc = new THREE.Raycaster(V3(x, y, 400), V3(0, 0, -1), 0, 1000);
  const hits = rc.intersectObjects(meshes, false);
  if (!hits.length) return null;
  const h = hits[0];
  const n = h.face.normal.clone().normalize();
  return h.point.clone().add(n.multiplyScalar(offset));
}

export function coronaryPaths(meshes) {
  const proj = (xy, a, b) =>
    xy.slice(a, b).map(([x, y]) => projectToSurface(x, y, meshes, 1.0)).filter(Boolean);
  return {
    lad: [V3(28, 506, 36)].concat(proj(LAD_XY, 2, LAD_XY.length)),
    rca: [V3(1, 496, 38)].concat(proj(RCA_XY, 1, 8)),
  };
}

export function densify(points, step) {
  const out = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    const L = a.distanceTo(b), k = Math.max(1, Math.ceil(L / step));
    for (let j = 0; j < k; j++) out.push(a.clone().lerp(b, j / k));
  }
  if (points.length > 0) out.push(points[points.length - 1].clone());
  return out;
}

/* ---------- Sol ventrikül duvar bölgeleri ---------- */
export function computeLVRegions(lvGeo, septumGeo, mitralC, aorticC, sigmaDeg = 38) {
  const pos = lvGeo.attributes.position;
  const n = pos.count;
  const c = V3(0, 0, 0);
  for (let i = 0; i < n; i++) c.add(V3(pos.getX(i), pos.getY(i), pos.getZ(i)));
  c.divideScalar(n);

  const base = mitralC.clone().add(aorticC).multiplyScalar(0.5);
  let apex = null, bestD = -1;
  for (let i = 0; i < n; i++) {
    const v = V3(pos.getX(i), pos.getY(i), pos.getZ(i));
    const d = v.distanceToSquared(base);
    if (d > bestD) { bestD = d; apex = v; }
  }
  const a = apex.clone().sub(base).normalize();

  const sp = septumGeo.attributes.position, sC = V3(0, 0, 0);
  for (let i = 0; i < sp.count; i++) sC.add(V3(sp.getX(i), sp.getY(i), sp.getZ(i)));
  sC.divideScalar(sp.count);

  const perp = (v) => v.clone().sub(a.clone().multiplyScalar(v.dot(a)));
  const s = perp(sC.clone().sub(c)).normalize();
  const e2 = a.clone().cross(s).normalize();
  const theta = (v) => Math.atan2(v.dot(e2), v.dot(s));

  const sgn = theta(perp(V3(0, 0, 1))) < 0 ? -1 : 1;
  const D = Math.PI / 180;
  const ang = {
    septal: 0,
    anterior: sgn * 65 * D,
    lateral: sgn * 155 * D,
    inferior: -sgn * 100 * D,
  };
  const sigma = sigmaDeg * Math.PI / 180;
  const wrap = (d) => {
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return d;
  };
  const W = {
    septal: new Float32Array(n),
    anterior: new Float32Array(n),
    lateral: new Float32Array(n),
    inferior: new Float32Array(n),
  };
  for (let i = 0; i < n; i++) {
    const r = perp(V3(pos.getX(i), pos.getY(i), pos.getZ(i)).sub(c));
    const th = theta(r);
    for (const k in W) {
      const d = wrap(th - ang[k]) / sigma;
      const g = Math.exp(-0.5 * d * d);
      W[k][i] = smoothstep(0.18, 0.9, g);
    }
  }
  return { weights: W, angles: ang, axis: a, apex, base, centroid: c, septalCentroid: sC };
}

/* ---------- Anatomik Bilgiler & Sözlük (Tıkla-Açıkla) ---------- */
export const REGION_INFO = {
  anterior: {
    ad: 'Ön (anterior) duvar',
    ekg: 'V3–V4',
    art: 'LAD (sol ön inen arter)',
    desc: 'Sol ventrikülün ön yüzeyi; en geniş kas kütlesine sahiptir.',
  },
  inferior: {
    ad: 'Alt (inferior) duvar',
    ekg: 'II, III, aVF',
    art: 'RCA / posterior inen dal',
    desc: 'Diyaframa oturan alt yüzey; sağ koroner arter beslemesi hakimdir.',
  },
  lateral: {
    ad: 'Yan (lateral) duvar',
    ekg: 'I, aVL, V5–V6',
    art: 'LCx (sirkumfleks arter)',
    desc: 'Sol serbest duvar; yüksek lateral (I, aVL) ve apikal lateral (V5–V6) derivasyonlar yansıtır.',
  },
  septal: {
    ad: 'Septal duvar',
    ekg: 'V1–V2',
    art: 'LAD septal dalları',
    desc: 'İki ventrikülü ayıran interventriküler septum; ileti demetlerini (His-Purkinje) barındırır.',
  },
};

export const INFO = {
  lv: {
    title: 'Sol Ventrikül (LV)',
    category: 'Kas Duvarı',
    desc: 'Oksijenli kanı aort yoluyla tüm vücuda pompalayan en kalın duvarlı oda. Miyokard enfarktüsünün en sık etkilediği ana yapıdır.',
  },
  rv: {
    title: 'Sağ Ventrikül (RV)',
    category: 'Kas Duvarı',
    desc: 'Oksijeni azalmış kanı pulmoner arter yoluyla akciğerlere pompalar. Ön yüzde yer aldığı için V1–V2 derivasyonlarına yakındır.',
  },
  septum: {
    title: 'İnterventriküler Septum',
    category: 'Kas Duvarı',
    desc: 'Sol ve sağ ventrikülü ayıran septum. V1–V2 derivasyonları septal aktiviteyi yansıtır; LAD septal dallarınca beslenir.',
  },
  ra: {
    title: 'Sağ Atriyum (RA)',
    category: 'Odacık',
    desc: 'Üst ve alt vena kavadan gelen venöz kanı toplar; triküspit kapak ile sağ ventriküle iletir. Sinoatriyal düğüm (SA) burada yer alır.',
  },
  la: {
    title: 'Sol Atriyum (LA)',
    category: 'Odacık',
    desc: 'Akciğerlerden dört pulmoner ven ile gelen oksijenli kanı toplar; mitral kapak ile sol ventriküle boşaltır.',
  },
  'valve-mitral': {
    title: 'Mitral Kapak',
    category: 'Kapak',
    desc: 'Sol atriyum ile sol ventrikül arasındaki biküspit (iki yapraklı) atriyoventriküler kapak.',
  },
  'valve-tricuspid': {
    title: 'Triküspit Kapak',
    category: 'Kapak',
    desc: 'Sağ atriyum ile sağ ventrikül arasındaki üç yapraklı kapak.',
  },
  'valve-aortic': {
    title: 'Aort Kapağı',
    category: 'Kapak',
    desc: 'Sol ventrikülden aorta çıkışı denetler; sinüslerinden sağ ve sol koroner arterler köken alır.',
  },
  'valve-pulmonary': {
    title: 'Pulmoner Kapak',
    category: 'Kapak',
    desc: 'Sağ ventrikülden pulmoner artere kan çıkışını kontrol eden yarımay kapak.',
  },
  papillary: {
    title: 'Papiller Kaslar',
    category: 'İç Yapı',
    desc: 'Korda tendinealar aracılığıyla kapak yaprakçıklarını tutarak sistol esnasında regürjitasyonu (geri kaçışı) önler.',
  },
  lad: {
    title: 'LAD – Sol Ön İnen Arter',
    category: 'Koroner Arter',
    desc: 'Ön interventriküler olukta seyreder; ön duvar, septum ve apeksi besler. Tıkanması ön duvar MI (V3–V4) oluşturur.',
  },
  rca: {
    title: 'RCA – Sağ Koroner Arter',
    category: 'Koroner Arter',
    desc: 'Sağ atriyoventriküler olukta ilerler; sağ ventrikül ve alt duvarı (inferior MI: II, III, aVF) besler.',
  },
};

export const VINFO = {
  aorta: {
    title: 'Aort',
    category: 'Büyük Damar',
    desc: 'Sol ventrikülden çıkan ana atardamar; çıkan aort, arkus aorta ve inen aort bölümlerinden oluşur.',
  },
  brakiyosefalik: {
    title: 'Brakiyosefalik Gövde',
    category: 'Arkus Dalı',
    desc: 'Arkus aortanın ilk ve en büyük dalı; sağ kol ve başın sağ yarısına kan sağlar.',
  },
  'sol-karotis': {
    title: 'Sol Ortak Karotis',
    category: 'Arkus Dalı',
    desc: 'Arkus aortadan çıkarak boynun sol tarafı ve beyne kan götürür.',
  },
  'sol-subklavyen': {
    title: 'Sol Subklavyen Arter',
    category: 'Arkus Dalı',
    desc: 'Sol kola ve beyin tabanına (vertebral arter dalıyla) kan taşır.',
  },
  'pulmoner-govde': {
    title: 'Pulmoner Gövde',
    category: 'Büyük Damar',
    desc: 'Sağ ventrikülden oksijensiz kanı alıp sağ ve sol pulmoner arterlere dallanır.',
  },
  'sag-pulmoner-arter': {
    title: 'Sağ Pulmoner Arter',
    category: 'Büyük Damar',
    desc: 'Oksijeni azalmış kanı sağ akciğere iletir.',
  },
  'sol-pulmoner-arter': {
    title: 'Sol Pulmoner Arter',
    category: 'Büyük Damar',
    desc: 'Oksijeni azalmış kanı sol akciğere iletir.',
  },
  'vena-kava-superior': {
    title: 'Üst Vena Kava (SVC)',
    category: 'Büyük Damar',
    desc: 'Baş, boyun, kollar ve göğüsten gelen oksijensiz kanı sağ atriyuma boşaltır.',
  },
  'vena-kava-inferior': {
    title: 'Alt Vena Kava (IVC)',
    category: 'Büyük Damar',
    desc: 'Gövdenin alt yarısı ve bacaklardan gelen venöz kanı sağ atriyuma getirir.',
  },
};

export function getVesselInfo(id) {
  if (VINFO[id]) return VINFO[id];
  if (id.includes('pulmoner-ven') || /-pv$/.test(id)) {
    return {
      title: 'Pulmoner Venler',
      category: 'Büyük Damar',
      desc: 'Akciğerlerde oksijenlenen temiz kanı sol atriyuma taşıyan 4 ana ven.',
    };
  }
  return { title: id, category: 'Damar', desc: 'Kardiyovasküler sistem damarı.' };
}

export const CAMERA_VIEWS = {
  anterior: [0, 0.12, 1],
  posterior: [0, 0.12, -1],
  'left-lateral': [1, 0.1, 0.12],
  'right-lateral': [-1, 0.1, 0.12],
  superior: [0, 1, 0.1],
  inferior: [0, -1, 0.1],
};
