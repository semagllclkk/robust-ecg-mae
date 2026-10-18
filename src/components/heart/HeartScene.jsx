import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import {
  CAMERA_VIEWS,
  INFO,
  REGION_INFO,
  addEpicardialFat,
  buildTubeGeometry,
  computeLVRegions,
  coronaryPaths,
  densify,
  getVesselInfo,
  lcxPath,
  paintGeometry,
  smoothstep,
  srgb,
  vesselSpecs,
} from './heartLogic.js';

// Rapor 2.5 duvarları → sol ventrikül bölge ağırlıkları (Anterior/Septal: V1–V4)
const WALL_TO_LV = {
  anterior: ['anterior', 'septal'],
  inferior: ['inferior'],
  lateral: ['lateral'],
};
const LV_TO_WALL = { anterior: 'anterior', septal: 'anterior', inferior: 'inferior', lateral: 'lateral' };
const CORONARY_WALL = { lad: 'anterior', rca: 'inferior', lcx: 'lateral' };

const expandWalls = (walls) => walls.flatMap((wall) => WALL_TO_LV[wall] ?? []);

// Modelde UV olmadığından doku kabartması nesne uzayında prosedürel gürültüyle üretilir
const TISSUE_BUMP_GLSL = `
varying vec3 vTissuePos;
float tissueHash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float tissueNoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(tissueHash(i), tissueHash(i + vec3(1, 0, 0)), f.x),
        mix(tissueHash(i + vec3(0, 1, 0)), tissueHash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(tissueHash(i + vec3(0, 0, 1)), tissueHash(i + vec3(1, 0, 1)), f.x),
        mix(tissueHash(i + vec3(0, 1, 1)), tissueHash(i + vec3(1, 1, 1)), f.x), f.y),
    f.z);
}
vec3 tissuePerturb(vec3 surfPos, vec3 surfNorm, vec2 dHdxy, float faceDir) {
  vec3 sigmaX = dFdx(surfPos);
  vec3 sigmaY = dFdy(surfPos);
  vec3 r1 = cross(sigmaY, surfNorm);
  vec3 r2 = cross(surfNorm, sigmaX);
  float det = dot(sigmaX, r1) * faceDir;
  vec3 grad = sign(det) * (dHdxy.x * r1 + dHdxy.y * r2);
  return normalize(abs(det) * surfNorm - grad);
}
`;

function addTissueBump(material, strength) {
  if (!strength) return material;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.tissueBump = { value: strength };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vTissuePos;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvTissuePos = position;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform float tissueBump;\n${TISSUE_BUMP_GLSL}`)
      .replace(
        '#include <normal_fragment_maps>',
        `#include <normal_fragment_maps>
        float tissueH = tissueNoise(vTissuePos * 0.32) * 0.6 + tissueNoise(vTissuePos * 0.9) * 0.4;
        normal = tissuePerturb(-vViewPosition, normal, vec2(dFdx(tissueH), dFdy(tissueH)) * tissueBump, faceDirection);`,
      );
  };
  material.customProgramCacheKey = () => `tissue-bump-${strength}`;
  return material;
}

// Nemli epikard: düşük parlaklıkta clearcoat (sheen katmanı tümleşik GPU'da pahalı olduğundan yok)
function tissueMaterial(opts = {}, bump = 0.012) {
  return addTissueBump(new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.58,
    metalness: 0.0,
    clearcoat: 0.32,
    clearcoatRoughness: 0.42,
    envMapIntensity: 0.85,
    ...opts,
  }), bump);
}

function contactShadow(box) {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(40,48,64,0.42)');
  g.addColorStop(0.55, 'rgba(40,48,64,0.14)');
  g.addColorStop(1, 'rgba(40,48,64,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const width = (box.max.x - box.min.x) * 1.3;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(width, width * 0.6),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set((box.min.x + box.max.x) / 2, box.min.y - 0.4, (box.min.z + box.max.z) / 2);
  mesh.renderOrder = -1;
  return mesh;
}

function geoCentroid(geo) {
  const p = geo.attributes.position;
  const c = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    c.x += p.getX(i);
    c.y += p.getY(i);
    c.z += p.getZ(i);
  }
  return c.divideScalar(p.count);
}

function geoBoxCenter(geo) {
  geo.computeBoundingBox();
  return geo.boundingBox.getCenter(new THREE.Vector3());
}

export default function HeartScene({
  bpm = 72,
  beatActive = true,
  cameraView = 'anterior',
  highlightedRegion = null,
  highlightStrength = 1,
  affectedRegions = [],
  layers = { vessels: true, coronary: true, atria: true, ventricles: true, inner: true },
  wallOpacity = 1.0,
  explode = 0,
  onSelectInfo = null,
  onLoaded = null,
}) {
  const containerRef = useRef(null);
  const internalRef = useRef({
    flyTo: null,
    ready: false,
    layersMap: null,
    wallParts: null,
    explodeGroups: null,
  });

  const propsRef = useRef({});
  propsRef.current = {
    bpm,
    beatActive,
    cameraView,
    highlightedRegion,
    highlightStrength,
    affectedRegions,
    layers,
    wallOpacity,
    explode,
    onSelectInfo,
    onLoaded,
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let stopped = false;
    let animationFrameId = null;

    // Sahne, kamera, renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight || 1, 1, 400);
    camera.position.set(0, 3.5, 30);
    scene.add(camera);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.0;
    controls.minDistance = 8;
    controls.maxDistance = 80;
    const target = new THREE.Vector3(0, 0.4, 0);
    controls.target.copy(target);

    // Aydınlatma
    camera.add(new THREE.HemisphereLight(0xf2f6ff, 0x6b3a32, 0.85));
    const keyLight = new THREE.DirectionalLight(0xfff4ea, 2.1);
    keyLight.position.set(8, 14, 12);
    camera.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x9fb8ff, 0.6);
    fillLight.position.set(-12, -4, 6);
    camera.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffd9c8, 0.95);
    rimLight.position.set(-4, 6, -14);
    camera.add(rimLight);

    // Stüdyo Çevresi (PMREM)
    const envScene = new THREE.Scene();
    envScene.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(50, 32, 16),
        new THREE.MeshBasicMaterial({ color: 0x8e99a8, side: THREE.BackSide })
      )
    );
    const createPanel = (w, h, hex, k, p) => {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(hex).multiplyScalar(k), side: THREE.DoubleSide })
      );
      m.position.set(p[0], p[1], p[2]);
      m.lookAt(0, 0, 0);
      envScene.add(m);
    };
    createPanel(34, 22, 0xfff2e6, 7, [22, 32, 26]);
    createPanel(26, 26, 0x9fb8ff, 2.6, [-36, 10, 12]);
    createPanel(44, 10, 0xffffff, 3.2, [0, -32, -18]);
    createPanel(22, 22, 0xffd6c0, 2.2, [0, 16, -42]);
    const pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromScene(envScene, 0.04).texture;
    pm.dispose();

    // Kalp Kök Grubu
    const root = new THREE.Group();
    scene.add(root);

    const parts = {};
    const beatItems = [];
    const pickables = [];
    const sceneLayers = { vessels: [], coronary: [], atria: [], ventricles: [], inner: [] };
    const explodeGroups = [];
    let regionW = null;
    let baseCol = {};
    let coronaryMeshes = {};

    const hl = { walls: [], regions: [], level: 0, target: 0, strength: 1 };
    const heatOut = [0, 0, 0];

    function setHeat(w, pulse, out) {
      const a = smoothstep(0.0, 0.55, w);
      const k = smoothstep(0.25, 1.0, w);
      const g = 0.85 + 0.3 * pulse;
      // Rapor 2.5: hasarlı segment kırmızı gradyanla (koyu → parlak kırmızı) vurgulanır
      out[0] = (0.5 + (0.95 - 0.5) * k) * g;
      out[1] = (0.035 + (0.012 - 0.035) * k) * g;
      out[2] = (0.035 + (0.018 - 0.035) * k) * g;
      return a;
    }

    function blend(arr, target, alpha) {
      arr[0] += (target[0] - arr[0]) * alpha;
      arr[1] += (target[1] - arr[1]) * alpha;
      arr[2] += (target[2] - arr[2]) * alpha;
    }

    const px = [0, 0, 0];
    const blink = [0, 0, 0];

    // Vertex renkleri: taban → kopuk kanal yanıp sönmesi → model hasar vurgusu (her zaman üstte)
    function paintMesh(mesh, base, manualW, affectedW, pulse, blinkAlpha) {
      const col = mesh.geometry.attributes.color.array;
      const n = mesh.geometry.attributes.position.count;
      for (let i = 0; i < n; i++) {
        const i3 = i * 3;
        px[0] = base[i3];
        px[1] = base[i3 + 1];
        px[2] = base[i3 + 2];
        const wA = affectedW ? affectedW(i) : 0;
        if (wA >= 0.01) blend(px, blink, smoothstep(0.01, 0.85, wA) * blinkAlpha);
        const wM = manualW ? manualW(i) : 0;
        if (wM >= 0.004) {
          const aM = setHeat(Math.min(1, wM), pulse, heatOut) * Math.min(1, wM * 1.15);
          blend(px, heatOut, aM);
        }
        col[i3] = px[0];
        col[i3 + 1] = px[1];
        col[i3 + 2] = px[2];
      }
      mesh.geometry.attributes.color.needsUpdate = true;
    }

    let painted = false;

    function recolorAll(pulse, now) {
      if (!regionW || !parts.lv || !parts.septum) return;

      const currentProps = propsRef.current;
      const affectedWalls = currentProps.affectedRegions || [];
      const affectedLV = expandWalls(affectedWalls);
      const manualLV = hl.regions;
      const level = hl.level * hl.strength;
      const hasManual = manualLV.length > 0 && level > 0.001;

      // Vurgu yoksa taban renkler bir kez geri yüklenir; sonraki karelerde GPU'ya veri gönderilmez
      if (!hasManual && !affectedWalls.length) {
        if (!painted) return;
        painted = false;
      } else {
        painted = true;
      }

      // Pembe yanıp sönme (~1.4 Hz): sinyali eksik (kopuk derivasyonlu) duvar
      const blinkT = (Math.sin(now * 0.009) + 1) / 2;
      blink[0] = 1.3 + (1.7 - 1.3) * blinkT;
      blink[1] = 0.45 + (1.05 - 0.45) * blinkT;
      blink[2] = 0.75 + (1.25 - 0.75) * blinkT;
      const blinkAlpha = 0.3 + 0.3 * blinkT;

      const maxOver = (list, field) => (i) => {
        let w = 0;
        for (let j = 0; j < list.length; j++) {
          const arr = field[list[j]];
          if (arr && arr[i] > w) w = arr[i];
        }
        return w;
      };

      const manualLVWeight = maxOver(manualLV, regionW.lv);
      paintMesh(
        parts.lv,
        baseCol.lv,
        hasManual ? (i) => manualLVWeight(i) * level : null,
        affectedLV.length ? maxOver(affectedLV, regionW.lv) : null,
        pulse,
        blinkAlpha,
      );

      const sepMan = hasManual && manualLV.includes('septal');
      const sepAff = affectedLV.includes('septal');
      paintMesh(
        parts.septum,
        baseCol.septum,
        sepMan ? (i) => regionW.septumAll[i] * level : null,
        sepAff ? () => 0.85 : null,
        pulse,
        blinkAlpha,
      );

      // Koroner arter parıltısı: duvarı besleyen arter vurgulanır
      Object.entries(coronaryMeshes).forEach(([key, mesh]) => {
        const wall = CORONARY_WALL[key];
        const emissive = mesh.material.emissive;
        if (affectedWalls.includes(wall)) {
          const k = 0.7 * blinkAlpha;
          emissive.setRGB(blink[0] * k, blink[1] * k, blink[2] * k);
        } else if (hasManual && hl.walls.includes(wall)) {
          emissive.setRGB(0.9 * level, 0.12 * level, 0.04 * level);
        } else {
          emissive.setRGB(0, 0, 0);
        }
      });
    }

    // Kamera slerp / tween
    let tween = null;
    function flyTo(dir, dist) {
      const d0 = camera.position.clone().sub(controls.target);
      const d1 = dir.clone().normalize();
      tween = {
        t: 0,
        dur: 750,
        dist0: d0.length(),
        dist1: dist || d0.length(),
        dir0: d0.clone().normalize(),
        dir1: d1,
        q: new THREE.Quaternion().setFromUnitVectors(d0.clone().normalize(), d1),
      };
      controls.enabled = false;
    }
    internalRef.current.flyTo = flyTo;

    // GLB Yükleme
    const loader = new GLTFLoader();
    loader.load(
      '/heart.glb',
      (gltf) => {
        if (stopped) return;
        try {
          const nodes = {};
          gltf.scene.updateMatrixWorld(true);
          gltf.scene.traverse((o) => {
            if (o.isMesh) nodes[o.name.replace(/^VH_M_/, '')] = o;
          });

          const need = [
            'heart_left_ventricle',
            'heart_right_ventricle',
            'interventricular_septum',
            'right_cardiac_atrium',
            'left_cardiac_atrium',
            'mitral_valve',
            'tricuspid_valve',
            'aortic_valve',
            'pulmonary_valve',
          ];
          for (const n of need) {
            if (!nodes[n]) throw new Error('Modelde beklenen yapı yok: ' + n);
          }

          Object.values(nodes).forEach((m) => {
            if (m.parent) m.parent.remove(m);
            m.geometry.scale(1000, 1000, 1000);
            m.geometry.computeBoundingBox();
          });

          const lv = nodes.heart_left_ventricle;
          const rv = nodes.heart_right_ventricle;
          const sep = nodes.interventricular_septum;
          const ra = nodes.right_cardiac_atrium;
          const la = nodes.left_cardiac_atrium;
          const valves = [
            nodes.mitral_valve,
            nodes.tricuspid_valve,
            nodes.aortic_valve,
            nodes.pulmonary_valve,
          ];
          const papillary = Object.keys(nodes)
            .filter((n) => n.startsWith('papillary_muscle_of_heart_'))
            .map((n) => nodes[n]);

          Object.assign(parts, { lv, rv, septum: sep, ra, la });
          lv.name = 'lv';
          rv.name = 'rv';
          sep.name = 'septum';
          ra.name = 'ra';
          la.name = 'la';
          nodes.mitral_valve.name = 'valve-mitral';
          nodes.tricuspid_valve.name = 'valve-tricuspid';
          nodes.aortic_valve.name = 'valve-aortic';
          nodes.pulmonary_valve.name = 'valve-pulmonary';
          papillary.forEach((m) => {
            m.name = 'papillary';
          });

          const baseColors = { lv: 0x9e352d, rv: 0xa63d33, septum: 0x8f3029, ra: 0x96404a, la: 0x8f3d46 };
          const colors = {};
          const mats = {
            lv: tissueMaterial(),
            rv: tissueMaterial({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
            septum: tissueMaterial(),
            ra: tissueMaterial({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
            la: tissueMaterial(),
          };

          ['lv', 'rv', 'septum', 'ra', 'la'].forEach((k, i) => {
            colors[k] = paintGeometry(parts[k].geometry, srgb(baseColors[k]), 0.16, 0.085, i * 17.3);
            parts[k].material = mats[k];
          });

          valves.forEach((m, i) => {
            paintGeometry(m.geometry, srgb(0xdccdb0), 0.07, 0.15, 40 + i);
            m.material = tissueMaterial({ roughness: 0.55, clearcoat: 0 }, 0);
          });

          papillary.forEach((m, i) => {
            paintGeometry(m.geometry, srgb(0x9c4b41), 0.1, 0.15, 60 + i);
            m.material = tissueMaterial({ roughness: 0.5, clearcoat: 0 }, 0);
          });

          const mitralC = geoCentroid(nodes.mitral_valve.geometry);
          const aorticC = geoCentroid(nodes.aortic_valve.geometry);
          const allChambers = new THREE.Box3();
          [lv, rv, sep, ra, la].forEach((m) => allChambers.union(m.geometry.boundingBox));
          const heartCenter = allChambers.getCenter(new THREE.Vector3());
          const cVent = geoBoxCenter(lv.geometry).add(geoBoxCenter(rv.geometry)).multiplyScalar(0.5);
          const cRA = geoBoxCenter(ra.geometry);
          const cLA = geoBoxCenter(la.geometry);

          function mkGroup(name, anchor, k) {
            const g = new THREE.Group();
            g.name = name;
            root.add(g);
            let dir = anchor.clone().sub(heartCenter);
            if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
            explodeGroups.push({ group: g, dir: dir.normalize(), k });
            return g;
          }

          const gLV = mkGroup('g-lv', geoBoxCenter(lv.geometry), 1.0);
          const gRV = mkGroup('g-rv', geoBoxCenter(rv.geometry), 1.0);
          const gSep = mkGroup('g-sep', geoBoxCenter(sep.geometry), 0.55);
          const gRA = mkGroup('g-ra', cRA, 1.0);
          const gLA = mkGroup('g-la', cLA, 1.0);
          const gIn = mkGroup('g-inner', mitralC, 0.3);

          gLV.add(lv);
          gRV.add(rv);
          gSep.add(sep);
          gRA.add(ra);
          gLA.add(la);
          valves.concat(papillary).forEach((m) => gIn.add(m));

          sceneLayers.ventricles.push(lv, rv, sep);
          sceneLayers.atria.push(ra, la);
          sceneLayers.inner.push(...valves, ...papillary);

          [lv, rv, sep].forEach((m) => beatItems.push({ obj: m, kind: 'vent', c: cVent }));
          beatItems.push({ obj: ra, kind: 'atria', c: cRA });
          beatItems.push({ obj: la, kind: 'atria', c: cLA });
          valves.concat(papillary).forEach((m) => beatItems.push({ obj: m, kind: 'vent', c: cVent }));
          pickables.push(lv, rv, sep, ra, la, ...valves, ...papillary);

          // Koroner damarlar
          root.updateMatrixWorld(true);
          const wallMeshes = [lv, rv, ra, la, sep];
          wallMeshes.forEach((m) => {
            m.material.side = THREE.DoubleSide;
          });
          const cor = coronaryPaths(wallMeshes);
          // LV Bölgeleri (17 segment / EKG türetimi)
          const reg = computeLVRegions(lv.geometry, sep.geometry, mitralC, aorticC, 38);
          cor.lcx = lcxPath(lv, reg, cor.lad[0]);
          wallMeshes.forEach((m) => {
            m.material.side = THREE.FrontSide;
          });

          const coronaryMat = () =>
            new THREE.MeshStandardMaterial({
              color: srgb(0xd23a2f),
              roughness: 0.32,
              emissive: new THREE.Color(0, 0, 0),
              envMapIntensity: 1.1,
            });

          if (cor.lad.length > 3) {
            const g = buildTubeGeometry(cor.lad, (t) => 2.15 - 0.9 * t, { segs: 120, radial: 14, capEnd: true });
            const m = new THREE.Mesh(g, coronaryMat());
            m.name = 'lad';
            coronaryMeshes.lad = m;
            gLV.add(m);
            sceneLayers.coronary.push(m);
            pickables.push(m);
            beatItems.push({ obj: m, kind: 'vent', c: cVent });
          }

          if (cor.rca.length > 3) {
            const g = buildTubeGeometry(cor.rca, (t) => 2.05 - 0.6 * t, { segs: 100, radial: 14, capEnd: true });
            const m = new THREE.Mesh(g, coronaryMat());
            m.name = 'rca';
            coronaryMeshes.rca = m;
            gRV.add(m);
            sceneLayers.coronary.push(m);
            pickables.push(m);
            beatItems.push({ obj: m, kind: 'vent', c: cVent });
          }

          if (cor.lcx.length > 3) {
            const g = buildTubeGeometry(cor.lcx, (t) => 1.9 - 0.7 * t, { segs: 110, radial: 14, capEnd: true });
            const m = new THREE.Mesh(g, coronaryMat());
            m.name = 'lcx';
            coronaryMeshes.lcx = m;
            gLV.add(m);
            sceneLayers.coronary.push(m);
            pickables.push(m);
            beatItems.push({ obj: m, kind: 'vent', c: cVent });
          }

          // Epikardiyal Yağ (koroner oluklar boyunca)
          const fatLine = [];
          ['lad', 'rca', 'lcx'].forEach((k) => {
            if (cor[k] && cor[k].length > 1) fatLine.push(...densify(cor[k], 2.0));
          });
          ['lv', 'rv', 'ra', 'la'].forEach((k, i) => {
            addEpicardialFat(parts[k].geometry, colors[k], fatLine, srgb(0xe3c47e), 10.0, 11 + i * 5);
          });

          ['lv', 'septum'].forEach((k) => {
            baseCol[k] = new Float32Array(colors[k]);
          });

          // Büyük Damarlar
          vesselSpecs().forEach((sp) => {
            const g = buildTubeGeometry(sp.pts, sp.r, { segs: 72, radial: 18, capEnd: !!sp.capEnd });
            paintGeometry(g, srgb(sp.hex), 0.07, 0.11, sp.id.length * 3.1);
            const m = new THREE.Mesh(
              g,
              new THREE.MeshStandardMaterial({
                color: 0xffffff,
                vertexColors: true,
                roughness: 0.42,
                envMapIntensity: 0.9,
              })
            );
            m.name = 'vessel:' + sp.id;
            m.userData.vessel = sp;
            const owner = { lv: gLV, rv: gRV, ra: gRA, la: gLA }[sp.owner];
            if (owner) owner.add(m);
            sceneLayers.vessels.push(m);
            pickables.push(m);
            beatItems.push({
              obj: m,
              kind: sp.owner === 'ra' || sp.owner === 'la' ? 'atria-t' : 'vent-t',
              c: sp.owner === 'ra' ? cRA : sp.owner === 'la' ? cLA : cVent,
              root: sp.pts[0].clone(),
            });
          });

          regionW = {
            lv: reg.weights,
            septumAll: new Float32Array(sep.geometry.attributes.position.count).fill(0.95),
          };

          // Boyutlandırma ve merkezleme
          root.scale.setScalar(0.1);
          root.position.copy(heartCenter).multiplyScalar(-0.1);
          root.updateMatrixWorld(true);

          // Yumuşak temas gölgesi (açık arka planda modeli zemine oturtur)
          const heartBox = new THREE.Box3().setFromObject(root);
          scene.add(contactShadow(heartBox));

          internalRef.current.ready = true;
          internalRef.current.layersMap = sceneLayers;
          internalRef.current.wallParts = ['lv', 'rv', 'ra', 'la'].map((k) => parts[k]);
          internalRef.current.explodeGroups = explodeGroups;

          if (propsRef.current.onLoaded) {
            propsRef.current.onLoaded();
          }
        } catch (err) {
          console.error('Anatomik kalp modeli işleme hatası:', err);
        }
      },
      undefined,
      (err) => {
        console.error('GLB yüklenemedi:', err);
      }
    );

    // Tıklama tespiti (Raycasting)
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let pointerDownInfo = null;

    const onPointerDown = (e) => {
      pointerDownInfo = { x: e.clientX, y: e.clientY, t: performance.now() };
    };

    const onPointerUp = (e) => {
      if (!pointerDownInfo) return;
      const moved = Math.hypot(e.clientX - pointerDownInfo.x, e.clientY - pointerDownInfo.y);
      const dt = performance.now() - pointerDownInfo.t;
      pointerDownInfo = null;
      if (moved > 6 || dt > 600) return;

      const rect = renderer.domElement.getBoundingClientRect();
      ndc.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      ray.setFromCamera(ndc, camera);

      const activePickables = pickables.filter((o) => o.visible && (!o.material || o.material.opacity > 0.25));
      const hits = ray.intersectObjects(activePickables, false);
      if (!hits.length) {
        if (propsRef.current.onSelectInfo) propsRef.current.onSelectInfo(null);
        return;
      }

      const hit = hits[0];
      const o = hit.object;
      const nm = o.name;

      if (nm.startsWith('vessel:')) {
        const vId = nm.slice(7);
        const info = getVesselInfo(vId);
        if (propsRef.current.onSelectInfo) {
          propsRef.current.onSelectInfo(info);
        }
        return;
      }

      if (nm === 'lv' && regionW) {
        const f = hit.face;
        const W = regionW.lv;
        let best = 'anterior';
        let maxVal = -1;
        Object.keys(W).forEach((k) => {
          const val = W[k][f.a] + W[k][f.b] + W[k][f.c];
          if (val > maxVal) {
            maxVal = val;
            best = k;
          }
        });
        const r = REGION_INFO[best] || REGION_INFO.anterior;
        if (propsRef.current.onSelectInfo) {
          propsRef.current.onSelectInfo({
            title: `${INFO.lv.title} · ${r.ad}`,
            category: INFO.lv.category,
            desc: r.desc,
            leads: r.ekg,
            artery: r.art,
            wall: LV_TO_WALL[best],
          });
        }
        return;
      }

      const foundInfo = INFO[nm];
      if (foundInfo && propsRef.current.onSelectInfo) {
        propsRef.current.onSelectInfo(foundInfo);
      }
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);

    // Pencere boyutu güncelleme
    function resize() {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    // Animasyon Döngüsü
    const clock = new THREE.Clock();
    let phase = 0;
    const bump = (p, a, b, pw = 1) =>
      p > a && p < b ? Math.pow(Math.sin((Math.PI * (p - a)) / (b - a)), pw) : 0;
    let curOpacity = 1;
    let lastAppliedOpacity = 1;

    // Sahne ekranda değilse çizim döngüsü durdurulur
    let onScreen = true;
    const visibility = new IntersectionObserver(([entry]) => {
      onScreen = entry.isIntersecting;
      if (onScreen && !animationFrameId && !stopped) {
        clock.getDelta();
        animate();
      }
    });
    visibility.observe(container);

    function animate() {
      if (stopped) return;
      if (!onScreen) {
        animationFrameId = null;
        return;
      }
      animationFrameId = requestAnimationFrame(animate);

      const dt = Math.min(clock.getDelta(), 0.1);
      const currentProps = propsRef.current;
      const bpmVal = currentProps.bpm || 72;
      const isBeat = currentProps.beatActive;

      phase = (phase + (dt * bpmVal) / 60.0) % 1.0;

      // Atım
      const sysA = isBeat ? bump(phase, 0.22, 0.36, 1.3) : 0;
      const sysV = isBeat ? bump(phase, 0.4, 0.64, 1.6) : 0;
      const sV = 1 - 0.065 * sysV;
      const sA = 1 - 0.05 * sysA;

      for (let i = 0; i < beatItems.length; i++) {
        const it = beatItems[i];
        const o = it.obj;
        if (it.kind === 'vent') {
          o.scale.setScalar(sV);
          o.position.copy(it.c).multiplyScalar(1 - sV);
        } else if (it.kind === 'atria') {
          o.scale.setScalar(sA);
          o.position.copy(it.c).multiplyScalar(1 - sA);
        } else if (it.kind === 'vent-t') {
          o.position.copy(it.c).sub(it.root).multiplyScalar(1 - sV);
        } else if (it.kind === 'atria-t') {
          o.position.copy(it.c).sub(it.root).multiplyScalar(1 - sA);
        }
      }

      // Parçalama (Exploded view)
      const ex = (currentProps.explode || 0) * 0.55;
      for (let i = 0; i < explodeGroups.length; i++) {
        const g = explodeGroups[i];
        g.group.position.copy(g.dir).multiplyScalar(ex * g.k);
      }

      // Katman Görünürlükleri
      if (currentProps.layers) {
        Object.keys(currentProps.layers).forEach((key) => {
          const list = sceneLayers[key];
          if (list) {
            const visible = currentProps.layers[key];
            for (let j = 0; j < list.length; j++) {
              list[j].visible = visible;
            }
          }
        });
      }

      // XAI vurgusu (model tahmini veya kullanıcı seçimi) ve kopuk kanal yanıp sönmesi
      const activeHighlight = currentProps.highlightedRegion;
      if (activeHighlight) {
        // Söndürme sırasında son bölge korunur ki vurgu yumuşakça kaybolsun
        hl.walls = [activeHighlight];
        hl.regions = expandWalls(hl.walls);
      }
      hl.strength = currentProps.highlightStrength ?? 1;
      hl.target = activeHighlight ? 1 : 0;
      hl.level += (hl.target - hl.level) * Math.min(1, dt * 6);
      if (Math.abs(hl.target - hl.level) < 0.003) hl.level = hl.target;

      const now = performance.now();
      const pulseVal = sysV * 0.8 + 0.2 * Math.sin(now / 380);
      recolorAll(pulseVal, now);

      const wantOpacity = currentProps.wallOpacity ?? 1.0;
      curOpacity += (wantOpacity - curOpacity) * Math.min(1, dt * 6);

      if (Math.abs(curOpacity - lastAppliedOpacity) > 0.002) {
        const wallPartsList = ['lv', 'rv', 'ra', 'la'];
        wallPartsList.forEach((k) => {
          if (parts[k] && parts[k].material) {
            const m = parts[k].material;
            const tr = curOpacity < 0.995;
            if (m.transparent !== tr) {
              m.transparent = tr;
              m.depthWrite = !tr;
              m.side = tr ? THREE.DoubleSide : THREE.FrontSide;
              m.needsUpdate = true;
            }
            m.opacity = curOpacity;
            parts[k].renderOrder = tr ? 2 : 0;
          }
        });
        lastAppliedOpacity = curOpacity;
      }

      // Kamera yumuşak süzülme (Tween)
      if (tween) {
        tween.t += dt * 1000;
        const k0 = Math.min(1, tween.t / tween.dur);
        const k = k0 < 0.5 ? 2 * k0 * k0 : 1 - Math.pow(-2 * k0 + 2, 2) / 2;
        const qi = new THREE.Quaternion().slerp(tween.q, k);
        const d = tween.dir0.clone().applyQuaternion(qi).multiplyScalar(tween.dist0 + (tween.dist1 - tween.dist0) * k);
        camera.position.copy(controls.target).add(d);
        camera.lookAt(controls.target);
        if (k0 >= 1) {
          tween = null;
          controls.enabled = true;
        }
      }

      controls.update();
      renderer.render(scene, camera);
    }

    animate();

    return () => {
      stopped = true;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      observer.disconnect();
      visibility.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      controls.dispose();
      // GPU belleğini boşalt (aksi hâlde her yeniden bağlanmada geometri/doku birikir)
      scene.traverse((object) => {
        object.geometry?.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => {
          if (!material) return;
          Object.values(material).forEach((value) => value?.isTexture && value.dispose());
          material.dispose();
        });
      });
      scene.environment?.dispose();
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Kamera açısı değiştiğinde flyTo
  useEffect(() => {
    const coords = CAMERA_VIEWS[cameraView];
    if (coords && internalRef.current.flyTo) {
      internalRef.current.flyTo(new THREE.Vector3(coords[0], coords[1], coords[2]), 30);
    }
  }, [cameraView]);

  return (
    <div
      aria-label="3D Anatomik Kalp Modeli"
      className="relative h-full w-full select-none"
      ref={containerRef}
      role="img"
    />
  );
}
