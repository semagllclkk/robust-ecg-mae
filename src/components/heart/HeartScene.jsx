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
  paintGeometry,
  smoothstep,
  srgb,
  vesselSpecs,
} from './heartLogic.js';

function tissueMaterial(opts = {}) {
  return new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.42,
    metalness: 0.0,
    clearcoat: 0.45,
    clearcoatRoughness: 0.3,
    envMapIntensity: 1.0,
    ...opts,
  });
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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
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
    camera.add(new THREE.HemisphereLight(0xbfd4ff, 0x3a1612, 0.5));
    const keyLight = new THREE.DirectionalLight(0xfff1e2, 1.8);
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
        new THREE.MeshBasicMaterial({ color: 0x151a24, side: THREE.BackSide })
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

    const hl = { region: null, level: 0, target: 0 };
    const heatOut = [0, 0, 0];

    function setHeat(w, pulse, out) {
      const a = smoothstep(0.0, 0.55, w);
      const k = smoothstep(0.25, 1.0, w);
      const g = 0.85 + 0.3 * pulse;
      out[0] = (1.45 + (1.3 - 1.45) * k) * g;
      out[1] = (1.0 + (0.07 - 1.0) * k) * g;
      out[2] = (0.2 + (0.05 - 0.2) * k) * g;
      return a;
    }

    function recolorAll(pulse, now) {
      if (!regionW || !parts.lv || !parts.septum || !parts.rv) return;

      const currentProps = propsRef.current;
      const affected = currentProps.affectedRegions || [];
      const manualRegion = hl.region;
      const manualLevel = hl.level;

      const hasAffected = affected.length > 0;
      const hasManual = manualRegion && manualLevel > 0.001;

      // Pembe-beyaz yanıp sönme hesabı (~1.8 Hz ritmik yanıp sönme)
      const blinkT = (Math.sin(now * 0.009) + 1) / 2; // 0 (pembe) ile 1 (beyaz)
      const blinkR = 1.35 + (1.90 - 1.35) * blinkT;
      const blinkG = 0.35 + (1.90 - 0.35) * blinkT;
      const blinkB = 0.65 + (1.90 - 0.65) * blinkT;
      const blinkAlpha = 0.65 + 0.35 * Math.sin(now * 0.009);

      // LV Vertex Renklendirme
      const lvCol = parts.lv.geometry.attributes.color.array;
      const lvBase = baseCol.lv;
      const nLV = parts.lv.geometry.attributes.position.count;

      for (let i = 0; i < nLV; i++) {
        const i3 = i * 3;
        let r = lvBase[i3];
        let g = lvBase[i3 + 1];
        let b = lvBase[i3 + 2];

        // 1) Manuel XAI vurgusu (sarı-kırmızı ısı haritası)
        if (hasManual && regionW.lv[manualRegion]) {
          const wM = regionW.lv[manualRegion][i] * manualLevel;
          if (wM >= 0.004) {
            const aM = setHeat(Math.min(1, wM), pulse, heatOut) * Math.min(1, wM * 1.15);
            r += (heatOut[0] - r) * aM;
            g += (heatOut[1] - g) * aM;
            b += (heatOut[2] - b) * aM;
          }
        }

        // 2) Kopan kanallara bağlı pembe-beyaz yanıp sönme
        if (hasAffected) {
          let maxAffW = 0;
          for (let j = 0; j < affected.length; j++) {
            const affReg = affected[j];
            if (regionW.lv[affReg]) {
              const w = regionW.lv[affReg][i];
              if (w > maxAffW) maxAffW = w;
            }
          }
          if (maxAffW >= 0.01) {
            const aAff = smoothstep(0.01, 0.85, maxAffW) * blinkAlpha;
            r += (blinkR - r) * aAff;
            g += (blinkG - g) * aAff;
            b += (blinkB - b) * aAff;
          }
        }

        lvCol[i3] = r;
        lvCol[i3 + 1] = g;
        lvCol[i3 + 2] = b;
      }
      parts.lv.geometry.attributes.color.needsUpdate = true;

      // Septum Vertex Renklendirme
      const sepCol = parts.septum.geometry.attributes.color.array;
      const sepBase = baseCol.septum;
      const nSep = parts.septum.geometry.attributes.position.count;
      const sepAff = affected.includes('septal');
      const sepMan = manualRegion === 'septal' && hasManual;

      for (let i = 0; i < nSep; i++) {
        const i3 = i * 3;
        let r = sepBase[i3];
        let g = sepBase[i3 + 1];
        let b = sepBase[i3 + 2];

        if (sepMan) {
          const wM = regionW.septumAll[i] * manualLevel;
          const aM = setHeat(Math.min(1, wM), pulse, heatOut) * Math.min(1, wM * 1.15);
          r += (heatOut[0] - r) * aM;
          g += (heatOut[1] - g) * aM;
          b += (heatOut[2] - b) * aM;
        }

        if (sepAff) {
          const aAff = 0.85 * blinkAlpha;
          r += (blinkR - r) * aAff;
          g += (blinkG - g) * aAff;
          b += (blinkB - b) * aAff;
        }

        sepCol[i3] = r;
        sepCol[i3 + 1] = g;
        sepCol[i3 + 2] = b;
      }
      parts.septum.geometry.attributes.color.needsUpdate = true;

      // RV Vertex Renklendirme
      const rvCol = parts.rv.geometry.attributes.color.array;
      const rvBase = baseCol.rv;
      const nRV = parts.rv.geometry.attributes.position.count;
      const rvAff = affected.includes('septal');
      const rvMan = manualRegion === 'septal' && hasManual;

      for (let i = 0; i < nRV; i++) {
        const i3 = i * 3;
        let r = rvBase[i3];
        let g = rvBase[i3 + 1];
        let b = rvBase[i3 + 2];

        if (rvMan) {
          const wM = regionW.rvSoft[i] * manualLevel;
          const aM = setHeat(Math.min(1, wM), pulse, heatOut) * Math.min(1, wM * 1.15);
          r += (heatOut[0] - r) * aM;
          g += (heatOut[1] - g) * aM;
          b += (heatOut[2] - b) * aM;
        }

        if (rvAff) {
          const aAff = regionW.rvSoft[i] * blinkAlpha;
          r += (blinkR - r) * aAff;
          g += (blinkG - g) * aAff;
          b += (blinkB - b) * aAff;
        }

        rvCol[i3] = r;
        rvCol[i3 + 1] = g;
        rvCol[i3 + 2] = b;
      }
      parts.rv.geometry.attributes.color.needsUpdate = true;

      // Koroner Damar Parıltısı (Emissive Glow)
      const isLadAff = affected.includes('anterior') || affected.includes('septal');
      const isLadMan = manualRegion === 'anterior' || manualRegion === 'septal';
      if (coronaryMeshes.lad && coronaryMeshes.lad.material) {
        if (isLadAff) {
          coronaryMeshes.lad.material.emissive.setRGB(blinkR * 0.7 * blinkAlpha, blinkG * 0.7 * blinkAlpha, blinkB * 0.7 * blinkAlpha);
        } else if (isLadMan && hasManual) {
          coronaryMeshes.lad.material.emissive.setRGB(0.9 * manualLevel, 0.28 * manualLevel, 0.04 * manualLevel);
        } else {
          coronaryMeshes.lad.material.emissive.setRGB(0, 0, 0);
        }
      }

      const isRcaAff = affected.includes('inferior');
      const isRcaMan = manualRegion === 'inferior';
      if (coronaryMeshes.rca && coronaryMeshes.rca.material) {
        if (isRcaAff) {
          coronaryMeshes.rca.material.emissive.setRGB(blinkR * 0.7 * blinkAlpha, blinkG * 0.7 * blinkAlpha, blinkB * 0.7 * blinkAlpha);
        } else if (isRcaMan && hasManual) {
          coronaryMeshes.rca.material.emissive.setRGB(0.9 * manualLevel, 0.28 * manualLevel, 0.04 * manualLevel);
        } else {
          coronaryMeshes.rca.material.emissive.setRGB(0, 0, 0);
        }
      }
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

          const baseColors = { lv: 0xa13a34, rv: 0xa9433b, septum: 0x8e302b, ra: 0x8d3b47, la: 0x853a45 };
          const colors = {};
          const mats = {
            lv: tissueMaterial(),
            rv: tissueMaterial({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
            septum: tissueMaterial(),
            ra: tissueMaterial({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
            la: tissueMaterial(),
          };

          ['lv', 'rv', 'septum', 'ra', 'la'].forEach((k, i) => {
            colors[k] = paintGeometry(parts[k].geometry, srgb(baseColors[k]), 0.11, 0.085, i * 17.3);
            parts[k].material = mats[k];
          });

          valves.forEach((m, i) => {
            paintGeometry(m.geometry, srgb(0xdccdb0), 0.07, 0.15, 40 + i);
            m.material = tissueMaterial({ roughness: 0.55, clearcoat: 0.15 });
          });

          papillary.forEach((m, i) => {
            paintGeometry(m.geometry, srgb(0x9c4b41), 0.1, 0.15, 60 + i);
            m.material = tissueMaterial({ roughness: 0.5 });
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
          wallMeshes.forEach((m) => {
            m.material.side = THREE.FrontSide;
          });

          const coronaryMat = () =>
            new THREE.MeshPhysicalMaterial({
              color: srgb(0x8f1c1c),
              roughness: 0.34,
              clearcoat: 0.7,
              clearcoatRoughness: 0.25,
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

          // Epikardiyal Yağ
          const fatLine = [];
          ['lad', 'rca'].forEach((k) => {
            if (cor[k] && cor[k].length > 1) fatLine.push(...densify(cor[k], 2.0));
          });
          ['lv', 'rv', 'ra', 'la'].forEach((k, i) => {
            addEpicardialFat(parts[k].geometry, colors[k], fatLine, srgb(0xd9b25f), 9.0, 11 + i * 5);
          });

          ['lv', 'rv', 'septum'].forEach((k) => {
            baseCol[k] = new Float32Array(colors[k]);
          });

          // Büyük Damarlar
          vesselSpecs().forEach((sp) => {
            const g = buildTubeGeometry(sp.pts, sp.r, { segs: 110, radial: 30, capEnd: !!sp.capEnd });
            paintGeometry(g, srgb(sp.hex), 0.07, 0.11, sp.id.length * 3.1);
            const m = new THREE.Mesh(
              g,
              new THREE.MeshPhysicalMaterial({
                color: 0xffffff,
                vertexColors: true,
                roughness: 0.5,
                clearcoat: 0.3,
                clearcoatRoughness: 0.45,
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

          // LV Bölgeleri (17 segment / EKG türetimi)
          const reg = computeLVRegions(lv.geometry, sep.geometry, mitralC, aorticC, 38);
          regionW = {
            lv: reg.weights,
            septumAll: new Float32Array(sep.geometry.attributes.position.count).fill(0.95),
            rvSoft: new Float32Array(rv.geometry.attributes.position.count).fill(0.5),
          };

          // Boyutlandırma ve merkezleme
          root.scale.setScalar(0.1);
          root.position.copy(heartCenter).multiplyScalar(-0.1);
          root.updateMatrixWorld(true);

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
          propsRef.current.onSelectInfo({
            ...info,
            extra: 'Görselleştirme amaçlı prosedürel büyük damar.',
          });
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
        const baseI = INFO.lv;
        if (propsRef.current.onSelectInfo) {
          propsRef.current.onSelectInfo({
            title: `${baseI.title} • ${r.ad}`,
            category: baseI.category,
            desc: baseI.desc,
            leads: r.ekg,
            artery: r.art,
            extra: `${r.ad} — İlişkili EKG: ${r.ekg} (${r.art})`,
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

    function animate() {
      if (stopped) return;
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

      // XAI ve Kopan Sinyal Vurgulama
      const activeHighlight = currentProps.highlightedRegion;
      const affectedList = currentProps.affectedRegions || [];
      hl.region = activeHighlight;
      hl.target = activeHighlight ? 1 : 0;
      hl.level += (hl.target - hl.level) * Math.min(1, dt * 6);
      if (Math.abs(hl.target - hl.level) < 0.003) hl.level = hl.target;

      const now = performance.now();
      const pulseVal = sysV * 0.8 + 0.2 * Math.sin(now / 380);
      recolorAll(pulseVal, now);

      // Saydamlık (Septal vurguda veya kopuklukta iç yapıyı görebilmek için otomatik şeffaflaşma)
      const userOpacity = currentProps.wallOpacity ?? 1.0;
      const isSeptalActive = activeHighlight === 'septal' || affectedList.includes('septal');
      const wantOpacity = Math.min(userOpacity, isSeptalActive ? 0.35 : 1.0);
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
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      controls.dispose();
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
