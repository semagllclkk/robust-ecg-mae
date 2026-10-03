import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createHeartModel, disposeHeartModel } from './heartModel.js';

const CAMERA_POSITIONS = {
  anterior: [0, 0, 18],
  posterior: [0, 0, -18],
  'left-lateral': [18, 0, 0],
  superior: [0, 18, 2],
};

const HIGHLIGHT_STYLES = {
  anterior: { ventricles: [0xffaa00, 0.6], lad: [0xffff00, 0.8] },
  inferior: { ventricles: [0xff3300, 0.7] },
  lateral: {
    leftAtrium: [0x00ff88, 0.6],
    ventricles: [0x00aa55, 0.5],
  },
  septal: { rightAtrium: [0x3388ff, 0.7] },
};

const REGION_PARTS = {
  anterior: ['ventricles', 'lad'],
  inferior: ['ventricles'],
  lateral: ['ventricles', 'leftAtrium'],
  septal: ['ventricles', 'rightAtrium'],
};

export default function HeartScene({
  affectedRegions,
  bpm,
  cameraView,
  highlightedRegion,
}) {
  const containerRef = useRef(null);
  const sceneStateRef = useRef({});
  const propsRef = useRef({});

  propsRef.current = { affectedRegions, bpm, highlightedRegion };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e14);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(10, 15, 10);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x58a6ff, 0.5);
    fillLight.position.set(-10, -10, -10);
    scene.add(fillLight);

    const { group: heart, parts } = createHeartModel();
    scene.add(heart);

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

    const initialPosition = CAMERA_POSITIONS.anterior;
    camera.position.set(...initialPosition);
    controls.target.set(0, 0, 0);
    controls.update();

    let animationFrame = 0;
    let stopped = false;
    let elapsed = 0;
    let lastFrame = performance.now();

    function animate(now) {
      if (stopped) return;
      animationFrame = window.requestAnimationFrame(animate);

      elapsed += (now - lastFrame) / 1000;
      lastFrame = now;
      const { bpm: currentBpm, affectedRegions: affected, highlightedRegion: selected } =
        propsRef.current;
      const beatCycle = ((elapsed * currentBpm) / 60) % 1;
      let contraction = 0;
      if (beatCycle > 0.35 && beatCycle < 0.55) {
        contraction = Math.sin(((beatCycle - 0.35) / 0.2) * Math.PI);
        parts.ventricles.rotation.y = contraction * 0.05;
      } else {
        parts.ventricles.rotation.y = 0;
      }
      const scale = 1 - contraction * 0.08;
      heart.scale.set(scale, scale * 1.02, scale);

      const brokenParts = new Set(
        affected.flatMap((region) => REGION_PARTS[region] ?? []),
      );
      const selectedStyles = HIGHLIGHT_STYLES[selected] ?? {};
      const errorIntensity = 0.45 + 0.35 * ((Math.sin(elapsed * 7) + 1) / 2);

      for (const [partName, mesh] of Object.entries(parts)) {
        const [color, intensity] =
          selectedStyles[partName] ??
          (brokenParts.has(partName) ? [0xff4d6d, errorIntensity] : [0x000000, 0]);
        mesh.material.emissive.setHex(color);
        mesh.material.emissiveIntensity = intensity;
      }

      controls.update();
      renderer.render(scene, camera);
    }

    sceneStateRef.current = { camera, controls };
    animate();

    return () => {
      stopped = true;
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      disposeHeartModel(heart);
      renderer.dispose();
      renderer.domElement.remove();
      sceneStateRef.current = {};
    };
  }, []);

  useEffect(() => {
    const cameraPosition = CAMERA_POSITIONS[cameraView];
    const { camera, controls } = sceneStateRef.current;
    if (!cameraPosition || !camera || !controls) return;
    camera.position.set(...cameraPosition);
    controls.target.set(0, 0, 0);
    controls.update();
  }, [cameraView]);

  return (
    <div
      aria-label="Fareyle döndürülebilen 3D anatomik kalp modeli"
      className="absolute inset-0"
      ref={containerRef}
      role="img"
    />
  );
}
