import * as THREE from 'three';

function createVessel(points, radius, color) {
  const curve = new THREE.CatmullRomCurve3(points);
  const geometry = new THREE.TubeGeometry(curve, 20, radius, 8, false);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.3,
  });
  return new THREE.Mesh(geometry, material);
}

export function createHeartModel() {
  const group = new THREE.Group();
  const parts = {};

  const myocardiumMaterial = new THREE.MeshStandardMaterial({
    color: 0x8a151b,
    roughness: 0.35,
    metalness: 0.15,
    emissive: 0x220002,
    emissiveIntensity: 0.2,
  });
  const ventriclesGeometry = new THREE.SphereGeometry(3.2, 32, 32);
  const vertices = ventriclesGeometry.attributes.position;

  for (let index = 0; index < vertices.count; index += 1) {
    let x = vertices.getX(index);
    let y = vertices.getY(index);
    let z = vertices.getZ(index);

    if (y < 0) {
      const taper = 1 + (y / 3.5) * 0.65;
      x *= taper;
      z *= taper;
      x -= Math.sin(y * 0.3) * 0.5;
    } else {
      z *= 1.1;
    }
    vertices.setXYZ(index, x, y, z);
  }
  ventriclesGeometry.computeVertexNormals();

  parts.ventricles = new THREE.Mesh(ventriclesGeometry, myocardiumMaterial);
  parts.ventricles.position.set(0, -0.6, 0);
  group.add(parts.ventricles);

  const atriumMaterial = new THREE.MeshStandardMaterial({
    color: 0x6e1017,
    roughness: 0.5,
    metalness: 0.1,
  });
  parts.leftAtrium = new THREE.Mesh(
    new THREE.SphereGeometry(1.6, 24, 24),
    atriumMaterial,
  );
  parts.leftAtrium.position.set(1.4, 2.2, -0.8);
  parts.leftAtrium.scale.set(1.1, 0.9, 1.2);
  group.add(parts.leftAtrium);

  parts.rightAtrium = new THREE.Mesh(
    new THREE.SphereGeometry(1.5, 24, 24),
    atriumMaterial.clone(),
  );
  parts.rightAtrium.position.set(-1.6, 2, 0.2);
  parts.rightAtrium.scale.set(1.2, 0.8, 1);
  group.add(parts.rightAtrium);

  const aortaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.2, 1.8, 0.5),
    new THREE.Vector3(0.1, 3.8, 0.3),
    new THREE.Vector3(0.8, 4.4, -0.4),
    new THREE.Vector3(1.6, 3.6, -1.2),
    new THREE.Vector3(1.4, 1, -1.6),
  ]);
  group.add(
    new THREE.Mesh(
      new THREE.TubeGeometry(aortaCurve, 32, 0.7, 16, false),
      new THREE.MeshStandardMaterial({
        color: 0x9b1c1c,
        roughness: 0.3,
        metalness: 0.2,
      }),
    ),
  );

  const pulmonaryCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.4, 1.5, 1.2),
    new THREE.Vector3(-0.8, 3, 0.8),
    new THREE.Vector3(-1.8, 3.4, -0.2),
  ]);
  const pulmonaryMaterial = new THREE.MeshStandardMaterial({
    color: 0x224488,
    roughness: 0.35,
    metalness: 0.1,
  });
  group.add(
    new THREE.Mesh(
      new THREE.TubeGeometry(pulmonaryCurve, 24, 0.6, 16, false),
      pulmonaryMaterial,
    ),
  );

  const venaCavaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-2.2, 4, -0.6),
    new THREE.Vector3(-2, 2.2, -0.4),
    new THREE.Vector3(-1.8, 0.2, -0.2),
  ]);
  group.add(
    new THREE.Mesh(
      new THREE.TubeGeometry(venaCavaCurve, 20, 0.55, 16, false),
      pulmonaryMaterial.clone(),
    ),
  );

  parts.lad = createVessel(
    [
      new THREE.Vector3(0, 1.8, 2.4),
      new THREE.Vector3(0.2, 0.5, 2.7),
      new THREE.Vector3(-0.1, -1.2, 2.2),
      new THREE.Vector3(-0.4, -2.6, 1.2),
    ],
    0.1,
    0xd12424,
  );
  group.add(parts.lad);

  group.rotation.set(0.15, -0.2, 0);
  return { group, parts };
}

export function disposeHeartModel(group) {
  group.traverse((object) => {
    if (!object.isMesh) return;
    object.geometry.dispose();
    if (Array.isArray(object.material)) {
      object.material.forEach((material) => material.dispose());
    } else {
      object.material.dispose();
    }
  });
}
