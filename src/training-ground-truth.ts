import * as THREE from "three";
import type { SweepPlan } from "./sweep";
import type { StoreRenderer } from "./sim/renderer";
import { setCameraPose } from "./sim/camera-pose.ts";

// Optional training-only supervision. RGB, the camera and the intersected meshes
// are the same objects used by the renderer; package bounds are never used as GT.
export function trainingGroundTruth(view: StoreRenderer, plan: SweepPlan, cols = 24, rows = 32) {
  view.scene.updateMatrixWorld(true);
  const cameras = plan.poses.map(p => {
    const camera = new THREE.PerspectiveCamera(54, 720 / 960, 0.05, 100);
    setCameraPose(camera, p, { x: p.lookX, y: p.lookY, z: p.lookZ }, p.rollDegrees);
    return camera;
  });
  const ray = new THREE.Raycaster();
  const meshes: THREE.Mesh[] = [];
  view.scene.traverse(o => { if (o instanceof THREE.Mesh) meshes.push(o); });
  function identity(object: THREE.Object3D) {
    for (let o: THREE.Object3D | null = object; o; o = o.parent)
      if (o.userData.instanceId) return o.userData;
    return { shelfId: "", instanceId: "", kind: "background", skuId: "" };
  }
  const samples = cameras.map(camera => Array.from({ length: cols * rows }, (_, i) => {
    const pixel = [(i % cols + 0.5) * 720 / cols - 0.5, (Math.floor(i / cols) + 0.5) * 960 / rows - 0.5];
    ray.setFromCamera(new THREE.Vector2((pixel[0] + 0.5) / 720 * 2 - 1, 1 - (pixel[1] + 0.5) / 960 * 2), camera);
    const hit = ray.intersectObjects(meshes, false)[0];
    const id = hit ? identity(hit.object) : identity(view.scene);
    return { pixel, point: hit && id.shelfId === plan.shelf.id ? hit.point.clone() : null, id };
  }));
  const pairs = cameras.flatMap((camera, source) => cameras.flatMap((_, target) => source === target ? [] : [{
    source, target, points: samples[target].map(sample => {
      const base = { target: sample.pixel, source: [0, 0], status: "background", instanceId: sample.id.instanceId,
        skuId: sample.id.skuId, kind: sample.id.kind, world: sample.point?.toArray() ?? null };
      if (!sample.point) return base;
      const ndc = sample.point.clone().project(camera);
      base.source = [(ndc.x + 1) * 720 / 2 - 0.5, (1 - ndc.y) * 960 / 2 - 0.5];
      if (ndc.z < -1 || ndc.z > 1 || Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1) return { ...base, status: "outside" };
      const delta = sample.point.clone().sub(camera.position);
      const distance = delta.length();
      ray.set(camera.position, delta.normalize());
      ray.far = distance + 0.005;
      const first = ray.intersectObjects(meshes, false)[0];
      ray.far = Infinity;
      return { ...base, status: first && Math.abs(first.distance - distance) <= 0.004 &&
        identity(first.object).instanceId === sample.id.instanceId ? "visible" : "occluded" };
    }),
  }]));
  return { schemaVersion: 2, kind: "rendered_3d", imageSize: [720, 960], queryGrid: [cols, rows],
    pixelCoordinates: "Zero-based pixel centers. Actual mesh intersections including curved labels. Background ignored.",
    visibilityToleranceMeters: 0.004, pairs };
}
export type TrainingGroundTruth = ReturnType<typeof trainingGroundTruth>;
