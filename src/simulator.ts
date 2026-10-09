import * as THREE from "three";
import { StoreRenderer } from "./sim/renderer";
import { rng } from "./sim/store";
import type { Shelf, SKU } from "./types";
import { createSweepPlan, type SweepOptions, type SweepPlan } from "./sweep";
import { shelfProducts, createMerchandising } from "./sim/structure";
import { renderOrthographicGT, type OrthographicGT } from "./ground-truth";
import { productLabel } from "./sim/products";
import { captureCameraPose, cameraPoseGroundTruth, type CameraPoseGT, type CameraGroundTruth } from "./sim/camera-pose";

export type Sweep = SweepPlan & { frames: string[]; groundTruth: OrthographicGT; cameraGroundTruth: CameraGroundTruth };

function labelMaterial(sku: SKU) {
  const random = rng(sku.labelSeed);
  const canvas = document.createElement("canvas");
  canvas.width = 192; canvas.height = 192;
  const ctx = canvas.getContext("2d")!;
  const hue = sku.hue * 360;
  ctx.fillStyle = `hsl(${hue} 45% 91%)`;
  ctx.fillRect(0, 0, 192, 192);
  ctx.fillStyle = `hsl(${hue} 60% 32%)`;
  ctx.fillRect(8, 8, 176, 36);
  ctx.font = "bold 21px sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(["FIELD", "DAILY", "PURE"][Math.floor(random() * 3)], 20, 34);
  ctx.font = "bold 26px sans-serif";
  ctx.fillStyle = "#26353b";
  ctx.fillText(sku.name, 10, 82);
  ctx.font = "14px sans-serif";
  ctx.fillText(["ORIGINAL", "CLASSIC", "LIGHT", "NATURAL"][Math.floor(random() * 4)], 10, 108);
  for (let j = 0; j < 30; j++) {
    ctx.fillStyle = random() > 0.5 ? "#273340" : "#ffffff";
    ctx.fillRect(10 + j * 5, 125, 2 + random() * 3, 30 + random() * 15);
  }
  ctx.fillStyle = "#273340";
  ctx.font = "15px monospace";
  ctx.fillText(`${Math.floor(10000000 + random() * 90000000)}`, 12, 183);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshLambertMaterial({ map: texture, side: THREE.DoubleSide });
}

function addLabels(view: StoreRenderer, shelf: Shelf) {
  const group = new THREE.Group();
  group.position.set(shelf.x, 0, shelf.z);
  group.rotation.y = -shelf.angle;
  const materials = new Map<string, THREE.MeshLambertMaterial>();
  const catalog = (shelf.merchandising ?? createMerchandising(shelf, view.store.seed)).catalog;
  for (const p of shelfProducts(shelf, view.store.seed)) {
    let material = materials.get(p.skuId);
    if (!material) {
      material = labelMaterial(catalog.find(sku => sku.id === p.skuId)!);
      materials.set(p.skuId, material);
    }
    const label = productLabel(catalog.find(sku => sku.id === p.skuId)!, material, p.side);
    label.position.add(new THREE.Vector3(p.x, p.y, p.z));
    group.add(label);
  }
  view.scene.add(group);
  return () => {
    group.traverse(o => { if (o instanceof THREE.Mesh) o.geometry.dispose(); });
    for (const material of materials.values()) { material.map?.dispose(); material.dispose(); }
    view.scene.remove(group);
  };
}

export function makeSweep(view: StoreRenderer, seed: number, options: number | SweepOptions = 9): Sweep {
  const plan = createSweepPlan(seed, options);
  view.setStore(plan.store);
  const cleanup = addLabels(view, plan.shelf);
  view.camera.fov = 54;
  view.camera.updateProjectionMatrix();
  try {
    const cameraPoses: CameraPoseGT[] = [];
    const frames = plan.poses.map((p, i) => {
      view.renderPose(p, { x: p.lookX, y: p.lookY, z: p.lookZ }, p.rollDegrees);
      cameraPoses.push(captureCameraPose(view.camera, i, p.timestamp));
      return view.renderer.domElement.toDataURL("image/jpeg", 0.95);
    });
    return { ...plan, frames, cameraGroundTruth: cameraPoseGroundTruth(cameraPoses), groundTruth: renderOrthographicGT(view, plan) };
  } finally { cleanup(); }
}
