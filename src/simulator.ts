import * as THREE from "three";
import { StoreRenderer } from "../../src/sim/renderer";
import { rng } from "../../src/sim/store";
import type { Shelf } from "../../src/types";
import { createSweepPlan, type SweepOptions, type SweepPlan } from "./sweep";

export type Sweep = SweepPlan & { frames: string[] };

// Distinct label/barcode texture creates appearance features, never coordinates or shelf IDs.
function addLabels(view: StoreRenderer, shelf: Shelf) {
  const r = rng(view.store.seed + 23817);
  const group = new THREE.Group();
  group.position.set(shelf.x, 0, shelf.z);
  group.rotation.y = -shelf.angle;
  const columns = shelf.productColumns ?? 8;
  for (let side of [-1, 1])
    for (let layer = 0; layer < 4; layer++)
      for (let col = 0; col < columns; col++) {
        const canvas = document.createElement("canvas");
        canvas.width = 192;
        canvas.height = 128;
        const ctx = canvas.getContext("2d")!;
        ctx.fillStyle = `hsl(${r() * 360} 50% 88%)`;
        ctx.fillRect(0, 0, 192, 128);
        ctx.fillStyle = `hsl(${r() * 360} 70% 28%)`;
        ctx.fillRect(8, 8, 176, 20);
        ctx.font = "bold 23px sans-serif";
        ctx.fillStyle = "#26353b";
        ctx.fillText(
          ["TEA", "MILK", "JUICE", "COFFEE", "OATS"][Math.floor(r() * 5)],
          10,
          58,
        );
        // Decorative random label marks: sampled once per product, shared across every camera frame.
        for (let j = 0; j < 30; j++) {
          ctx.fillStyle = r() > 0.5 ? "#273340" : "#ffffff";
          ctx.fillRect(10 + j * 5, 74, 2 + r() * 3, 25 + r() * 20);
        }
        ctx.fillStyle = "#273340";
        ctx.font = "16px monospace";
        ctx.fillText(`${Math.floor(1000 + r() * 9000)}`, 12, 125);
        const texture = new THREE.CanvasTexture(canvas);
        texture.colorSpace = THREE.SRGBColorSpace;
        const material = new THREE.MeshLambertMaterial({
          map: texture,
          side: THREE.DoubleSide,
        });
        const label = new THREE.Mesh(
          new THREE.PlaneGeometry((shelf.width / (columns * 1.25)) * 0.9, 0.15),
          material,
        );
        if (side < 0) label.rotation.y = Math.PI;
        label.position.set(
          -shelf.width / 2 + ((col + 0.5) * shelf.width) / columns,
          0.32 + (layer * (shelf.height - 0.3)) / 4,
          side * shelf.depth * 0.451,
        );
        group.add(label);
      }
  view.scene.add(group);
  return () => {
    group.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        const m = o.material as THREE.MeshLambertMaterial;
        m.map?.dispose();
        m.dispose();
      }
    });
    view.scene.remove(group);
  };
}

export function makeSweep(
  view: StoreRenderer,
  seed: number,
  options: number | SweepOptions = 9,
): Sweep {
  const plan = createSweepPlan(seed, options);
  view.setStore(plan.store);
  const cleanup = addLabels(view, plan.shelf);
  view.camera.fov = 54;
  view.camera.updateProjectionMatrix();
  try {
    const frames = plan.poses.map((p) => {
      view.renderPose(p, { x: p.lookX, y: p.lookY, z: p.lookZ });
      return view.renderer.domElement.toDataURL("image/jpeg", 0.95);
    });
    return { ...plan, frames };
  } finally {
    cleanup();
  }
}
