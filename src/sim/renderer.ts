import * as THREE from "three";
import type { Store } from "../types";
import { rng } from "./store";
export class StoreRenderer {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  store: Store;
  constructor(
    canvas: HTMLCanvasElement,
    store: Store,
    width: number,
    height: number,
  ) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setSize(width, height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.camera = new THREE.PerspectiveCamera(54, width / height, 0.05, 100);
    this.store = store;
    this.setStore(store);
  }
  setStore(store: Store) {
    this.scene.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
          m.map?.dispose();
          m.dispose();
        }
      }
    });
    this.scene.clear();
    this.store = store;
    this.scene.background = new THREE.Color("#d9dce0");
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x737780, 2));
    const light = new THREE.DirectionalLight(0xffffff, 2);
    light.position.set(2, 6, -3);
    this.scene.add(light);
    const r = rng(store.seed);
    for (const s of store.shelves) {
      const g = new THREE.Group();
      g.position.set(s.x, 0, s.z);
      g.rotation.y = -s.angle;
      const box = (
        w: number,
        h: number,
        d: number,
        x: number,
        y: number,
        z: number,
        color: THREE.ColorRepresentation,
      ) => {
        const m = new THREE.Mesh(
          new THREE.BoxGeometry(w, h, d),
          new THREE.MeshLambertMaterial({ color }),
        );
        m.position.set(x, y, z);
        g.add(m);
      };
      box(s.width, s.height, 0.06, 0, s.height / 2, 0, 0x737a81);
      for (let l = 0; l < 4; l++) {
        box(s.width, 0.06, s.depth, 0, 0.18 + l * 0.45, 0, 0xb9bfc3);
        for (let side of [-1, 1])
          for (let c = 0; c < (s.productColumns ?? 16); c++) {
            const n = s.productColumns ?? 16;
            box(
              (s.width / n) * 0.8,
              0.23 + r() * 0.12,
              0.12 + r() * 0.12,
              -s.width / 2 + ((c + 0.5) * s.width) / n,
              0.34 + l * 0.45,
              side * s.depth * 0.28,
              new THREE.Color().setHSL((c % 4) / 4, 0.5, 0.5),
            );
          }
      }
      this.scene.add(g);
    }
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(store.width, store.depth),
      new THREE.MeshLambertMaterial({ color: 0xbbb9b3 }),
    );
    floor.rotation.x = -Math.PI / 2;
    this.scene.add(floor);
  }
  renderPose(
    p: { x: number; y?: number; z: number },
    look: { x: number; y?: number; z: number },
  ) {
    this.camera.position.set(p.x, p.y ?? 1, p.z);
    this.camera.lookAt(look.x, look.y ?? 1, look.z);
    this.camera.updateMatrixWorld(true);
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this.scene.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        for (const m of Array.isArray(o.material) ? o.material : [o.material])
          m.dispose();
      }
    });
    this.renderer.dispose();
  }
}
