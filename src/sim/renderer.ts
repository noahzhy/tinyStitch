import * as THREE from "three";
import type { Store } from "../types";
import { shelfBays, shelfProducts, createMerchandising } from "./structure";
import { productTemplate } from "./products";
import { setCameraPose } from "./camera-pose";
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
      const bays = shelfBays(s);
      for (const bay of bays) {
        for (const y of bay.layerHeights)
          box(bay.width, 0.06, s.depth, bay.x, y, 0, 0xb9bfc3);
        for (const side of [-1, 1])
          box(0.035, s.height, s.depth, bay.x + side * bay.width / 2, s.height / 2, 0, 0x8c969d);
      }
      const catalog = (s.merchandising ?? createMerchandising(s, store.seed)).catalog;
      const templates = new Map<string, THREE.Group>();
      for (const p of shelfProducts(s, store.seed)) {
        let template = templates.get(p.skuId);
        if (!template) {
          template = productTemplate(catalog.find(sku => sku.id === p.skuId)!);
          templates.set(p.skuId, template);
        }
        const item = template.clone();
        item.position.set(p.x, p.y, p.z);
        g.add(item);
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
    rollDegrees = 0,
  ) {
    setCameraPose(this.camera, p, look, rollDegrees);
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this.scene.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
          m.map?.dispose();
          m.dispose();
        }
      }
    });
    this.renderer.dispose();
  }
}
