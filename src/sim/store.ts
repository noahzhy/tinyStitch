import type { Store } from "../types";
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function generateStore(
  seed: number,
  _count = 8,
  _extra = 0,
  _flag = false,
): Store {
  const r = rng(seed);
  return {
    seed,
    width: 18,
    depth: 28,
    shelves: [
      {
        id: "shelf-0",
        x: 0,
        z: 0,
        width: 3 + r() * 3,
        height: 2,
        depth: 0.65,
        angle: 0,
        productColumns: 16,
      },
    ],
    route: [],
  };
}
export function free(store: Store, x: number, z: number, pad: number) {
  return (
    Math.abs(x) < store.width / 2 - pad &&
    Math.abs(z) < store.depth / 2 - pad &&
    store.shelves.every((s) => {
      const c = Math.cos(s.angle),
        t = Math.sin(s.angle),
        dx = x - s.x,
        dz = z - s.z;
      return (
        Math.abs(dx * c + dz * t) > s.width / 2 + pad ||
        Math.abs(-dx * t + dz * c) > s.depth / 2 + pad
      );
    })
  );
}
