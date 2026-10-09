import { generateStore, free, rng } from "./sim/store.ts";
import type { Store, Shelf, StockOptions, BayOptions } from "./types";
import { createBays, createMerchandising, stockOptions, resolveBayLayout } from "./sim/structure.ts";
import { DEFAULT_OPTIONS } from "./defaults.ts";

export type RandomRange = { min: number; max: number };
export type SweepOptions = BayOptions & {
  shelfLength?: number | RandomRange;
  frames?: number | RandomRange;
  view?: { mode?: "fixed" | "handheld"; yaw?: number; pitch?: number; jitter?: number };
  stock?: StockOptions;
};
export type SweepPlan = {
  store: Store;
  shelf: Shelf;
  side: number;
  distance: number;
  estimatedOverlap: number;
  options: SweepOptions;
  poses: {
    x: number;
    y: number;
    z: number;
    lookX: number;
    lookY: number;
    lookZ: number;
    yawDegrees: number;
    pitchDegrees: number;
    rollDegrees: number;
    timestamp: number;
  }[];
};

// Smooth, seeded controls model small hand-held drift instead of unrelated frame jumps.
function smoothSignal(random: () => number) {
  const knots = Array.from({ length: 7 }, () => random() * 2 - 1);
  return (t: number) => {
    const at = Math.min(5, Math.floor(t * 6)), f = t * 6 - at;
    const blend = f * f * (3 - 2 * f);
    return knots[at] * (1 - blend) + knots[at + 1] * blend;
  };
}

export function resolveShelfLength(seed: number, value?: number | RandomRange) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
    throw Error("场景种子必须为 32 位非负整数");
  return value === undefined
    ? Math.max(...generateStore(seed).shelves.map(s => s.width))
    : pick(value, 1, 12, false, rng(seed ^ 0x745acc), "货架长度");
}
function pick(
  value: number | RandomRange,
  min: number,
  max: number,
  integer: boolean,
  random: () => number,
  label: string,
) {
  const validate = (n: number) =>
    Number.isFinite(n) &&
    n >= min &&
    n <= max &&
    (!integer || Number.isInteger(n));
  if (typeof value === "number") {
    if (!validate(value))
      throw Error(`${label}必须为 ${min}–${max}${integer ? " 的整数" : ""}`);
    return value;
  }
  if (
    !value ||
    !validate(value.min) ||
    !validate(value.max) ||
    value.min > value.max
  )
    throw Error(
      `${label}随机范围必须在 ${min}–${max} 内，下限不能大于上限${integer ? "，且必须为整数" : ""}`,
    );
  if (integer)
    return value.min + Math.floor(random() * (value.max - value.min + 1));
  return Math.max(
    value.min,
    Math.min(
      value.max,
      Math.round((value.min + random() * (value.max - value.min)) * 100) / 100,
    ),
  );
}

export function createSweepPlan(
  seed: number,
  input: number | SweepOptions = 9,
): SweepPlan {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
    throw Error("场景种子必须为 32 位非负整数");
  const options: SweepOptions =
    typeof input === "number" ? { frames: input } : input;
  const count = pick(
    options.frames ?? 9,
    2,
    48,
    true,
    rng(seed ^ 0x43f21a),
    "图片张数",
  );
  const length = resolveShelfLength(seed, options.shelfLength);
  const store = generateStore(seed, 8, 0, false);
  const view = { ...DEFAULT_OPTIONS.view, ...options.view };
  if (!["fixed", "handheld"].includes(view.mode)) throw Error("拍摄模式必须为 fixed 或 handheld");
  const stock = stockOptions(options.stock);
  for (const [key, min, max, label] of [
    ["yaw", -25, 25, "偏航角"], ["pitch", -20, 20, "俯仰角"],
    ["jitter", 0, 15, "逐帧角度扰动"],
  ] as const) {
    if (!Number.isFinite(view[key]) || view[key] < min || view[key] > max)
      throw Error(`${label}必须在 ${min}–${max} 度内`);
  }
  let candidates = [...store.shelves].sort((a, b) => b.width - a.width);
  // Explicit options reserve an aisle for the requested shelf and capture span.
  const configured = typeof input !== "number";
  let span: number | undefined;
  let plannedDistance: number | undefined;
  if (configured) {
    const target = candidates[0];
    target.width = length;
    target.productColumns = Math.max(4, Math.round(target.width / 0.23));
    const normalDistance = target.height * 1.13;
    const visiblePerDistance = 0.75 * 2 * Math.tan((54 * Math.PI) / 360);
    const footprint = Math.max(
      normalDistance * visiblePerDistance,
      target.width / (0.8 + 0.45 * (count - 1)),
    );
    plannedDistance = footprint / visiblePerDistance;
    span = Math.max(target.width * 0.25, target.width - 0.8 * footprint);
    const viewOffset = Math.abs(Math.tan(view.yaw * Math.PI / 180) * (target.depth / 2 + plannedDistance));
    const sin = Math.abs(Math.sin(target.angle));
    store.width = Math.max(
      store.width,
      target.width + 4,
      span + 4 + 2 * plannedDistance * sin + 2 * viewOffset,
    );
    store.depth = Math.max(
      store.depth,
      plannedDistance + 18 + target.width * sin,
    );
    target.x = 0;
    target.z =
      -store.depth / 2 +
      plannedDistance +
      target.depth / 2 +
      1.6 +
      (span / 2) * sin;
    const rearStart =
      target.z + (target.width / 2) * sin + target.depth / 2 + 2.5;
    store.shelves
      .filter((s) => s !== target)
      .forEach((s, i) => {
        s.x = (((i % 3) - 1) * store.width) / 4;
        s.z = rearStart + Math.floor(i / 3) * 2.6;
      });
    candidates = [target];
  }
  for (const shelf of candidates) {
    const bays = resolveBayLayout(shelf.width, options);
    shelf.bays = createBays(shelf, bays.layers);
    shelf.merchandising = createMerchandising(shelf, seed, stock);
    for (const side of [-1, 1]) {
      const distance = plannedDistance ?? shelf.height * 1.13;
      const travel = span ?? shelf.width * 1.15;
      const c = Math.cos(shelf.angle),
        s = Math.sin(shelf.angle);
      const random = rng(seed ^ 0x17ca31);
      const handRandom = rng(seed ^ 0x633cb2);
      const [yawSway, pitchSway, rollSway, xSway, ySway, zSway] = Array.from({ length: 6 }, () => smoothSignal(handRandom));
      const handheld = view.mode === "handheld";
      let timestamp = 0;
      const poses = Array.from({ length: count }, (_, i) => {
        const t = i / (count - 1);
        const progress = handheld ? t + 0.018 * Math.sin(2 * Math.PI * t) : t;
        // Shift the camera path sideways to keep the same capture span when looking obliquely.
        const u = (progress - 0.5) * travel -
          Math.tan(view.yaw * Math.PI / 180) * (shelf.depth / 2 + distance) + (handheld ? xSway(t) * 0.02 : 0);
        const v = side * (shelf.depth / 2 + distance + (handheld ? zSway(t) * 0.035 : 0));
        const yawDegrees = view.yaw + (handheld ? yawSway(t) : random() * 2 - 1) * view.jitter;
        const pitchDegrees = view.pitch + (handheld ? pitchSway(t) : random() * 2 - 1) * view.jitter;
        const rollDegrees = handheld ? rollSway(t) * Math.min(0.7, view.jitter * 0.4) : 0;
        const yaw = yawDegrees * Math.PI / 180;
        const pitch = pitchDegrees * Math.PI / 180;
        const dx = Math.sin(yaw) * Math.cos(pitch);
        const dz = -side * Math.cos(yaw) * Math.cos(pitch);
        const x = shelf.x + u * c - v * s;
        const z = shelf.z + u * s + v * c;
        const y = shelf.height * 0.51 + (handheld ? ySway(t) * 0.025 : 0);
        if (i > 0) timestamp += handheld ? 0.45 + handRandom() * 0.1 : 0.5;
        return {
          x, y, z,
          lookX: x + dx * c - dz * s,
          lookZ: z + dx * s + dz * c,
          lookY: y - Math.sin(pitch),
          yawDegrees, pitchDegrees, rollDegrees, timestamp,
        };
      });
      if (
        poses.every((p, i) => Array.from({ length: 6 }, (_, j) => {
          const previous = poses[Math.max(0, i - 1)], t = j / 5;
          return free(store, previous.x * (1 - t) + p.x * t, previous.z * (1 - t) + p.z * t, 0.22);
        }).every(Boolean))
      ) {
        if (configured)
          store.route = poses.map(({ x, z, lookX, lookZ }) => ({
            x,
            z,
            lookX,
            lookZ,
          }));
        return {
          store,
          shelf,
          side,
          distance,
          poses,
          options: { ...options, bayMode: bays.mode, bayWidth: bays.targetWidth,
            bayLayers: options.bayLayers ?? DEFAULT_OPTIONS.bayLayers, view, stock },
          estimatedOverlap: Math.max(
            0,
            1 -
              travel /
                (count - 1) /
                (distance * 0.75 * 2 * Math.tan((54 * Math.PI) / 360)),
          ),
        };
      }
    }
  }
  throw Error("这个场景没有可用的货架横移拍摄空间，请更换种子");
}
