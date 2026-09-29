import { generateStore, free, rng } from "../../src/sim/store";
import type { Store, Shelf } from "../../src/types";

export type RandomRange = { min: number; max: number };
export type SweepOptions = {
  shelfLength?: number | RandomRange;
  frames?: number | RandomRange;
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
  }[];
};
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
  const length =
    options.shelfLength === undefined
      ? undefined
      : pick(
          options.shelfLength,
          1,
          12,
          false,
          rng(seed ^ 0x745acc),
          "货架长度",
        );
  const store = generateStore(seed, 8, 0, false);
  let candidates = [...store.shelves].sort((a, b) => b.width - a.width);
  // Explicit options reserve an aisle for the requested shelf. Numeric legacy calls keep
  // the original geometry and route exactly, so existing training samples remain reproducible.
  const configured = typeof input !== "number";
  let span: number | undefined;
  let plannedDistance: number | undefined;
  if (configured) {
    const target = candidates[0];
    target.width = length ?? target.width;
    target.productColumns = Math.max(4, Math.round(target.width / 0.23));
    const normalDistance = target.height * 1.13;
    const visiblePerDistance = 0.75 * 2 * Math.tan((54 * Math.PI) / 360);
    const footprint = Math.max(
      normalDistance * visiblePerDistance,
      target.width / (0.8 + 0.45 * (count - 1)),
    );
    plannedDistance = footprint / visiblePerDistance;
    span = Math.max(target.width * 0.25, target.width - 0.8 * footprint);
    const sin = Math.abs(Math.sin(target.angle));
    store.width = Math.max(
      store.width,
      target.width + 4,
      span + 4 + 2 * plannedDistance * sin,
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
    for (const side of [-1, 1]) {
      const distance = plannedDistance ?? shelf.height * 1.13;
      const travel = span ?? shelf.width * 1.15;
      const c = Math.cos(shelf.angle),
        s = Math.sin(shelf.angle);
      const poses = Array.from({ length: count }, (_, i) => {
        const u = (i / (count - 1) - 0.5) * travel;
        const v = side * (shelf.depth / 2 + distance);
        const aim = u + 0.04 * shelf.width * Math.sin(i * 0.7 + seed);
        return {
          x: shelf.x + u * c - v * s,
          z: shelf.z + u * s + v * c,
          y: shelf.height * 0.51 + 0.015 * Math.sin(i + seed),
          lookX: shelf.x + aim * c,
          lookZ: shelf.z + aim * s,
          lookY: shelf.height * 0.51 + 0.02 * Math.cos(i * 0.8 + seed),
        };
      });
      if (
        Array.from({ length: 101 }, (_, i) => {
          const t = i / 100;
          return free(
            store,
            poses[0].x * (1 - t) + poses.at(-1)!.x * t,
            poses[0].z * (1 - t) + poses.at(-1)!.z * t,
            0.22,
          );
        }).every(Boolean)
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
          options,
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
