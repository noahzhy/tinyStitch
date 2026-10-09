import type { SweepOptions } from "./sweep";

export const DEFAULT_SEED = 42;
export const DEFAULT_OPTIONS = {
  shelfLength: { min: 2, max: 8 }, frames: 9, bayMode: "auto", bayWidth: 1, bayLayers: [3, 4, 5],
  view: { mode: "handheld", yaw: 0, pitch: 0, jitter: 15 },
  stock: { emptyRate: 0.18, facingsMin: 2, facingsMax: 5, depthCopies: 3 },
} satisfies SweepOptions;
