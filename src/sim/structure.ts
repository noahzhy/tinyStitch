import type { Shelf, ShelfBay, StockOptions, Merchandising, BayOptions, SKU, ProductShape } from "../types";
import { rng } from "./store.ts";
import { DEFAULT_OPTIONS } from "../defaults.ts";

export const MAX_BAYS = 24;
export const PRODUCT_GAP = 0.004; // 4 mm between planned packages, including empty slots.
function validateLayers(layers: number[]) {
  if (!Array.isArray(layers) || layers.length < 1 || layers.length > MAX_BAYS ||
      layers.some(n => !Number.isInteger(n) || n < 1 || n > 8))
    throw Error(`子货架必须为 1–${MAX_BAYS} 个，每个子货架层板数量为 1–8 的整数`);
}

export function resolveBayLayout(length: number, options: BayOptions = {}) {
  // An old explicit layer list retains its manually configured count.
  const mode = options.bayMode ?? (options.bayLayers === undefined ? "auto" : "manual");
  if (mode !== "auto" && mode !== "manual") throw Error("子货架数量模式必须为 auto 或 manual");
  const targetWidth = options.bayWidth ?? DEFAULT_OPTIONS.bayWidth;
  if (!Number.isFinite(targetWidth) || targetWidth < 0.5 || targetWidth > 3)
    throw Error("目标子货架宽度必须在 0.5–3 模拟米内");
  const template = options.bayLayers ?? DEFAULT_OPTIONS.bayLayers;
  validateLayers(template);
  if (!Number.isFinite(length) || length <= 0) throw Error("货架长度必须为正数");
  const count = mode === "auto" ? Math.max(1, Math.ceil(length / targetWidth - 1e-10)) : template.length;
  if (count > MAX_BAYS) throw Error(`子货架数量不能超过 ${MAX_BAYS} 个`);
  return { mode, targetWidth, layers: Array.from({ length: count }, (_, i) => template[i % template.length]) };
}

// Positions are local to the shelf. Each bay has its own independently spaced boards.
export function createBays(shelf: Shelf, layers: number[]): ShelfBay[] {
  validateLayers(layers);
  const width = shelf.width / layers.length;
  return layers.map((count, i) => ({
    id: `${shelf.id}/bay-${i + 1}`,
    x: -shelf.width / 2 + (i + 0.5) * width,
    width,
    layers: count,
    layerHeights: Array.from({ length: count }, (_, l) =>
      0.15 + l * (shelf.height - 0.25) / count),
  }));
}

export function shelfBays(shelf: Shelf) {
  return shelf.bays ?? createBays(shelf, [4]);
}

export function stockOptions(input: StockOptions = {}): Required<StockOptions> {
  const options = { emptyRate: 0.18, facingsMin: 2, facingsMax: 5, depthCopies: 3, ...input };
  if (!Number.isFinite(options.emptyRate) || options.emptyRate < 0 || options.emptyRate > 1)
    throw Error("空位比例必须在 0–1 内");
  if (![options.facingsMin, options.facingsMax].every(n => Number.isInteger(n) && n >= 1 && n <= 12) || options.facingsMin > options.facingsMax)
    throw Error("同款连续列数范围必须在 1–12 内，下限不能大于上限");
  if (!Number.isInteger(options.depthCopies) || options.depthCopies < 1 || options.depthCopies > 4)
    throw Error("每列纵深容量必须为 1–4 的整数");
  return options;
}

export function createMerchandising(shelf: Shelf, seed: number, input: StockOptions = {}): Merchandising {
  const options = stockOptions(input);
  const random = rng(seed ^ 0x27ab5f);
  // Stock changes never move or collapse the planned facings and empty spaces.
  const stockRandom = rng(seed ^ 0x7831a2);
  const bays = shelfBays(shelf);
  const columns = Math.max(1, Math.round(bays[0].width / 0.23));
  const cellWidth = (bays[0].width - Math.min(0.08, bays[0].width * 0.15)) / columns;
  const minGap = Math.min(...bays.map(b => (shelf.height - 0.25) / b.layers));
  const maxDepth = (shelf.depth / 2 - 0.055 - 0.012 * (options.depthCopies - 1)) / options.depthCopies;
  const catalog: SKU[] = Array.from({ length: 16 }, (_, i) => {
    // Each quartet includes all shapes, including the short packs that fit tall
    // layer counts. Width never exceeds the previous 0.88 * cellWidth limit.
    const shape: ProductShape = (["box", "bottle", "can", "jar"] as const)[i % 4];
    const hue = ((Math.floor(i / 4) * 0.23) + (i % 4) * 0.025 + random() * 0.015) % 1;
    const variation = random();
    const width = shape === "box" ? cellWidth * (0.4 + variation * 0.48) :
      Math.min(cellWidth * 0.88, maxDepth) * ((shape === "bottle" ? 0.65 : 0.75) + variation * (shape === "bottle" ? 0.35 : 0.25));
    const sampledHeight = i < 4 ? (minGap - 0.04) * (0.65 + random() * 0.2) : 0.16 + random() * 0.23;
    return {
      id: `sku-${String(i + 1).padStart(3, "0")}`,
      name: [["OATS", "CEREAL", "RICE", "TEA"], ["JUICE", "MILK", "TEA", "COFFEE"],
        ["COFFEE", "COCOA", "JUICE", "TEA"], ["COCOA", "COFFEE", "OATS", "TEA"]][i % 4][Math.floor(i / 4)],
      shape, stackable: shape !== "bottle", hue, width,
      height: shape === "box" ? sampledHeight : Math.min(sampledHeight, width * (shape === "bottle" ? 4.5 : shape === "can" ? 2.4 : 1.8)),
      // Round containers retain a circular footprint and fit the selected depth
      // capacity. Boxes retain their original depth allowance.
      depth: shape === "box" ? maxDepth : width,
      labelSeed: Math.floor(random() * 4294967296),
    };
  });
  const slots: Merchandising["slots"] = [], products: Merchandising["products"] = [];
  for (const [b, bay] of bays.entries()) {
    const margin = Math.min(0.025, bay.width * 0.15);
    const left = bay.x - bay.width / 2 + margin, right = bay.x + bay.width / 2 - margin;
    for (const [layer, boardY] of bay.layerHeights.entries()) {
      // Board heights describe their centers; leave room for the 3 cm lower
      // half of the board above, as well as 5 mm package clearance.
      const nextBoard = bay.layerHeights[layer + 1];
      const ceiling = nextBoard === undefined ? shelf.height : nextBoard - 0.03;
      const allowed = catalog.filter(s => s.height < ceiling - boardY - 0.035);
      // A small row-level assortment creates category blocks and repeats, rather than random packs.
      const assortment = Array.from({ length: Math.min(3, allowed.length) }, () => allowed[Math.floor(random() * allowed.length)]);
      for (const side of [-1, 1]) {
        let col = 0, block = 0, previous = "", cursor = left;
        while (cursor < right) {
          const fits = (s: SKU) => cursor + s.width <= right + 1e-10;
          const remaining = assortment.filter(fits);
          const choices = remaining.filter(s => s.id !== previous);
          // Fill the tail with another fitting SKU rather than leaving a large
          // artificial gap because the preferred assortment has wider packs.
          const pool = choices.length ? choices : remaining.length ? remaining : allowed.filter(fits);
          if (!pool.length) break;
          const sku = pool[Math.floor(random() * pool.length)];
          previous = sku.id;
          const stackCapacity = sku.stackable ? Math.max(1, Math.floor((ceiling - boardY - 0.035) / sku.height + 1e-10)) : 1;
          const capacity = options.depthCopies * stackCapacity;
          const facings = options.facingsMin + Math.floor(random() * (options.facingsMax - options.facingsMin + 1));
          const blockEmptyRate = options.emptyRate * 0.55;
          const blockEmpty = stockRandom() < blockEmptyRate;
          const slotEmptyRate = (options.emptyRate - blockEmptyRate) / (1 - blockEmptyRate);
          for (let j = 0; j < facings && fits(sku); j++, col++) {
            const emptyDraw = stockRandom(), countDraw = stockRandom(), recessionDraw = stockRandom();
            const stock = blockEmpty || emptyDraw < slotEmptyRate ? 0 : 1 + Math.floor(countDraw * capacity);
            const id = `${bay.id}/row-${layer + 1}/side-${side}/col-${col + 1}`;
            const x = cursor + sku.width / 2;
            slots.push({ id, bay: b, layer, col, side, skuId: sku.id, block, x, width: sku.width, boardY, stock, capacity,
              depthCapacity: options.depthCopies, stackCapacity });
            // Partly depleted rows may be pulled forward or sit slightly recessed.
            const usedDepth = Math.ceil(stock / stackCapacity);
            const start = Math.floor(recessionDraw * (options.depthCopies - usedDepth + 1));
            for (let k = 0; k < stock; k++) {
              // Fill each depth position from the bottom before starting another
              // stack, so every upper package has a physical support beneath it.
              const depthIndex = start + Math.floor(k / stackCapacity), stackIndex = k % stackCapacity;
              products.push({
                id: `${id}/depth-${depthIndex + 1}/stack-${stackIndex + 1}`, slotId: id, skuId: sku.id, shape: sku.shape,
                bay: b, layer, col, side, depthIndex, stackIndex, x,
                y: boardY + 0.03 + (stackIndex + 0.5) * sku.height,
                z: side * (shelf.depth / 2 - 0.02 - sku.depth / 2 - depthIndex * (sku.depth + 0.012)),
                width: sku.width, height: sku.height, depth: sku.depth,
              });
            }
            cursor += sku.width + PRODUCT_GAP;
          }
          block++;
        }
      }
    }
  }
  return { options, catalog, slots, products };
}

// Shared by meshes, labels and export; the layout is generated only once per shelf.
export function shelfProducts(shelf: Shelf, seed: number) {
  return (shelf.merchandising ?? createMerchandising(shelf, seed)).products;
}
