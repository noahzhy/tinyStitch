export type ShelfBay = {
  id: string;
  x: number;
  width: number;
  layers: number;
  layerHeights: number[];
};
export type BayOptions = { bayMode?: "auto" | "manual"; bayWidth?: number; bayLayers?: number[] };
export type StockOptions = { emptyRate?: number; facingsMin?: number; facingsMax?: number; depthCopies?: number };
export type ProductShape = "box" | "bottle" | "can" | "jar";
export type SKU = { id: string; name: string; shape: ProductShape; stackable: boolean; hue: number; width: number; height: number; depth: number; labelSeed: number };
export type ProductSlot = {
  id: string; bay: number; layer: number; col: number; side: number;
  skuId: string; block: number; x: number; width: number; boardY: number; stock: number; capacity: number; depthCapacity: number; stackCapacity: number;
};
export type ShelfProduct = {
  id: string; slotId: string; skuId: string; shape: ProductShape; bay: number; layer: number; col: number; side: number; depthIndex: number; stackIndex: number;
  x: number; y: number; z: number; width: number; height: number; depth: number;
};
export type Merchandising = { options: Required<StockOptions>; catalog: SKU[]; slots: ProductSlot[]; products: ShelfProduct[] };
export type Shelf = {
  id: string;
  x: number;
  z: number;
  width: number;
  height: number;
  depth: number;
  angle: number;
  productColumns?: number;
  bays?: ShelfBay[];
  merchandising?: Merchandising;
};
export type Store = {
  seed: number;
  width: number;
  depth: number;
  shelves: Shelf[];
  route: { x: number; z: number; lookX: number; lookZ: number }[];
};
