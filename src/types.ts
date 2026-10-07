export type Shelf = {
  id: string;
  x: number;
  z: number;
  width: number;
  height: number;
  depth: number;
  angle: number;
  productColumns?: number;
};
export type Store = {
  seed: number;
  width: number;
  depth: number;
  shelves: Shelf[];
  route: { x: number; z: number; lookX: number; lookZ: number }[];
};
