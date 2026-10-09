import * as THREE from "three";
import type { SweepPlan } from "./sweep";
import type { StoreRenderer } from "./sim/renderer";
import { setCameraPose } from "./sim/camera-pose.ts";

type Point = { x: number; y: number };
const RGB_WIDTH = 720, RGB_HEIGHT = 960, FOV = 54;

// Clip the reference-plane rectangle by the actual camera frustum. Unlike
// corner-ray intersection, this also handles views partly facing off the plane.
function clip(polygon: Point[], distance: (p: Point) => number): Point[] {
  const result: Point[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i], b = polygon[(i + 1) % polygon.length];
    const da = distance(a), db = distance(b);
    if (da >= 0) result.push(a);
    if ((da >= 0) !== (db >= 0)) {
      const t = da / (da - db);
      result.push({ x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) });
    }
  }
  return result;
}
function area(polygon: Point[]) {
  return Math.abs(polygon.reduce((sum, p, i) => {
    const q = polygon[(i + 1) % polygon.length];
    return sum + p.x * q.y - q.x * p.y;
  }, 0)) / 2;
}

export function orthographicLayout(plan: SweepPlan) {
  const { shelf, side } = plan;
  const shelfToWorld = new THREE.Matrix4().makeRotationY(-shelf.angle);
  shelfToWorld.setPosition(shelf.x, 0, shelf.z);
  const referenceZ = side * shelf.depth / 2;
  const camera = new THREE.PerspectiveCamera(FOV, RGB_WIDTH / RGB_HEIGHT, 0.05, 100);
  const polygons = plan.poses.map((p, frame) => {
    setCameraPose(camera, p, { x: p.lookX, y: p.lookY, z: p.lookZ }, p.rollDegrees);
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    let polygon = [{ x: -shelf.width / 2, y: 0 }, { x: shelf.width / 2, y: 0 },
      { x: shelf.width / 2, y: shelf.height }, { x: -shelf.width / 2, y: shelf.height }];
    for (const plane of frustum.planes) polygon = clip(polygon, q =>
      plane.distanceToPoint(new THREE.Vector3(q.x, q.y, referenceZ).applyMatrix4(shelfToWorld)));
    return { frame, timestamp: p.timestamp, polygon: area(polygon) > 1e-10 ? polygon : [] };
  });
  const points = polygons.flatMap(p => p.polygon);
  if (!points.length) throw Error("无法生成正交 GT：拍摄视野没有覆盖货架参考平面");
  const observedBounds = {
    minX: Math.min(...points.map(p => p.x)), maxX: Math.max(...points.map(p => p.x)),
    minY: Math.min(...points.map(p => p.y)), maxY: Math.max(...points.map(p => p.y)),
  };
  const pixelsPerMeter = RGB_HEIGHT / shelf.height;
  const width = Math.max(1, Math.ceil((observedBounds.maxX - observedBounds.minX) * pixelsPerMeter));
  const height = Math.max(1, Math.ceil((observedBounds.maxY - observedBounds.minY) * pixelsPerMeter));
  // Expand symmetrically by less than one pixel so scale is exactly uniform.
  const cx = (observedBounds.minX + observedBounds.maxX) / 2;
  const cy = (observedBounds.minY + observedBounds.maxY) / 2;
  const bounds = { minX: cx - width / pixelsPerMeter / 2, maxX: cx + width / pixelsPerMeter / 2,
    minY: cy - height / pixelsPerMeter / 2, maxY: cy + height / pixelsPerMeter / 2 };
  const mapping = [[side * pixelsPerMeter, 0, side === 1 ? -bounds.minX * pixelsPerMeter : bounds.maxX * pixelsPerMeter],
    [0, -pixelsPerMeter, bounds.maxY * pixelsPerMeter], [0, 0, 1]];
  return {
    schemaVersion: 1, projection: "orthographic", width, height, pixelsPerMeter, side,
    referencePlane: { coordinates: "shelf-local", z: referenceZ },
    bounds, observedBounds, shelfToWorldColumnMajor: shelfToWorld.toArray(),
    shelfXYToImage: mapping,
    pixelCoordinates: "Image edges start at (0,0); pixel centers are (column+0.5,row+0.5). Image +u follows side * shelf-local +X, +v follows -Y.",
    coverage: "Union of RGB camera frusta intersected with the front reference plane and shelf rectangle. Geometric coverage, not an occlusion or product-surface visibility mask. White=covered, black=uncovered; orthographic PNG is transparent outside coverage.",
    products: "Original 3D package shapes, labels and stacks rendered frontally with an orthographic camera; depths are preserved in the scene.",
    files: { image: "gt/orthographic.png", coverageMask: "gt/coverage.png" },
    frames: polygons,
  };
}

export type OrthographicMetadata = ReturnType<typeof orthographicLayout>;
export type OrthographicGT = { image: string; coverageMask: string; metadata: OrthographicMetadata };

export function renderOrthographicGT(view: StoreRenderer, plan: SweepPlan): OrthographicGT {
  const metadata = orthographicLayout(plan);
  const { width, height, bounds, side, pixelsPerMeter, shelfXYToImage: m } = metadata;
  const mask = document.createElement("canvas");
  mask.width = width; mask.height = height;
  const ctx = mask.getContext("2d")!;
  ctx.fillStyle = "white";
  // Fill each footprint separately: gaps between views must remain uncovered.
  for (const { polygon } of metadata.frames) {
    if (!polygon.length) continue;
    ctx.beginPath();
    polygon.forEach((p, i) => {
      const x = m[0][0] * p.x + m[0][2], y = m[1][1] * p.y + m[1][2];
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.closePath(); ctx.fill();
  }
  const pixels = ctx.getImageData(0, 0, width, height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const covered = pixels.data[i + 3] >= 128 ? 255 : 0;
    pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = covered;
    pixels.data[i + 3] = covered;
  }
  ctx.putImageData(pixels, 0, 0);
  const camera = new THREE.OrthographicCamera(-width / pixelsPerMeter / 2, width / pixelsPerMeter / 2,
    height / pixelsPerMeter / 2, -height / pixelsPerMeter / 2, 0.05, 100);
  const shelfToWorld = new THREE.Matrix4().fromArray(metadata.shelfToWorldColumnMajor);
  const center = new THREE.Vector3((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, metadata.referencePlane.z);
  camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, side * 3)).applyMatrix4(shelfToWorld));
  camera.lookAt(center.clone().applyMatrix4(shelfToWorld));
  camera.updateMatrixWorld(true);
  const size = view.renderer.getSize(new THREE.Vector2());
  const image = document.createElement("canvas");
  image.width = width; image.height = height;
  try {
    view.renderer.setSize(width, height, false);
    view.renderer.render(view.scene, camera);
    const out = image.getContext("2d")!;
    out.drawImage(view.renderer.domElement, 0, 0);
    out.globalCompositeOperation = "destination-in";
    out.drawImage(mask, 0, 0);
  } finally { view.renderer.setSize(size.x, size.y, false); }
  // The image uses transparent coverage; the separate mask is opaque binary PNG.
  for (let i = 3; i < pixels.data.length; i += 4) pixels.data[i] = 255;
  ctx.putImageData(pixels, 0, 0);
  return { image: image.toDataURL("image/png"), coverageMask: mask.toDataURL("image/png"), metadata };
}
