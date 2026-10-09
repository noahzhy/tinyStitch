import * as THREE from "three";
import type { SKU } from "../types";

// SKU dimensions are the complete package bounds, including shoulder, lid and
// cap. The origin is at its center, with vertical +Y and the label facing +Z.
export function productTemplate(sku: SKU) {
  const group = new THREE.Group(), { width: w, height: h, depth: d } = sku;
  const body = new THREE.MeshLambertMaterial({ color: new THREE.Color().setHSL(sku.hue, 0.52, 0.48) });
  const cap = new THREE.MeshLambertMaterial({ color: new THREE.Color().setHSL(sku.hue, 0.45, 0.22) });
  const metal = new THREE.MeshLambertMaterial({ color: 0xcbd2d5 });
  const cylinder = (bottom: number, top: number, radiusBottom: number, radiusTop: number, material: THREE.Material) => {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radiusTop * w / 2, radiusBottom * w / 2,
      (top - bottom) * h, 32), material);
    mesh.position.y = ((bottom + top) / 2 - 0.5) * h;
    mesh.scale.z = d / w;
    group.add(mesh);
  };
  if (sku.shape === "bottle") {
    cylinder(0, 0.68, 1, 1, body);
    cylinder(0.68, 0.84, 1, 0.42, body);
    cylinder(0.84, 0.95, 0.42, 0.42, body);
    cylinder(0.95, 1, 0.5, 0.5, cap);
  } else if (sku.shape === "can") {
    cylinder(0.03, 0.97, 0.97, 0.97, body);
    cylinder(0, 0.03, 1, 1, metal);
    cylinder(0.97, 1, 1, 1, metal);
  } else if (sku.shape === "jar") {
    cylinder(0, 0.84, 1, 1, body);
    cylinder(0.84, 0.91, 1, 0.9, body);
    cylinder(0.91, 1, 1, 1, cap);
  } else {
    group.add(new THREE.Mesh(new THREE.BoxGeometry(w, h, d), body));
  }
  // Only materials used by the meshes need to survive until scene disposal.
  if (sku.shape === "box" || sku.shape === "can") cap.dispose();
  if (sku.shape !== "can") metal.dispose();
  return group;
}

export function productLabel(sku: SKU, material: THREE.Material, side: number) {
  if (sku.shape === "box") {
    const label = new THREE.Mesh(new THREE.PlaneGeometry(sku.width * 0.92, sku.height * 0.85), material);
    label.position.z = side * (sku.depth / 2 + 0.0004);
    if (side < 0) label.rotation.y = Math.PI;
    return label;
  }
  const [bottom, top] = sku.shape === "bottle" ? [0.1, 0.6] : sku.shape === "jar" ? [0.12, 0.76] : [0.12, 0.88];
  const radius = sku.width / 2 * (sku.shape === "can" ? 0.97 : 1) + 0.0004;
  const label = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, (top - bottom) * sku.height,
    32, 1, true, -Math.PI * 0.4, Math.PI * 0.8), material);
  label.scale.z = sku.depth / sku.width;
  label.position.y = ((bottom + top) / 2 - 0.5) * sku.height;
  if (side < 0) label.rotation.y = Math.PI;
  return label;
}
