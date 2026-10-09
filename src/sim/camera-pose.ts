import * as THREE from "three";

type Position = { x: number; y?: number; z: number };
export function setCameraPose(camera: THREE.Camera, position: Position, look: Position, rollDegrees = 0) {
  camera.position.set(position.x, position.y ?? 1, position.z);
  camera.lookAt(look.x, look.y ?? 1, look.z);
  camera.rotateZ(rollDegrees * Math.PI / 180);
  camera.updateMatrixWorld(true);
}

function rows(matrix: THREE.Matrix4) {
  const e = matrix.elements;
  return Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => e[c * 4 + r]));
}

// Read the camera that actually rendered this RGB frame. Do not use the input
// shelf-relative yaw/pitch as world Euler angles, or rebase to the first frame.
export function captureCameraPose(camera: THREE.Camera, frame: number, timestamp: number) {
  const c2w = camera.matrixWorld.clone();
  const position = new THREE.Vector3().setFromMatrixPosition(c2w);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(c2w).normalize();
  const euler = new THREE.Euler().setFromQuaternion(quaternion, "XYZ");
  const angles = [euler.x, euler.y, euler.z];
  const cvC2w = c2w.clone().multiply(new THREE.Matrix4().makeScale(1, -1, -1));
  return {
    frame, image: `rgb/${String(frame).padStart(4, "0")}.jpg`, timestamp_s: timestamp,
    position_m: position.toArray(), euler_xyz_rad: angles,
    euler_xyz_deg: angles.map(a => a * 180 / Math.PI),
    pose6d_m_rad: [...position.toArray(), ...angles], quaternion_xyzw: quaternion.toArray(),
    T_c2w: rows(c2w), T_w2c: rows(c2w.clone().invert()),
    opencv: { T_c2w: rows(cvC2w), T_w2c: rows(cvC2w.clone().invert()) },
  };
}
export type CameraPoseGT = ReturnType<typeof captureCameraPose>;

export function cameraPoseGroundTruth(frames: CameraPoseGT[]) {
  return {
    schemaVersion: 1,
    source: "Actual RGB rendering camera.matrixWorld, captured immediately after each RGB render; synthetic scene ground truth, not estimated or first-frame aligned.",
    world: { handedness: "right", origin: "Store floor center (0,0,0)",
      axes: { x: "store width", y: "up", z: "store depth" }, length_unit: "meter (1 scene unit = 1 meter)" },
    camera: { axes: { x: "right", y: "up", z: "backward; viewing direction is -Z" } },
    rotation: { euler_order: "XYZ (intrinsic)", composition: "R_c2w = Rx(rx) * Ry(ry) * Rz(rz)",
      angle_unit: "radian; euler_xyz_deg additionally provided", quaternion_order: "xyzw", quaternion_direction: "camera-to-world",
      pose6d_fields: ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"],
      note: "World Euler angles, not the shelf-relative input yaw/pitch/roll. Euler angles are nonunique and may jump at branch cuts or gimbal lock; use quaternion or matrices for geometry." },
    matrices: { storage: "Nested row-major 4x4 arrays; multiply column vectors", T_c2w: "p_world = T_c2w * p_camera",
      T_w2c: "p_camera = T_w2c * p_world" },
    opencv: { axes: { x: "right", y: "down", z: "forward" },
      conversion: "T_c2w_cv = T_c2w * diag(1,-1,-1,1); world coordinates unchanged. Use per-frame opencv.T_w2c with intrinsics.json." },
    frames,
  };
}
export type CameraGroundTruth = ReturnType<typeof cameraPoseGroundTruth>;

export function cameraPoseCSV(gt: CameraGroundTruth) {
  const header = "frame,image,timestamp_s,x_m,y_m,z_m,rx_rad,ry_rad,rz_rad,rx_deg,ry_deg,rz_deg,qx,qy,qz,qw";
  return [header, ...gt.frames.map(p => [p.frame, p.image, p.timestamp_s, ...p.pose6d_m_rad,
    ...p.euler_xyz_deg, ...p.quaternion_xyzw].join(","))].join("\n") + "\n";
}
