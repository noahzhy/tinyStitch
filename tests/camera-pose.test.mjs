import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { setCameraPose, captureCameraPose } from '../src/sim/camera-pose.ts';
import { createSweepPlan } from '../src/sweep.ts';

const matrix = rows => new THREE.Matrix4().set(...rows.flat());
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const sameMatrix = (a, b) => a.elements.forEach((v, i) => near(v, b.elements[i]));

test('camera GT records known world translation and actual optical-axis roll', () => {
  const camera = new THREE.PerspectiveCamera();
  setCameraPose(camera, { x: 1, y: 2, z: 3 }, { x: 1, y: 2, z: 2 }, 30);
  const p = captureCameraPose(camera, 7, 3.5);
  assert.deepEqual(p.position_m, [1, 2, 3]);
  assert.equal(p.image, 'rgb/0007.jpg');
  assert.equal(p.timestamp_s, 3.5);
  p.euler_xyz_rad.forEach((v, i) => near(v, i === 2 ? Math.PI / 6 : 0));
  near(p.quaternion_xyzw[2], Math.sin(Math.PI / 12));
  sameMatrix(matrix(p.T_c2w).multiply(matrix(p.T_w2c)), new THREE.Matrix4());
  const transformedOrigin = new THREE.Vector3().applyMatrix4(matrix(p.T_c2w));
  assert.deepEqual(transformedOrigin.toArray(), p.position_m);
});

test('GT uses camera world matrix including parent transform, not local coordinates', () => {
  const parent = new THREE.Group(), camera = new THREE.PerspectiveCamera();
  parent.position.set(5, 6, 7); parent.rotation.y = 0.5;
  camera.position.set(1, 2, 3); camera.rotation.set(0.1, 0.2, 0.3);
  parent.add(camera); parent.updateMatrixWorld(true);
  const p = captureCameraPose(camera, 0, 0);
  assert.notDeepEqual(p.position_m, camera.position.toArray());
  sameMatrix(matrix(p.T_c2w), camera.matrixWorld);
  const q = new THREE.Quaternion(...p.quaternion_xyzw);
  sameMatrix(new THREE.Matrix4().compose(new THREE.Vector3(...p.position_m), q, new THREE.Vector3(1, 1, 1)), camera.matrixWorld);
});

test('Euler XYZ and quaternion recover rotation across branch cuts and gimbal lock', () => {
  for (const angles of [[0.2, -0.3, 0.4], [0, Math.PI, 0], [0.4, Math.PI / 2, -0.7], [-2, -Math.PI / 2, 1]]) {
    const camera = new THREE.PerspectiveCamera();
    camera.rotation.set(...angles, 'XYZ'); camera.updateMatrixWorld(true);
    const p = captureCameraPose(camera, 0, 0), [rx, ry, rz] = p.euler_xyz_rad;
    const recovered = new THREE.Matrix4().makeRotationX(rx)
      .multiply(new THREE.Matrix4().makeRotationY(ry)).multiply(new THREE.Matrix4().makeRotationZ(rz));
    sameMatrix(recovered, camera.matrixWorld);
    near(Math.hypot(...p.quaternion_xyzw), 1);
  }
});

test('OpenCV world extrinsics reproduce rendered projections for every jittered frame', () => {
  const plan = createSweepPlan(42, { shelfLength: 6, frames: 12 });
  const camera = new THREE.PerspectiveCamera(54, 720 / 960, 0.05, 100);
  const focal = 960 / (2 * Math.tan(27 * Math.PI / 180));
  for (const [i, pose] of plan.poses.entries()) {
    setCameraPose(camera, pose, { x: pose.lookX, y: pose.lookY, z: pose.lookZ }, pose.rollDegrees);
    const p = captureCameraPose(camera, i, pose.timestamp);
    sameMatrix(matrix(p.opencv.T_c2w).multiply(matrix(p.opencv.T_w2c)), new THREE.Matrix4());
    for (const local of [[0.2, 0.1, 2], [-0.3, -0.4, 3], [0, 0, 1]]) {
      const world = new THREE.Vector3(...local).applyMatrix4(matrix(p.opencv.T_c2w));
      const cv = world.clone().applyMatrix4(matrix(p.opencv.T_w2c));
      const ndc = world.clone().project(camera);
      assert.ok(cv.z > 0);
      near(focal * cv.x / cv.z + 359.5, (ndc.x + 1) * 360 - 0.5, 1e-8);
      near(focal * cv.y / cv.z + 479.5, (1 - ndc.y) * 480 - 0.5, 1e-8);
    }
    const forward = new THREE.Vector3(0, 0, -1).transformDirection(matrix(p.T_c2w));
    const look = new THREE.Vector3(pose.lookX - pose.x, pose.lookY - pose.y, pose.lookZ - pose.z).normalize();
    near(forward.distanceTo(look), 0);
  }
});
