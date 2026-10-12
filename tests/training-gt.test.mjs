import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { trainingGroundTruth } from '../src/training-ground-truth.ts';
import { parseCLI, resolveCLI } from '../scripts/cli-options.mjs';

test('training GT is opt-in and does not change scene options', () => {
  const normal = resolveCLI(parseCLI(['--out', '/tmp/train-gt']));
  const training = resolveCLI(parseCLI(['--out', '/tmp/train-gt', '--training-gt']));
  assert.equal(normal.trainingGT, false); assert.equal(training.trainingGT, true);
  assert.deepEqual(normal.options, training.options);
});

test('mesh ray GT distinguishes visibility, physical instances and occlusion', () => {
  const scene = new THREE.Scene();
  const back = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  back.userData = { shelfId: 'test', instanceId: 'back', skuId: 'sku', kind: 'product' };
  scene.add(back);
  const front = new THREE.Mesh(new THREE.PlaneGeometry(.4, 4), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  front.position.set(.5, 0, 1); front.userData = { shelfId: 'test', instanceId: 'front', skuId: 'sku', kind: 'product' };
  scene.add(front);
  const pose = x => ({ x, y: 0, z: 3, lookX: x, lookY: 0, lookZ: 0, rollDegrees: 0 });
  const gt = trainingGroundTruth({ scene }, { shelf: { id: 'test' }, poses: [pose(0), pose(1)] }, 12, 12);
  const points = gt.pairs.flatMap(p => p.points);
  assert.ok(points.some(p => p.status === 'visible'));
  assert.ok(points.some(p => p.status === 'occluded'));
  assert.ok(points.some(p => p.instanceId === 'front'));
  assert.ok(points.some(p => p.instanceId === 'back'));
  assert.ok(points.filter(p => p.status === 'visible').every(p => p.world.length === 3));
});
