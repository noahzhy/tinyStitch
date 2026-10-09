import test from 'node:test';
import assert from 'node:assert/strict';
import { createSweepPlan, resolveShelfLength } from '../src/sweep.ts';
import { crc32, makeZip, sequenceFiles } from '../src/archive.ts';
import { shelfProducts, resolveBayLayout, PRODUCT_GAP } from '../src/sim/structure.ts';
import { orthographicLayout } from '../src/ground-truth.ts';
import { productTemplate, productLabel } from '../src/sim/products.ts';
import * as THREE from 'three';
import { setCameraPose, captureCameraPose, cameraPoseGroundTruth } from '../src/sim/camera-pose.ts';

const text = new TextEncoder();
const fakeGT = plan => ({ image: 'data:image/png;base64,' + btoa('png'), coverageMask: 'data:image/png;base64,' + btoa('mask'), metadata: orthographicLayout(plan) });
const fakeCameraGT = plan => cameraPoseGroundTruth(plan.poses.map((p, i) => {
  const camera = new THREE.PerspectiveCamera();
  setCameraPose(camera, p, { x: p.lookX, y: p.lookY, z: p.lookZ }, p.rollDegrees);
  return captureCameraPose(camera, i, p.timestamp);
}));
test('fixed seed and settings reproduce the same scene and ordered poses', () => {
  const options = { shelfLength: 6, frames: 12 };
  const first = createSweepPlan(42, options);
  assert.deepEqual(first, createSweepPlan(42, options));
  assert.equal(first.shelf.width, 6);
  assert.equal(first.poses.length, 12);
  assert.notDeepEqual(first.poses, createSweepPlan(43, options).poses);
  assert.ok(first.estimatedOverlap > 0 && first.estimatedOverlap < 1);
});

test('random ranges stay within bounds and invalid settings are rejected', () => {
  const options = { shelfLength: { min: 3, max: 4 }, frames: { min: 8, max: 12 } };
  for (const seed of [0, 42, 2026, 4294967295]) {
    const plan = createSweepPlan(seed, options);
    assert.ok(plan.shelf.width >= 3 && plan.shelf.width <= 4);
    assert.ok(plan.poses.length >= 8 && plan.poses.length <= 12);
  }
  for (const invalid of [{ frames: 1 }, { frames: 49 }, { shelfLength: 0 }, { frames: { min: 12, max: 8 } }]) {
    assert.throws(() => createSweepPlan(42, invalid));
  }
  assert.throws(() => createSweepPlan(-1));
});

test('ZIP uses standard signatures, offsets, UTF-8 names and CRC32', () => {
  assert.equal(crc32(text.encode('123456789')), 0xcbf43926);
  const files = [{ name: 'rgb/0000.jpg', bytes: text.encode('image') }, { name: '参数.json', bytes: text.encode('{}') }];
  const zip = makeZip(files), v = new DataView(zip.buffer);
  const end = zip.length - 22;
  assert.equal(v.getUint32(end, true), 0x06054b50);
  assert.equal(v.getUint16(end + 10, true), 2);
  let entry = v.getUint32(end + 16, true);
  for (const file of files) {
    assert.equal(v.getUint32(entry, true), 0x02014b50);
    assert.equal(v.getUint16(entry + 8, true), 0x800);
    const nameLength = v.getUint16(entry + 28, true);
    assert.equal(new TextDecoder().decode(zip.subarray(entry + 46, entry + 46 + nameLength)), file.name);
    const local = v.getUint32(entry + 42, true);
    assert.equal(v.getUint32(local, true), 0x04034b50);
    assert.equal(v.getUint32(local + 14, true), crc32(file.bytes));
    const offset = local + 30 + v.getUint16(local + 26, true);
    assert.deepEqual(zip.subarray(offset, offset + file.bytes.length), file.bytes);
    entry += 46 + nameLength;
  }
  assert.equal(entry, end);
});

test('sequence export preserves image order and includes matching calibration and timestamps', () => {
  const frames = ['first', 'second'].map(s => 'data:image/jpeg;base64,' + btoa(s));
  const scene = createSweepPlan(42, { shelfLength: 3, frames: 2, bayLayers: [2, 5], view: { mode: 'fixed', yaw: 12, pitch: 8, jitter: 0 } });
  const files = sequenceFiles({ ...scene, frames, groundTruth: fakeGT(scene), cameraGroundTruth: fakeCameraGT(scene) });
  assert.deepEqual(files.slice(0, 2).map(f => f.name), ['rgb/0000.jpg', 'rgb/0001.jpg']);
  assert.equal(new TextDecoder().decode(files[1].bytes), 'second');
  const json = name => JSON.parse(new TextDecoder().decode(files.find(f => f.name === name).bytes));
  assert.deepEqual(json('timestamps.json'), [0, 0.5]);
  assert.equal(json('intrinsics.json').width, 720);
  assert.equal(json('generation/scene.json').poses.length, frames.length);
  assert.equal(json('intrinsics.json').K[0][0], json('intrinsics.json').K[1][1]);
  assert.deepEqual(json('generation/scene.json').shelf.bays.map(b => b.layers), [2, 5]);
  assert.equal(json('generation/scene.json').poses[1].yawDegrees, 12);
  assert.equal(json('generation/scene.json').poses[1].pitchDegrees, 8);
  assert.equal(json('generation/config.json').seed, 42);
  assert.deepEqual(json('generation/config.json').options, scene.options);
  assert.equal(json('gt/metadata.json').projection, 'orthographic');
  assert.equal(new TextDecoder().decode(files.find(f => f.name === 'gt/orthographic.png').bytes), 'png');
  assert.equal(new TextDecoder().decode(files.find(f => f.name === 'gt/coverage.png').bytes), 'mask');
  assert.equal(json('generation/scene.json').groundTruth, undefined);
  assert.equal(json('generation/scene.json').cameraGroundTruth, undefined);
  assert.deepEqual(json('gt/camera_poses.json').frames.map(p => p.timestamp_s), scene.poses.map(p => p.timestamp));
  assert.deepEqual(json('gt/camera_poses.json').frames.map(p => p.image), ['rgb/0000.jpg', 'rgb/0001.jpg']);
  const csv = new TextDecoder().decode(files.find(f => f.name === 'gt/camera_poses.csv').bytes).trim().split('\n');
  assert.equal(csv.length, 3);
  assert.equal(csv[0].split(',').length, 16);
  assert.deepEqual(csv[1].split(',').slice(3, 9).map(Number), json('gt/camera_poses.json').frames[0].pose6d_m_rad);
  assert.throws(() => sequenceFiles({ ...scene, frames, groundTruth: fakeGT(scene), cameraGroundTruth: cameraPoseGroundTruth([]) }), /帧数不一致/);
});

test('stock plan has repeated SKU blocks, consistent physical packs, depth copies and empty slots', () => {
  const plan = createSweepPlan(42, { shelfLength: 6, bayLayers: [3, 4, 5], stock: { emptyRate: 0.25, facingsMin: 3, facingsMax: 3, depthCopies: 3 } });
  const { catalog, slots, products } = plan.shelf.merchandising;
  assert.ok(slots.some(s => s.stock === 0));
  assert.ok(slots.some(s => s.stock > 1));
  assert.equal(products.length, slots.reduce((n, s) => n + s.stock, 0));
  for (const p of products) {
    const sku = catalog.find(s => s.id === p.skuId);
    const slot = slots.find(s => s.id === p.slotId);
    assert.deepEqual([p.width, p.height, p.depth], [sku.width, sku.height, sku.depth]);
    assert.equal(slot.skuId, sku.id);
    assert.equal(p.shape, sku.shape);
    assert.ok(slot.stock > 0 && p.depthIndex < slot.capacity);
  }
  let repeats = 0;
  for (let i = 1; i < slots.length; i++) {
    const a = slots[i - 1], b = slots[i];
    if (a.bay === b.bay && a.layer === b.layer && a.side === b.side && a.block === b.block) {
      assert.equal(a.skuId, b.skuId);
      assert.equal(b.col, a.col + 1);
      repeats++;
    }
  }
  assert.ok(repeats > slots.length / 2);
  assert.deepEqual(plan, createSweepPlan(42, plan.options));
});

test('mixed package shapes vary widths within the old limit and fit shelf and depth capacity', () => {
  for (const depthCopies of [1, 3, 4]) {
    const plan = createSweepPlan(42, { shelfLength: 6, bayLayers: [2, 5, 8], stock: { depthCopies, emptyRate: 0 } });
    const { catalog, products } = plan.shelf.merchandising;
    assert.deepEqual(new Set(catalog.map(s => s.shape)), new Set(['box', 'bottle', 'can', 'jar']));
    assert.deepEqual(new Set(products.map(s => s.shape)), new Set(['box', 'bottle', 'can', 'jar']));
    const bay = plan.shelf.bays[0], columns = Math.round(bay.width / 0.23);
    const oldMax = (bay.width - Math.min(0.08, bay.width * 0.15)) / columns * 0.88;
    const maxDepth = (plan.shelf.depth / 2 - 0.055 - 0.012 * (depthCopies - 1)) / depthCopies;
    assert.ok(catalog.every(s => s.width <= oldMax && s.depth <= maxDepth));
    assert.ok(Math.max(...catalog.map(s => s.width)) / Math.min(...catalog.map(s => s.width)) > 1.6);
    for (const s of catalog.filter(s => s.shape !== 'box')) assert.equal(s.width, s.depth);
    for (const p of products) {
      const b = plan.shelf.bays[p.bay];
      assert.ok(p.x - p.width / 2 > b.x - b.width / 2 && p.x + p.width / 2 < b.x + b.width / 2);
      assert.ok(Math.abs(p.z) + p.depth / 2 < plan.shelf.depth / 2);
      const nextBoard = b.layerHeights[p.layer + 1];
      const ceiling = nextBoard === undefined ? plan.shelf.height : nextBoard - 0.03;
      assert.ok(p.y + p.height / 2 < ceiling);
    }
    const rows = new Map();
    for (const p of products) {
      const key = p.slotId + '/stack-' + p.stackIndex;
      const previous = rows.get(key);
      if (previous) assert.ok(Math.abs(p.z - previous.z) >= p.depth + 0.0119);
      rows.set(key, p);
    }
  }
});

test('stacking uses local bay clearance, preserves support and combines vertical and depth capacities', () => {
  const plan = createSweepPlan(42, { shelfLength: 6, bayLayers: [2, 5, 8], stock: { emptyRate: 0, depthCopies: 3 } });
  const { catalog, slots, products } = plan.shelf.merchandising;
  assert.ok(products.some(p => p.stackIndex > 0));
  assert.equal(new Set(products.map(p => p.id)).size, products.length);
  const byPosition = new Map(products.map(p => [`${p.slotId}/${p.depthIndex}/${p.stackIndex}`, p]));
  for (const slot of slots) {
    const sku = catalog.find(s => s.id === slot.skuId), bay = plan.shelf.bays[slot.bay];
    const nextBoard = bay.layerHeights[slot.layer + 1];
    const ceiling = nextBoard === undefined ? plan.shelf.height : nextBoard - 0.03;
    const expected = sku.stackable ? Math.floor((ceiling - slot.boardY - 0.035) / sku.height + 1e-10) : 1;
    assert.equal(slot.stackCapacity, expected);
    assert.equal(slot.capacity, slot.depthCapacity * slot.stackCapacity);
    assert.equal(slot.depthCapacity, 3);
    assert.ok(slot.stock <= slot.capacity);
  }
  for (const p of products) {
    const slot = slots.find(s => s.id === p.slotId);
    assert.ok(p.stackIndex < slot.stackCapacity && p.depthIndex < slot.depthCapacity);
    const bay = plan.shelf.bays[p.bay], nextBoard = bay.layerHeights[p.layer + 1];
    const ceiling = nextBoard === undefined ? plan.shelf.height : nextBoard - 0.03;
    assert.ok(p.y + p.height / 2 <= ceiling - 0.0049);
    if (p.shape === 'bottle') assert.equal(p.stackIndex, 0);
    if (p.stackIndex) {
      const below = byPosition.get(`${p.slotId}/${p.depthIndex}/${p.stackIndex - 1}`);
      assert.ok(below, 'stacked packages must have a supporting package');
      assert.equal(below.skuId, p.skuId);
      assert.ok(Math.abs(p.y - below.y - p.height) < 1e-10);
      assert.deepEqual([p.x, p.z], [below.x, below.z]);
    } else assert.ok(Math.abs(p.y - p.height / 2 - slot.boardY - 0.03) < 1e-10);
  }
});

test('package meshes respect exported bounds and curved labels sit on their own bodies', () => {
  const { catalog } = createSweepPlan(42, { shelfLength: 6 }).shelf.merchandising;
  for (const sku of catalog) {
    const template = productTemplate(sku);
    const bounds = new THREE.Box3().setFromObject(template), size = bounds.getSize(new THREE.Vector3());
    for (const [actual, expected] of [[size.x, sku.width], [size.y, sku.height], [size.z, sku.depth]])
      assert.ok(Math.abs(actual - expected) < 1e-7);
    assert.ok(bounds.getCenter(new THREE.Vector3()).length() < 1e-7);
    assert.equal(template.children.length, sku.shape === 'box' ? 1 : sku.shape === 'bottle' ? 4 : 3);
    const material = new THREE.MeshBasicMaterial();
    for (const side of [-1, 1]) {
      const label = productLabel(sku, material, side);
      if (sku.shape === 'box') {
        assert.equal(label.geometry.type, 'PlaneGeometry');
        assert.ok(Math.abs(label.position.z - side * (sku.depth / 2 + 0.0004)) < 1e-10);
      } else {
        assert.equal(label.geometry.type, 'CylinderGeometry');
        const top = label.position.y + label.geometry.parameters.height / 2;
        const bodyTop = (sku.shape === 'bottle' ? 0.68 : sku.shape === 'jar' ? 0.84 : 0.97) * sku.height - sku.height / 2;
        assert.ok(top < bodyTop);
        assert.ok(label.geometry.parameters.radiusTop > sku.width / 2 * (sku.shape === 'can' ? 0.97 : 1));
      }
      label.geometry.dispose();
    }
    template.traverse(o => { if(o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
    material.dispose();
  }
});

test('all-empty and no-empty stock extremes are supported without changing camera poses', () => {
  const options = { shelfLength: 4, bayLayers: [3, 5], view: { mode: 'handheld' } };
  const full = createSweepPlan(42, { ...options, stock: { emptyRate: 0 } });
  const empty = createSweepPlan(42, { ...options, stock: { emptyRate: 1 } });
  assert.ok(full.shelf.merchandising.slots.every(s => s.stock >= 1));
  assert.ok(empty.shelf.merchandising.slots.every(s => s.stock === 0));
  assert.equal(empty.shelf.merchandising.products.length, 0);
  assert.deepEqual(full.poses, empty.poses);
  for (const stock of [{ emptyRate: -0.1 }, { emptyRate: 1.1 }, { facingsMin: 4, facingsMax: 2 }, { depthCopies: 5 }, { depthCopies: 1.5 }])
    assert.throws(() => createSweepPlan(42, { stock }));
});

test('width-based packing keeps small gaps while empty slots retain their planned positions', () => {
  for (const seed of [0, 42, 2026]) {
    const options = { shelfLength: 6, bayLayers: [2, 5, 8] };
    const full = createSweepPlan(seed, { ...options, stock: { emptyRate: 0 } });
    const partial = createSweepPlan(seed, { ...options, stock: { emptyRate: 0.35 } });
    const empty = createSweepPlan(seed, { ...options, stock: { emptyRate: 1 } });
    const footprint = plan => plan.shelf.merchandising.slots.map(({ stock, ...slot }) => slot);
    assert.deepEqual(footprint(full), footprint(partial));
    assert.deepEqual(footprint(full), footprint(empty));
    assert.ok(partial.shelf.merchandising.slots.some(s => s.stock === 0));
    assert.ok(partial.shelf.merchandising.slots.some(s => s.stock > 0));
    const rows = new Map();
    for (const slot of full.shelf.merchandising.slots) {
      const key = `${slot.bay}/${slot.layer}/${slot.side}`;
      if (!rows.has(key)) rows.set(key, []);
      rows.get(key).push(slot);
      const sku = full.shelf.merchandising.catalog.find(s => s.id === slot.skuId);
      assert.equal(slot.width, sku.width);
    }
    for (const row of rows.values()) {
      const bay = full.shelf.bays[row[0].bay];
      assert.ok(row.reduce((sum, s) => sum + s.width, 0) > bay.width * 0.8);
      for (let i = 1; i < row.length; i++) {
        const gap = row[i].x - row[i].width / 2 - row[i - 1].x - row[i - 1].width / 2;
        assert.ok(Math.abs(gap - PRODUCT_GAP) < 1e-10, 'adjacent planned packages should be 4 mm apart');
      }
    }
    for (const p of partial.shelf.merchandising.products) {
      const slot = partial.shelf.merchandising.slots.find(s => s.id === p.slotId);
      assert.equal(p.x, slot.x);
      assert.ok(slot.stock > 0);
    }
  }
});

test('handheld capture is smooth, bounded and exports its real nonuniform sample times', () => {
  const plan = createSweepPlan(42, { frames: 48, shelfLength: 6, view: { mode: 'handheld', jitter: 1.5 } });
  assert.deepEqual(plan, createSweepPlan(42, plan.options));
  assert.equal(plan.poses[0].timestamp, 0);
  const intervals = [];
  for (const [i, p] of plan.poses.entries()) {
    assert.ok(Math.abs(p.y - plan.shelf.height * 0.51) <= 0.025);
    assert.ok(Math.abs(p.yawDegrees) <= 1.5 && Math.abs(p.pitchDegrees) <= 1.5 && Math.abs(p.rollDegrees) <= 0.7);
    if (i) {
      const previous = plan.poses[i - 1];
      const dt = p.timestamp - previous.timestamp;
      assert.ok(dt >= 0.45 - 1e-12 && dt <= 0.55 + 1e-12);
      assert.ok(Math.abs(p.yawDegrees - previous.yawDegrees) < 1);
      intervals.push(dt);
    }
  }
  assert.ok(new Set(intervals).size > 1);
  assert.ok(new Set(plan.poses.map(p => p.y)).size > 1);
  const fakeFrames = plan.poses.map(() => 'data:image/jpeg;base64,' + btoa('jpeg'));
  const files = sequenceFiles({ ...plan, frames: fakeFrames, groundTruth: fakeGT(plan), cameraGroundTruth: fakeCameraGT(plan) });
  assert.deepEqual(JSON.parse(new TextDecoder().decode(files.find(f => f.name === 'timestamps.json').bytes)), plan.poses.map(p => p.timestamp));
});

test('automatic bays scale with actual length, repeat layer template and preserve manual layout', () => {
  for (const length of [1, 3, 6, 12]) {
    const plan = createSweepPlan(42, { shelfLength: length });
    assert.equal(plan.shelf.bays.length, length);
    assert.ok(plan.shelf.bays.every(b => b.width === 1));
    assert.deepEqual(plan.shelf.bays.map(b => b.layers), Array.from({ length }, (_, i) => [3, 4, 5][i % 3]));
    assert.deepEqual(plan, createSweepPlan(42, plan.options));
  }
  const custom = createSweepPlan(42, { shelfLength: 12, bayMode: 'auto', bayWidth: 0.5, bayLayers: [2, 6] });
  assert.equal(custom.shelf.bays.length, 24);
  assert.equal(custom.shelf.bays[23].layers, 6);
  assert.equal(createSweepPlan(42, { shelfLength: 12, bayLayers: [2, 5] }).shelf.bays.length, 2);
  for (const seed of [1, 42, 2026]) {
    const value = { min: 3.1, max: 7.9 }, length = resolveShelfLength(seed, value);
    const plan = createSweepPlan(seed, { shelfLength: value });
    assert.equal(plan.shelf.bays.length, Math.ceil(length));
    assert.equal(plan.shelf.width, length);
    assert.ok(plan.shelf.bays.every(b => b.width <= 1));
  }
  for (const options of [{ bayMode: 'other' }, { bayWidth: 0 }, { bayWidth: 3.1 }])
    assert.throws(() => resolveBayLayout(6, options));
});

function knownView(x = 0, y = 1, z = -2.325) {
  return { x, y, z, lookX: x, lookY: y, lookZ: z + 1, yawDegrees: 0, pitchDegrees: 0, rollDegrees: 0, timestamp: 0 };
}
function knownPlan(width = 4, poses = [knownView()]) {
  const plan = createSweepPlan(42, { shelfLength: width });
  return { ...plan, shelf: { ...plan.shelf, x: 0, z: 0, angle: 0 }, side: -1, poses };
}
test('orthographic GT crops to actual frustum coverage and exports an invertible metric map', () => {
  const gt = orthographicLayout(knownPlan());
  const halfWidth = 2 * Math.tan(27 * Math.PI / 180) * 0.75;
  assert.ok(Math.abs(gt.observedBounds.minX + halfWidth) < 1e-10);
  assert.ok(Math.abs(gt.observedBounds.maxX - halfWidth) < 1e-10);
  assert.equal(gt.observedBounds.minY, 0);
  assert.equal(gt.observedBounds.maxY, 2);
  assert.equal(gt.height, 960);
  assert.equal(gt.width, Math.ceil(2 * halfWidth * 480));
  const m = gt.shelfXYToImage;
  assert.ok(Math.abs(m[0][0] * gt.bounds.maxX + m[0][2]) < 1e-10);
  assert.ok(Math.abs(m[0][0] * gt.bounds.minX + m[0][2] - gt.width) < 1e-10);
  assert.ok(Math.abs(m[1][1] * gt.bounds.maxY + m[1][2]) < 1e-10);
  assert.equal(gt.referencePlane.z, -0.325);
});
test('GT combines every view, retains separated footprints and rejects absent reference-plane coverage', () => {
  const poses = [knownView(-4, 1, -0.825), knownView(4, 1, -0.825)];
  const gt = orthographicLayout(knownPlan(12, poses));
  assert.ok(gt.observedBounds.minX < -4 && gt.observedBounds.maxX > 4);
  assert.equal(gt.frames.length, 2);
  assert.ok(gt.frames[0].polygon.every(p => p.x < -3));
  assert.ok(gt.frames[1].polygon.every(p => p.x > 3));
  const away = { ...knownView(), lookZ: -3.325 };
  assert.throws(() => orthographicLayout(knownPlan(4, [away])), /没有覆盖/);
  assert.throws(() => orthographicLayout(knownPlan(4, [])), /没有覆盖/);
  const tilted = { ...knownView(), lookX: 0.25, lookY: 0.8, rollDegrees: 12 };
  assert.notDeepEqual(orthographicLayout(knownPlan(4, [tilted])).frames[0].polygon, orthographicLayout(knownPlan()).frames[0].polygon);
});
test('GT coverage and metric mapping respect rotated shelves and opposite viewing sides', () => {
  const plan = knownPlan();
  const angle = 0.6, c = Math.cos(angle), s = Math.sin(angle);
  const rotated = { ...plan, shelf: { ...plan.shelf, x: 7, z: 9, angle }, poses: plan.poses.map(p => ({ ...p,
    x: 7 + p.x * c - p.z * s, z: 9 + p.x * s + p.z * c,
    lookX: 7 + p.lookX * c - p.lookZ * s, lookZ: 9 + p.lookX * s + p.lookZ * c })) };
  const original = orthographicLayout(plan), gt = orthographicLayout(rotated);
  for (const key of ['minX', 'maxX', 'minY', 'maxY']) assert.ok(Math.abs(gt.observedBounds[key] - original.observedBounds[key]) < 1e-10);
  const reverse = orthographicLayout({ ...plan, side: 1, poses: [{ ...knownView(), z: 2.325, lookZ: 1.325 }] });
  assert.equal(reverse.shelfXYToImage[0][0], 480);
  assert.equal(reverse.referencePlane.z, 0.325);
  assert.equal(reverse.width, original.width);
});

test('bays have independent board heights and products fit their own compartments', () => {
  const { shelf } = createSweepPlan(42, { shelfLength: 6, bayLayers: [2, 5, 8] });
  const bays = shelf.bays;
  assert.deepEqual(bays.map(b => b.layerHeights.length), [2, 5, 8]);
  assert.equal(bays.reduce((sum, b) => sum + b.width, 0), shelf.width);
  assert.equal(bays[0].x - bays[0].width / 2, -shelf.width / 2);
  assert.equal(bays.at(-1).x + bays.at(-1).width / 2, shelf.width / 2);
  assert.notEqual(bays[0].layerHeights[1], bays[1].layerHeights[1]);
  const products = shelfProducts(shelf, 42);
  for (const p of products) {
    const b = bays[p.bay];
    assert.ok(p.width > 0 && p.height > 0 && p.depth > 0);
    assert.ok(p.x - p.width / 2 > b.x - b.width / 2);
    assert.ok(p.x + p.width / 2 < b.x + b.width / 2);
    assert.ok(p.y - p.height / 2 >= b.layerHeights[p.layer] + 0.029);
    const nextBoard = b.layerHeights[p.layer + 1];
    assert.ok(p.y + p.height / 2 < (nextBoard === undefined ? shelf.height : nextBoard - 0.03));
    assert.ok(Math.abs(p.z) + p.depth / 2 < shelf.depth / 2);
  }
  for (const bayLayers of [[], [0], [9], [2.5], Array(25).fill(4)])
    assert.throws(() => createSweepPlan(42, { bayLayers }));
});

test('camera angles control actual look directions and seeded per-frame variation', () => {
  const options = { shelfLength: 6, frames: 12, view: { mode: 'fixed', yaw: 20, pitch: 10, jitter: 0 } };
  const plan = createSweepPlan(42, options);
  for (const p of plan.poses) {
    const dx = p.lookX - p.x, dy = p.lookY - p.y, dz = p.lookZ - p.z;
    const c = Math.cos(plan.shelf.angle), s = Math.sin(plan.shelf.angle);
    const localX = dx * c + dz * s, localZ = -dx * s + dz * c;
    assert.ok(Math.abs(Math.hypot(dx, dy, dz) - 1) < 1e-12);
    assert.ok(Math.abs(Math.atan2(localX, -plan.side * localZ) * 180 / Math.PI - 20) < 1e-10);
    assert.ok(Math.abs(-Math.asin(dy) * 180 / Math.PI - 10) < 1e-10);
  }
  const front = createSweepPlan(42, { ...options, view: { mode: 'fixed', yaw: 0, pitch: 0, jitter: 0 } });
  for (let i = 0; i < plan.poses.length; i++) {
    const p = plan.poses[i], f = front.poses[i];
    // Intersect both view rays with the shelf reference plane: yaw changes viewpoint,
    // while retaining the same horizontal sequence coverage.
    const planeZ = plan.shelf.z;
    const hitX = p.x + (p.lookX - p.x) * (planeZ - p.z) / (p.lookZ - p.z);
    assert.ok(Math.abs(hitX - f.x) < 1e-10);
  }
  const jittered = createSweepPlan(42, { ...options, view: { mode: 'fixed', yaw: 20, pitch: 10, jitter: 3 } });
  assert.deepEqual(jittered, createSweepPlan(42, jittered.options));
  assert.ok(new Set(jittered.poses.map(p => p.yawDegrees)).size > 1);
  for (const p of jittered.poses) {
    assert.ok(Math.abs(p.yawDegrees - 20) <= 3);
    assert.ok(Math.abs(p.pitchDegrees - 10) <= 3);
  }
  assert.deepEqual(jittered.poses, createSweepPlan(42, { ...jittered.options, bayLayers: [2, 8] }).poses);
  for (const view of [{ yaw: 26 }, { pitch: -21 }, { jitter: -1 }, { jitter: NaN }])
    assert.throws(() => createSweepPlan(42, { view }));
});
