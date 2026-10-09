import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCLI, resolveCLI } from '../scripts/cli-options.mjs';
import { createSweepPlan } from '../src/sweep.ts';
import { DEFAULT_OPTIONS } from '../src/defaults.ts';

test('web, CLI and core generation default to handheld with independent ±15 degree yaw/pitch', () => {
  const cli = resolveCLI(parseCLI(['--out', '/tmp/default-capture']));
  assert.deepEqual(cli.options.view, DEFAULT_OPTIONS.view);
  assert.deepEqual(cli.options.view, { mode: 'handheld', yaw: 0, pitch: 0, jitter: 15 });
  for (const options of [{ frames: 48 }, { ...cli.options, frames: 48 }]) {
    const plan = createSweepPlan(42, options);
    assert.deepEqual(plan.options.view, cli.options.view);
    for (const p of plan.poses) {
      assert.ok(p.yawDegrees >= -15 && p.yawDegrees <= 15);
      assert.ok(p.pitchDegrees >= -15 && p.pitchDegrees <= 15);
    }
    assert.ok(plan.poses.some(p => Math.abs(p.yawDegrees) > 10));
    assert.ok(plan.poses.some(p => Math.abs(p.pitchDegrees) > 10));
    assert.ok(plan.poses.some(p => p.yawDegrees !== p.pitchDegrees));
  }
  assert.throws(() => createSweepPlan(42, { view: { jitter: 16 } }));
});

test('CLI exposes web generation settings and deterministic batch parameters', () => {
  const args = parseCLI(['--out', '/tmp/test-shelf', '--seed', '100', '--count', '2', '--length', '3:8', '--frames', '8:16', '--layers', '2,5', '--capture', 'handheld', '--jitter', '1.5', '--empty', '0.25', '--facings', '2:4', '--depth', '3', '--zip']);
  const result = resolveCLI(args);
  assert.equal(result.seed, 100); assert.equal(result.count, 2); assert.equal(result.zip, true);
  assert.deepEqual(result.options.bayLayers, [2, 5]);
  assert.deepEqual(result.options.shelfLength, { min: 3, max: 8 });
  assert.equal(result.options.stock.emptyRate, 0.25);
  assert.notDeepEqual(createSweepPlan(result.seed, result.options).poses, createSweepPlan(result.seed + 1, result.options).poses);
});

test('web config round trips into CLI and explicit flags override only selected settings', () => {
  const plan = createSweepPlan(42, { shelfLength: 6, frames: 12, bayLayers: [3, 4, 5], view: { mode: 'handheld', pitch: 6 }, stock: { emptyRate: 0.25 } });
  const config = JSON.parse(JSON.stringify({ schemaVersion: 1, seed: 42, options: plan.options }));
  const same = resolveCLI(parseCLI(['--out', '/tmp/replay']), config);
  assert.deepEqual(createSweepPlan(same.seed, same.options), plan);
  const changed = resolveCLI(parseCLI(['--out', '/tmp/changed', '--seed', '43', '--frames', '16']), config);
  assert.equal(changed.seed, 43); assert.equal(changed.options.frames, 16);
  assert.deepEqual(changed.options.stock, config.options.stock);
  assert.deepEqual(changed.options.view, config.options.view);
});

test('invalid CLI/config values fail before rendering or writing data', () => {
  assert.throws(() => resolveCLI(parseCLI([])));
  for (const flags of [['--seed', '4294967295', '--count', '2'], ['--frames', 'bad'], ['--count', '0'], ['--layers', '2,,4'], ['--capture', 'other'], ['--empty', '2'], ['--facings', '5:2'], ['--length', '1:2:3']])
    assert.throws(() => resolveCLI(parseCLI(['--out', '/tmp/new', ...flags])));
  assert.throws(() => parseCLI(['--unknown', '1']));
  for (const config of [{ schemaVersion: 2 }, { options: null }, { options: { mismatch: 1 } }, { options: { stock: null } }, { options: { view: { what: 1 } } }])
    assert.throws(() => resolveCLI(parseCLI(['--out', '/tmp/new']), config));
});

test('CLI auto count follows length and explicit layer lists retain manual compatibility', () => {
  const run = flags => resolveCLI(parseCLI(['--out', '/tmp/auto', '--length', '6', ...flags]));
  const auto = run([]);
  assert.equal(createSweepPlan(auto.seed, auto.options).shelf.bays.length, 6);
  const pattern = run(['--layers', '2,5', '--bay-mode', 'auto', '--bay-width', '0.5']);
  assert.deepEqual(createSweepPlan(pattern.seed, pattern.options).shelf.bays.map(b => b.layers), Array.from({ length: 12 }, (_, i) => i % 2 ? 5 : 2));
  const manual = run(['--layers', '2,5']);
  assert.equal(createSweepPlan(manual.seed, manual.options).shelf.bays.length, 2);
  for (const flags of [['--bay-mode', 'bad'], ['--bay-width', '0.1']]) assert.throws(() => run(flags));
});
