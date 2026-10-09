import { readFile, writeFile, mkdir, mkdtemp, rename, lstat } from 'node:fs/promises';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
import { makeZip } from '../src/archive.ts';
import { help, parseCLI, resolveCLI } from './cli-options.mjs';

async function exists(path) { try { await lstat(path); return true; } catch (e) { if (e.code === 'ENOENT') return false; throw e; } }
async function main() {
  const values = parseCLI(process.argv.slice(2));
  if (values.help) { console.log(help); return; }
  const config = values.config ? JSON.parse(await readFile(values.config, 'utf8')) : {};
  const request = resolveCLI(values, config);
  const output = resolve(request.out);
  if (await exists(output)) throw Error(`输出目录已存在，不会覆盖：${output}`);
  const root = fileURLToPath(new URL('../', import.meta.url));
  let server, browser, staging;
  try {
    server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 0, strictPort: false, open: false } });
    await server.listen();
    const executablePath = request.browser ?? process.env.CHROME_PATH;
    browser = await puppeteer.launch({ ...(executablePath ? { executablePath } : { channel: 'chrome' }), headless: true, args: ['--enable-unsafe-swiftshader'], timeout: 30000 });
    const page = await browser.newPage();
    page.setDefaultTimeout(120000);
    const port = server.httpServer.address().port;
    await page.goto(`http://127.0.0.1:${port}/generate.html`);
    await page.waitForFunction(() => typeof window.generateShelfCLI === 'function');
    await mkdir(dirname(output), { recursive: true });
    staging = await mkdtemp(join(dirname(output), '.shelf-generating-'));
    const manifest = { schemaVersion: 1, status: 'running', renderer: 'Three.js / headless Chromium', seed: request.seed, count: request.count, options: request.options, sequences: [] };
    for (let i = 0; i < request.count; i++) {
      const seed = request.seed + i;
      const subdir = request.count === 1 ? '' : `seed-${seed}`;
      console.log(`生成 ${i + 1}/${request.count}，seed=${seed}`);
      const rendered = await page.evaluate(({ seed, options }) => window.generateShelfCLI(seed, options), { seed, options: request.options });
      const files = rendered.map(file => ({ name: file.name, bytes: Buffer.from(file.base64, 'base64') }));
      for (const file of files) {
        if (!/^(rgb\/\d{4}\.jpg|gt\/(orthographic\.png|coverage\.png|metadata\.json|camera_poses\.(json|csv))|generation\/(scene|config)\.json|(intrinsics|timestamps|capture)\.json)$/.test(file.name)) throw Error(`无效输出文件：${file.name}`);
        const path = join(staging, subdir, file.name);
        await mkdir(dirname(path), { recursive: true });
        await writeFile(path, file.bytes, { flag: 'wx' });
      }
      if (request.zip) await writeFile(join(staging, subdir, 'sequence.zip'), makeZip(files), { flag: 'wx' });
      const scene = JSON.parse(files.find(f => f.name === 'generation/scene.json').bytes.toString());
      const gt = JSON.parse(files.find(f => f.name === 'gt/metadata.json').bytes.toString());
      manifest.sequences.push({ seed, directory: subdir || '.', frames: scene.poses.length, bays: scene.shelf.bays.length, groundTruth: { image: gt.files.image, width: gt.width, height: gt.height, cameraPoses: "gt/camera_poses.json", cameraPosesCSV: "gt/camera_poses.csv" }, emptySlots: scene.shelf.merchandising.slots.filter(s => s.stock === 0).length, slots: scene.shelf.merchandising.slots.length });
      await writeFile(join(staging, 'manifest.json'), JSON.stringify(manifest, null, 2));
    }
    manifest.status = 'complete';
    await writeFile(join(staging, 'manifest.json'), JSON.stringify(manifest, null, 2));
    if (await exists(output)) throw Error(`输出目录已存在，不会覆盖：${output}`);
    await rename(staging, output);
    staging = undefined;
    console.log(`已生成全部数据：${output}`);
  } catch (error) {
    if (staging) {
      await writeFile(join(staging, 'FAILED.json'), JSON.stringify({ status: 'failed', reason: error.message }, null, 2));
      console.error(`未完成的数据保留于：${staging}`);
    }
    throw error;
  } finally { await browser?.close(); await server?.close(); }
}
main().catch(error => { console.error(`生成失败：${error.message}`); process.exitCode = 1; });
