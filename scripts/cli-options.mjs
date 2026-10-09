import { parseArgs } from 'node:util';
import { DEFAULT_OPTIONS, DEFAULT_SEED } from '../src/defaults.ts';
import { createSweepPlan } from '../src/sweep.ts';

export const help = `生成货架 RGB、正交 GT、覆盖掩码、相机真实世界 6D 位姿 GT（JSON/CSV）、内参、SKU 陈列和空位数据（需要本机 Chrome/Chromium）

npm run generate -- --out ./output --seed 42 --length 6 --frames 12 --bay-mode auto --layers 3,4,5
  --config FILE       复用网页下载的 shelf-config.json 或 generation/config.json
  --out DIR           新的输出目录；已有目录不会覆盖（必填）
  --seed N            场景种子，默认 42
  --count N           连续种子批量生成，默认 1，最多 1000
  --length N|MIN:MAX  货架长度，默认 2:8 模拟米
  --frames N|MIN:MAX  图片数量，默认 9
  --bay-mode MODE     auto 随长度伸缩（默认）或 manual 手动数量
  --bay-width N       自动模式目标宽度 0.5–3 模拟米，默认 1
  --layers 3,4,5      层板配置；单独指定时使用手动数量，配合 --bay-mode auto 则循环填充
  --capture MODE      handheld 仿真拍摄（默认）或 fixed 固定路线
  --yaw N --pitch N  偏航、俯仰角（度），默认 0
  --jitter N          偏航、俯仰扰动幅度 0–15°，默认各 ±15°
  --empty N           空位概率 0–1，默认 0.18
  --facings MIN:MAX   同款连续列数，默认 2:5
  --depth N           每列纵深容量 1–4，默认 3
  --zip               同时保存每组的 sequence.zip
  --browser FILE      Chrome/Chromium 可执行文件；也可设置 CHROME_PATH
  --help              显示帮助
`;

export function parseCLI(args) {
  return parseArgs({ args, options: {
    ...Object.fromEntries(['config', 'out', 'seed', 'count', 'length', 'frames', 'layers', 'bay-mode', 'bay-width', 'capture', 'yaw', 'pitch', 'jitter', 'empty', 'facings', 'depth', 'browser'].map(key => [key, { type: 'string' }])),
    zip: { type: 'boolean', default: false }, help: { type: 'boolean', default: false },
  }}).values;
}

function number(text, label) {
  if (typeof text !== 'string' || text.trim() === '' || !Number.isFinite(Number(text))) throw Error(`${label}必须是有效数字`);
  return Number(text);
}
function range(text, label) {
  const values = text.split(':').map(v => number(v, label));
  if (values.length === 1) return values[0];
  if (values.length === 2) return { min: values[0], max: values[1] };
  throw Error(`${label}使用数值或 MIN:MAX`);
}
export function resolveCLI(values, config = {}) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) throw Error('生成配置必须为 JSON 对象');
  if (config.schemaVersion !== undefined && config.schemaVersion !== 1) throw Error('不支持此生成配置版本');
  const supplied = config.options === undefined ? {} : config.options;
  if (!supplied || typeof supplied !== 'object' || Array.isArray(supplied)) throw Error('options 必须为对象');
  for (const key of Object.keys(supplied))
    if (!['shelfLength', 'frames', 'bayMode', 'bayWidth', 'bayLayers', 'view', 'stock'].includes(key)) throw Error(`未知生成参数：${key}`);
  for (const [key, allowed] of [['view', ['mode', 'yaw', 'pitch', 'jitter']], ['stock', ['emptyRate', 'facingsMin', 'facingsMax', 'depthCopies']]]) {
    if (supplied[key] !== undefined && (!supplied[key] || typeof supplied[key] !== 'object' || Array.isArray(supplied[key]))) throw Error(`${key} 必须为对象`);
    for (const field of Object.keys(supplied[key] ?? {})) if (!allowed.includes(field)) throw Error(`未知生成参数：${key}.${field}`);
  }
  const options = { ...DEFAULT_OPTIONS, ...supplied, view: { ...DEFAULT_OPTIONS.view, ...supplied.view }, stock: { ...DEFAULT_OPTIONS.stock, ...supplied.stock } };
  if (supplied.bayLayers !== undefined && supplied.bayMode === undefined) options.bayMode = 'manual';
  const seed = values.seed === undefined ? (config.seed ?? config.store?.seed ?? DEFAULT_SEED) : number(values.seed, '种子');
  const count = values.count === undefined ? 1 : number(values.count, '批量数量');
  if (!Number.isInteger(count) || count < 1 || count > 1000) throw Error('批量数量必须为 1–1000 的整数');
  if (seed + count - 1 > 4294967295) throw Error('批量种子超出 32 位范围');
  if (values.length !== undefined) options.shelfLength = range(values.length, '货架长度');
  if (values.frames !== undefined) options.frames = range(values.frames, '图片张数');
  if (values.layers !== undefined) {
    options.bayLayers = values.layers.split(',').map(v => number(v, '层板数量'));
    options.bayMode = 'manual';
  }
  if (values['bay-mode'] !== undefined) options.bayMode = values['bay-mode'];
  if (values['bay-width'] !== undefined) options.bayWidth = number(values['bay-width'], '目标子货架宽度');
  if (values.capture !== undefined) options.view.mode = values.capture;
  for (const key of ['yaw', 'pitch', 'jitter']) if (values[key] !== undefined) options.view[key] = number(values[key], key);
  if (values.empty !== undefined) options.stock.emptyRate = number(values.empty, '空位概率');
  if (values.depth !== undefined) options.stock.depthCopies = number(values.depth, '纵深容量');
  if (values.facings !== undefined) {
    const facings = range(values.facings, '连续列数');
    options.stock.facingsMin = typeof facings === 'number' ? facings : facings.min;
    options.stock.facingsMax = typeof facings === 'number' ? facings : facings.max;
  }
  createSweepPlan(seed, options); // Validate before starting Chromium or creating output files.
  if (!values.out?.trim()) throw Error('请使用 --out 指定新的输出目录');
  return { seed, count, options, out: values.out, zip: values.zip, browser: values.browser };
}
