import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => a.replace(/^--/, "").split("=")),
);
const seed = Number(args.seed || 42),
  count = Number(args.count || 1),
  output = path.resolve(args.output || "examples/simulated");
if (
  !Number.isInteger(seed) ||
  seed < 0 ||
  seed > 4294967295 ||
  !Number.isInteger(count) ||
  count < 1 ||
  count > 500
)
  throw Error("seed 必须是 32 位非负整数，count 必须为 1–500");
const rangeOrNumber = (key, lower, upper) =>
  args[key] === "random"
    ? {
        min: Number(args[`${key}-min`] ?? lower),
        max: Number(args[`${key}-max`] ?? upper),
      }
    : Number(args[key]);
const options =
  args.length !== undefined || args.frames !== undefined
    ? {
        ...(args.length !== undefined
          ? { shelfLength: rangeOrNumber("length", 2, 8) }
          : {}),
        frames: args.frames !== undefined ? rangeOrNumber("frames", 8, 16) : 9,
      }
    : 9;
const browser = await chromium.launch({
  headless: true,
  channel: "chrome",
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
try {
  await page.goto(args.url || "http://127.0.0.1:5180");
  await page.waitForFunction(() => window.shelfStitch);
  for (let i = 0; i < count; i++) {
    const sample = await page.evaluate(
      ({ seed, options }) => window.shelfStitch.generate(seed, options),
      { seed: seed + i, options },
    );
    const dest = count === 1 ? output : path.join(output, String(seed + i));
    const previous = await fs
      .readFile(path.join(dest, "generation/scene.json"), "utf8")
      .then(JSON.parse)
      .catch((e) => {
        if (e.code === "ENOENT") return null;
        throw e;
      });
    const existing = await fs.readdir(path.join(dest, "rgb")).catch((e) => {
      if (e.code === "ENOENT") return [];
      throw e;
    });
    if (
      existing.length &&
      (!previous ||
        ["store", "shelf", "poses", "side", "distance"].some(
          (k) => JSON.stringify(previous[k]) !== JSON.stringify(sample[k]),
        ) ||
        existing.filter((f) => f.endsWith(".jpg")).length !==
          sample.frames.length)
    )
      throw Error(
        "已有输出目录的场景或图片数量不同，请指定新的 --output；原始图片未覆盖",
      );
    await fs.mkdir(path.join(dest, "rgb"), { recursive: true });
    await fs.mkdir(path.join(dest, "generation"), { recursive: true });
    for (let j = 0; j < sample.frames.length; j++)
      await fs.writeFile(
        path.join(dest, "rgb", `${String(j).padStart(4, "0")}.jpg`),
        Buffer.from(sample.frames[j].split(",")[1], "base64"),
      );
    const { frames, ...metadata } = sample;
    await fs.writeFile(
      path.join(dest, "generation", "scene.json"),
      JSON.stringify(metadata, null, 2),
    );
    console.log(
      JSON.stringify({
        seed: seed + i,
        frames: sample.frames.length,
        shelf: sample.shelf.id,
        shelf_length: sample.shelf.width,
        estimated_overlap: sample.estimatedOverlap,
        output: dest,
      }),
    );
  }
  await page.screenshot({
    path: path.join(output, "../simulator.png"),
    fullPage: true,
  });
} finally {
  await browser.close();
}
