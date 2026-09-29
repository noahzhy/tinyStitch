import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
const out = path.resolve(
  process.env.tinyStitch_QA_OUTPUT || "examples/validation",
);
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  channel: "chrome",
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
const page = await browser.newPage({
  baseURL: "http://127.0.0.1:5180",
  viewport: { width: 1440, height: 1080 },
});
const errors = [],
  requests = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("request", (r) => {
  if (r.url().endsWith("/api/stitch")) requests.push(JSON.parse(r.postData()));
});
try {
  await page.goto("http://127.0.0.1:5180");
  await page.waitForFunction(() => window.tinyStitch);
  await page.getByText("拼接服务就绪").waitFor();
  await page.getByLabel("货架长度模式").selectOption("custom");
  await page.getByLabel("货架长度", { exact: true }).fill("6");
  await page.getByLabel("图片张数", { exact: true }).fill("12");
  await page.getByRole("button", { name: "生成拍摄", exact: true }).click();
  assert.equal(await page.locator(".thumbs button").count(), 12);
  assert.match(await page.locator(".source-note").textContent(), /6\.00/);
  const generated = await page.locator(".photo-stage img").getAttribute("src");
  await page.getByRole("button", { name: "生成拍摄", exact: true }).click();
  assert.equal(
    await page.locator(".photo-stage img").getAttribute("src"),
    generated,
  );
  await page.getByRole("button", { name: "生成完整拼图" }).click();
  await page.getByAltText("自动拼接的完整货架图片").waitFor({ timeout: 60000 });
  const first = await page
    .getByAltText("自动拼接的完整货架图片")
    .getAttribute("src");
  let job = await (
    await page.request.get(
      first.replace("/files/", "/jobs/").split("/panorama.png")[0],
    )
  ).json();
  assert.equal(job.status, "completed");
  assert.equal(job.result.report.input_count, 12);
  assert.equal(job.result.report.method, "jepa");
  assert.ok(job.result.report.model.trained_steps > 0);
  assert.equal(job.result.report.model.checkpoint_sha256.length, 64);
  assert.ok(job.result.report.size[0] > 720);
  assert.deepEqual(Object.keys(requests[0]), ["rgb_id", "method"]);
  const png = await page.request.get(first);
  assert.equal(
    (await png.body()).subarray(0, 8).toString("hex"),
    "89504e470d0a1a0a",
  );
  await page.screenshot({
    path: path.join(out, "desktop.png"),
    fullPage: true,
  });
  await page.getByLabel("场景种子").fill("2026");
  await page.getByLabel("货架长度模式").selectOption("random");
  await page.getByLabel("货架长度下限").fill("3");
  await page.getByLabel("货架长度上限").fill("4");
  await page.getByLabel("图片张数模式").selectOption("random");
  await page.getByLabel("图片张数下限").fill("8");
  await page.getByLabel("图片张数上限").fill("12");
  await page.getByRole("button", { name: "生成拍摄", exact: true }).click();
  const randomCount = await page.locator(".thumbs button").count();
  assert.ok(randomCount >= 8 && randomCount <= 12);
  const randomLength = Number(
    (await page.locator(".source-note").textContent()).match(
      /长度\s*(\d+\.\d+)/,
    )[1],
  );
  assert.ok(randomLength >= 3 && randomLength <= 4);
  await page.getByRole("button", { name: "生成完整拼图" }).click();
  await page.getByAltText("自动拼接的完整货架图片").waitFor({ timeout: 60000 });
  const second = await page
    .getByAltText("自动拼接的完整货架图片")
    .getAttribute("src");
  assert.notEqual(first, second);
  const secondJob = await (
    await page.request.get(
      second.replace("/files/", "/jobs/").split("/panorama.png")[0],
    )
  ).json();
  assert.notEqual(
    secondJob.result.report.input_digest,
    job.result.report.input_digest,
  );
  assert.equal(secondJob.result.report.input_count, randomCount);
  // Export generated RGB, then import through the actual file picker and re-run stitching.
  const zip = await page.request.get(
    `/api/inputs/${requests[1].rgb_id}/download`,
  );
  assert.equal((await zip.body()).subarray(0, 2).toString(), "PK");
  const source = path.resolve("examples/simulated/rgb");
  await page
    .getByLabel("导入拍摄图片", { exact: true })
    .setInputFiles(
      (await fs.readdir(source))
        .filter((f) => f.endsWith(".jpg"))
        .map((f) => path.join(source, f)),
    );
  await page.getByText("已按文件名数字顺序导入。", { exact: false }).waitFor();
  await page.getByRole("button", { name: "生成完整拼图" }).click();
  await page.getByAltText("自动拼接的完整货架图片").waitFor({ timeout: 60000 });
  assert.equal(requests.length, 3);
  assert.ok(
    requests.every((r) => Object.keys(r).join(",") === "rgb_id,method"),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(out, "mobile.png"), fullPage: true });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  assert.deepEqual(errors, []);
  const result = {
    custom_and_random_generation: true,
    seed_reproducible: true,
    actual_stitch: true,
    changed_input: true,
    import_and_recompute: true,
    rgb_only_contract: true,
    mobile_no_overflow: true,
    console_errors: errors,
    sample_report: job.result.report,
  };
  await fs.writeFile(
    path.join(out, "e2e.json"),
    JSON.stringify(result, null, 2),
  );
  console.log(
    JSON.stringify({
      ...result,
      sample_report: {
        size: job.result.report.size,
        seconds: job.result.report.seconds,
      },
    }),
  );
} finally {
  await browser.close();
}
