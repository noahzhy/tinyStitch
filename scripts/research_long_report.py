"""Regenerate the long-shelf two-stage report from saved experiment artifacts."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from backend.research.metrics import image_metrics

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/research"
OUT = ART / "long-shelf-selected-v2"
read = lambda p: json.loads(p.read_text())


def neural(group, name, n):
    folder = ART / group / name / str(n)
    report = read(folder / "report.json")
    target = (
        ROOT
        / "data/research"
        / ("long-shelf-native60-v1" if "native" in name else f"long-shelf{n}-v1")
        / str(report["scene_seed"])
    )
    gt = np.asarray(Image.open(folder / "gt.png").convert("RGB"))
    depth = np.load(target / "orthographic/geometry.npz")["depth"]
    for row in report["stage2"]:
        rgb = np.asarray(Image.open(folder / (row["mode"] + "-rgb.png")).convert("RGB"))
        row["quality_png"] = image_metrics(rgb, gt, depth)
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    (folder.parent / "summary.json").write_text(json.dumps([report], indent=2))
    return report, folder


old = {n: neural("long-shelf-selected-v2", f"{n}-local", n) for n in (10, 30, 60)}
new = {n: neural("long-shelf-render-v1", f"{n}-local", n) for n in (10, 30, 60)}
global12 = neural("long-shelf-selected-v2", "60-global12", 60)[0]
native_old = neural("long-shelf-selected-v2", "60-native-local", 60)
native_new = neural("long-shelf-render-v1", "60-native-local", 60)
stage1 = read(OUT / "stage1.json")
text = """# 长货架：第一阶段位姿 → 第二阶段正交渲染与估计单应矩阵对照

日期：2026-10-07。**已修正“帧数只增加采样密度”的实验设定，完成长货架训练及实际图像比较。位姿改善；渲染仍有噪声，不宣称全面优于单应拼接或通过产线验收。**

## 输入与数据协议

| 输入数 | 货架长度 | 拍摄跨度 | 每层商品数 | 输入分辨率 | 正交GT宽度 |
|---|---:|---:|---:|---|---:|
| 10 | 3.86m | 1.26m | 15 | 256×192 | 256 |
| 30 | 6.66m | 4.06m | 27 | 256×192 | 512 |
| 60 | 10.86m | 8.26m | 43 | 256×192，另有1280×960对照 | 768 |

长度是模拟器单位，固定相邻相机间距0.14m、相机距离2.6m、54°源水平视场；真实推理不恢复米制尺度。商品尺寸基本固定，通过增加列数延长货架，未把原有商品横向拉伸。RGB和正交GT来自同一不透明三维盒体场景，保留层板遮挡；原生照片从几何直接重渲染，不是256图片放大。

训练仅用60张长货架732000–732003，验证732004；独立测试732005，以及另行生成的短/中货架734005、733005。10/30/60不再是同一货架的嵌套子集；每种长度只测一个独立货架，不能当作产线失败率或泛化统计。按长度扩大GT输出宽度，避免把长货架强制压成256宽。

第一阶段实际工作尺寸256×192。原生1280×960的颜色采样保留原图，空间VAE仍在256×192编码。新位姿模型训练的是固定间距、近正面横向拍摄，不能据此声称对任意六自由度轨迹有效。

## 第一阶段：相对位置

保持JEPA、深度、VAE和原渲染场不变，仅训练关系位姿头。6000步、seed20261007。V1使用最大间距8的图像对，相邻尺度严重偏大；V2限制为间距1/2，与推理邻接图一致。两者从同一原检查点初始化，属于训练配对策略对照。V2验证选5400步。

| 输入数 | 原跨度/GT | 新跨度/GT | 原中心平均误差 | 新中心平均误差 | 原最大旋转误差 | 新最大旋转误差 | 深度AbsRel |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
for n in (10, 30, 60):
    a = next(
        x
        for x in stage1
        if x["input_count"] == n and "render-multicount" in x["checkpoint"]
    )
    b = next(
        x for x in stage1 if x["input_count"] == n and "selected" in x["checkpoint"]
    )
    u, v = a["pose"], b["pose"]
    text += f"| {n} | {u['extent_ratio']:.3f} | {v['extent_ratio']:.3f} | {u['center_l2_mean']:.5f} | {v['center_l2_mean']:.5f} | {u['rotation_max_deg']:.2f}° | {v['rotation_max_deg']:.2f}° | {b['depth_abs_rel']*100:.2f}% |\n"
text += """
中心误差使用中间相机和源深度中位数定义的规范尺度，不是米。深度网络没有在本轮训练，上表深度保持原状。V1的60张测试跨度1.878倍、中心误差0.78646，是保留的失败候选。

验证场景V2的链式中心误差0.06228、位姿图0.11816，因此本轮选择关闭位姿图；未用测试结果选择。修复了训练代码在验证选择关闭时仍继承旧配置`pose_graph=True`的问题。原始强制图版本[long-pose-v2](../long-pose-v2/report.json)仍保留，选择后的检查点[long-pose-selected-v2](../long-pose-selected-v2/report.json)只有该配置更改，位姿头权重相同。强制图在60张测试中心误差更小，不改变基于验证的选择。

完整第一阶段结果：[stage1.json](stage1.json)。目标正交图没有进入第一阶段评测或拟合。长序列剩余旋转误差仍会影响商品边缘，不能仅因跨度接近1就宣称几何通过。

## 第二阶段：局部查询与长货架训练

新增`--local-sources 12`：以预测相机中心到目标正交光线的横向距离选择附近照片，按窄列分块查询；全部输入参与特征与位姿估计，单个渲染块最多12张。排序后还原像素顺序。它不使用目标RGB、目标深度或真值源相机挑选视图。匹配GT相机仅定义此处的评测光线。

渲染器再训练1800步、seed20261008，每训练场景固定2048个正交GT有效像素，源几何冻结为预测结果；局部查询缓存按目标列排序、64射线分块。训练使用训练货架的正交RGB/深度GT，验证GT选步800；测试货架目标没有加载进训练。验证缓存PSNR11.84→14.20 dB，是缓存像素上的选步值，不能当作全图或独立场景分数。新场与旧场有训练数据、采样和像素中心配置变化，因此整体比较不能归因于单一设计。

| 输入数 | 短货架训练场 PSNR | 长货架训练场 PSNR | 短场 SSIM | 长场 SSIM | 长场GT有效区域支持率 |
|---|---:|---:|---:|---:|---:|
"""
for n in (10, 30, 60):
    a, b = old[n][0]["stage2"][0], new[n][0]["stage2"][0]
    text += f"| {n} | {a['psnr_gt_valid']:.2f} | {b['psnr_gt_valid']:.2f} | {a['quality_png']['ssim_gt_valid']:.3f} | {b['quality_png']['ssim_gt_valid']:.3f} | {b['supported_fraction_gt_valid']*100:.1f}% |\n"
a = old[60][0]["stage2"][0]
b = global12["stage2"][0]
text += f"\n相同新位姿＋短场下，全序列均匀12张与局部12张的60输入对照：PSNR {b['psnr_gt_valid']:.2f}→{a['psnr_gt_valid']:.2f} dB，支持率 {b['supported_fraction_gt_valid']*100:.1f}%→{a['supported_fraction_gt_valid']*100:.1f}%。局部查询没有提升本例覆盖率，不能直接称其解决了长货架缺口。它提供了随场景位置切换照片的预算机制，仍需改进置信度与可见性选择。\n"
text += """
## 第二阶段：估计单应矩阵的真实图像比较

复用原`backend.stitch`的SIFT和现有JEPA描述子、双向ratio、RANSAC、相邻/近邻图、共享平面优化与中心接缝/窄羽化。研究包装允许60张，不改变现有网页API的48张上限。两种单应矩阵均只由同一组源RGB估计；没有使用GT单应、目标RGB对齐、人工裁剪或挑选最优测试变换。输入哈希已与新方法核对。

传统拼图原本是透视平面画布，不能直接与正交GT逐像素比较。这里公开采用中间源相机的**单个正面平面**适配到正交评测画布：平面深度取新方法预测源深度的中位数，两种方法共享同一GT目标相机及规范尺度。GT源深度只用于与NVS一致的评分尺度定义，不用于估计图像间H。这个适配明确保留传统方法的平面假设，不能将数字解释成原始拼图观感；原始透视拼图`native-mosaic.png`和适配结果均保存。它也不是无需标签的推理入口验收。

同一固定GT正深度区域评分，未按预测掩码删掉失败像素。下表所有PSNR及SSIM统一由8-bit PNG计算，避免把神经浮点分数与传统PNG分数混用。

| 输入数 | 方法 | PNG PSNR | SSIM | 边缘PNG PSNR | GT有效区域支持率 |
|---|---|---:|---:|---:|---:|
"""
for n in (10, 30, 60):
    for baseline in read(OUT / f"homography{n}/summary.json"):
        if baseline["status"] == "ok":
            q = baseline["metrics"]
            text += f"| {n} | {baseline['method'].upper()}估计H | {q['psnr_png_gt_valid']:.2f} | {q['ssim_gt_valid']:.3f} | {q['edge_psnr_png']:.2f} | {baseline['supported_fraction_gt_valid']*100:.1f}% |\n"
        else:
            text += f"| {n} | {baseline['method'].upper()}估计H | 失败 | — | — | — |\n"
    b = new[n][0]["stage2"][0]
    q = b["quality_png"]
    text += f"| {n} | 长货架神经渲染 | {q['psnr_png_gt_valid']:.2f} | {q['ssim_gt_valid']:.3f} | {q['edge_psnr_png']:.2f} | {b['supported_fraction_gt_valid']*100:.1f}% |\n"
# Galleries use full-frame outputs; fixed regions use normalized GT coordinates.
for n in (10, 30, 60):
    gt = old[n][1] / "gt.png"
    w, h = Image.open(gt).size
    panels = [
        ("Simulator orthographic GT", gt),
        (
            "Estimated SIFT homography",
            OUT / f"homography{n}/sift/orthographic-adapted-rgb.png",
        ),
        (
            "Estimated JEPA homography",
            OUT / f"homography{n}/jepa/orthographic-adapted-rgb.png",
        ),
        ("Neural: short-shelf field", old[n][1] / "predicted_all-rgb.png"),
        ("Neural: long-shelf field", new[n][1] / "predicted_all-rgb.png"),
    ]
    canvas = Image.new("RGB", (w, 5 * (h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(panels):
        draw.text((8, i * (h + 24) + 5), label, fill="black")
        if path.exists():
            canvas.paste(Image.open(path).convert("RGB"), (0, i * (h + 24) + 24))
    canvas.save(OUT / f"comparison-{n}.png")
    text += (
        f"\n### {n}张、不同货架长度的同范围输出\n\n![{n}张全图](comparison-{n}.png)\n"
    )
    crop_canvas = Image.new("RGB", (3 * 200, 5 * 120), "white")
    draw = ImageDraw.Draw(crop_canvas)
    for i, (label, path) in enumerate(panels):
        draw.text((5, i * 120 + 2), label, fill="black")
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        for j, x in enumerate((0.08, 0.44, 0.80)):
            crop = image.crop(
                (round(w * x), round(h * 0.18), round(w * (x + 0.12)), round(h * 0.72))
            )
            crop.thumbnail((195, 96), Image.Resampling.NEAREST)
            # Enlarge tiny crops to reveal actual artifacts without smoothing.
            scale = min(195 / crop.width, 96 / crop.height)
            crop = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.NEAREST,
            )
            crop_canvas.paste(crop, (j * 200, i * 120 + 22))
    crop_canvas.save(OUT / f"crops-{n}.png")
    text += f"\n![固定左中右商品和层板区域](crops-{n}.png)\n"
text += """
局部图显示：单应拼图商品纹理通常更清晰，但存在透视/层板错位和长序列累计形变；神经渲染仍存在密度选择噪声、错位和边缘伪影。PSNR增益不等于结构细节或观感全面获胜，应结合SSIM及图像检查。

## 60张1280×960原生输入控制

同一10.86m独立货架从模拟几何重渲染到原生输入，目标仍为768宽正交GT。几何工作尺寸256×192，输出不是1280像素级几何验收。

| 方法 | PNG PSNR | SSIM | 支持率 | 状态 |
|---|---:|---:|---:|---|
"""
for baseline in read(OUT / "homography60-native/summary.json"):
    if baseline["status"] == "ok":
        q = baseline["metrics"]
        text += f"| {baseline['method'].upper()}估计H | {q['psnr_png_gt_valid']:.2f} | {q['ssim_gt_valid']:.3f} | {baseline['supported_fraction_gt_valid']*100:.1f}% | 完成 |\n"
    else:
        text += f"| {baseline['method'].upper()}估计H | — | — | — | 匹配图断连 |\n"
for label, record in [("短货架训练场", native_old), ("长货架训练场", native_new)]:
    b = record[0]["stage2"][0]
    q = b["quality_png"]
    text += f"| {label} | {q['psnr_png_gt_valid']:.2f} | {q['ssim_gt_valid']:.3f} | {b['supported_fraction_gt_valid']*100:.1f}% | 完成 |\n"
text += """
现有JEPA在原生60张控制中未得到连接图，明确计为失败，没有用SIFT回退或伪造完整输出。单个失败例不是失败率统计。

## 资源、验证和复现

47项测试通过，包含商品尺寸/长度、固定拍摄间距、原生重渲染RGB及GT一致、局部源选择、列排序还原、缓存与显式子集一致及训练梯度、平面适配解析校验，以及原有API/几何/EMA/检查点测试。CPU并行实验的耗时和峰值RSS保存于每个report.json，不据此宣称加速；LPIPS、真实轨迹、多种曝光、多个独立场景与产线画质验收仍未完成。

模型选择只用验证货架，旧权重、失败候选与网页接口未替换。长货架渲染GT训练明确标记于[训练协议](../long-render-v1/report.json)。本轮新权重：`long-pose-selected-v2/model.pt`、`long-render-v1/model.pt`。训练从头建新头/新场，没有中断恢复实现。

使用新输出目录复现，已有模型不会被覆盖：

```bash
.venv/bin/python -m backend.research.cli generate --output data/research/long-shelf60-v1 --scenes 6 --seed 732000 --views 60 --capture-step .14 --target-width 768 --no-exposure-jitter
.venv/bin/python -m backend.research.cli generate --output data/research/long-shelf30-v1 --scenes 6 --seed 733000 --views 30 --capture-step .14 --target-width 512 --no-exposure-jitter
.venv/bin/python -m backend.research.cli generate --output data/research/long-shelf10-v1 --scenes 6 --seed 734000 --views 10 --capture-step .14 --target-width 256 --no-exposure-jitter
.venv/bin/python -m backend.research.cli pose-adapt --data data/research/long-shelf60-v1 --checkpoint artifacts/research/render-multicount-v3/model.pt --output artifacts/research/long-pose-reproduce --steps 6000 --max-pair-gap 2
# 修复后的训练代码自动按验证选择关闭位姿图，无需复制或人工修补检查点。
.venv/bin/python -m backend.research.cli render-adapt --data data/research/long-shelf60-v1 --checkpoint artifacts/research/long-pose-selected-v2/model.pt --output artifacts/research/long-render-reproduce --steps 1800 --rays 2048 --input-counts 60 --feature-center-mapping --local-sources 12
.venv/bin/python -m backend.research.cli multiview --scene data/research/long-shelf60-v1/732005 --checkpoint artifacts/research/long-render-v1/model.pt --output artifacts/research/long-render-eval-reproduce --width 768 --geometry-width 256 --samples 32 --counts 60 --local-sources 12
.venv/bin/python -m backend.research.cli homography-benchmark --scene data/research/long-shelf60-v1/732005 --reference artifacts/research/long-shelf-render-v1/60-local/60 --output artifacts/research/long-homography-reproduce
.venv/bin/python -m backend.research.cli generate-native --scene data/research/long-shelf60-v1/732005 --output data/research/long-shelf-native60-v1/732005 --width 1280 --height 960 --target-width 768
.venv/bin/python -m backend.research.long_geometryeval --data-roots data/research/long-shelf10-v1 data/research/long-shelf30-v1 data/research/long-shelf60-v1 --checkpoints artifacts/research/render-multicount-v3/model.pt artifacts/research/long-pose-selected-v2/model.pt --output artifacts/research/long-shelf-selected-v2/stage1.json
PYTHONPATH=. .venv/bin/python scripts/research_long_report.py
.venv/bin/python -m pytest -q
```

后续优先在源视图之间修正深度和遮挡可见性，并控制长序列的旋转漂移；先做真值源几何的渲染上限控制，再决定提高几何编码分辨率或更改密度网络。当前证据不支持只通过加深颜色网络来解决这些问题。
"""
(OUT / "REPORT.md").write_text(text)
print(OUT / "REPORT.md")

# Global sequence attention is selected on validation before these test reports.
global_files = [
    ART / "long-shelf-global-v1" / f"{n}-local" / str(n) / "report.json"
    for n in (10, 30, 60)
]
global_files.append(ART / "long-shelf-global-v1/60-native-local/60/report.json")
if all(p.exists() for p in global_files) and (OUT / "global-stage1.json").exists():
    rows = read(OUT / "global-stage1.json")
    global_text = "\n## 全局特征：减少位姿累计漂移的实际对照\n\n新增两层全序列Transformer（128维、4头），输入各视图JEPA token均值/方差、局部位姿、序列位置；所有10–60视图交互后输出受限位姿残差，固定中间相机。不是把远距无重叠图片强行做配对，也没有测试时读取GT或目标图。它是序列上下文的研究原型，尚未建立显式三维全局地标，序列位置先验也可能依赖本轮固定间距拍摄。\n\n只训练该全局模块，JEPA、局部位姿、深度、VAE、渲染场冻结；732000–003训练、732004验证，测试场景未加载进训练。连续10/30/60帧窗口包含不同长度区域，1600步、seed20261009。验证选择中心误差加0.01倍最大旋转角，而非测试渲染质量。\n\n| 输入数 | 局部跨度/GT | 全局跨度/GT | 局部中心误差 | 全局中心误差 | 局部最大旋转 | 全局最大旋转 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for n in (10, 30, 60):
        a = next(
            x
            for x in rows
            if x["input_count"] == n and "selected-v2" in x["checkpoint"]
        )["pose"]
        b = next(
            x
            for x in rows
            if x["input_count"] == n and "global-pose" in x["checkpoint"]
        )["pose"]
        global_text += f"| {n} | {a['extent_ratio']:.3f} | {b['extent_ratio']:.3f} | {a['center_l2_mean']:.5f} | {b['center_l2_mean']:.5f} | {a['rotation_max_deg']:.2f}° | {b['rotation_max_deg']:.2f}° |\n"
    global_text += "\n| 输入 | 同一长场＋局部位姿 PNG PSNR/SSIM | 同一长场＋全局特征 PNG PSNR/SSIM |\n|---|---:|---:|\n"
    for n in (10, 30, 60):
        record, folder = neural("long-shelf-global-v1", f"{n}-local", n)
        assert record["input_sha256"] == new[n][0]["input_sha256"]
        a = new[n][0]["stage2"][0]["quality_png"]
        b = record["stage2"][0]["quality_png"]
        global_text += f"| {n}张256输入 | {a['psnr_png_gt_valid']:.2f}/{a['ssim_gt_valid']:.3f} | {b['psnr_png_gt_valid']:.2f}/{b['ssim_gt_valid']:.3f} |\n"
        gt = old[n][1] / "gt.png"
        w, h = Image.open(gt).size
        panels = [
            ("Orthographic GT", gt),
            (
                "SIFT estimated homography",
                OUT / f"homography{n}/sift/orthographic-adapted-rgb.png",
            ),
            ("Local poses + neural field", new[n][1] / "predicted_all-rgb.png"),
            ("Global features + SAME neural field", folder / "predicted_all-rgb.png"),
        ]
        canvas = Image.new("RGB", (w, 4 * (h + 24)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (label, path) in enumerate(panels):
            draw.text((8, i * (h + 24) + 5), label, fill="black")
            canvas.paste(Image.open(path).convert("RGB"), (0, i * (h + 24) + 24))
        canvas.save(OUT / f"global-comparison-{n}.png")
    record, folder = neural("long-shelf-global-v1", "60-native-local", 60)
    a = native_new[0]["stage2"][0]["quality_png"]
    b = record["stage2"][0]["quality_png"]
    global_text += f"| 60张1280原生输入 | {a['psnr_png_gt_valid']:.2f}/{a['ssim_gt_valid']:.3f} | {b['psnr_png_gt_valid']:.2f}/{b['ssim_gt_valid']:.3f} |\n"
    global_text += "\n![60张全局特征、局部位姿与单应矩阵对照](global-comparison-60.png)\n\n同一渲染器传播后的图像表现与几何结果分开看；全局特征不能消除平面拼接的深度视差，也不能自动解决源深度噪声及遮挡错误。测试新模型只能给出本轮轨迹上的证据，不能据此宣称累计变形已完全消除。新模块测试验证零初始化保留原位姿、合法旋转、固定规范、远距视图梯度及检查点恢复。\n\n[全局训练选步](../long-global-pose-v1/report.json)、[独立几何控制](global-stage1.json)、[全局渲染](../long-shelf-global-v1/60-local/60/report.json)。\n\n```bash\n.venv/bin/python -m backend.research.cli global-pose-adapt --data data/research/long-shelf60-v1 --checkpoint artifacts/research/long-render-v1/model.pt --output artifacts/research/long-global-reproduce --steps 1600\n.venv/bin/python -m backend.research.cli multiview --scene data/research/long-shelf60-v1/732005 --checkpoint artifacts/research/long-global-pose-v1/model.pt --output artifacts/research/long-global-eval-reproduce --width 768 --geometry-width 256 --samples 32 --counts 60 --local-sources 12\n```\n"
    text = text.replace(
        "## 输入与数据协议",
        "最新全局特征对照：60张独立长货架的跨度1.058→1.031倍、最大旋转2.57°→1.19°；同一渲染器原生输入PNG PSNR14.78→15.26 dB，SSIM仍低。详见文末全局特征控制。\n\n## 输入与数据协议",
    )
    smoke = ART / "long-shelf-global-v1/rgb-only-mps/report.json"
    if smoke.exists():
        evidence = read(smoke)
        global_text += f"\nMPS全局模型实际完成仅RGB＋内参的10张1280×960推理：64宽输出、{evidence['elapsed_seconds']:.2f}秒、覆盖{evidence['coverage']*100:.1f}%。输入目录没有labels/heldout，未做测试场景拟合。该序列是既有设备烟测目录，64宽仅验证可执行及标签隔离，不是长货架或产线画质验收。见[MPS报告](../long-shelf-global-v1/rgb-only-mps/report.json)。\n"
    (OUT / "REPORT.md").write_text(text + global_text)
