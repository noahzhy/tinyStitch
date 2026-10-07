"""Build first-stage evidence report from completed, seed-isolated experiments."""

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="artifacts/research/grouped-layout-eval-v1")
parser.add_argument(
    "--training-report", default="artifacts/research/grouped-matching-v1/report.json"
)
options = parser.parse_args()
root = Path(options.root)
records = []
for p in sorted(root.glob("*/report.json")):
    r = json.loads(p.read_text())
    if not r.get("groundtruth_evaluation"):
        continue
    r["experiment"] = p.parent.name
    r["low_confidence"] = bool(
        not r["success"] or r.get("after", {}).get("withheld_match_p95_px", 99) > 5
    )
    records.append(r)
(root / "metrics.json").write_text(json.dumps(records, indent=2))
training = json.loads(Path(options.training_report).read_text())
lines = [
    "# 多层数、分组商品：第一阶段全局对齐实验",
    "",
    "本轮只评估源图匹配、货架朝向及全局 homography 对齐；没有训练或评估第二阶段神经渲染。正面校正输出仍是透视图像的旋转校正与拼接，不是带深度重建的正交新视角。",
    "",
    "## 数据与训练",
    "",
    "两套数据各 24 个场景、每场景 60 帧，货架长 10.86 m，步长 0.14 m，输入 256×192。固定近正面与带偏转采集分别使用 seed 736000 和 737000 开始；各 12 train / 6 val / 6 test，场景种子隔离。带偏转数据增加约 ±6° yaw、±1.5° roll。关闭曝光变化以隔离布局/姿态影响。",
    "六个布局为 2/3/4/5/6/4 层，分别每 3/6/9/4/7/整排位置同款；层间距不均匀，前五种有随机空位，商品有前后深度差。商品/货架为不透明射线首交表面。",
    "采用现有 64 维卷积 DenseJEPA 小模型，用源图 GT 深度与相机投影生成双向可见对应，EMA 目标分支及预测+匹配损失。没有使用目标正交图，也没有读取测试集训练。模型不是新 ViT 或单独的货架位姿网络。",
    f'MPS 实训 {training["steps"]} 步，batch {training["batch"]}，耗时 {training["elapsed_seconds"]:.1f} s；验证选中 step {training["selected_step"]}。检索准确率 {training["initial_validation"]["visible_crossview_retrieval_accuracy"]*100:.2f}% → {training["selected_validation_accuracy"]*100:.2f}%。检索候选是每对 128 个可见位置，重复商品存在歧义；当前数值仍低，不能据此宣称特征已可靠。',
    "冻结新匹配权重后进行测试。新增约束：LSD 层板横线/商品竖边 → 标定消失方向 → SO(3) 朝向 → 全序列旋转观测先验与二阶平滑 → 统一前向相机基底下鲁棒全局平移。并非强制输入相机朝向相同，也不预设层数。拟合只读 RGB 与内参；GT 只在拟合后评分。",
    "",
    "## 每场景结果",
    "",
    "商品误差是在真实可见商品正面采样点上，映射到中间相机后的像素误差。朝向方案先转换回相同中间相机坐标规范评分，避免把输出坐标变化误当作改善。背板为另一深度平面的诊断，不能与商品误差混为一谈。局部匹配 P95 使用奇数序号匹配；新拟合仅用偶数，但原始初始化用全部匹配，因此不是严格独立匹配测试。",
    "",
    "|场景/方法/约束|商品均误差 原→新 px|背板均误差 原→新 px|局部匹配 P95 原→新 px|朝向均误差 原→全局 °|",
    "|---|---:|---:|---:|---:|",
]
# Put experimental conclusions before the detailed per-scene table.
conclusion = [
    "## 结论",
    "",
    "共享尺度/全局位移在固定近正面采集下有效，但不充分：六场景商品均误差 SIFT 52.82→10.20 px，JEPA 36.10→5.26 px。整排同款 SIFT 仍 36.10 px、局部匹配 P95 25.68 px，说明连通与优化收敛不保证关联正确。",
    "相机偏转时，SIFT 共享基底方案为 16.68 px，加入货架朝向及全局旋转正则后为 10.71 px；JEPA 为 19.24→13.56 px。朝向估计六场景均误差 1.03→0.75°。这是朝向约束对第一阶段有效的证据，不是正交重建或画质验收。",
    "带偏转整排同款仍失败于重复关联：SIFT 商品误差 24.68 px、匹配 P95 47.73 px，JEPA 33.50 px、P95 37.74 px。朝向误差仅 0.74°，因此主要问题不是朝向，而是错误商品位置关联；尚未解决这类歧义。",
    "训练提升了跨图检索数值，但旧/新 JEPA 在固定整排同款优化后分别 6.00/6.27 px，不能声称新训练改善了最终拼接。",
    "36 个协议内拟合均返回结果，硬断连/异常为 0/36；低置信度标准为优化器未收敛或奇数匹配 P95>5 px（工作分辨率经验阈值，非校准概率）。各六场景组分别为近正面 SIFT 1/6、JEPA 1/6，偏转共享基底 SIFT 2/6、JEPA 1/6，偏转朝向方案 SIFT 1/6、JEPA 1/6。高 GT 误差的结果保留，不从均值中删除。",
    "本轮 53 项测试通过；覆盖消失方向无穷点/偏转、结构不足、分组确定性与跨图遮挡/空重叠，并保留既有几何和梯度测试。仅生成器轴对齐商品、固定内参/距离的长货架，不覆盖多个不同朝向的货架平面。",
    "",
]
if root == Path("artifacts/research/grouped-layout-eval-v1"):
    lines[2:2] = conclusion
for r in records:
    b = r["baseline_products_gt_evaluation"]["mean_px"]
    a = r["global_products_gt_evaluation"]["mean_px"]
    bp = r["baseline_gt_evaluation"]["backplane_mean_px"]
    ap = r["global_gt_evaluation"]["backplane_mean_px"]
    d = r.get("before", {}).get("withheld_match_p95_px")
    e = r.get("after", {}).get("withheld_match_p95_px")
    angular = "—"
    if "raw_orientation_error_degrees" in r:
        angular = f"{r['raw_orientation_error_degrees']['mean']:.2f}→{r['orientation_error_degrees']['mean']:.2f}"
    lines.append(
        f"|{r['experiment']}|{b:.2f}→{a:.2f}|{bp:.2f}→{ap:.2f}|{f'{d:.2f}→{e:.2f}' if d is not None else '—'}|{angular}|"
    )
lines += [
    "",
    "## 分组平均（六种布局等权）",
    "",
    "|输入/方法/约束|完成场景数|商品均误差 原→新 px|",
    "|---|---:|---:|",
]
for prefix in [
    "front-sift-translation",
    "front-jepa-translation",
    "tilt-sift-translation",
    "tilt-jepa-translation",
    "tilt-sift-orientation",
    "tilt-jepa-orientation",
]:
    rs = [r for r in records if r["experiment"].startswith(prefix)]
    if rs:
        lines.append(
            f"|{prefix}|{len(rs)}/6|{np.mean([r['baseline_products_gt_evaluation']['mean_px'] for r in rs]):.2f}→{np.mean([r['global_products_gt_evaluation']['mean_px'] for r in rs]):.2f}|"
        )
lines += [
    "",
    "## 可复现命令",
    "",
    "```sh",
    ".venv/bin/python -m backend.research.cli generate --output data/research/grouped-varied-v1 --scenes 24 --seed 736000 --views 60 --capture-step .14 --target-width 768 --varied-layout --no-exposure-jitter",
    ".venv/bin/python -m backend.research.cli generate --output data/research/grouped-varied-tilt-v1 --scenes 24 --seed 737000 --views 60 --capture-step .14 --target-width 768 --varied-layout --capture-tilt --no-exposure-jitter",
    ".venv/bin/python -m backend.research.cli matching-adapt --data data/research/grouped-varied-v1 --output artifacts/research/grouped-matching-rerun --steps 1200 --device mps",
    ".venv/bin/python scripts/research_grouped_eval.py --output artifacts/research/grouped-eval-rerun --checkpoint artifacts/research/grouped-matching-rerun/shelf-jepa.pt",
    ".venv/bin/python -m backend.research.cli homography-global --scene data/research/grouped-varied-tilt-v1/737018 --output artifacts/research/orientation-rerun --method sift --model orientation",
    ".venv/bin/python scripts/research_grouped_report.py --root artifacts/research/grouped-eval-rerun --training-report artifacts/research/grouped-matching-rerun/report.json",
    ".venv/bin/python -m pytest -q",
    "```",
    "",
    "匹配训练默认禁止覆盖已有权重。评测脚本复用已完成 report.json；如要完整重跑，选择新的输出根目录或先备份已有实验目录。训练状态保存，但本轮 CLI 未实现断点恢复。权重为独立 grouped-matching-v1/shelf-jepa.pt，原 shelf-jepa.pt 保留。",
    "几何对应边界测试发现浮点零边界问题，修复了 1e-6 容差；本次保存权重是在修复前训练，严格重现该权重需使用原零容差。该修复只影响边界有效性，后续重训会使用修复后的规则。",
    "",
    "只含 RGB 与 intrinsics.json 的 60 帧入口已独立运行成功；变换与启用 GT 评分的运行一致，见 rgb-only-verification.json。",
    "",
    "## 图像检查",
    "",
    "![独立布局的中间输入帧](dataset-preview.png)",
    "",
    "![带偏转二层货架的正面校正拼接](tilt-sift-orientation-737018/front-facing.png)",
    "",
    "本轮图像用于检查结构与局部错位。拼图有效区以 alpha 保存；缺失区保持透明。商品侧面/遮挡与深度差仍存在，单应方案不能让所有三维表面同时成为正交视图。当前仅合成 256×192 证据，尚未证明真实 1280×960 或任意摆放方向适用。",
]
(root / "REPORT.md").write_text("\n".join(lines) + "\n")
canvas = Image.new("RGB", (768, 420), "white")
draw = ImageDraw.Draw(canvas)
for n in range(6):
    im = Image.open(f"data/research/grouped-varied-v1/{736018+n}/rgb/0030.png").convert(
        "RGB"
    )
    x = (n % 3) * 256
    y = (n // 3) * 210
    canvas.paste(im, (x, y + 18))
    draw.text(
        (x + 5, y + 2),
        f'layers {[2,3,4,5,6,4][n]}, group {[3,6,9,4,7,"full row"][n]}',
        fill="black",
    )
canvas.save(root / "dataset-preview.png")
print(f"{len(records)} completed reports")
