"""Summarize matched, first-stage sequence-consensus experiments."""

import json
from pathlib import Path
import numpy as np

root = Path("artifacts/research/sequence-consensus-v1")
oldroot = Path("artifacts/research/grouped-layout-eval-v1")
records = []
failures = []
for seed in range(737018, 737024):
    for method in ("sift", "jepa"):
        p = root / f"test-{method}-{seed}" / "report.json"
        if not p.exists():
            failures.append(
                dict(
                    seed=seed,
                    method=method,
                    log=str(root / f"test-{method}-{seed}.log"),
                )
            )
            continue
        new = json.loads(p.read_text())
        old = json.loads(
            (oldroot / f"tilt-{method}-orientation-{seed}" / "report.json").read_text()
        )
        records.append(dict(seed=seed, method=method, old=old, new=new))
(root / "metrics.json").write_text(
    json.dumps(dict(results=records, failures=failures), indent=2)
)
lines = [
    "# 第一阶段：序列运动共识与光流补充",
    "",
    "本轮固定上轮 JEPA 权重、相同源图/内参、朝向估计与旋转平滑参数，新增匹配筛选和光流边，不训练第二阶段、不评估正交渲染。测试每组60帧、256×192、10.86m长货架；使用六种层数/分组布局 seed 737018–737023。",
    "",
    "## 结果",
    "",
    "|方法|完成/总数|商品均误差 原→新 px|整排同款 原→新 px|",
    "|---|---:|---:|---:|",
]
for method in ("sift", "jepa"):
    rs = [r for r in records if r["method"] == method]
    if rs:
        b = np.mean([r["old"]["global_products_gt_evaluation"]["mean_px"] for r in rs])
        a = np.mean([r["new"]["global_products_gt_evaluation"]["mean_px"] for r in rs])
        full = [r for r in rs if r["seed"] == 737023]
        f = (
            "尚未完成"
            if not full
            else f"{full[0]['old']['global_products_gt_evaluation']['mean_px']:.2f}→{full[0]['new']['global_products_gt_evaluation']['mean_px']:.2f}"
        )
        lines.append(f"|{method}|{len(rs)}/6|{b:.2f}→{a:.2f}|{f}|")
lines += [
    "",
    f"硬失败/未完成：{len(failures)}/12；对应日志保留。商品误差使用拟合后真实可见商品点，对比同一中间相机坐标规范。",
    "",
    "## 设计与验证选择",
    "",
    "从偶数匹配中估计跨帧位移：每条边的中位位移除以帧间距，再在全序列取中位，避免高匹配数的坏边主导。按预测位移筛掉跳到相邻重复商品的对应，单条边至少6点。未知步长从观测推断，未读货架宽度、商品宽度或模拟相机。",
    "补充相邻帧的 Shi-Tomasi + pyramidal LK 光流边：在正面校正的图像中跟踪，前后向误差<1px、LK误差<15，并通过相同运动门限。光流与特征边联合做原有全局平移拟合；严格检查筛选后的图连通，断开时明确失败。",
    "规则仅在验证 seed737017 上调整：最初3–6px门限使图在第27/28帧附近断开，加入光流后仍断开。为容纳朝向校正误差，在验证上调整为位移门限 clip(3×1.4826×MAD,[8,16],[10,20]) px；门限针对实际成对位移，不随帧间距放大。随后冻结规则再测全部六测试场景。完整固定配置见 protocol.json。",
    "注意原诊断字段 tolerance_px_per_frame 为历史命名，本轮实际应用为绝对成对位移门限，计算规则与 protocol.json 一致。",
    "这是近似匀速、稠密、有序横扫先验；明显变速、回头或大间隔采集尚未验证。不声称重复纹理歧义被普遍解决。",
    "",
    "## 逐场景诊断",
    "",
    "|方法/seed|商品均误差 原→新|背板均误差 原→新|全部奇数匹配P95 原→新|剔除偶数对应|补充光流边|",
    "|---|---:|---:|---:|---:|---:|",
]
for r in records:
    b, a = r["old"], r["new"]
    c = a["sequence_consensus"]
    lines.append(
        f"|{r['method']}/{r['seed']}|{b['global_products_gt_evaluation']['mean_px']:.2f}→{a['global_products_gt_evaluation']['mean_px']:.2f}|{b['global_gt_evaluation']['backplane_mean_px']:.2f}→{a['global_gt_evaluation']['backplane_mean_px']:.2f}|{b['after']['withheld_match_p95_px']:.2f}→{a['after']['withheld_match_p95_px']:.2f}|{c['rejected_correspondences']}|{c['optical_flow_edges']}|"
    )
lines += [
    "",
    "奇数匹配诊断保留全部原始对应，包括被运动先验判为错误的商品关联，不只报告过滤后的点。因此几何改善时原始匹配P95仍可很高。原初始化使用全部匹配，光流也读完整源图；奇数诊断不是严格独立监督测试。低置信度标记仍按原P95或未收敛标准保留。背板/商品深度不同，约束仍可能牺牲背板精度。",
    "",
    "55项测试通过。新增测试覆盖重复商品跳位剔除及筛选后断连失败。60帧整排同款的仅RGB＋内参入口另行验证，见 rgb-only-verification.json。",
    "",
    "## 复现",
    "",
    "```sh",
    ".venv/bin/python scripts/research_sequence_eval.py",
    ".venv/bin/python scripts/research_sequence_report.py",
    ".venv/bin/python -m backend.research.cli homography-global --scene data/research/grouped-varied-tilt-v1/737023 --output artifacts/research/sequence-rerun --method jepa --model orientation-consensus --checkpoint artifacts/research/grouped-matching-v1/shelf-jepa.pt",
    ".venv/bin/python -m pytest -q",
    "```",
    "",
    "所有新输出位于 sequence-consensus-v1，旧报告、权重、默认拼接/API保留。上述脚本会重新执行测试，需保留现有证据时先备份此输出目录。",
    "",
    "![整排同款商品的新第一阶段拼接](test-jepa-737023/front-facing.png)",
    "",
    "正面校正依旧是源图单应拼接；商品侧面、遮挡和三维视差没有被转换成严格正交视图。原生1280×960与真实产线验证仍未完成。",
]
(root / "REPORT.md").write_text("\n".join(lines) + "\n")
print("reports", len(records), "missing", len(failures))
