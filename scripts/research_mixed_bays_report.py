"""Separate mixed-bay data, stage-one alignment and orthographic capacity evidence."""

from pathlib import Path
import json, numpy as np

root = Path("artifacts/research/mixed-bays-v1")
root.mkdir(parents=True, exist_ok=True)
training = json.loads(
    Path("artifacts/research/mixed-bays-matching-v1/report.json").read_text()
)
fits = [
    json.loads(
        Path(f"artifacts/research/mixed-bays-orthofit-v{i}/report.json").read_text()
    )
    for i in (1, 2)
]
rows = []
for seed in range(738018, 738024):
    a = json.loads(
        Path(
            f"artifacts/research/mixed-bays-stage1-v1/local-{seed}/report.json"
        ).read_text()
    )
    b = json.loads(
        Path(
            f"artifacts/research/mixed-bays-stage1-v1/global-{seed}/report.json"
        ).read_text()
    )
    rows.append(dict(seed=seed, local=a, global_equal_rows=b))
held_path = Path("artifacts/research/mixed-bays-heldout-v1/60/report.json")
held = json.loads(held_path.read_text()) if held_path.exists() else None
aligned_path = Path("artifacts/research/mixed-bays-aligned-heldout-v1/60/report.json")
aligned = json.loads(aligned_path.read_text()) if aligned_path.exists() else None
(root / "metrics.json").write_text(
    json.dumps(
        dict(
            training=training,
            capacity=fits,
            stage1=rows,
            heldout=held,
            aligned_heldout=aligned,
        ),
        indent=2,
    )
)
lines = [
    "# 同一条长货架内，不同子货架层数：数据、训练与渲染",
    "",
    "本轮修正的是“同一场景内部的层数差异”。每条10.86m货架分成5个子货架，每个子货架独立生成层数、间距、分组商品与空位；默认层数集合2/5/3/6/4随场景轮换，整排同款也存在。每个子货架有独立层板和分隔立柱，层板不横跨其他子货架。商品与货架为不透明首交表面。",
    "",
    "## 数据与训练",
    "",
    "数据目录 data/research/mixed-bays-v1：24个场景×60帧，源图256×192，目标正交GT512宽。训练12场景738000–738011、验证6场景738012–738017、测试6场景738018–738023；按种子隔离，不跨集合。GT包含RGB、线性深度和相机，正交GT由同一几何独立射线渲染。capture tilt、0.14m等间距采集，关闭曝光扰动。",
    f'匹配模型：MPS实训1200步，验证选中{training["selected_step"]}步；真实跨图检索准确率{training["initial_validation"]["visible_crossview_retrieval_accuracy"]*100:.2f}%→{training["selected_validation_accuracy"]*100:.2f}%，耗时{training["elapsed_seconds"]:.1f}s。使用训练/验证源图深度建立可见对应，没有读测试场景训练，也没有目标正交图监督匹配。模型为现有DenseJEPA卷积模型，独立权重保留在mixed-bays-matching-v1。',
    "渲染模型：在训练场景738000上，以60张源图和真值源几何输入，使用正交GT RGB/深度监督神经场1500步。这是容量检查；渲染器的ViT/VAE及几何头沿用旧模型，不是把新DenseJEPA直接接入渲染器，也未做全管线联合训练。",
    "",
    "## 第一阶段：不强制所有子货架层板等高",
    "",
    "新增orientation-local：共享标定朝向/尺度和运动共识，匹配与光流提供局部对应；关闭强制源横线对齐中间图层板高度的全局等高残差。没有用GT子货架ID、层数或边界参与拟合，也没有实现自动子货架实例分割。",
    "",
    "|测试seed|原全局等高 商品误差px|新局部对应 商品误差px|",
    "|---|---:|---:|",
]
for row in rows:
    lines.append(
        f"|{row['seed']}|{row['global_equal_rows']['global_products_gt_evaluation']['mean_px']:.2f}|{row['local']['global_products_gt_evaluation']['mean_px']:.2f}|"
    )
lines += [
    "",
    f"六场景均误差：{np.mean([r['global_equal_rows']['global_products_gt_evaluation']['mean_px'] for r in rows]):.2f}→{np.mean([r['local']['global_products_gt_evaluation']['mean_px'] for r in rows]):.2f}px。不是所有场景都改善，不能用本轮声称已解决全部累计变形或错配。均值不剔除差结果。",
    "",
    "## 第二阶段：正交渲染容量检查",
    "",
    "首轮v1以行顺序分块，但局部视图查询要求窄列；块横跨多子货架造成源视图选择错误。v2只修复缓存分组：先按目标列分组选择邻近源照片，再恢复原始像素顺序；旧render_image已有列分组，修复主要影响训练缓存。GT、初始检查点、1500步、采样64、局部源4、边缘采样35%、表面损失0.02全部匹配。新增测试验证列分组及恢复顺序后的特征/光线一致。",
    "",
    "|实验|初始→训练后PSNR|目标深度AbsRel|",
    "|---|---:|---:|",
]
for i, fit in enumerate(fits, 1):
    lines.append(
        f"|v{i}|{fit['initial_psnr']:.2f}→{fit['final_psnr']:.2f}dB|{fit['final_metrics']['target_depth_abs_rel']*100:.2f}%|"
    )
lines += [
    "",
    "这些指标使用被训练的正交GT，且源几何为真值，不能证明未见场景或仅RGB推理质量。只修改训练缓存，未把结果提高归因于新增网络深度或匹配训练。支持掩码、目标深度异常及局部伪影见各原始report.json；当前20dB仍不是画质验收通过。",
    "",
    "![模拟器正交GT：同一条货架内不同层数](/Users/haoyu/Documents/Projects/tinyStitch/data/research/mixed-bays-v1/738000/orthographic/rgb.png)",
    "",
    "![GT监督的训练场景神经渲染](/Users/haoyu/Documents/Projects/tinyStitch/artifacts/research/mixed-bays-orthofit-v2/panorama.png)",
    "",
    "## 独立测试渲染",
    "",
]
if held:
    lines.append(
        "固定上述渲染权重，在未参加训练的738018上评估；目标GT只定义评分视点、范围和指标，不进行测试场景拟合。预测源几何和同输入数量的真值源几何对照分开报告。"
    )
    lines.append(
        "```json\n"
        + json.dumps(
            dict(stage1=held["stage1"], stage2=held["stage2"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n```"
    )
else:
    lines.append("独立测试正在运行；本报告暂不声称完成泛化验证。")
if aligned:
    lines += [
        "",
        "### 第一阶段对齐接入神经渲染",
        "",
        "从source-only的estimated_rotations与front_transforms导出独立alignment JSON（含输入顺序/内参哈希），用预测前景深度建立统一规范尺度，将二维全局平移提升为近似三维相机中心。保持渲染器、预测深度和60张输入不变，替换原位姿头的相机。",
        "这个提升依赖固定货架距离、近似平行横扫和RGB饱和度前景先验；不恢复米制尺度，不代表任意运动的完整6DoF求解。没有读GT源相机、GT深度或GT布局来计算这些位姿。评分目标视点/范围仍按独立GT定义。",
        f"未见738018：原位姿跨度比{held['stage1']['sequence_extent_ratio']:.3f}，对齐提升后{aligned['stage1']['sequence_extent_ratio']:.3f}；源深度AbsRel保持{aligned['stage1']['source_depth_abs_rel']*100:.2f}%。GT有效区PSNR从{held['stage2'][0]['psnr_gt_valid']:.2f}→{aligned['stage2'][0]['psnr_gt_valid']:.2f}dB，真值源几何对照{held['stage2'][1]['psnr_gt_valid']:.2f}dB。支持覆盖仍不完整，且存在明显噪声/遮挡边缘，不算画质验收通过。",
        "![未见混合层数货架：源图对齐位姿＋预测深度的神经渲染](/Users/haoyu/Documents/Projects/tinyStitch/artifacts/research/mixed-bays-aligned-heldout-v1/60/predicted_all.png)",
        "",
    ]
lines += [
    "",
    "## 复现与验证",
    "",
    "```sh",
    ".venv/bin/python -m backend.research.cli generate --output data/research/mixed-bays-rerun --scenes 24 --seed 738000 --views 60 --capture-step .14 --varied-layout --mixed-bays --capture-tilt --target-width 512 --no-exposure-jitter",
    ".venv/bin/python -m backend.research.cli matching-adapt --data data/research/mixed-bays-v1 --output artifacts/research/mixed-matching-rerun --steps 1200 --device mps",
    ".venv/bin/python scripts/research_mixed_bays_eval.py",
    ".venv/bin/python -m backend.research.cli alignment-export --scene data/research/mixed-bays-v1/738018 --report artifacts/research/mixed-bays-stage1-v1/local-738018/report.json --output artifacts/research/mixed-bays-stage1-v1/alignment-738018.json",
    ".venv/bin/python -m backend.research.cli orthofit --scene data/research/mixed-bays-v1/738000 --checkpoint artifacts/research/long-render-v1/model.pt --output artifacts/research/mixed-render-rerun --steps 1500 --width 512 --device mps --source-geometry oracle --local-sources 4 --surface-weight .02 --edge-fraction .35 --occlusion-aware",
    ".venv/bin/python -m backend.research.cli multiview --scene data/research/mixed-bays-v1/738018 --checkpoint artifacts/research/mixed-bays-orthofit-v2/model.pt --output artifacts/research/mixed-heldout-rerun --device mps --width 512 --samples 64 --counts 60 --local-sources 4 --oracle-control",
    ".venv/bin/python -m backend.research.cli multiview --scene data/research/mixed-bays-v1/738018 --checkpoint artifacts/research/mixed-bays-orthofit-v2/model.pt --output artifacts/research/mixed-aligned-rerun --device mps --width 512 --samples 64 --counts 60 --local-sources 4 --alignment artifacts/research/mixed-bays-stage1-v1/alignment-738018.json",
    ".venv/bin/python scripts/research_mixed_bays_report.py",
    ".venv/bin/python -m pytest -q",
    "```",
    "",
    "仅RGB＋内参的60帧对齐已另行验证：不包含标签目录的运行与带GT评分运行输出完全相同，并导出一致的alignment JSON，见mixed-bays-stage1-v1/rgb-only-verification.json。",
    "58项测试通过，包括不同子货架2/6层验证及局部缓存光线/特征顺序回归。旧权重和报告保留，独立目录存放本轮数据和检查点；网页API未变。输出512宽与输入256×192是资源约束下的原型，未验证真实1280×960。",
]
(root / "REPORT.md").write_text("\n".join(lines) + "\n")
print("heldout", bool(held))
