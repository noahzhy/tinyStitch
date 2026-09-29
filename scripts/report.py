"""Generate a human-readable report from actual learned-model and baseline runs."""

import base64
import html
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(file):
    return json.loads(file.read_text())


history = read(ROOT / "artifacts/models/history.json")
jepa = read(ROOT / "examples/validation/jepa/benchmark.json")
sift = read(ROOT / "examples/validation/sift/benchmark.json")
deployment = read(ROOT / "artifacts/deployment-validation.json")
e2e = read(ROOT / "examples/validation/e2e.json")
training = read(ROOT / "artifacts/models/training-data.json")
assert (
    e2e["sample_report"]["model"]["checkpoint_sha256"]
    == deployment["model"]["checkpoint_sha256"]
)
summary = {}
for name, bench in [("jepa", jepa), ("sift", sift)]:
    valid = [r for r in bench["records"] if r["status"] == "completed"]
    summary[name] = {
        "completed": len(valid),
        "total": bench["total"],
        "mean_seconds": float(np.mean([r["seconds"] for r in valid]))
        if valid
        else None,
        "median_front_plane_error_px": float(
            np.median([r["independent_front_plane_error_px"] for r in valid])
        )
        if valid
        else None,
        "rss_at_completion_bytes": bench["rss_at_completion_bytes"],
    }
report = {
    "models": summary,
    "deployment": deployment,
    "end_to_end": {k: v for k, v in e2e.items() if k != "sample_report"},
    "train_layouts": len(training["train_layouts"]),
    "val_layouts": len(training["val_layouts"]),
    "trained_steps": history[-1]["step"],
    "training_seconds": history[-1]["elapsed_seconds"],
    "initial_validation": training["initial_validation"],
    "final_validation": history[-1]["validation"],
    "validation_scope": "16 fixed augmented image pairs spread over all 9 validation layouts; matching among up to 128 known positions, not full shelf accuracy",
    "scope": "Synthetic same-side shelf sweeps with labels and slight camera jitter. No real-store validation. No claim of JEPA superiority; no isolated JEPA ablation.",
}
(ROOT / "artifacts/report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
rows = ""
for name, m in summary.items():
    rows += f"<tr><th>{name.upper()}</th><td>{m['completed']}/{m['total']}</td><td>{m['mean_seconds']:.2f}</td><td>{m['median_front_plane_error_px']:.3f}</td></tr>"
sample = ROOT / "examples/simulated/jepa-result/panorama.png"
encoded = base64.b64encode(sample.read_bytes()).decode()
html_report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>shelfStitch JEPA 实际实验报告</title><style>body{{font-family:system-ui,sans-serif;color:#283b4a;background:#f4f7f8;line-height:1.8}}main{{max-width:1000px;margin:36px auto;padding:0 20px}}h1{{font-size:28px}}h2{{font-size:20px;margin-top:28px}}.lead,table,pre{{background:white;border-radius:8px;padding:20px}}.lead{{border-left:4px solid #258578}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #e1e9ee}}img{{width:100%;background:repeating-conic-gradient(#e4e9ec 0 25%,white 0 50%) 0/20px 20px}}pre{{overflow:auto;font-size:11px}}.muted{{font-size:12px;color:#7a8d9d}}.scroll{{overflow:auto}}</style><main><p class="muted">tinyLayout / shelfStitch · 已实际训练与调用的 JEPA 思路实验</p><h1>RGB 货架连续照片 → 完整拼图</h1><div class="lead">主流程使用已训练 JEPA 学习描述子，自动估计变换并融合输入照片。使用 SIFT 仅检测关键点位置，主流程不计算 SIFT 描述子。保留显式 SIFT 描述子对照。<p>独立测试 {jepa["total"]} 个未见布局，每个 9 张 720×960 图片。JEPA 成功连接 {jepa["completed"]}/{jepa["total"]}。本次 JEPA 耗时更高，正面平面点误差略大于 SIFT，对照未显示 JEPA 优势。这只能证明本次合成实验中的闭环，不能证明真实门店泛化。</p></div><h2>实际训练</h2><p>47 个训练布局 / 423 张图片，9 个验证布局 / 81 张图片；按布局隔离。1600 步：前 400 步 EMA 目标潜在预测，后 1200 步保留 JEPA 并加入对应点 InfoNCE。M4 MPS 训练约 {history[-1]["elapsed_seconds"] / 60:.2f} 分钟。编码器 76,448 参数、训练分支 221,088 参数。权重按验证增强对应点区分率选择。</p><p>验证 16 个固定增强对覆盖全部验证布局，至多 128 个候选位置；对应点区分率从 {training["initial_validation"]["augmentation_matching_accuracy"]:.3f} 到 {history[-1]["validation"]["augmentation_matching_accuracy"]:.3f}，特征标准差 {history[-1]["validation"]["feature_std"]:.3f}。这些指标不等于完整货架配准精度。模型是自研 JEPA 思路与对应点辅助项的混合实验，不是 Meta I-JEPA/V-JEPA 复现。</p><h2>未见布局的实测与 SIFT 对照</h2><div class="scroll"><table><tr><th>方法</th><th>成功 / 总数</th><th>每组平均秒</th><th>正面平面点误差中位数 px</th></tr>{rows}</table></div><p class="muted">测试种子 920000–920011，未进入训练/验证。误差通过独立投影的同一货架正面物理点计算：先运行 RGB-only 拼接，评估端再读取模拟制作信息。不能反映背景、商品凸起或所有接缝像素的误差。耗时包括图像加载、模型/特征提取、匹配、优化与输出；首个 JEPA 任务还包含 PyTorch 导入与 MPS 编译冷启动，后续任务可热。没有测试后调阈值或按结果选择案例。</p><h2>实际 JEPA 样例</h2><p>固定演示种子 42：输入 9 张，输出 {deployment["sample_size"][0]}×{deployment["sample_size"][1]}。不是训练/独立测试集中的布局；这里只用于展示界面与文件格式。透明边界表示没拍到的区域，照片中的背景也会保留。</p><img alt="真实 JEPA 描述子产生的货架拼图" src="data:image/png;base64,{encoded}"><h2>实现与限制</h2><p>上下文遮挡 → 潜在特征预测 → stop-gradient EMA 目标；方差/协方差约束防坍塌。部署只保留 stride=4、64 维密集编码器，用学习描述子找跨图对应，再做 RANSAC、整组共享平面优化、最近视图接缝与窄范围融合。没有将 JEPA 预测解码成新增商品。</p><p>单个平面模型无法彻底消除商品深度差带来的视差。重复商品、反光、模糊、不足重叠仍会失败或产生重影。纯 JEPA 潜在损失不足以保证像素级对应，本实验使用了辅助对应点训练；没有隔离消融，不能把效果全部归因于 JEPA。真实货架照片尚未测试；手机模型部署和精确尺度不在本次范围。</p><h2>真实调用与隔离验证</h2><p>删除模拟制作信息后，使用独立目录的 RGB 仍实际运行训练权重；更换输入生成新摘要和新输出。模型 SHA-256 {deployment["model"]["checkpoint_sha256"]}。网页验收包含模拟拍摄、实际 JEPA 拼接、原始序列导出、照片重新导入、再次调用 JEPA、PNG 下载和手机宽度布局。目标分支无梯度与 EMA 更新通过测试。</p><details><summary>完整机器可读记录</summary><pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre></details><p class="muted">参考 <a href="https://github.com/facebookresearch/jepa">Meta JEPA</a> 与 <a href="https://arxiv.org/abs/2301.08243">I-JEPA 论文</a>。源文件与部署权重保存在 shelfStitch/backend 与 artifacts/models。</p></main></html>"""
(ROOT / "artifacts/report.html").write_text(html_report)
print(json.dumps(summary))
