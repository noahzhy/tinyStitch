"""Summarize saved paired experiments; never train or select using test labels."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'artifacts/research'
OUT = A / 'mixed-bays-optimization-v1'
OUT.mkdir(exist_ok=True)
paths = {
 'baseline': ['mixed-bays-aligned-heldout-v1','mixed-bays-single-scene-test19-v1'],
 'renderer_only': ['mixed-bays-multiscene-heldout-v1','mixed-bays-multiscene-test19-v1'],
 'depth_dynamic_scale': ['mixed-bays-depth-heldout18-v1','mixed-bays-depth-heldout19-v1'],
 'level_camera': ['mixed-bays-level-heldout18-v1','mixed-bays-level-heldout19-v1'],
 'combined_dynamic_scale': ['mixed-bays-combined-test18-v1','mixed-bays-combined-test19-v1'],
 'stable_combined': ['mixed-bays-stable-test18-v1','mixed-bays-stable-test19-v1'],
}
reports = {key:[json.loads((A/p/'60/report.json').read_text()) for p in names] for key,names in paths.items()}
def psnr(r): return next(s['psnr_gt_valid'] for s in r['stage2'] if s['mode']=='predicted_all')
verification=[]
for i,seed in enumerate([738018,738019]):
 base=A/paths['baseline'][i]/'60'
 stable=A/paths['stable_combined'][i]/'60'
 manual=A/f'mixed-bays-fixedscale-test{18+i}-v1'/'60'
 r=reports['stable_combined'][i]
 checks=dict(seed=seed, cameras_equal_baseline=bool(np.array_equal(np.load(base/'cameras.npy'),np.load(stable/'cameras.npy'))), automatic_equals_manual=bool(np.array_equal(np.load(manual/'cameras.npy'),np.load(stable/'cameras.npy')) and (manual/'predicted_all.png').read_bytes()==(stable/'predicted_all.png').read_bytes()), same_input_hash=r['input_sha256']==reports['baseline'][i]['input_sha256'], no_test_fitting=not r['gt_used_for_test_fitting'], weights_frozen=r['weights_frozen'], frozen_source_calibration=r['alignment_pose']['frozen_reference_scale'])
 assert all(v for k,v in checks.items() if k!='seed'),checks
 verification.append(checks)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
(OUT/'metrics.json').write_text(json.dumps(reports,indent=2))
rows='\n'.join('| '+key+' | '+' | '.join(f'{psnr(r):.3f}' for r in rs)+' |' for key,rs in reports.items())
r18,r19=reports['stable_combined']
b18,b19=reports['baseline']
report=f'''# 不同子货架层数：分阶段优化实测

本轮完成多场景渲染器训练、深度训练及两场独立测试的正交渲染。稳定组合结果为 **{psnr(r18):.2f}/{psnr(r19):.2f} dB**，原结果 **{psnr(b18):.2f}/{psnr(b19):.2f} dB**。提升有限，边缘噪声、错位与局部缺失仍然存在，尚未通过画质验收。

## 第一阶段：深度与相机尺度

同一条货架有5个不同层数的子货架，包含成组/整排重复商品及遮挡。12个训练场景738000–738011，6个验证场景738012–738017。深度适配1200步，验证选择1100步；测试场景不加载到训练进程。

| 测试场景 | 原源深度AbsRel | 新源深度AbsRel | 相机跨度比 |
| --- | --- | --- | --- |
| 738018 | {b18['stage1']['source_depth_abs_rel']:.4f} | {r18['stage1']['source_depth_abs_rel']:.4f} | {r18['stage1']['sequence_extent_ratio']:.4f} |
| 738019 | {b19['stage1']['source_depth_abs_rel']:.4f} | {r19['stage1']['source_depth_abs_rel']:.4f} | {r19['stage1']['sequence_extent_ratio']:.4f} |

问题定位：原位姿提升将预测前景深度同时用于相机尺度校准。替换深度头时，相机尺度也变化，不能将该结果视为纯深度消融。改进保留冻结的原深度头，从每次输入RGB重新估计相机规范；新深度头只提供渲染几何。自动校准与手工固定源预测尺度的结果逐像素一致，相机数组与原基线完全相同，详见 verification.json。没有读取测试真值尺度；不宣称恢复米制尺寸。

近水平相机先验另作对照：验证场景朝向误差0.823°→0.462°，但渲染未改善，因此仅保留可选 orientation-local-level，不纳入稳定组合。该先验限于近水平横扫。

## 第二阶段：正交渲染

渲染器从旧权重热启动，12训练/6验证场景，1800步，验证缓存PSNR17.907→19.052 dB。训练使用真值源相机、深度及合成正交GT，冻结编码器/VAE/位姿分支，仅优化场网络。每场景缓存2048条光线，训练32采样点；测试64点。与旧单场景训练预算和采样不同，不能把收益归因于单一结构改动。

| 分支 | 测试738018 PSNR | 测试738019 PSNR |
| --- | --- | --- |
{rows}

真值源几何下，单独新渲染器从17.746/17.425提升至18.442/18.043 dB；预测几何下单独换渲染器几乎无收益。深度直接替换及动态尺度组合均恶化画质。稳定组合同时采用验证选择的新渲染器、新深度头和冻结源尺度校准分支。

颜色融合在验证738012上：weighted真值几何PSNR17.532，consensus17.254，nearest17.322。保留weighted；另外两种仅作为失败对照，不作为改进结果。

### 测试协议与限制

每场景60张256×192源图，10.86m货架，固定14cm步距；正交输出512×94。目标相机和输出范围由GT提供，仅用于固定范围评测，测试目标RGB/深度/外参不用于拟合。测试过程冻结权重，不执行场景优化。源位姿来自RGB与内参的全局对齐，不采用测试GT几何；局部每列块查询4张源图。

稳定结果有效GT区域支持比例为{r18['stage2'][0]['supported_fraction_gt_valid']:.1%}/{r19['stage2'][0]['supported_fraction_gt_valid']:.1%}。耗时{r18['elapsed_seconds']:.1f}/{r19['elapsed_seconds']:.1f}秒；进程峰值RSS {r18['peak_rss_mb']:.0f}/{r19['peak_rss_mb']:.0f}MB，不等于完整GPU内存。训练实际使用CPU，渲染使用MPS。这两个测试场景用于配对诊断；尺度修复由测试诊断引出，仍需新场景冻结规则后验证。没有新增真实产线、1280×960或全部6测试场景验收证据。

渲染训练报告早期 source_geometry 文本记录错误，已按实际CLI参数修正为oracle；原始元数据另存 report-original-metadata.json，权重未因此修改。各原始报告与权重均保留。

## 复现

在项目根目录执行，输出目录须使用新名称：

```sh
.venv/bin/python -m backend.research.cli render-adapt --data data/research/mixed-bays-v1 --checkpoint artifacts/research/mixed-bays-orthofit-v2/model.pt --output artifacts/research/reproduce-render --steps 1800 --rays 2048 --input-counts 60 --feature-center-mapping --local-sources 4 --source-geometry oracle --warm-field
.venv/bin/python -m backend.research.cli depth-adapt --data data/research/mixed-bays-v1 --checkpoint artifacts/research/mixed-bays-orthofit-v2/model.pt --output artifacts/research/reproduce-depth --steps 1200 --consistency-weight .5 --edge-weight 1 --skip-test
.venv/bin/python -m backend.research.combine --renderer artifacts/research/reproduce-render/model.pt --depth artifacts/research/reproduce-depth/model.pt --output artifacts/research/reproduce-combined
.venv/bin/python -m backend.research.cli multiview --scene data/research/mixed-bays-v1/738018 --checkpoint artifacts/research/reproduce-combined/model.pt --output artifacts/research/reproduce-test18 --device mps --width 512 --samples 64 --counts 60 --local-sources 4 --alignment artifacts/research/mixed-bays-stage1-v1/alignment-738018.json
.venv/bin/python scripts/research_mixed_bays_optimization_report.py
```

将最后渲染命令中的738018及alignment文件替换为738019可复现第二场景。脚本只汇总现有实验并断言配对条件，不训练或重新选择模型。

稳定检查点：../mixed-bays-stable-combined-v1/model.pt。默认网页API和旧检查点未替换。

![测试738018 原结果](../mixed-bays-aligned-heldout-v1/60/predicted_all-rgb.png)
![测试738018 稳定组合](../mixed-bays-stable-test18-v1/60/predicted_all-rgb.png)
![测试738018 正交GT](../mixed-bays-stable-test18-v1/60/gt.png)
'''
(OUT/'REPORT.md').write_text(report)
print(json.dumps(verification,indent=2))
