"""Summarize depth-only paired tests and validation selection."""
from pathlib import Path
import json
import numpy as np
A=Path(__file__).resolve().parents[1]/'artifacts/research'
folders=['mixed-bays-geometry-baseline-v2','mixed-bays-geometry-depth-control-v2','mixed-bays-geometry-depth-v2']
rows={f:json.loads((A/f/'geometry-evaluation.json').read_text())['rows'] for f in folders}
train={f:json.loads((A/f/'report.json').read_text()) for f in folders[1:]}
out=A/'mixed-bays-geometry-optimization-v2';out.mkdir(exist_ok=True)
mean=lambda f,key:np.mean([r[key] for r in rows[f]])
table='\n'.join(f'| {f} | {mean(f,"abs_rel")*100:.3f}% | {mean(f,"edge_abs_rel")*100:.3f}% | {mean(f,"multiscale_gradient"):.5f} |' for f in folders)
detail='\n'.join('| '+str(rows[folders[0]][i]['seed'])+' | '+' | '.join(f'{rows[f][i]["abs_rel"]*100:.3f}%' for f in folders)+' |' for i in range(6))
(out/'metrics.json').write_text(json.dumps(dict(test=rows,training=train),indent=2))
(out/'REPORT.md').write_text(f'''# 第一阶段几何：深度边界优化

本轮仅训练和评估几何深度，不渲染、不以第二阶段画质选模型。原相机朝向和全局位姿规则未改；冻结源尺度校准分支保留，因此深度训练不会改变原相机规范。

## 实际结果

6个独立测试场景738018–738023，各60张256×192输入。先由冻结模型完成预测，随后读取源深度标签计算指标；无测试场景拟合，无目标视角监督。后4个场景扩展了此前两场测试覆盖。

| 模型 | 全像素AbsRel均值 | 深度边界AbsRel均值 | 多尺度log深度差误差 |
| --- | --- | --- | --- |
{table}

| 场景 | 原模型 | 同预算继续训练 | 多尺度监督 |
| --- | --- | --- | --- |
{detail}

边界定义：GT相邻像素log深度跳变>0.015的两侧有效像素，不包含GT无效像素。这里没有尺度对齐后再挑选最佳尺度；沿用统一场景规范。指标反映合成深度监督效果，不代表真实产线泛化。

多尺度分支相对同预算对照，6个场景整体AbsRel均更低，多尺度深度差误差也更低；但紧邻遮挡边界AbsRel略高于对照，不能宣称遮挡边缘问题已解决。按预先使用的验证整体AbsRel选取多尺度分支为本轮候选，同预算对照继续保留。实际收益包含继续训练，不能全部归因于新增损失。

## 方法和公平对照

新增可选 `--multiscale-weight`：在2/4/8像素间距分别监督水平/垂直log深度差；不跨边界施加平滑先验。保留原像素、单像素梯度、边界与跨视图一致性监督。

两组从同一稳定检查点启动，同一随机种子20261007，1800步、学习率0.0002，12训练场景/6验证场景，冻结JEPA/VAE/位姿/渲染器，仅训练深度头及refiner。测试场景不进入训练缓存。选权重只按验证整体AbsRel，边界指标仅用于独立报告，避免在测试集选择损失权重。

同预算对照验证选择step {train[folders[1]]['protocol']['selected_step']}；多尺度分支step {train[folders[2]]['protocol']['selected_step']}。新增loss测试覆盖正确深度零损失、模糊边界非零损失、有效梯度与无效区域；完整测试63项通过。

本轮不改变默认网页API，不覆盖旧模型；是否推广多尺度分支需同时参考对照及逐场景结果。相机局部朝向波动尚未进一步改善，深度图仍存在商品边缘平滑问题。

## 复现

```sh
.venv/bin/python -m backend.research.cli depth-adapt --data data/research/mixed-bays-v1 --checkpoint artifacts/research/mixed-bays-stable-combined-v1/model.pt --output artifacts/research/reproduce-geometry --steps 1800 --consistency-weight .5 --edge-weight 1 --multiscale-weight 1 --skip-test
.venv/bin/python -m scripts.research_geometry_depth_eval --checkpoint artifacts/research/reproduce-geometry/model.pt --output artifacts/research/reproduce-geometry
```

对照将multiscale-weight改为0并使用独立输出目录。汇总已保存实验：`.venv/bin/python -m scripts.research_geometry_report`。
''')
print(table)
