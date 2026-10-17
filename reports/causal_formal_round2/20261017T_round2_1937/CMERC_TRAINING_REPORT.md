# IEMOCAP Causal CMERC：正式训练已完成

状态：**80/80 epoch完成，独立验收通过**。CL、HSCL、CC全部启用，严格使用本批真实因果教师生成的difficulty，完整来源见DIFFICULTY_PROVENANCE_REPORT.md。

实际初始化SHA256与Baseline完全一致：`de943257bf89c6aa53abc205c794f22f17e79aa40ed0a1701a08b1cd6aa0752f`；同一train/validation划分、特征字节、backbone、seed和80轮预算。课程自然改变早期训练对话数/optimizer步数，不声称每轮样本数或总step数与无CL相等。

- 验证W-F1选择第48轮，记录值63.39%；保留两位小数选模且并列取早。
- best.pt SHA256：`d4e1501b1082f3f4cc951b65bbac5fc0b78779d3306ce671e3a4c613355650f4`。
- latest.pt SHA256：`ebfa7644d4b67d5671d41b57d41dc82813fb0ab50273635dea044abaf263897b`；完成80轮并保留Adam、RNG、课程进度与历史best绑定。
- difficulty SHA256：`73791cfb5755710e74424329bf8f3e4dc51d6ba464d19d6c8b71566cf9e73463`；teacher best SHA256：`1d9bbae19f0e4f8bd976bc3e4285f93c14f114cc54e6d28afcc76be8f90dd3bc`。
- 逐轮记录训练耗时合计1502.78秒；峰值allocated GPU显存4065.14MiB；进程墙钟时间见PIPELINE_STATE.json。

## 实际损失与梯度

| 分量 | 系数 | 原始loss的epoch均值 | 实际加权loss的epoch均值 | 原始梯度范数范围 | 加权梯度范数范围 |
|---|---:|---:|---:|---:|---:|
| cc_loss | 0.005 | 113.231381 | 0.566157 | [1006.519282, 12919.880972] | [5.032596, 64.599405] |
| classification_loss | 1.0 | 0.625354 | 0.625354 | [1.867188, 24.875493] | [1.867188, 24.875493] |
| cscl_loss | 0.05 | 2.629258 | 0.131463 | [0.931151, 5.561153] | [0.046558, 0.278058] |
| mscl_loss | 0.05 | 4.077027 | 0.203851 | [1.299140, 7.797830] | [0.064957, 0.389891] |

loss为各epoch批均值再取均值，非全话语加权平均。梯度范围来自每epoch首训练batch真实autograd，非以原始loss大小猜测；所有训练batch均在更新前检查total loss与活跃梯度有限。各分项范数不能相加成总梯度，也不等同Adam最终更新份额；更新还受方向、Adam状态和weight_decay=.00003影响。每轮记录真实参数更新，分类梯度非零；辅助模型只读、不在优化器中。

辅助策略保留legacy_epoch_snapshot：训练函数开始同步、辅助eval，主模型train且正常dropout；batch0权重相同，后续batch存在既有权重滞后。训练CC混合了视图、dropout模式与滞后，不能解释成纯信息删除效应。分项CEL评估则全部使用相同checkpoint权重与eval模式。same_step_same_mode属于后续独立策略实验，不作为修复替换本配置。

本批没有中断/恢复重启；保存状态满足已验证的完整resume协议，未把仅加载模型权重称为续训。本轮不重复第一轮大规模合成审查或声称跨GPU逐位确定。

证据：CMERC_VERIFICATION.json、CMERC_EPOCH_METRICS.json、runs/cmerc/*/{metrics.jsonl,training_diagnostics.jsonl,run_manifest.json,best.pt,latest.pt}、cmerc_training.log与PIPELINE_STATE.json。
