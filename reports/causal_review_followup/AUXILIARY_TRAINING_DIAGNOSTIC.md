# 辅助训练分支诊断

保留 `legacy_epoch_snapshot` 为默认：每次训练函数开始把主模型权重复制到三辅助模型并设为 eval。主模型在 train 中每 batch 更新。实际两 step 记录证明 batch 0 主/辅助权重相同，batch 1 权重已不同；dropout=.15 时 batch 0 虽权重相同，主 train / 辅助 eval 仍有模式差异。因此原协议的 confidence 差值不能一律解释为“仅信息替换”的效应。逐项 CC 和实际 HSCL 标签保存在 `TRAINING_INTEGRATION_cpu.json` 每配置 `batches` 字段。

新增 `same_step_same_mode` 是单独显式实验配置，默认行为未替换。每 batch 用当前主模型同步辅助模型；full confidence reference 与所有 view 同权重、eval、ATen 后端。classification pass 保持 train 和正常 dropout，有梯度；HSCL 使用 classification hidden，有梯度；HSCL 的离散参考标签由同权重/eval confidence 分数生成，无标签梯度；removed view 始终 no_grad；CC full reference 用主模型 eval 前向保留梯度，view 固定，无辅助模型优化。前向之后恢复主模型原 mode。

GPU 的局部 ATen 后端是 cuDNN eval RNN 不支持 backward 的工程修复，不改变所要求的 eval/dropout 语义。保持 reference 和所有 view 后端一致，避免把 cuDNN/ATen 数值差异也掺入置信度比较。正常 dropout 的另一个 train 集成分支已执行。

两种 policy 的 batch 2 权重差、五类 CC、五组 HSCL 标签和每损失梯度都已记录。它们属于不同实验协议，不能将跨 policy 的指标差异全部归因为替换策略或时间因果性；正式实验必须固定 policy。

prefix_mean 的可逆性、首句完整概率 no-op 和 GNN-edge ablation 边界由 `AUXILIARY_VIEW_DIAGNOSTIC.md` 与完整因果回归单独报告。本文件只解释真实训练函数的辅助权重滞后和模式差异。
