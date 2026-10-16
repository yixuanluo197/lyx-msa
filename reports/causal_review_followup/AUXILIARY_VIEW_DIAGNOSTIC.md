# 辅助视图与真实训练分支诊断

替换策略为 `prefix_mean_v1`，没有新增或偷偷改成严格缺失/零值 policy。

`m_t=(x_1+...+x_t)/t` 的实测最大误差 2.38418579e-07；逆式 `x_t=t*m_t-(t-1)*m_(t-1)` 最大误差 4.76837158e-07。完整均值前缀可逆，所以这是前缀平滑视图，不是模态完全移除。

同权重、同 eval 的完整概率向量首句 no-op 共 60 项通过；含 GRU/LSTM/Transformer、av_encoder/residue 两值及五种 view，不只比较 argmax。

24 项历史梯度检查通过：Model_Intra/Inter 只删除 GNN 中对应 speaker 的时间边；编码器和定向超图仍保留历史。不得称为完全切断同/异 speaker 上下文。

## 真实训练函数两个 optimizer step

下面为实际训练函数的 full(CC+HSCL) CPU 配置。每个策略均从相同初始化与数据种子开始。记录的是使用置信度视图时的权重差和 mode；不是用训练后再同步权重掩盖滞后。

| policy | dropout | batch 1 主/辅助权重差 | batch 2 权重差 | 平均CC | main mode / auxiliary mode |
| --- | ---: | ---: | ---: | ---: | --- |
| legacy_epoch_snapshot | 0.0 | 0 | 0.00100004673 | 0.0519703627 | True / False |
| legacy_epoch_snapshot | 0.15 | 0 | 0.00100004673 | 0.87695539 | True / False |
| same_step_same_mode | 0.0 | 0 | 0 | 0.0350345373 | True / False |
| same_step_same_mode | 0.15 | 0 | 0 | 0.035117805 | True / False |

legacy 的 epoch snapshot 在第二步确有权重滞后，第一步也因主 train/辅助 eval 而有正常 dropout 模式差异。故原 confidence 差不全是信息替换的作用。same_step 是另一项显式实验配置，保持默认 legacy 不变。每一步五类 CC 与实际生成的 HSCL 二值标签均存入原始 JSON，不能只报总体 loss。

## 梯度策略与后端

分类 pass 始终保持原训练模式及正常 dropout，保留梯度。legacy CC 参考分类分数，辅助视图固定为 epoch snapshot/eval/no_grad；HSCL 用分类 hidden 且标签只读。same_step 在每 batch 同步后，full confidence reference 与五视图使用同权重/eval/ATen；full 为 CC 保留梯度，视图和离散标签 no_grad，HSCL 特征仍来自 train pass。比较 confidence 时不混合后端，完成后恢复主模型 mode。

GPU cuDNN eval RNN 不支持反向的原始失败保留，ATen 局部修复没有重新开启 eval dropout。四损失各自未加权和按实际系数加权的梯度范数、实际参数更新及辅助无梯度状态见 `TRAINING_INTEGRATION_cpu.json`、`TRAINING_INTEGRATION_cuda.json`。

没有将离线总损失的跨样本梯度误判为单次前向未来泄漏；本报告的未来梯度结论针对固定参数下的单位置 scalar 输出。没有真实教师时，CL 仅用明确标注的合成 manifest 验证流程。

原始分项说明：[AUXILIARY_VIEW_CORE.md](AUXILIARY_VIEW_CORE.md)、[AUXILIARY_TRAINING_DIAGNOSTIC.md](AUXILIARY_TRAINING_DIAGNOSTIC.md)；完整数值：[CAUSALITY_REGRESSION.json](CAUSALITY_REGRESSION.json)、[TRAINING_INTEGRATION.json](TRAINING_INTEGRATION.json)。
