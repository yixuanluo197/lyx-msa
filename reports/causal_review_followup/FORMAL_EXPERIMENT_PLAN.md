# 正式因果实验计划（仅计划，未启动）

## 开始条件

代码级 CPU/GPU 因果回归、有限 batch、恢复与真实特征检查具有本轮证据。完整 CL 尚缺基于同一划分真实训练、带完整 provenance 的 causal baseline best checkpoint 及由它生成的不可变 difficulty。先解决该依赖，再提交完整矩阵；不能把合成流程成功、两步真实特征诊断权重或旧非因果 difficulty 当作这一依赖。

## 固定项

- 同一 causal backbone、生产维度、特征字节版本、预处理、训练预算、验证选模规则和指标实现。
- IEMOCAP split_seed 固定1475，MELD 固定官方 train/validation IDs；训练 seed 另行登记。初始建议配对训练 seeds `[1937,2025,3407]`，正式开始前固定，不能根据测试结果选择。
- 固定 replacement policy=`prefix_mean_v1`；它是可逆前缀平滑，不称完整模态移除。CEL 分为 text/audio/visual、intra/inter GNN-edge 五项，跨 policy 不直接比较。
- 默认实验固定 `legacy_epoch_snapshot`；`same_step_same_mode` 若比较，作为单独预注册的辅助协议因子，使用自己的 run manifest，不当作同一实现的小补丁。
- checkpoint 只按 validation W-F1 选；正式测试集只用于方案冻结后的最终评估，不能中途挑 epoch/seed/超参数。
- 保留当前入口每次训练函数调用重置 RNG 的 legacy 行为，并记录 backend/dropout；改成 continuous RNG 或其他后端应另建实验身份，不能继承其他批次名。

## 顺序

1. 先做 matched causal baseline：CL=off、HSCL=off、CC=off。每个 seed 的模型与完整 causal CMERC 使用同一初始化/数据划分和训练协议。
2. 核验每个 baseline best 的实际 train/valid 特征/ID 哈希、split/train seed、epoch/best_epoch、选模、架构、policy 和 SHA。由对应真实权重在训练 IDs 上生成 difficulty；独占保存 hash-addressed artifact/manifest，消费者绑定同 seed 和 SHA。
3. 完整 causal CMERC：CL+HSCL+CC，在同一配对 seed 下运行，验证集选模。
4. 组件消融：full minus CL、full minus HSCL、full minus CC；必要时再加入单组件配置，但不能看测试结果后临时追加有利组合。
5. 每 seed 保存所有计划运行、失败/重启、epoch日志、选模对象、完整状态与环境；汇总配对差及变异，不只选最好 seed。GPU 设备按当前服务器归属与健康白名单排队，目录与运行代码冻结，不能同步修改正在运行版本。

## 报告

W-F1、ECE（固定 bins=15）和五路分项 CEL 在同一 replacement/auxiliary policy 下报告；补充 Macro-F1、校准指标与每 seed 配对差。区分原始值、校准视图分量、开关与实际加权梯度。统计方法与汇总规则在启动前固定。

本轮不配置80/40轮主实验或多 seed 作业，不生成“已完成”性能表。非因果与因果不仅删除未来边，还改变 RNN 宽度/方向、超图操作、归一化和替换策略；两者指标差不能全部解释为“只删除未来信息”。原始特征提取链未审计，也不宣称原始流端到端时间因果性。
