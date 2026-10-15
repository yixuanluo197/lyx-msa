# 最近三组 CMERC 实验档案

本次将原版 V100 复现、XP 多 seed、V100 多 seed 三个实验系列的相关代码与证据整理为公开档案。V100 多 seed 的 V1 失败诊断归入同一系列，单独标记，不混入 V2 正式统计。

| 实验系列 | 服务器与设备 | 正式范围 | 状态 | 入口 |
| --- | --- | --- | --- | --- |
| 原版单 seed 复现 | `biggpu` / V100 | 2 数据集 × baseline/CMERC × official_test/validation，共 8 run | 已完成 | [01_v100_single_seed](01_v100_single_seed/README.md) |
| XP 多 seed | 本地 `user23` / TITAN Xp | 10 个 CMERC + 2 个 baseline，共 12 run | 已完成 | [02_xp_multiseed](02_xp_multiseed/README.md) |
| V100 多 seed V2 | `biggpu` / V100 | 5 seed × 2 数据集 × baseline/CMERC，共 20 run | 已完成 | [03_v100_multiseed](03_v100_multiseed/README.md) |

## 如何交给 GPT 审查

先提供本页，再指定需要审查的实验目录。每组的 `code/` 对应其保留或冻结的源码，配置、逐轮指标、独立评估、最终汇总与报告在同一档案下。各组 `export_manifest.json` 记录原文件来源和 SHA-256；实验本身已有的 manifest、快照清单与校验记录继续保留。

建议审查请求：

> 请先读取 experiment_records/README.md 和三组 README，区分服务器、代码版本、数据划分、选模协议、随机状态和课程策略。按各组 code/、配置及逐轮指标核对最终报告，重点检查训练与测试隔离、seed 汇总、checkpoint 选取、失败处理及版本一致性。请指出实际读取的文件，以文件行号和原始指标支持每项结论，不将不同协议或 V1/V2 混成一组。

## 可比性与产物边界

- 原版复现的两套选模协议分别保留；XP 使用逐轮测试集选模的 `official_test`；V100 V2 使用验证集选模，以及固定 split seed、continuous RNG、ATen RNN、capacity_safe 批次排列。三组不能直接混算均值。
- XP 的两个 baseline 只是单 seed 诊断，不能支持五 seed 配对提升统计。V100 V2 才有两种模型的完整配对五 seed。
- V100 V1 的容量失败与中间进度报告是历史证据；最终结论使用 V2 完整结果。`first3` 是中间三 seed 版本，不能覆盖完整五 seed 汇总。
- 原版复现未保存启动时的源码哈希。其 `code/` 来自原服务器留存工作目录的本次导出，不能将公开提交或导出哈希倒填为运行时冻结版本。
- 大数据、模型权重、逐样本预测、二进制数组、缓存和原始日志保留在实验服务器，不进入公开档案。保留了逐轮聚合指标与独立评估的聚合指标。
- 完整的 checkpoint／预测重算脚本仍需要原服务器产物；仅克隆公开仓库可以进行代码与聚合证据审查，不能无条件重做全部逐样本验证。
- 各组源码可能保留绝对实验路径。重跑时先依照该组 README 准备环境和外部数据，不能直接执行历史启动命令覆盖旧目录。

本次整理没有启动新训练、修改现有训练实现或覆盖已有实验产物。
