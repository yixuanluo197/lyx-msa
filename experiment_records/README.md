# CMERC 实验档案

历史01/02/03分别保留原版V100复现、旧XP多seed和V100多seed的代码与证据；V100的V1失败诊断单独标记，不混入V2正式统计。04为新增的XP主研究基线，20项五seed配对训练与独立评估已完成并通过技术验收。

| 实验系列 | 服务器与设备 | 正式范围 | 状态 | 入口 |
| --- | --- | --- | --- | --- |
| 原版单 seed 复现 | `biggpu` / V100 | 2 数据集 × baseline/CMERC × official_test/validation，共 8 run | 已完成 | [01_v100_single_seed](01_v100_single_seed/README.md) |
| XP 多 seed | 本地 `user23` / TITAN Xp | 10 个 CMERC + 2 个 baseline，共 12 run | 已完成 | [02_xp_multiseed](02_xp_multiseed/README.md) |
| V100 多 seed V2 | `biggpu` / V100 | 5 seed × 2 数据集 × baseline/CMERC，共 20 run | 已完成 | [03_v100_multiseed](03_v100_multiseed/README.md) |
| XP 主研究基线 | 本地 `user23` / TITAN Xp | 5 seed × 2 数据集 × baseline/CMERC，共20 run；validation/continuous/ATen/capacity_safe | 已完成并通过验收 | [04_xp_mainline_validation](04_xp_mainline_validation/README.md) |

## 如何交给 GPT 审查

先提供本页，再指定需要审查的实验目录。每组的 `code/` 对应其保留或冻结的源码，配置、逐轮指标、独立评估、最终汇总与报告在同一档案下。各组 `export_manifest.json` 记录原文件来源和 SHA-256；实验本身已有的 manifest、快照清单与校验记录继续保留。

建议审查请求：

> 请先读取 experiment_records/README.md 和四组 README，区分服务器、代码版本、数据划分、选模协议、随机状态和课程策略。按各组 code/、配置及逐轮指标核对最终报告，重点检查训练与测试隔离、seed 汇总、checkpoint 选取、失败处理及版本一致性。请指出实际读取的文件，以文件行号和原始指标支持每项结论，不将不同协议或 V1/V2 混成一组。

## 可比性与产物边界

- 原版01的两套选模协议分别保留；旧XP02使用逐轮测试集选模的 `official_test`；V100 V2 03与XP04使用验证集选模、固定split seed、continuous RNG、ATen RNN和capacity_safe。不同组独立展示，不合并均值，不把环境差异直接归因于显卡。
- 旧XP02的两个baseline只是单seed诊断，不能支持五seed配对提升统计。V100 V2 03和XP04均有两种模型的完整五seed配对；04的主线登记依据完整性与协议验收。
- V100 V1 的容量失败与中间进度报告是历史证据；最终结论使用 V2 完整结果。`first3` 是中间三 seed 版本，不能覆盖完整五 seed 汇总。
- 原版复现未保存启动时的源码哈希。其 `code/` 来自原服务器留存工作目录的本次导出，不能将公开提交或导出哈希倒填为运行时冻结版本。
- 大数据、模型权重、逐样本预测、二进制数组、缓存和原始日志保留在实验服务器，不进入公开档案。保留了逐轮聚合指标与独立评估的聚合指标。
- 完整的 checkpoint／预测重算脚本仍需要原服务器产物；仅克隆公开仓库可以进行代码与聚合证据审查，不能无条件重做全部逐样本验证。
- 各组源码可能保留绝对实验路径。重跑时先依照该组 README 准备环境和外部数据，不能直接执行历史启动命令覆盖旧目录。

历史01/02/03原目录保持不变。04新增20个完整正式run，另保留1个因资源竞争失效的部分训练尝试；训练逻辑继承V100 V2冻结源码，资源适配与恢复单独审计，未自动提交或推送远程。

## 当前主研究基线

04 为 XP 主研究基线：非因果、验证集选模、固定外部课程、五 seed 配对。20项完整性与协议验收通过后登记，分数不作为登记门槛。历史01/02/03原目录不变；不同协议/环境的统计分别展示。大型权重、逐样本预测和原始日志保留原批次及独立备份，公开04仅提供代码、配置和审查材料。
