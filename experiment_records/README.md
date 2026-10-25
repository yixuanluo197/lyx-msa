# CMERC 实验档案

历史01/02/03分别保留原版V100复现、旧XP多seed和V100多seed的代码与证据；V100的V1失败诊断单独标记，不混入V2正式统计。04为XP主研究基线，20项五seed配对训练与独立评估已完成并通过技术验收。05为官方README流程的4项单Seed对照，单独保留测试集选模及兼容补丁记录，不改变04的主线登记。

06为接续04/05的3项最小消融：MELD关闭CC，IEMOCAP为Baseline/CMERC改变Batch排列。全部采用验证集选模与最终一次测试，只提供单Seed根因定位线索，不替代04的五Seed主线。

07固定XP04完整CMERC，仅比较两个新学习率与历史0.0001控制，共4项训练；验证集选模且不做最终测试。IEMOCAP的峰值改善伴随末段下降，MELD有待第二Seed确认的候选；不与历史测试集分数混报。

08接续07选定的MELD学习率0.0002，以预留seed2024完成唯一一次40轮确认，采用相同验证协议、零测试前向。最佳验证值未提高，末段均值改善作为辅助线索保留；暂缓五Seed和Batch8。

| 实验系列 | 服务器与设备 | 正式范围 | 状态 | 入口 |
| --- | --- | --- | --- | --- |
| 原版单 seed 复现 | `biggpu` / V100 | 2 数据集 × baseline/CMERC × official_test/validation，共 8 run | 已完成 | [01_v100_single_seed](01_v100_single_seed/README.md) |
| XP 多 seed | 本地 `user23` / TITAN Xp | 10 个 CMERC + 2 个 baseline，共 12 run | 已完成 | [02_xp_multiseed](02_xp_multiseed/README.md) |
| V100 多 seed V2 | `biggpu` / V100 | 5 seed × 2 数据集 × baseline/CMERC，共 20 run | 已完成 | [03_v100_multiseed](03_v100_multiseed/README.md) |
| XP 主研究基线 | 本地 `user23` / TITAN Xp | 5 seed × 2 数据集 × baseline/CMERC，共20 run；validation/continuous/ATen/capacity_safe | 已完成并通过验收 | [04_xp_mainline_validation](04_xp_mainline_validation/README.md) |
| 官方 README 单 Seed 对照 | 本地 `user23` / TITAN Xp GPU3 | IEMOCAP seed1475、MELD seed67137，各含baseline/CMERC，共4 run；official_test_selection | 已完成并通过验收 | [05_official_readme_reproduction](05_official_readme_reproduction/README.md) |
| 最小消融根因验证 | 本地 `user23` / TITAN Xp GPU3 | MELD seed67137 w/o CC；IEMOCAP seed1475 source_order Baseline/CMERC，共3 run；validation/continuous/ATen | 已完成并通过验收 | [06_minimal_ablation_validation](06_minimal_ablation_validation/README.md) |
| 完整CMERC学习率诊断 | 本地 `user23` / TITAN Xp GPU3 | IEMOCAP seed1475、MELD seed67137，各比较0.00005和0.0002，共4 run；validation/continuous/ATen，零测试前向 | 已完成并通过验收 | [07_learning_rate_diagnosis](07_learning_rate_diagnosis/README.md) |
| MELD学习率第二Seed确认 | 本地 `user23` / TITAN Xp GPU3 | MELD seed2024、lr0.0002，唯一40轮完整CMERC；validation/continuous/ATen，零测试前向 | 训练与技术验收完成；主指标收益未确认 | [08_lr_second_seed_confirmation](08_lr_second_seed_confirmation/GITHUB_DELIVERY.md) |
| MELD Batch Size诊断 | 本地 `user23` / TITAN Xp GPU1 | MELD seed2024、lr0.0001、Batch8，完成唯一40轮；验证选模、零测试前向 | 最佳验证65.89%/第9轮，末10轮63.048%；独立验收通过 | [09_batch_size_diagnosis](09_batch_size_diagnosis/GITHUB_DELIVERY.md) |
| MELD Dropout诊断 | 本地 `user23` / TITAN Xp GPU3 | MELD seed2024、lr0.0001、Batch16、Dropout0.5，唯一40轮；验证选模、零测试前向 | 最佳66.49%/e15、末10轮62.775%；验收通过，第二Seed候选 | [10_dropout_diagnosis](10_dropout_diagnosis/README.md) |
| MELD Dropout第二Seed确认 | 本地 `user23` / TITAN Xp GPU0 | MELD seed67137、lr0.0001、Batch16、Dropout0.5，唯一40轮；验证选模、零新增测试前向 | 最佳66.41%/e17、末10轮62.440%；验收通过，混合证据 | [11_dropout_second_seed_confirmation](11_dropout_second_seed_confirmation/README.md) |
| MELD HSCL最小机制消融 | 本地 `user23` / TITAN Xp GPU0 | MELD seed67137、Dropout0.4、lr0.0001、Batch16、w/o HSCL，唯一40轮；保留CC和CL，验证选模、零新增测试前向 | 最佳65.69%/e5、末10轮62.794%；验收通过，混合证据 | [12_hscl_ablation_validation](12_hscl_ablation_validation/README.md) |
| 辅助损失梯度与伪标签诊断 | 本地 `user23` / TITAN Xp GPU0 | XP04 e8/e40与XP12 e5/e40，同32训练Batch；768前向、512独立autograd；训练/step/测试均0 | 128单元及独立验收通过；唯一优先系数证据不足 | [13_auxiliary_loss_diagnostic](13_auxiliary_loss_diagnostic/README.md) |

## 如何交给 GPT 审查

先提供本页，再指定需要审查的实验目录。历史组的 `code/` 对应其保留或冻结的源码；05的公开入口指向原始独立目录 `experiments/official_readme_reproduction/`，其中 `official_code/` 和 `runtime_code/` 分别为原始与兼容修复副本。配置、逐轮指标、独立评估、最终汇总与报告均保留。历史 `export_manifest.json` 以及05的 `publication_manifest.json` 记录来源或发布文件 SHA-256；实验本身已有的 manifest、快照清单与校验记录继续保留。

建议审查请求：

> 请先读取 experiment_records/README.md 和指定组的 README，区分服务器、代码版本、数据划分、选模协议、随机状态和课程策略。按各组源码、配置及逐轮指标核对最终报告，重点检查训练与测试隔离、seed 汇总、checkpoint 选取、失败处理及版本一致性。请指出实际读取的文件，以文件行号和原始指标支持每项结论，不将不同协议或 V1/V2 混成一组。

## 可比性与产物边界

- 原版01的两套选模协议分别保留；旧XP02使用逐轮测试集选模的 `official_test`；V100 V2 03与XP04使用验证集选模、固定split seed、continuous RNG、ATen RNN和capacity_safe。不同组独立展示，不合并均值，不把环境差异直接归因于显卡。
- 05保留官方 `official_test_selection`、RNG段首重置、cuDNN GRU及完整原始训练池。其固定checkpoint独立进程重评不消除此前测试选轮偏差；每数据集仅一个Seed，不能替代五Seed统计，也不能把与04的差值直接视为同协议因果效应。
- 06保持04对应Seed的划分与验证选模，分别登记CC开关或Batch排列；历史完整CMERC跨物理卡与CUDA非确定性限制严格因果解释。IEMOCAP验证负增量缩小不等于绝对提升，MELD去CC的验证证据混合；暂缓最终五Seed。
- 07保持04完整模型及全部辅助项，仅改变学习率；新增训练没有测试评估。MELD按预登记最佳值优先规则首选0.0002，但末段只提高0.074个百分点，不代表稳健改善；IEMOCAP尚无满足资格的第二Seed候选。本批不追加Seed、Batch8、Baseline或最终测试。
- 08以预留seed2024独立确认07的MELD候选。最佳验证65.46%低于XP04同Seed的66.01%，末10轮均值62.034%高于61.223%；按原规则为主指标未改善，不据末段结果更换确认门槛或宣称两Seed显著。保留同卡配对及共享资源限制，不新增测试、Baseline或Batch8。
- 旧XP02的两个baseline只是单seed诊断，不能支持五seed配对提升统计。V100 V2 03和XP04均有两种模型的完整五seed配对；04的主线登记依据完整性与协议验收。
- V100 V1 的容量失败与中间进度报告是历史证据；最终结论使用 V2 完整结果。`first3` 是中间三 seed 版本，不能覆盖完整五 seed 汇总。
- 原版复现未保存启动时的源码哈希。其 `code/` 来自原服务器留存工作目录的本次导出，不能将公开提交或导出哈希倒填为运行时冻结版本。
- 大数据、模型权重、逐样本预测、二进制数组、缓存和原始日志保留在实验服务器，不进入公开档案。保留了逐轮聚合指标与独立评估的聚合指标。
- 完整的 checkpoint／预测重算脚本仍需要原服务器产物；仅克隆公开仓库可以进行代码与聚合证据审查，不能无条件重做全部逐样本验证。
- 各组源码可能保留绝对实验路径。重跑时先依照该组 README 准备环境和外部数据，不能直接执行历史启动命令覆盖旧目录。

历史01/02/03原目录保持不变。04新增20个完整正式run，另保留1个因资源竞争失效的部分训练尝试；训练逻辑继承V100 V2冻结源码，资源适配与恢复单独审计，未自动提交或推送远程。

## 当前主研究基线

04 为 XP 主研究基线：非因果、验证集选模、固定外部课程、五 seed 配对。20项完整性与协议验收通过后登记，分数不作为登记门槛。历史01/02/03原目录不变；不同协议/环境的统计分别展示。大型权重、逐样本预测和原始日志保留原批次及独立备份，公开04仅提供代码、配置和审查材料。

09按登记规则为最佳验证主指标未改善（65.89% vs 66.01%），末10轮改善1.825个百分点仅作辅助线索；暂不追加第二Seed或最终五Seed。两loader分组与历史GPU物理卡均变化，限制严格机制归因。

10仅提高Dropout0.4→0.5，保持XP04同Seed及其余解析科学参数。最佳和末段W-F1均提高，但验证分类Loss升高，单Seed不证明稳定收益或过拟合根因；只建议另行登记第二Seed，本组未执行额外训练或测试。

11复用10的seed2024，以seed67137做唯一第二Seed确认：最佳+1.17 pp但末段−0.469 pp、波动增大；两Seed最佳均提高而末段方向不一致，未满足原登记完整一致性规则，暂缓五Seed。验证分类Loss未改善；偶然历史测试字段访问事件已公开说明，未据测试决策、零新增测试前向。

12只关闭HSCL优化贡献：最佳验证+0.450pp、末段−0.115pp，属混合证据。原始MSCL/CSCL仍计算，两项有效贡献0；CC梯度在独立probe已证实，CL和其他科学参数保持。末段分类Loss略降不等于F1稳定改善；不追加Seed、五Seed或测试。04～11历史文件未变。
