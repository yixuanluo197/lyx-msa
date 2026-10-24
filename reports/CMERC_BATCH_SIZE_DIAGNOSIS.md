# CMERC Batch Size诊断：资源阻塞，尚未训练

本文件是阶段状态报告，不是实验完成报告。唯一登记任务为MELD完整非因果CMERC，seed2024/split_seed1475、LR=0.0001、Batch8、40轮。当前GPU首Batch探针0次、正式训练0次，不存在本次验证F1或Checkpoint；结果未知。

## 历史对照与待完成比较

| 指标 | Batch16（XP04真实归档） | Batch8（09） |
|---|---:|---|
| 最佳验证W-F1 | 66.01% | 待训练 |
| 最佳Epoch | 9 | 待训练 |
| 末10轮验证W-F1均值 | 61.223% | 待训练 |
| 40轮优化步数 | 2548（CPU重建） | 预计5096（尚无实际步数） |

控制只读取原始run_manifest和40行训练/验证metrics，不重复训练，不读取历史最终测试结果。实际新增Loss及曲线尚不存在，不能分析本次稳定性、过拟合或提升，亦不能决定第二Seed或最终五Seed。

## 已完成审计与CPU预检

- 12个实际训练源字节同08，保留07/08的skip-final-test工程门禁；模型、损失公式和权重不改。训练循环AST与XP04相同。
- 科学参数逐项约束仅Batch16→8，LR固定0.0001；Adam/L2=3e-5、dropout0.4、GRU/ATen、continuous、capacity_safe、hyper/direct、concat_DHT/BN/L3/K3，CC/MSCL/CSCL=.002/.15/.15、CL=.4/1，验证WeightedF1严格改善选模、平局最早。
- 相同1038训练/114验证对话及原课程数据SHA，split_seed1475，完整生效配置和argv预登记。
- 8项CPU训练保护测试、5项纯合成结果规则测试通过；CUDA未初始化。40轮采样容量重放通过，B8最多167超边/429节点，小于1000/5200；训练池与整轮扁平ID顺序未改变，无过滤、无容量重试。
- B16原日志容量/retry重放匹配；B8旧/新Sampler AST及逐批输出一致。B8预计5096步，B16重建2548步，验证600/320批。CPU预检不代替GPU实际显存及前后向检查。
- 04–08十个历史根目录共1691个文件和30个符号链接已建立训练前保全记录；完成后另行比对，不提前宣称最终保全验收通过。

## 当前资源阻碍及执行状态

资源采样时间`2026-10-24T13:39:37.994647+00:00`：本地user23四张TITAN Xp均有高负载任务。登记的GPU3 UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456有新增外部任务，利用率99%、空闲2399 MiB，未达free>=5000 MiB、util<=65及进程门禁。实时状态见state/status.json。

后台监督器仍处于preflight_admission，只监测资源，不占用训练GPU、不终止其他项目、不迁移服务器。资源满足登记条件后，自动执行一次独立首Adam探针；四项Loss/总Loss与梯度有限且参数更新、GPU释放后重新检查资源，再从头执行唯一一次40轮正式训练。探针不向正式训练提供模型、Adam或RNG。技术失败保留证据，不按分数重训，不追加实验。

正式训练后CPU finalizer会独立核验40train/40validation/0test、真实Batch轨迹、best/latest模型/Adam/RNG/参数、冻结输入和历史保全，再生成本报告的正式结果版本与20个PNG/SVG图。状态以state/status.json及state/finalization.json为准；不能把已登记、已排队、CPU通过称为训练成功。

## 比较边界和预登记判断

原--batch-size同时控制train与validation分组，因此本次两者都16→8。模型双向GRU直接处理未pack的padded序列，BN含padding，验证分组也可能改变固定权重预测。W-F1仍按固定完整验证池统一计算，不能将本次称为仅训练分组改变的对照。

Batch变化同时改变每Epoch Adam/L2更新次数、BN统计、HSCL跨对话正负对、未归一化CC总和尺度和dropout实际RNG消耗；原Loss为逐Batch均值汇总，不能直接把跨Batch数值解释为统一每样本归一化目标。课程难度作者文件生成历史未知且含validation键；保持同源配对，不称完整预处理链无验证信息。综合pickle/test-loader元数据仍按原代码加载，但禁止测试loader遍历或测试前向，不计算新测试F1。

最佳验证F1提高且末10轮均值不降，才视作第二Seed确认候选；仅最佳提高为混合线索；最佳未提高则本Seed不支持主指标改善。单Seed不证明跨Seed稳定收益或统计显著。本批不自动启动第二Seed、其他学习率、其他Batch、Baseline或五Seed训练。

## 证据入口

[独立批次](../experiments/batch_size_diagnosis/20261024T131956Z/README.md)、[训练前登记](../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)、[源码协议审计](../experiments/batch_size_diagnosis/20261024T131956Z/reports/PROTOCOL_AUDIT.md)、[配置/完整argv](../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)、[冻结清单](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/frozen_execution.json)、[数据SHA](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/data_inputs.json)、[环境](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/environment.json)、[CPU保护测试](../experiments/batch_size_diagnosis/20261024T131956Z/preflight/guard_tests.json)、[容量重放](../experiments/batch_size_diagnosis/20261024T131956Z/preflight/capacity_cpu.json)、[上传时监督器快照](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/publication_status.json)、[执行交接](../experiments/batch_size_diagnosis/20261024T131956Z/HANDOFF.md)。

## 本次GitHub发布状态

截至`2026-10-24T14:13:23.658649+00:00`，GPU探针0次、正式训练0次，仍为资源准入排队。GitHub保存上传时的报告与监督器快照，不会实时反映服务器队列；运行后的正式结果需另行同步。原始数据、权重、实时state和原始日志未上传，发布不触发新的训练或评估。
