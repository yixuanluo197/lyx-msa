# 第09组Batch Size诊断：完成报告发布

唯一登记正式训练已在本地user23 GPU1完成40轮，并通过独立CPU验收。MELD完整非因果CMERC，seed2024/split_seed1475、LR0.0001、Batch8；验证集W-F1选模，全程零测试前向。

| 指标 | Batch16（XP04） | Batch8（09） |
|---|---:|---:|
| 最佳验证W-F1 | 66.01% | 65.89% |
| 最佳Epoch | 9 | 9 |
| 末10轮验证均值 | 61.223% | 63.048% |

最佳值下降0.12个百分点，末段提高1.825个百分点；按预登记规则为主指标未改善，暂不追加第二Seed或最终五Seed。后期过拟合迹象较轻属于辅助线索，尚未证明跨Seed收益或内部机制因果。原参数同时改变train/validation分组，且控制GPU3、本次GPU1；这些限制保留。

## 公开证据

- [完整中文报告](../../reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)、[独立实验目录](../../experiments/batch_size_diagnosis/20261024T131956Z/README.md)、[预登记](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)
- [配置及完整argv](../../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)、[实际解析参数](../../experiments/batch_size_diagnosis/20261024T131956Z/results/MELD_cmerc_batch8_lr1e-4_seed2024/effective_args.json)、[环境](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/environment.json)
- [对照CSV](../../experiments/batch_size_diagnosis/20261024T131956Z/results/comparison.csv)、[逐轮曲线CSV](../../experiments/batch_size_diagnosis/20261024T131956Z/results/curves.csv)、[各项Loss](../../experiments/batch_size_diagnosis/20261024T131956Z/results/loss_curves.csv)、[汇总JSON](../../experiments/batch_size_diagnosis/20261024T131956Z/results/metrics_summary.json)
- [独立验收](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/final_verification.json)、[完成前哈希及指标复核](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/completion_recheck.json)、[20件图及SHA](../../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/manifest.json)
- [资源门禁修订v2](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/RESOURCE_ADMISSION_AMENDMENT_V2.md)、[GPU0/1/3选择修订v3](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/XP_DEVICE_SELECTION_AMENDMENT_V3.md)、[导航脚本启动竞态修复记录](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/delivery_recovery.json)

## 发布边界

92项运行冻结源/输入、1691项历史文件保全通过；best/latest为第9/40轮，模型及Adam有限并保存完整随机状态，SHA见验收文件。450次资源采样显示共享GPU正式阶段最小空闲31MiB，未发生本任务OOM；不能声明独占或证明未影响其他任务吞吐。未终止其他项目进程，未追加训练。

当前分支已用完成实验的报告、结果和发布状态覆盖原09组待运行版本：原路径[publication_manifest.json](publication_manifest.json)及批次publication_status/publication_checks均表示唯一40轮训练已完成。先前提交`adea816600aa014652d78dcc11e488f60935465b`的0次训练快照已移至[历史发布目录](publication_history/adea816/README.md)，仅用于追溯。本次完整文件SHA见[publication_manifest.json](publication_manifest.json)；[final_publication_manifest.json](final_publication_manifest.json)为同内容副本。

数据与其本地链接、模型权重、原始训练/资源日志、逐Batch/逐Epoch原始观测、实时state和锁保留服务器，未上传。公开CSV为训练/验证指标和Loss汇总，不含新测试结果。冻结清单中的绝对路径及未公开文件SHA用于溯源，仅克隆GitHub仓库不能重算所有服务器验收。
