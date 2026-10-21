# 官方 README 四组单 Seed 对照：公开交付

本批已完成4项从头训练、4项固定最佳checkpoint独立进程重评及2项不更新参数的单Batch梯度诊断。完整实验目录保留在 [experiments/official_readme_reproduction](../../experiments/official_readme_reproduction/README.md)，本页仅提供GitHub导航和发布边界，不复制或替换原始实验。

官方固定commit：`815c7c060914f5b55db816c0ba3662b11d08552f`。执行服务器为本地 `user23`、TITAN Xp GPU3；IEMOCAP训练80轮，MELD训练40轮。实际解析参数、命令、环境、源码/数据/课程哈希及完整状态记录均已保留。

## 真实结果

| 数据集 | Seed | 模型 | 本次W-F1 % | XP04同Seed W-F1 % | 论文参考W-F1 % |
|---|---:|---|---:|---:|---:|
| IEMOCAP | 1475 | Baseline | 70.0432 | 66.9361 | 69.61 |
| IEMOCAP | 1475 | CMERC | 70.5108 | 68.2802 | 71.98 |
| MELD | 67137 | Baseline | 66.2257 | 65.4386 | 65.37 |
| MELD | 67137 | CMERC | 66.1301 | 65.7461 | 66.85 |

本次CMERC相对Baseline的增量为IEMOCAP **+0.4676**、MELD **−0.0956** 个百分点。绝对分数高于XP04对应Seed，但未恢复论文报告的方法增量。

本批使用官方 **`official_test_selection`**，XP04按验证集选模，训练池、RNG、GRU后端和batch也不同；上述差值仅作描述性比较。独立进程重评不会消除此前测试选轮偏差。每数据集一个Seed，不代表五Seed均值。作者难度文件来源及哈希已固定，但生成模型/历史未知；没有使用历史checkpoint初始化本次正式训练。

## 阅读与验证

以下路径均指向原实验目录：

1. [官方源码审计](../../experiments/official_readme_reproduction/reports/01_OFFICIAL_SOURCE_AUDIT.md)
2. [README与XP04协议差异](../../experiments/official_readme_reproduction/reports/02_README_PROTOCOL_COMPARISON.md)
3. [环境、资源与兼容预检](../../experiments/official_readme_reproduction/reports/03_ENVIRONMENT_PREFLIGHT.md)
4. [四组训练报告与验收](../../experiments/official_readme_reproduction/reports/04_FOUR_RUN_TRAINING_REPORT.md)
5. [数值与训练曲线比较](../../experiments/official_readme_reproduction/reports/05_XP04_VS_README_COMPARISON.md)
6. [差距诊断与最小后续对照建议](../../experiments/official_readme_reproduction/reports/06_F1_GAP_DIAGNOSIS_AND_NEXT_STEPS.md)

机器结果：[per_run.csv](../../experiments/official_readme_reproduction/results/per_run.csv)、[comparison.csv](../../experiments/official_readme_reproduction/results/comparison.csv)、[metrics_summary.json](../../experiments/official_readme_reproduction/results/metrics_summary.json)。验收记录：[final_verification.json](../../experiments/official_readme_reproduction/provenance/final_verification.json)；独立审查：[independent_review.md](../../experiments/official_readme_reproduction/provenance/independent_review.md)。

`official_code/`为固定上游副本，`runtime_code/`仅包含两处PyG接口兼容修复，独立diff位于 `provenance/compatibility.patch`。外部观测器记录Loss、预测和checkpoint，训练入口未改。运行分类为 `compatibility_patched_with_external_observer`，不称完全未修改官方运行。没有新增正式Seed、调参、重复训练或额外消融。

## GitHub与服务器产物边界

本次公开源码快照、作者已公开的小型难度文件、配置、六份报告、逐轮聚合指标、分项Loss曲线、聚合分类/目标/梯度诊断、PNG/SVG及溯源记录。

遵循仓库既有策略，以下产物仅保留在服务器：原始特征与数据链接、12个best/latest/final模型权重、逐样本预测NPZ、原始stdout/资源日志，以及逐Batch原始跟踪 `batches.jsonl`。`checkpoints/`中仅公开参数、最佳轮、状态和逐轮指标等小型JSON/JSONL，不含模型权重。公开JSON中的原始路径、日志或权重SHA是溯源信息，不能据此推断相应二进制已上传。

[publication_manifest.json](publication_manifest.json)记录本次实际提交文件的SHA-256和排除项；原实验 `provenance/delivery_manifest.json`保持不变，覆盖完整服务器交付（包含未上传文件），不是GitHub文件齐备性清单。原始目录中已记录文件的哈希没有因上传而重写。

克隆仓库后可以审查代码、配置、逐轮/聚合证据和图表；`verify_delivery.py`、Checkpoint重评及逐样本诊断仍需要服务器保留的权重和数组，不宣称在纯GitHub副本上已重新运行。准备数据和环境参见仓库 [复现运行指南](../../REPRODUCTION_GUIDE.md) 及冻结的 [上游README](../../experiments/official_readme_reproduction/official_code/README.md)。历史命令包含服务器绝对路径及输出防覆盖检查，迁移时须明确替换数据/环境/新输出目录，不能覆盖既有run。
