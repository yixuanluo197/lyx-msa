# 第06组：CMERC 最小消融根因验证

本组接续[XP04](../04_xp_mainline_validation/README.md)与[官方README对照](../05_official_readme_reproduction/README.md)。3项正式训练均已完成并通过独立CPU验收；原始实验保留在[独立批次目录](../../experiments/minimal_ablation_validation/20261023T043115Z/README.md)，不替换历史源码、权重或结果。

| 实验 / Seed | 最佳验证W-F1 % | 最终测试W-F1 % | XP04同Seed测试W-F1 % |
|---|---:|---:|---:|
| MELD w/o CC / 67137 | 65.7700 | 65.3096 | 65.7461 |
| IEMOCAP source_order Baseline / 1475 | 65.5800 | 66.8159 | 66.9361 |
| IEMOCAP source_order CMERC / 1475 | 64.9600 | 68.2501 | 68.2802 |

所有新运行按验证集W-F1选模、训练结束最终测试一次，沿用XP04划分、continuous RNG和ATen GRU。MELD仅关闭加权CC，保留HSCL、CL及原系数；IEMOCAP仅改变训练Batch排列。完整参数由历史manifest逐项核对，训练数学未修改。

MELD去CC的最佳验证值改善+0.5300个百分点，末10轮均值下降−0.1430，证据混合。IEMOCAP验证相对增量由−1.4300缩小至−0.6200，末段增量由−1.2690缩小至−1.0480；两臂绝对验证表现都下降，Baseline下降更多，不能把负增量缩小称作CMERC绝对提升。测试分数只描述固定选中模型，不用于调参或决定追加运行。

只有单Seed历史对照，完整模型跨物理卡及CUDA非确定性仍存在；作者难度生成过程未知。当前建议先做同卡控制和训练/验证诊断，暂缓最终五Seed，没有启动额外HSCL/CL消融。

## 主要材料

- [完整中文报告](../../reports/CMERC_MINIMAL_ABLATION_RESULTS.md)
- [训练前登记](../../experiments/minimal_ablation_validation/20261023T043115Z/reports/PREREGISTRATION.md)
- [每项指标](../../experiments/minimal_ablation_validation/20261023T043115Z/results/per_run.csv)与[XP04比较](../../experiments/minimal_ablation_validation/20261023T043115Z/results/comparison.csv)
- [汇总JSON](../../experiments/minimal_ablation_validation/20261023T043115Z/results/metrics_summary.json)与[曲线及哈希](../../experiments/minimal_ablation_validation/20261023T043115Z/results/figures/manifest.json)
- [源码补丁](../../experiments/minimal_ablation_validation/20261023T043115Z/provenance/source_order.patch)、[环境](../../experiments/minimal_ablation_validation/20261023T043115Z/provenance/environment.json)及[硬件边界](../../experiments/minimal_ablation_validation/20261023T043115Z/provenance/hardware_pairing.json)
- [最终验收](../../experiments/minimal_ablation_validation/20261023T043115Z/provenance/final_verification.json)：3项最终测试、逐轮覆盖/Loss、6个Checkpoint状态及1154个历史文件未变

## 发布与服务器产物

[publication_manifest.json](publication_manifest.json)列出本次实际发布文件的SHA-256及排除范围。公开源码、配置、逐轮聚合证据、报告和图表；数据、模型权重、逐样本数组、原始训练/资源日志、逐Batch跟踪及锁文件保留在服务器。原实验manifest保持不变，其中可能包含未上传文件的路径和SHA。

克隆后可审查公开证据；完整Checkpoint/逐样本重算需要服务器产物，不宣称纯GitHub副本已经重做全部验证。发布过程没有重新训练、追加Seed或访问测试集。
