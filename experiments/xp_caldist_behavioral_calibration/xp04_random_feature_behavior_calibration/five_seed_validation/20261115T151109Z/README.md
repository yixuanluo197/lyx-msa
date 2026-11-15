# XP04 五 Seed 置信度模块验证（20261115T151109Z）

**已完成并通过独立数值验证。** 主方法 E1_DA 的两个数据集、五个Seed三项校准指标均优于XP04；随机扰动行为相对E1的稳定增量尚未得到支持。

## 报告入口

- [完整数据结果与最终判断](FINAL_FIVE_SEED_REPORT.md)
- [预登记协议](EXPERIMENT_PROTOCOL.md) / [固定配置](preregistration.json)
- [方法及复核命令](reports/METHOD_AND_REPRODUCTION.md)
- [五Seed行为结果](behavior/DATA_RESULTS_REPORT.md) / [消融](behavior/BEHAVIOR_ABLATION_REPORT.md)
- [跨Seed稳定性与区间](behavior/CROSS_SEED_ANALYSIS.md)
- [正确/错误置信度调整](behavior/CONFIDENCE_ADJUSTMENT_REPORT.md)
- [第一批置信度基础映射实验](baseline/BASELINE_FIVE_SEED_REPORT.md)
- [GPU执行与完整性](reports/GPU_EXECUTION_REPORT.md)
- [总体验证](verification.json) / [独立数值验证](behavior/verification.json)

## 数据和代码

| 位置 | 内容 |
|---|---|
| baseline/original_predictions.csv | 8180条真正Validation原始输出，含完整类别概率与ID |
| baseline/ | XP04/E1/L0的OOF、折参数、指标与共享bootstrap抽样 |
| runs/ | 其余4Seed×2数据集的原始/扰动概率、log-probabilities、随机键与完成证据 |
| behavior/feature_matrix.csv | 每条话语一条行为特征，共8180条 |
| behavior/oof_predictions.csv | 七方法OOF置信度，共57260条 |
| behavior/per_seed_metrics.csv | 70行全部方法逐Seed指标 |
| behavior/aggregate_metrics.csv | 五Seed均值与样本标准差 |
| behavior/bootstrap_results.csv | 72行固定对比，共享Dialogue配对bootstrap2000次 |
| behavior/fold_parameters/ | 每折基础映射、标准化器、行为修正及内层选择参数 |
| scripts/ | 推理、分析、独立验证及资源汇总代码 |
| provenance/ | 冻结代码、输入哈希、Training统计量、审查和GPU元数据 |

Seed1475复用 [旧Pilot](../../pilot/20261113T025053Z/) 的joint扰动记录和 [统一模块Pilot](../../unified_module_pilot/20261115T145425Z/) 的冻结特征。没有重新运行该Seed的GPU扰动。没有上传checkpoint或原始特征文件。

第一批已单独提交为74ee78d4fa904665606b58989b66138f5d3526a4。第二批补齐四Seed扰动、七方法OOF、统计、完整派生数据和最终报告。文件完整性可运行 `python scripts/verify_delivery.py`；其只需Python标准库，验证根目录provenance/artifact_hashes.json中的相对路径和SHA256，不需要GPU或原始特征。

本轮完成后停止；不自动进入原始文本/音频/视频干扰。请先审核最终报告。Validation本身参与过checkpoint选择和方法设计，不能把本轮校准OOF称为独立测试集泛化结果。
