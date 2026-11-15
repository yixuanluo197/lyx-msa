# XP04 行为测量与联合校准后续实验（20261115T154720Z）

**两批实验均完成并通过独立验证。** 主方法没有稳定优于静态对照，当前不推进原始数据干扰。

## 结果入口

- [完整数据与最终研究判断](FINAL_MECHANISM_REPORT.md)
- [第一批测量诊断判断](DIAGNOSTIC_DECISION.md) / [重复测量完整报告](diagnostic/REPEATABILITY_REPORT.md)
- [第二批9方法校准结果](calibration/CONDITIONAL_CALIBRATION_REPORT.md) / [固定判据决策](calibration/FINAL_CONDITIONAL_DECISION.md)
- [数值精度审计](calibration/NUMERICAL_VERIFICATION_NOTE.md)
- [运行前冻结设计](DESIGN_AND_PLAN.md) / [预登记配置](preregistration.json)
- [总体验证](verification.json) / [诊断验证](diagnostic/verification.json) / [校准独立验证](calibration/independent_verification.json)

## 本轮完成的实验

1. 复用五Seed共163600条joint扰动，重建8180话语的静态/行为特征、16360组完整行为、49080组半样本行为，诊断Gaussian/Dropout的10次重复估计稳定性。
2. 在同一Dialogue OOF上固定比较XP04、E1、L0、T、T_DA、S、S_DA、S_DA_OFFSET、S_RAND。S含logit(C)、分类间隔、概率熵与原预测类别；DA为两种扰动各自D/A共4维；RAND为与标签无关的4维容量负对照。主比较始终为S_DA vs S。

IEMOCAP保留527话语/12Dialogue，MELD1109/114，每个Seed单独拟合；内层训练Dialogue选择C/修正正则，外层仅评价。五Seed共享原2000次Dialogue抽样。没有训练或加载CMERC、没有生成新扰动、没有修改原始特征或情绪分类、没有Test标签或温度缩放。

## 数据

| 路径 | 内容 |
|---|---|
| diagnostic/feature_matrix.csv | 8180行，每话语一条，静态、DA、μ/δ/JS及固定随机对照 |
| diagnostic/utterance_responses.csv | 16360组完整10重复行为与Monte Carlo误差 |
| diagnostic/split_responses.csv | 三种固定切分，共49080组半样本行为 |
| diagnostic/split_reliability.csv | 全部300条半样本相关性结果 |
| calibration/oof_predictions.csv | 73620条9方法OOF置信度 |
| calibration/per_seed_metrics.csv | 90行逐Seed指标，含Accuracy/F1、AUC等辅助量 |
| calibration/aggregate_metrics.csv | 五Seed均值、样本标准差 |
| calibration/bootstrap_results.csv | 60行固定配对比较与共享Dialogue区间 |
| calibration/parameters/ | 85外层折的全部内外层拟合与选参参数 |
| calibration/train_oof_gap.csv | 每折训练、内OOF、外OOF差距及模型容量诊断 |
| calibration/correct_wrong_adjustment.csv | 正确/错误样本置信度变化 |
| logs/ | 运行、工程/独立验证及首次数值阈值失败记录 |

原始概率及全部扰动概率复用 [上一轮五Seed实验](../../five_seed_validation/20261115T151109Z/) 与 [Seed1475 Pilot](../../pilot/20261113T025053Z/)，不重复上传。没有checkpoint或原始特征文件。

## 复核

只检查交付文件完整性：`python scripts/verify_delivery.py`，仅需Python标准库，不需要GPU或原始数据。

数值重算使用 `/data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/verify_diagnostic.py` 或 `scripts/verify_calibration.py`。脚本沿用原机器路径，需保留/映射preregistration.json中的源目录；验证器会刷新审计证据，应在独立副本中运行。工程检查为 `python -m unittest discover -s scripts -p test_conditional.py -v`。正式分析拒绝覆盖既有结果，复跑须使用新实验目录。

第一批已独立上传d1a37a2；provenance/diagnostic_delivery_hashes.json是该次提交的快照，当前README为第二批完成后的更新版本。当前完整交付以provenance/artifact_hashes.json为准。

## 解释边界

重复使用Validation的机制诊断，不是独立测试确认；checkpoint与方法曾用该Validation选择。五Seed不是五份独立数据，bootstrap不重拟合，区间未经多重比较修正。结果保持阴性，不依据某个消融或单独ECE挑选赢家。本轮完成后停止，等待研究者审核。
