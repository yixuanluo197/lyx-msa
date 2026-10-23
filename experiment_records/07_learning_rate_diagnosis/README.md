# 第07组：完整 CMERC 学习率诊断

接续[XP04](../04_xp_mainline_validation/README.md)、[官方 README 对照](../05_official_readme_reproduction/README.md)和[最小消融](../06_minimal_ablation_validation/README.md)。本批4项预登记正式训练均已完成并通过独立 CPU 验收；实验原目录位于 [experiments/learning_rate_diagnosis/20261023T084353Z](../../experiments/learning_rate_diagnosis/20261023T084353Z/README.md)。未修改或覆盖04、05、06历史实验。

## 验证集结果

下表均为验证集 Weighted F1 百分比；没有本轮测试集分数。0.0001使用XP04已有同Seed控制，不重复训练。

| 数据集 / Seed | 学习率 | 最佳验证W-F1 | 最佳Epoch | 末10轮验证均值 |
|---|---:|---:|---:|---:|
| IEMOCAP / 1475 | 0.0001（XP04） | 65.02 | 43 | 62.967 |
| IEMOCAP / 1475 | 0.00005 | 63.47 | 67 | 62.257 |
| IEMOCAP / 1475 | 0.0002 | 66.21 | 44 | 62.144 |
| MELD / 67137 | 0.0001（XP04） | 65.24 | 8 | 62.909 |
| MELD / 67137 | 0.00005 | 65.94 | 17 | 63.437 |
| MELD / 67137 | 0.0002 | 66.07 | 8 | 62.983 |

IEMOCAP的0.0002提高最佳值，但末段均值下降，属于混合证据，没有满足预登记资格的第二Seed候选。MELD两项均通过候选门槛，按预先规定的最佳值优先规则首选0.0002；其末段较控制只提高0.074个百分点，不能称为统计强证据或已经确定最优学习率。0.00005的MELD末段表现更好，这一取舍完整保留。

## 协议与结论边界

- 保持XP04非因果完整CMERC，CC、HSCL、CL及其系数均开启；batch16、划分、dropout、continuous RNG、ATen GRU和capacity_safe保持原值，只登记学习率变化。
- IEMOCAP固定80轮、MELD固定40轮；按验证集W-F1保存最佳Checkpoint，同分取首次最高轮。新增工程开关跳过最终测试，训练循环数学不变，独立补丁已保留。
- 四项共240轮、6178个训练Batch；Loss与梯度有限，零测试前向。原始综合特征文件仍含测试部分且既有代码会构造测试loader元数据；零测试前向不等于未加载含测试数据的文件。
- 三种学习率均有训练分类Loss下降而验证Loss上升的末段线索；本轮未计算验证ECE/CEL，不从辅助Loss大小推断校准改善。
- 单Seed、历史控制跨物理GPU、CUDA非确定性和作者难度生成历史未知限制因果与泛化结论。暂缓Batch8、Baseline及五Seed；2024仅预留为下一阶段独立确认Seed，尚未启动。

## 审查入口

- [完整中文报告](../../reports/CMERC_LEARNING_RATE_DIAGNOSIS.md)
- [训练前登记](../../experiments/learning_rate_diagnosis/20261023T084353Z/reports/PREREGISTRATION.md)与[候选资格实现澄清](../../experiments/learning_rate_diagnosis/20261023T084353Z/reports/ANALYSIS_RULE_CLARIFICATION.md)
- [真实参数与四项命令](../../experiments/learning_rate_diagnosis/20261023T084353Z/configs/registry.json)
- [每项结果](../../experiments/learning_rate_diagnosis/20261023T084353Z/results/per_run.csv)、[对照表](../../experiments/learning_rate_diagnosis/20261023T084353Z/results/comparison.csv)与[汇总JSON](../../experiments/learning_rate_diagnosis/20261023T084353Z/results/metrics_summary.json)
- [训练/验证曲线CSV](../../experiments/learning_rate_diagnosis/20261023T084353Z/results/curves.csv)、[分项Loss CSV](../../experiments/learning_rate_diagnosis/20261023T084353Z/results/loss_curves.csv)及[12张PNG/SVG图的哈希](../../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/manifest.json)
- [最终完整验收](../../experiments/learning_rate_diagnosis/20261023T084353Z/provenance/final_verification.json)：4项运行、8个Checkpoint、1357个历史文件及冻结输入检查

## GitHub与服务器边界

[publication_manifest.json](publication_manifest.json)记录本次实际发布文件的SHA-256及排除项。公开源码、作者已公开的小型难度文件、配置、预检摘要、逐轮聚合指标、报告、曲线和溯源信息；两个小型CPU测试日志作为审查证据单独公开。权重、原始数据及其链接、原始训练/资源日志、逐Batch跟踪和锁文件保留在服务器。

原实验manifest和哈希清单保持原字节，其中的服务器路径及未发布文件哈希不代表这些文件已上传。克隆仓库可审查公开证据；完整训练、Checkpoint重算和逐Batch复核仍需服务器数据、权重及环境。发布过程没有重新训练、执行模型推理、查看新测试结果或追加Seed。
