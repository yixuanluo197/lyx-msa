# XP04 C×μ、C×δ交互实验

用户批准的单一结构修正：让原始随机特征扰动μ/δ的系数随原C变化。复用两个数据集完整Validation、五Seed；无新GPU推理、无原始数据处理。

## 阅读顺序

1. [最终判断](FINAL_DECISION.md)：IEMOCAP比旧模块改善，仍未稳定超过E1；MELD未获益。
2. [五Seed数据报告](RESULTS.md)：完整均值、差值、改善方向和2000次Dialogue配对bootstrap。
3. [E1对照](E1_VS_INTERACTION.md)、[置信度调整](CONFIDENCE_ADJUSTMENT.md)。
4. [方法](METHOD.md)、[固定协议](INTERACTION_PROTOCOL.md)、preregistration.json。

## 数据与代码

- inputs/features.csv：8180条，每Dataset/Seed/Dialogue/Utterance一条；参考基线、外折与抽样文件本地保存。
- results/oof_predictions.csv：32720条，保存原C、原预测、校准概率及分数分解。
- results/per_seed_metrics.csv、aggregate_metrics.csv、bootstrap_results.csv：40条每Seed指标、五Seed均值/样本标准差及24项配对比较。
- results/fold_parameters/：全内外层中心、标准化器、系数、候选损失及训练/留出Dialogue IDs。
- scripts/：输入审计、实现、运行、工程测试和独立数值验证；vendor/mainline_core.py保持旧核心原字节。
- logs/：输入、工程测试、正式运行、独立审计日志。最初红灯来自执行记录；red_replay日志明确标记为重放，不冒充原始日志。

## 验证与重现

`python verify_delivery.py`为标准库只读交付检查。数值环境为/data2/yb/conda_envs/cmerc-xp-repro/bin/python，OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1；版本记录于provenance/environment.json。工程检查命令为`python scripts/test_interaction.py`。正式运行命令为`python scripts/run_interaction.py --run`，拒绝覆盖已完成结果；如需复现，在新的隔离目录复制代码、协议及inputs，保持results为空。`python scripts/verify_interaction.py`做深度数值审计，原绝对路径来源记录需在换机器时相应映射。GitHub副本以verify_delivery.py作可移植完整性检查。

原始概率与10次扰动明细复用上一阶段已发布产物，来源hash见inputs/source_hashes.json；不重复伪造或重新生成。当前所有结果仍是重复Validation上的探索性评估。
