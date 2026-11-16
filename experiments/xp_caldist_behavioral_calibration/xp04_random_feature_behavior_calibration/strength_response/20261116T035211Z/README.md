# 强度响应实验交付

0.05历史主条件与0.10敏感性条件；Seed1475、两个完整Validation、四模态、两干扰各10repeat。关键判断见[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)。完整中文结果见[STRENGTH_RESPONSE_REPORT.md](STRENGTH_RESPONSE_REPORT.md)。

结果CSV位于results/：原始预测1636条、行为汇总26176条、特征矩阵13088条、OOF预测65440条、校准指标80条。behavior_summary含全部话语及有效视觉子集，split_half_stability含零值占比、5vs5相关及固定方向AUC；matched_correct_wrong保存原始置信度配对。bootstrap_results和response_bootstrap复用历史2000次Dialogue抽样。

CPU重算：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/analyze_strength.py`。`--prepare`仅验证历史0.05并输出prepared/；`--behavior-only`完成行为分析但不拟合校准器。

更强响应不能替代增量校准证据；本次仅单Seed探索。完整拟合参数与输入哈希位于results/；独立审计和推理provenance见根目录交付文件。
