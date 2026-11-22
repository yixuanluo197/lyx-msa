# MELD K=2 μ、δ 可靠性评分

[总报告](FINAL_DECISION.md)记录六组评分器的对照。它们共用 72 条拟合话语和 69 条评估话语，其中评估组有 25 条原预测错误。评估 Dialogue 曾用于历史分析，本轮属于探索性实验。

- [指标汇总](results/scorer_metrics.csv)和[配对 Bootstrap 区间](results/paired_contrasts.csv)
- [正误分布](results/correct_wrong_distributions.csv)和[回归系数](results/coefficients.csv)
- [公式核验](results/formula_verification.json)、[冻结协议](EXPERIMENT_PROTOCOL.json)和[公开文件哈希](PUBLICATION_MANIFEST.json)

本地实验保留逐话语、逐版本记录以及完整抽样文件。[验证记录](verification.json)的 `sha256_manifest` 对应本地完整清单。公开汇总保留 Bootstrap 方案和 Dialogue 数量。

## 复现入口

`scripts/score_mu_delta.py` 读取[上一轮 MELD 评分实验](../20261122T041838Z_meld_real_scoring)的本地冻结输入，复算两个行为量并拟合六组逻辑回归。`scripts/verify_delivery.py` 复核公式、逐话语评分、指标和配对区间。脚本保留原实验目录结构与来源哈希校验，复跑使用 Python 3.10、NumPy 1.24.4、scikit-learn 1.2.2。
