# MELD D_pred 与 D_abs 探索性 Pilot

[实验总报告](FINAL_DECISION.md)汇总冻结模型推理和四组正确性评分。主对照使用同一批 34 条拟合话语和 29 条评估话语。结果来自以往已查看的 MELD Train Dialogue，属于探索性证据。

- [全部配对指标与区间](results/scoring/comparisons.csv)
- [响应分布与系数](D_PRED_D_ABS_COMPARISON.md)
- [独立性审计](INDEPENDENCE_AUDIT.md)
- [冻结协议](EXPERIMENT_PROTOCOL.json)和[公开文件哈希](PUBLICATION_MANIFEST.json)

逐话语结果、原始文本、候选改写、审核记录、完整重抽样记录、模型权重和特征保存在实验机。[验证记录](verification.json)中的 `sha256_manifest` 对应本地完整产物清单。

## 复现

脚本保留原实验机路径和来源锁。完整运行需要[历史评分实验](../20261122T041838Z_meld_real_scoring)的冻结资产与本轮审核记录。

- `scripts/audit_experiment_sources.py` 核查数据暴露和历史响应。`scripts/freeze_pilot.py` 分别运行 `develop`、`export` 和 `assemble`，生成固定角色、候选及审核锁。
- `scripts/token_preflight.py` 检查候选 Token。`scripts/run_pilot.py` 串联编码、冻结推理和 CPU 评分。`scripts/verify_delivery.py` 复核两种行为量、主对照区间和回退。

合成验证可单独运行。实验文本和评分环境使用 PyTorch 2.2.0、Transformers 4.42.3、scikit-learn 1.2.2。CMERC 推理环境使用 PyTorch 2.2.2 与 Transformers 4.45.1。

```bash
python scripts/test_infer_pilot.py
python scripts/score_pilot.py --self-test
```
