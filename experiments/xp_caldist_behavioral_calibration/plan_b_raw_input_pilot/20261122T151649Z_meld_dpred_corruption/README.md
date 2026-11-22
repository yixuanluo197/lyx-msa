# MELD D_pred 诊断与字符损坏 Pilot

[总报告](FINAL_DECISION.md)汇总了语义改写响应抵消、单次相邻字母换位的评分对照，以及历史音频响应。主评分比较有 27 条共同有效的拟合话语、12 条共同有效的评估话语。对照来自已经查看过的 MELD Train 评分组，属于探索性结果。

- [E0–E3 汇总指标](results/scoring/comparisons.csv)和[文本对照报告](TEXT_PERTURBATION_COMPARISON.md)
- [历史 D_pred 诊断](D_PRED_OFFLINE_DIAGNOSTIC.md)和[音频离线诊断](AUDIO_RESPONSE_DIAGNOSTIC.md)
- [冻结协议](EXPERIMENT_PROTOCOL.json)、[验证记录](verification.json)和[公开文件哈希](PUBLICATION_MANIFEST.json)

逐话语分数、完整预测向量、原始文本、拼写候选、盲审、模型权重、特征以及完整运行日志保存在实验机。`verification.json` 中的 `sha256_manifest` 对应本地完整产物清单。公开清单记录 GitHub 文件的哈希。报告中的数字与本地原始结果一致。

## 复现入口

历史输入和权重锁来自 [上一轮评分实验](../20261122T041838Z_meld_real_scoring)。这些脚本保留原实验机布局与本地数据检查。完整运行需要原实验机上的冻结资产。

- `scripts/audit_inputs.py` 核验冻结来源。`scripts/diagnose_dpred.py` 和 `scripts/diagnose_audio.py` 读取已保存的概率执行离线诊断。
- `scripts/orthographic_edits.py generate` 使用固定种子生成单处换位候选。`assemble` 读取保存的 AI 审核结论后锁定候选。随后 `scripts/token_preflight.py` 检查实际 Token 变化。
- `scripts/run_pilot.py` 串联冻结编码和推理、输入审核与 E0–E3 评分。`scripts/summarize_text.py` 从保存的概率复算文本响应。`scripts/verify_run.py` 核对主对照指标、2,000 次 Bootstrap 区间和缺失回退。

合成验证可以独立运行。文本环境使用 PyTorch 2.2.0、Transformers 4.42.3 与 scikit-learn 1.2.2。CMERC 环境使用 PyTorch 2.2.2 与 Transformers 4.45.1。

```bash
python scripts/test_infer_corruption.py
python scripts/compare_scores.py --self-test
```
