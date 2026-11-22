# MELD Seed 67137 可靠性评分试验

[实验报告](REAL_SCORING_REPORT.md)记录主对照。加入两版语义改写得到的 `D_pred` 后，Binary NLL 从 0.583462 升到 0.609387，配对 Dialogue Bootstrap 的改善区间跨零。主对照覆盖 69 条留出话语，完整留出组为 1,407 条。仅用原始最大 Softmax 拟合的单变量评分，在完整留出组的 Binary NLL 为 0.558154，直接用原始最大 Softmax 的参考值为 0.609219。

[汇总统计](results/scoring_summary.json)包括原始置信度参考值、两种干净特征组合的主对照和完整队列回退结果。[运行验证](verification.json)保留原始实验产物的哈希，包括本地留存文件。公开汇总去除了机器上的输入路径和逐 Dialogue 名单。逐话语得分、原始文本、AI 盲审记录、完整特征和权重保存在实验机。

## 复现依据

[冻结协议](configs/REAL_SCORING_PROTOCOL.json)和[分组锁](SPLIT_LOCK.json)记录样本范围与参数。原运行起点为 `330289ba38bb014ed9819dac6bb6d2f3ab79ce6e`。文本编码和打分使用 TEXT_repro 环境的 PyTorch 2.2.0、Transformers 4.42.3、scikit-learn 1.2.2。下游训练和推理使用 multimodalerc310 环境的 PyTorch 2.2.2 与 Transformers 4.45.1。

- [分组脚本](../20261121T175233Z_reliability_readiness/scripts/freeze_split.py)确定 Train 内四组 Dialogue。[准备脚本](scripts/prepare_data.py)核验分组清单与 Plan B 本地资产哈希，再分别导出基础训练和评分数据。
- [文本训练](scripts/train_text_branch.py)和[下游训练](scripts/run_downstream.py)运行新的 RoBERTa、M³Net 和 CMERC。[盲审清单](scripts/prepare_ai_roster.py)和[审核合并](scripts/assemble_ai_reviews.py)锁定两版改写。
- [评分入口](scripts/run_score_branch.py)串联冻结推理、概率审核和[冻结 CPU 评分器](../20261121T175233Z_reliability_readiness/scripts/score_heldout.py)。[报告脚本](scripts/summarize_scoring.py)使用保存的模型参数复算留出评分，并生成报告和验证记录。

训练和评分入口保留原实验机路径与版本检查。端到端复跑需要原运行锁定的本地特征、Dialogue 清单与审核记录。公开代码可核对研究流程和数值定义。
