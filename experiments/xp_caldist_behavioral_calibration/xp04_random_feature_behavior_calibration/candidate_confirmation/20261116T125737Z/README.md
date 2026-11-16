# XP04 Gaussian 单模态候选确认

IEMOCAP文本与MELD音频，σ=0.05；补齐五Seed、独立噪声重复、μ/δ消融。状态以`verification.json`和各run的`complete.json`为准。

- 预登记：`preregistration.json`、`EXPERIMENT_PLAN.md`
- Baseline核验：`XP04_BASELINE_AUDIT.md`
- 运行：`scripts/run_candidate.py`、`scripts/launch_candidates.py`
- 分析：`scripts/analyze_candidates.py`
- 独立复核：`scripts/verify_candidates.py`
- 主要结果与判断：分析完成后见`RESULTS_SUMMARY.md`与`FINAL_CANDIDATE_DECISION.md`

此实验不运行Test、不训练CMERC、不处理原始数据。候选源自既有Validation探索，本轮是跨Seed/随机扰动稳健性核查，不构成独立数据的确认性证据。五Seed共享同一批Validation话语。

## 优先阅读

- `CONFIRMATION_INTERPRETATION.md`：区分历史发现Seed与新增Seed，解释均值及随机重复的边界。
- `RESULTS_SUMMARY.md`：完整五Seed均值、样本标准差和改善Seed数。
- `FINAL_CANDIDATE_DECISION.md`：Bootstrap与Dialogue集中性判断。
- `results/`：完整CSV、OOF预测、校准器参数与独立验证证据。
