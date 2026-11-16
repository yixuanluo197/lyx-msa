# XP04 10次/30次随机特征扰动精度验证

Seed1475；IEMOCAP文本/MELD音频；Gaussian σ=0.05。只使用已有完整Validation原始及扰动概率，不运行模型、不新增随机特征、不使用Test。

原三个10次块分别保留，30次汇总为新独立登记的敏感性条件，不覆盖旧协议或报告。每条话语先汇总为一条校准记录。

- `EXPERIMENT_PLAN.md`、`preregistration.json`：固定范围、条件与比较。
- `inputs/`：选用的完整原始与扰动概率、身份及全局repeat索引。
- `results/`：行为特征、OOF概率、所有指标/参数/Bootstrap及独立验证。
- 最终状态以`verification.json`为准。

本轮只能讨论Seed1475的重复次数敏感性，不能作为跨Seed或独立数据证据。没有独立的第二组30次实验；前15/后15的分半诊断不能冒充30次可重复性验证。

## 阅读入口

- `RESEARCH_INTERPRETATION.md`：研究解释、全部10次块与30次对比、局部收益及限制。
- `RESULTS_SUMMARY.md`：指标摘要。
- `PRECISION_METHOD.md`：方法与协议。
- `MEASUREMENT_PRECISION_REPORT.md`：分半、零值与均值标准误。
- `OOF_CALIBRATION_REPORT.md`：完整模型对照、Bootstrap及Dialogue诊断。
- `FINAL_PRECISION_DECISION.md`：本轮结论。
