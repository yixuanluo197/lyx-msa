# 10次与30次随机扰动：结果摘要

固定强度Gaussian σ=0.05，只改变汇总次数。CPU复用49,080条现有扰动预测，1,636条原始话语，4个条件共6,544条行为记录、32,720条OOF预测。三个K10条件全部保留，未挑最好块。

| dataset | condition | method | ece15 | binary_brier | binary_nll |
|---|---|---|---|---|---|
| IEMOCAP | K10_block0 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K10_block0 | M_gaussian | 0.070949 | 0.220315 | 0.634945 |
| MELD | K10_block0 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K10_block0 | M_gaussian | 0.039572 | 0.192479 | 0.567729 |
| IEMOCAP | K10_block1 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K10_block1 | M_gaussian | 0.068211 | 0.221608 | 0.638023 |
| MELD | K10_block1 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K10_block1 | M_gaussian | 0.044262 | 0.192599 | 0.568210 |
| IEMOCAP | K10_block2 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K10_block2 | M_gaussian | 0.087496 | 0.221291 | 0.637272 |
| MELD | K10_block2 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K10_block2 | M_gaussian | 0.042241 | 0.192634 | 0.568326 |
| IEMOCAP | K30_pooled | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K30_pooled | M_gaussian | 0.081070 | 0.222261 | 0.639413 |
| MELD | K30_pooled | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K30_pooled | M_gaussian | 0.045950 | 0.192202 | 0.567321 |

| dataset | metric | K30_vs_E1 | K30_vs_historical_K10 |
|---|---|---|---|
| IEMOCAP | ece15 | -0.008703 | 0.010121 |
| IEMOCAP | binary_brier | 0.001051 | 0.001946 |
| IEMOCAP | binary_nll | 0.002349 | 0.004468 |
| MELD | ece15 | 0.002720 | 0.006379 |
| MELD | binary_brier | -0.000562 | -0.000277 |
| MELD | binary_nll | -0.001263 | -0.000409 |

- **IEMOCAP：**未达到ECE与至少一项proper loss同步改善的点估计；另有proper loss变差，需保留取舍；相对E1没有任何一项Dialogue区间完全小于0。仅作为本单Seed的新重复次数敏感性证据，不据此自动扩展其他Seed或原始数据。
- **MELD：**未达到ECE与至少一项proper loss同步改善的点估计；相对E1没有任何一项Dialogue区间完全小于0。仅作为本单Seed的新重复次数敏感性证据，不据此自动扩展其他Seed或原始数据。

方法与边界见PRECISION_METHOD.md；测量诊断见MEASUREMENT_PRECISION_REPORT.md；所有方法、Bootstrap及Dialogue影响见OOF_CALIBRATION_REPORT.md；结论见FINAL_PRECISION_DECISION.md。

本轮只有Seed1475、同一批已探索的真正Validation。K30由三个K10块合并而来，条件相互嵌套，不能视为独立重复。仅有一组30次，15+15分半不是独立30次复现实验。Bootstrap复用2000次Dialogue抽样，不重新拟合，未涵盖校准器拟合、模型Seed或随机扰动的不确定性；多重比较未校正。Validation也用于原checkpoint选择和候选探索。本轮不使用Test、不重跑CMERC、不生成新噪声、不更改强度、不训练E3新结构。

## 补充研究解释

精度变化、稀疏μ、IEMOCAP仅一个Dialogue生效及MELD指标取舍，详见[RESEARCH_INTERPRETATION.md](RESEARCH_INTERPRETATION.md)。
