# 重复次数精度实验判断

## 预登记主模型结果

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

## 研究判断

- **IEMOCAP：**未达到ECE与至少一项proper loss同步改善的点估计；另有proper loss变差，需保留取舍；相对E1没有任何一项Dialogue区间完全小于0。仅作为本单Seed的新重复次数敏感性证据，不据此自动扩展其他Seed或原始数据。
- **MELD：**未达到ECE与至少一项proper loss同步改善的点估计；相对E1没有任何一项Dialogue区间完全小于0。仅作为本单Seed的新重复次数敏感性证据，不据此自动扩展其他Seed或原始数据。

本实验目的为区分“估计不精确”和“增量可靠性信息不足”。分半相关/SE只说明测量性质，是否有额外校准价值必须由M_gaussian相对E1的OOF指标共同判断。不能只凭μ/δ非零、扰动更多或ECE单项下降宣布成功；同样不要求额外AUROC硬门槛。

本轮完成后停止，等待研究者审核。无自动补齐30次五Seed，无原始文本、面部遮挡或音色处理。

本轮只有Seed1475、同一批已探索的真正Validation。K30由三个K10块合并而来，条件相互嵌套，不能视为独立重复。仅有一组30次，15+15分半不是独立30次复现实验。Bootstrap复用2000次Dialogue抽样，不重新拟合，未涵盖校准器拟合、模型Seed或随机扰动的不确定性；多重比较未校正。Validation也用于原checkpoint选择和候选探索。本轮不使用Test、不重跑CMERC、不生成新噪声、不更改强度、不训练E3新结构。

## 补充研究解释

精度变化、稀疏μ、IEMOCAP仅一个Dialogue生效及MELD指标取舍，详见[RESEARCH_INTERPRETATION.md](RESEARCH_INTERPRETATION.md)。
