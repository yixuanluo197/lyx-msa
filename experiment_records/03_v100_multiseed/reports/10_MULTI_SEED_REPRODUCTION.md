# 原始非因果 CMERC 多随机种子复现

状态：全部预注册运行完成。有效完整运行 20/20。

预先固定训练 seeds：[1475, 67137, 2024, 3407, 42]。validation 选模，IEMOCAP split-seed 固定1475，MELD 使用官方划分；原始 BN 非因果路径。

所有统计只纳入完整80/40 epoch、严格验证集选模、独立最终评估、有限指标及同一数据划分的运行。样本标准差 ddof=1，未按测试成绩筛选seed。缺失/失败项保留在注册表中。

## 每 seed 实际分数

| 数据集 | 模型 | Seed | Acc | W-F1 | Macro-F1 | CEL | ECE | 最佳epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | baseline | 1475 | 66.9131 | 66.9061 | 66.1268 | 1419.2026 | 14.4769 | 61 |
| IEMOCAP | baseline | 67137 | 68.0222 | 68.1401 | 67.4872 | 1207.9184 | 11.9008 | 58 |
| IEMOCAP | baseline | 2024 | 67.6525 | 67.3245 | 66.7067 | 1060.7104 | 9.3045 | 46 |
| IEMOCAP | baseline | 3407 | 69.1312 | 68.9970 | 68.3045 | 1350.4091 | 14.0532 | 53 |
| IEMOCAP | baseline | 42 | 68.0838 | 67.9753 | 66.7583 | 1267.1329 | 12.1982 | 64 |
| IEMOCAP | cmerc | 1475 | 70.1787 | 70.5550 | 69.4038 | 877.1961 | 10.8439 | 65 |
| IEMOCAP | cmerc | 67137 | 67.0980 | 67.3478 | 66.4967 | 851.3002 | 8.6064 | 58 |
| IEMOCAP | cmerc | 2024 | 69.5009 | 69.8143 | 68.8722 | 922.5235 | 8.7148 | 69 |
| IEMOCAP | cmerc | 3407 | 68.4535 | 68.6172 | 67.7028 | 1131.4064 | 12.0032 | 75 |
| IEMOCAP | cmerc | 42 | 69.2545 | 69.3507 | 68.1373 | 871.9269 | 10.9125 | 65 |
| MELD | baseline | 1475 | 66.3602 | 65.4930 | 49.3872 | 601.9534 | 25.8669 | 6 |
| MELD | baseline | 67137 | 66.3985 | 65.7069 | 49.9655 | 572.9544 | 25.7191 | 7 |
| MELD | baseline | 2024 | 66.0920 | 65.3833 | 49.2713 | 671.4342 | 23.7285 | 10 |
| MELD | baseline | 3407 | 66.7816 | 65.8133 | 50.3385 | 534.9700 | 23.5796 | 9 |
| MELD | baseline | 42 | 66.6667 | 65.7353 | 49.3288 | 511.3234 | 25.6936 | 6 |
| MELD | cmerc | 1475 | 65.4023 | 64.4776 | 48.8458 | 839.1360 | 13.0096 | 18 |
| MELD | cmerc | 67137 | 66.8199 | 65.5110 | 49.1199 | 566.0607 | 19.6914 | 9 |
| MELD | cmerc | 2024 | 65.6705 | 64.9264 | 48.3823 | 677.0435 | 17.1963 | 10 |
| MELD | cmerc | 3407 | 66.7050 | 65.4923 | 49.7896 | 638.4580 | 16.9341 | 11 |
| MELD | cmerc | 42 | 66.6667 | 65.5419 | 48.6099 | 573.7872 | 20.3850 | 6 |

## 多 seed Mean ± Std

| 数据集 | 模型 | 有效/计划 | Acc | W-F1 | Macro-F1 | CEL | ECE | W-F1均值−论文 | CEL均值−论文 |
|---|---|---:|---|---|---|---|---|---:|---:|
| IEMOCAP | baseline | 5/5 | 67.9606 ± 0.8034 | 67.8686 ± 0.8033 | 67.0767 ± 0.8391 | 1261.0747 ± 137.8343 | 12.3867 ± 2.0566 | -1.7414 | -511.4653 |
| IEMOCAP | cmerc | 5/5 | 68.8971 ± 1.1799 | 69.1370 ± 1.2234 | 68.1226 ± 1.1206 | 930.8706 ± 115.0772 | 10.2161 ± 1.4931 | -2.8430 | -212.6294 |
| MELD | baseline | 5/5 | 66.4598 ± 0.2717 | 65.6264 ± 0.1804 | 49.6583 ± 0.4714 | 578.5271 ± 62.5003 | 24.9175 ± 1.1565 | +0.2564 | -221.0529 |
| MELD | cmerc | 5/5 | 66.2529 ± 0.6633 | 65.1898 ± 0.4731 | 48.9495 ± 0.5438 | 658.8971 ± 110.7874 | 17.4433 ± 2.9020 | -1.6602 | +53.8471 |

只有全部预注册五seed完成时，上表才是五seed结果；部分阶段的均值按有效数量标注，不外推为五seed结论。

## 同 seed 成对比较

| 数据集 | Seed | CMERC−Baseline W-F1 | CMERC−Baseline CEL |
|---|---:|---:|---:|
| IEMOCAP | 1475 | +3.6489 | -542.0065 |
| IEMOCAP | 67137 | -0.7923 | -356.6182 |
| IEMOCAP | 2024 | +2.4899 | -138.1869 |
| IEMOCAP | 3407 | -0.3798 | -219.0028 |
| IEMOCAP | 42 | +1.3755 | -395.2061 |
| MELD | 1475 | -1.0154 | +237.1826 |
| MELD | 67137 | -0.1959 | -6.8937 |
| MELD | 2024 | -0.4569 | +5.6093 |
| MELD | 3407 | -0.3210 | +103.4880 |
| MELD | 42 | -0.1934 | +62.4638 |

IEMOCAP 成对增益：W-F1 1.2684 ± 1.8797；CEL变化 -330.2041 ± 157.3369（负值表示降低）。

MELD 成对增益：W-F1 -0.4365 ± 0.3413；CEL变化 80.3700 ± 98.2972（负值表示降低）。

## V2批采样与资源记录

| 数据集 | 模型 | Seed | 采样重排次数Σ | 最大batch边数 | 最大batch节点数 | GPU峰值MiB |
|---|---|---:|---:|---:|---:|---:|
| IEMOCAP | baseline | 1475 | 0 | 973 | 2775 | 5803.13 |
| IEMOCAP | baseline | 67137 | 3 | 987 | 2817 | 5906.17 |
| IEMOCAP | baseline | 2024 | 4 | 986 | 2814 | 5991.82 |
| IEMOCAP | baseline | 3407 | 5 | 998 | 2850 | 5967.38 |
| IEMOCAP | baseline | 42 | 0 | 969 | 2763 | 5752.85 |
| IEMOCAP | cmerc | 1475 | 0 | 973 | 2775 | 6979.13 |
| IEMOCAP | cmerc | 67137 | 2 | 987 | 2817 | 7098.53 |
| IEMOCAP | cmerc | 2024 | 4 | 986 | 2814 | 7202.30 |
| IEMOCAP | cmerc | 3407 | 5 | 998 | 2850 | 7174.00 |
| IEMOCAP | cmerc | 42 | 0 | 969 | 2763 | 6903.05 |
| MELD | baseline | 1475 | 0 | 287 | 717 | 847.13 |
| MELD | baseline | 67137 | 0 | 281 | 699 | 849.56 |
| MELD | baseline | 2024 | 0 | 282 | 702 | 846.35 |
| MELD | baseline | 3407 | 0 | 286 | 714 | 843.35 |
| MELD | baseline | 42 | 0 | 283 | 705 | 850.35 |
| MELD | cmerc | 1475 | 0 | 300 | 756 | 943.06 |
| MELD | cmerc | 67137 | 0 | 315 | 801 | 939.54 |
| MELD | cmerc | 2024 | 0 | 300 | 756 | 939.48 |
| MELD | cmerc | 3407 | 0 | 315 | 801 | 939.11 |
| MELD | cmerc | 42 | 0 | 308 | 780 | 945.90 |

## 稳定性、最佳epoch与收敛

- IEMOCAP / baseline：最佳epoch [61, 58, 46, 53, 64]；W-F1 67.8686 ± 0.8033，CEL 1261.0747 ± 137.8343。
- IEMOCAP / cmerc：最佳epoch [65, 58, 69, 75, 65]；W-F1 69.1370 ± 1.2234，CEL 930.8706 ± 115.0772。
- MELD / baseline：最佳epoch [6, 7, 10, 9, 6]；W-F1 65.6264 ± 0.1804，CEL 578.5271 ± 62.5003。
- MELD / cmerc：最佳epoch [18, 9, 10, 11, 6]；W-F1 65.1898 ± 0.4731，CEL 658.8971 ± 110.7874。

IEMOCAP 在已完成的5个配对seed中，CMERC平均W-F1提高，CEL降低。增益标准差1.8797个百分点；小样本均值与标准差不足以证明所有随机种子均改善，完整配对分数保留上表。

MELD 在已完成的5个配对seed中，CMERC平均W-F1下降，CEL升高。增益标准差0.3413个百分点；小样本均值与标准差不足以证明所有随机种子均改善，完整配对分数保留上表。

同一数据集全部seed与Baseline/CMERC的train/validation/test IDs及hash必须一致。收敛图仅显示训练/验证Loss与W-F1；测试分数不用于选择epoch。SVG见机器汇总目录 figures/。

## 与历史单seed及论文的差异解释

论文参考 W-F1：IEMOCAP Baseline69.61 / CMERC71.98，MELD Baseline65.37 / CMERC66.85；CEL分别1772.54 / 1143.50 / 799.58 / 605.05。论文参考协议与validation规范协议不可视为完全等价。

本组初始化时播种，训练使用continuous RNG、评估保存/恢复RNG、epoch独立DataLoader随机序列；旧单seed保留每次函数调用重新播种的legacy行为。IEMOCAP当前固定split1475，其他训练seed不改变划分。新旧分数差异还包含随机过程策略/可能划分差异，不能全部归因于纯训练随机性。原单seed成果在 ../CMERC，未覆盖。历史分数保存在JSON historical_single_seed 字段，仅作参考。

PyTorch1.13的cuDNN GRU内部dropout状态无法通过公开RNG checkpoint完整恢复；为保证断点恢复一致，本组仅RNN采用ATen kernels，结构、数学公式与dropout概率不变，其他层保持原实现。checkpoint记录rnn_backend=aten，独立评估应用相同后端；历史官方checkpoint仍使用cuDNN。后端数值与随机序列差异也是新旧实验分数差异的可能原因。

V1真实训练seed67137第5epoch出现batch N=1001、E=1049，超过官方hyperedge_weight的1000容量并失败；原V1输出与失败记录单独保留。V2全部五seed与Baseline/CMERC统一使用capacity_safe：固定batch size16，仅在整轮随机排列产生超容量batch时重新排列，edge_capacity1000、node_capacity5200、最大重排10000；同一轮应使用该课程阶段的全部样本一次，训练/验证split、课程样本集合与模型数学结构不变。此条件重排改变batch联合分布及随机序列，是与V1/历史单seed比较的明确协议差异。V1部分/完整成绩均不混入V2统计，重排不以测试分数为条件，不筛选seed。

官方difficulty生成协议未完全公开，外部课程得分可能由包含本轮holdout的原训练数据生成。本组固定使用同一哈希difficulty并明确train/validation对话互斥，但无法宣称所有预处理阶段都不含validation信息。详细协议与资源审计见 [PROTOCOL_AND_RESOURCE_AUDIT.md](PROTOCOL_AND_RESOURCE_AUDIT.md)。

Acc、W-F1、Macro-F1、ECE单位均为百分数；ECE采用15等宽bin、max-softmax置信度；CEL沿用原论文五种信息移除后真实类别log置信度正向增长的累计定义。CEL总量只在同一数据集/样本与移除协议下比较。

## 未完成与失败注册表

| 数据集 | 模型 | Seed | 状态 | 原因 |
|---|---|---:|---|---|

机器汇总：`/data2/yb/paper/cmerc_seed_v2_20261014/results/multiseed_validation_v2/summary.json`，同目录registry.csv、per_seed.csv、paired.csv、aggregate.csv；生成命令：

```bash
python scripts/summarize_results.py --multiseed --run-root /data2/yb/paper/cmerc_seed_v2_20261014/runs/multiseed_validation_v2 --result-root /data2/yb/paper/cmerc_seed_v2_20261014/results/multiseed_validation_v2 --output /data2/yb/paper/cmerc_seed_v2_20261014/results/multiseed_validation_v2/summary.json --report /data2/yb/paper/cmerc_seed_v2_20261014/reports/10_MULTI_SEED_REPRODUCTION.md --seeds 1475,67137,2024,3407,42
```
