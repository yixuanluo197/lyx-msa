# 原始非因果 CMERC 多随机种子复现

状态：部分完成，尚未完成多 seed 复现。有效完整运行 0/20。

预先固定训练 seeds：[1475, 67137, 2024, 3407, 42]。validation 选模，IEMOCAP split-seed 固定1475，MELD 使用官方划分；原始 BN 非因果路径。

所有统计只纳入完整80/40 epoch、严格验证集选模、独立最终评估、有限指标及同一数据划分的运行。样本标准差 ddof=1，未按测试成绩筛选seed。缺失/失败项保留在注册表中。

## 每 seed 实际分数

| 数据集 | 模型 | Seed | Acc | W-F1 | Macro-F1 | CEL | ECE | 最佳epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|

## 多 seed Mean ± Std

| 数据集 | 模型 | 有效/计划 | Acc | W-F1 | Macro-F1 | CEL | ECE | W-F1均值−论文 | CEL均值−论文 |
|---|---|---:|---|---|---|---|---|---:|---:|
| IEMOCAP | baseline | 0/5 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 待完成 | 待完成 |
| IEMOCAP | cmerc | 0/5 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 待完成 | 待完成 |
| MELD | baseline | 0/5 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 待完成 | 待完成 |
| MELD | cmerc | 0/5 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 尚无完整运行 | 待完成 | 待完成 |

只有全部预注册五seed完成时，上表才是五seed结果；部分阶段的均值按有效数量标注，不外推为五seed结论。

## 同 seed 成对比较

| 数据集 | Seed | CMERC−Baseline W-F1 | CMERC−Baseline CEL |
|---|---:|---:|---:|

IEMOCAP 成对增益：W-F1 尚无完整运行；CEL变化 尚无完整运行（负值表示降低）。

MELD 成对增益：W-F1 尚无完整运行；CEL变化 尚无完整运行（负值表示降低）。

## 稳定性、最佳epoch与收敛

- IEMOCAP / baseline：最佳epoch []；W-F1 尚无完整运行，CEL 尚无完整运行。
- IEMOCAP / cmerc：最佳epoch []；W-F1 尚无完整运行，CEL 尚无完整运行。
- MELD / baseline：最佳epoch []；W-F1 尚无完整运行，CEL 尚无完整运行。
- MELD / cmerc：最佳epoch []；W-F1 尚无完整运行，CEL 尚无完整运行。

同一数据集全部seed与Baseline/CMERC的train/validation/test IDs及hash必须一致。收敛图仅显示训练/验证Loss与W-F1；测试分数不用于选择epoch。SVG见机器汇总目录 figures/。

## 与历史单seed及论文的差异解释

论文参考 W-F1：IEMOCAP Baseline69.61 / CMERC71.98，MELD Baseline65.37 / CMERC66.85；CEL分别1772.54 / 1143.50 / 799.58 / 605.05。论文参考协议与validation规范协议不可视为完全等价。

本组初始化时播种，训练使用continuous RNG、评估保存/恢复RNG、epoch独立DataLoader随机序列；旧单seed保留每次函数调用重新播种的legacy行为。IEMOCAP当前固定split1475，其他训练seed不改变划分。新旧分数差异还包含随机过程策略/可能划分差异，不能全部归因于纯训练随机性。原单seed成果在 ../CMERC，未覆盖。历史分数保存在JSON historical_single_seed 字段，仅作参考。

PyTorch1.13的cuDNN GRU内部dropout状态无法通过公开RNG checkpoint完整恢复；为保证断点恢复一致，本组仅RNN采用ATen kernels，结构、数学公式与dropout概率不变，其他层保持原实现。checkpoint记录rnn_backend=aten，独立评估应用相同后端；历史官方checkpoint仍使用cuDNN。后端数值与随机序列差异也是新旧实验分数差异的可能原因。

官方difficulty生成协议未完全公开，外部课程得分可能由包含本轮holdout的原训练数据生成。本组固定使用同一哈希difficulty并明确train/validation对话互斥，但无法宣称所有预处理阶段都不含validation信息。详细协议与资源审计见 [PROTOCOL_AND_RESOURCE_AUDIT.md](PROTOCOL_AND_RESOURCE_AUDIT.md)。

Acc、W-F1、Macro-F1、ECE单位均为百分数；ECE采用15等宽bin、max-softmax置信度；CEL沿用原论文五种信息移除后真实类别log置信度正向增长的累计定义。CEL总量只在同一数据集/样本与移除协议下比较。

## 未完成与失败注册表

| 数据集 | 模型 | Seed | 状态 | 原因 |
|---|---|---:|---|---|
| IEMOCAP | baseline | 1475 | pending | full epoch budget/final test not complete |
| IEMOCAP | baseline | 67137 | missing | training directory does not exist |
| IEMOCAP | baseline | 2024 | missing | training directory does not exist |
| IEMOCAP | baseline | 3407 | missing | training directory does not exist |
| IEMOCAP | baseline | 42 | missing | training directory does not exist |
| IEMOCAP | cmerc | 1475 | pending | full epoch budget/final test not complete |
| IEMOCAP | cmerc | 67137 | missing | training directory does not exist |
| IEMOCAP | cmerc | 2024 | missing | training directory does not exist |
| IEMOCAP | cmerc | 3407 | missing | training directory does not exist |
| IEMOCAP | cmerc | 42 | missing | training directory does not exist |
| MELD | baseline | 1475 | pending | full epoch budget/final test not complete |
| MELD | baseline | 67137 | missing | training directory does not exist |
| MELD | baseline | 2024 | missing | training directory does not exist |
| MELD | baseline | 3407 | missing | training directory does not exist |
| MELD | baseline | 42 | missing | training directory does not exist |
| MELD | cmerc | 1475 | pending | full epoch budget/final test not complete |
| MELD | cmerc | 67137 | missing | training directory does not exist |
| MELD | cmerc | 2024 | missing | training directory does not exist |
| MELD | cmerc | 3407 | missing | training directory does not exist |
| MELD | cmerc | 42 | missing | training directory does not exist |

机器汇总：`results/multiseed_validation/summary.json`，同目录registry.csv、per_seed.csv、paired.csv、aggregate.csv；生成命令：

```bash
python scripts/summarize_results.py --multiseed --run-root runs/multiseed_validation --result-root results/multiseed_validation --output results/multiseed_validation/summary.json --report reports/10_MULTI_SEED_REPRODUCTION.md --seeds 1475,67137,2024,3407,42
```
