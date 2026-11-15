# 五 Seed 统计比较说明

以下比较在正式五 Seed 行为分析结果产生前固定，共12对。冻结的7个方法、4维行为输入、内外Dialogue划分、off/1/0.1/0.01选择规则和calibration_core.py均不改变。

1. E1、L0、E1_MUDELTA、E1_DA、L0_MUDELTA、L0_DA 分别与 XP04 比较：6对，用于判断整体模块是否改善原始基线。
2. E1_MUDELTA vs E1、E1_DA vs E1、L0_MUDELTA vs L0、L0_DA vs L0：4对，用于隔离行为特征在相同基础映射之外的增量。
3. E1_DA vs E1_MUDELTA、L0_DA vs L0_MUDELTA：2对，用于比较修订的D/A与原始μ/δ行为表示。

每对比较均报告 ECE15、Binary Brier、Binary NLL，因此每数据集36行、共72行。差值为前者减后者，负值表示改善。主方法仍为 E1_DA，不根据消融结果替换主方法。

2000次Dialogue配对bootstrap直接复用baseline/bootstrap中同一抽样索引，五个模型Seed共享各次Dialogue抽样；每个Seed独立计算指标差，再对五Seed取均值。Seed仅表示同一数据划分下的模型随机性，不作为五份独立数据。
