# CMERC 复现结果对比

## 官方 checkpoint 与论文

| 数据集 | 指标 | 论文 CMERC | 官方 checkpoint 实测 | 绝对差 |
|---|---|---:|---:|---:|
| IEMOCAP | W-F1 | 71.98 | 71.9838 | +0.0038 |
| IEMOCAP | CEL | 1143.50 | 1143.4928 | -0.0072 |
| MELD | W-F1 | 66.85 | 66.8484 | -0.0016 |
| MELD | CEL | 605.05 | 605.0503 | +0.0003 |

四项差异均只来自报告精度的四舍五入，可判定作者提供的 checkpoint、真实特征、模型兼容补丁和 CEL 信息移除流程已严格对齐。

## 从头训练

8 个正式 run 正在执行。完成后将分别列出 `official_test` 和 `validation`，并报告：

- Accuracy、Weighted F1、Macro F1、CEL；
- 与论文相同方法参考值的绝对差；
- 同一数据集、同一选模协议下 CMERC 相对 baseline 的 W-F1 增益和 CEL 降幅；
- ECE/MCE/RMSCE 与显式定义下的 AUROC/AUPRC；
- 最佳 epoch、课程比例、耗时和峰值显存。

不得用官方 checkpoint 的结果替代从头训练结果，也不得把测试集选模结果与验证集选模结果混为一组。

## 暂不执行项目

- 论文只公开单个 IEMOCAP/MELD seed；五随机种子需要额外定义 seed 集并消耗约五倍训练资源，本轮主实验完成前不启动。
- w/o CL / w/o HSCL / w/o CC 仅在主实验成功后执行。本轮 GPU 5 已接近显存上限，先保证 8 个主 run 和独立评估完整落盘。
- 论文 Table 5 未充分说明 AUROC/AUPRC 的正类与 score 聚合，当前实现结果只能在明确假设下比较，不能作为严格论文复现结论。
