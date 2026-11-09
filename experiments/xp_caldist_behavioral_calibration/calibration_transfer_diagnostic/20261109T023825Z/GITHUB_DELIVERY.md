# CMERC Calibration Transfer Diagnostic 交付

本目录诊断 IEMOCAP 的 Validation 拟合温度为何无法稳定迁移。实验仅使用十个 XP04 CMERC checkpoint 已保存的真正 Validation 完整概率；没有读取 Test 预测或标签，没有重新训练、重新选模或修改 checkpoint。

## 诊断范围

- IEMOCAP：12 个 Validation 对话、527 条话语。
- MELD：114 个 Validation 对话、1109 条话语。
- 每个数据集五个模型 Seed 独立执行 Leave-One-Dialogue-Out Temperature Scaling。
- 对话层 50%／75%／100% 校准规模诊断；50% 和 75% 各重复 100 次。
- 外层 LODO、内层五折 grouped-by-dialogue CV，从固定候选中选择温度正则化 λ。
- 固定 OOF 预测执行 2000 次 dialogue-level paired bootstrap。

## 主要结果

- IEMOCAP 的 LODO 温度标准差为 `0.0516–0.1362`，MELD 仅为 `0.0016–0.0020`。
- IEMOCAP 只使用 50% 对话时，温度平均 seed 内标准差为 `0.3291`；MELD 为 `0.0181`。部分 IEMOCAP 子集会从全量的 `T>1` 翻转至 `T<1`。
- `Ses02F_impro06` 在 IEMOCAP 4/5 Seed 中是温度影响最大的对话，最大留出变化为 `−0.3591`。
- IEMOCAP neutral、frustrated 主要过置信，而 sad、angry 接近校准或略欠置信；MELD 的高支持 neutral、surprise、anger、joy 主要欠置信，但 sadness、fear、disgust 方向相反。
- IEMOCAP 普通 LODO TS 的 NLL 点估计在 5/5 Seed 改善，但五个 bootstrap 区间均跨零；MELD 三项指标均在 5/5 Seed 改善且区间排除零。
- 收缩温度到 `T=1` 的嵌套正则化没有稳定改善 NLL 或 Brier，不能解释或修复迁移失败。

## 研究判断

IEMOCAP 的主要问题是校准集规模小、对话组成影响大、类别校准方向冲突，以及温度估计方差较高。类别冲突提供了研究 class-conditional calibration 的动机，但当前 12 个对话不足以支持无约束 Classwise Temperature Scaling。行为可靠性信号路线仍应暂停。

下一项建议实验是使用更大且独立、按 session 分组的校准／评价分区，对 Global TS 与强正则化 hierarchical class-conditional temperature model 进行预注册比较。

## 验证与公开内容

目录包含预注册、六份正式报告、八份结果 CSV、六组 PNG/PDF/SVG 图片、分析与独立验证脚本、输入审计和 SHA-256 清单。`verification.json` 状态为 `passed`：独立重跑全部 630 个外层折、嵌套 λ 选择、2010 次样本规模拟合和 20000 次 bootstrap，八份结果 CSV 均逐字节一致。

本地绝对路径保留用于来源追踪。完整输入身份审计需要本地 XP04 checkpoint 与历史 Validation 概率；公开 CSV 足以复核报告中的全部统计结论。
