# Global Temperature Scaling 预注册

## 研究问题与比较

对 XP04 的 IEMOCAP、MELD 各五个 validation-selected CMERC checkpoint，分别用对应真正 Validation split 的多类别 NLL 拟合一个正的全局温度。温度冻结后，Test split 只评价一次。唯一比较为 Vanilla CMERC 与 Global Temperature Scaling；不使用行为特征，不训练或修改 CMERC。

## 温度搜索

- 固定范围：`T ∈ [0.05, 20]`。
- 主网格：2001个 `log(T)` 等距点；因上下界互为倒数，中点严格为 `T=1`。
- 细化：在网格最优点相邻的 `log(T)` 区间内执行128轮确定性黄金分割搜索。
- 目标：Validation 多类别 NLL，自然对数。
- 若最优点落在边界，记录边界状态，不扩展范围。
- 并列时优先选择 `|log(T)|` 更小者，再选择较小温度。

保存完整网格 NLL 曲线及细化最优点。参数文件必须包含 checkpoint SHA-256、Validation 身份哈希、拟合前后 NLL 和收敛/边界状态。

## 冻结与 Test 防泄漏

拟合脚本只解析 Validation 路径，不读取 Test 概率或 Test 指标。十个参数写入 `FROZEN_TEMPERATURES.json` 并计算参数载荷 SHA-256 后，Test 评价脚本才允许运行。只读输入审计可以核对 Test 文件身份和历史 Experiment 6 对应关系，但这些信息不进入温度目标或选择。

## 指标

主要指标为固定15等宽箱 ECE、XP04定义的多类别 Brier 和自然对数 NLL。附加报告 MCE15、RMSCE15、ECE10、平均置信度、Accuracy、Weighted-F1、Macro-F1。空箱不伪造零准确率或零置信度。

正温度必须保持每条话语 argmax 不变；任何预测类别变化都使 Test 评价失败。

## 统计

每个数据集、模型 Seed 独立报告。跨 Seed 只计算五个模型结果的均值、样本标准差和改善方向，不把同一 Test 话语跨 Seed 拼接成五倍样本。每个 Seed 使用2000次 dialogue-level paired bootstrap，所有同对话话语共同重采样，报告 GlobalTS 减 Vanilla 的95%百分位区间。

## 决策规则

- A：两个数据集多数 Seed 的 ECE15、Brier、NLL共同改善。
- B：ECE改善但 Brier 或 NLL退化。
- C：只在 IEMOCAP 稳定有效，MELD无效或方向相反。
- D：Test上没有稳定改善。

Test结果不得反向修改搜索区间、优化器、指标或温度。
