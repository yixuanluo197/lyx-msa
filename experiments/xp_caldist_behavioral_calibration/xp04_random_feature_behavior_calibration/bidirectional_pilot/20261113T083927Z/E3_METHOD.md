# XP04 E3 双向行为校准 Pilot：预先固定的方法

本文件在读取 E3 结果之前固定配置。仅使用 Seed 1475 的真正 Validation：IEMOCAP 527 条/12 个 Dialogue，MELD 1109 条/114 个 Dialogue。CMERC 预测及扰动结果来自 `pilot/20261113T025053Z/`，不运行 GPU、不重训模型、不读取 Test。

## 每条话语一行

主键为 `(dataset, seed, dialogue_id, utterance_index)`。从 `behavior_features.csv` 选取中度强度 0.05、10 次扰动已汇总的 μ、δ、JS。对于 `joint`、`text`、`audio`、`visual` 四种条件，分别对 Gaussian 与 Feature Dropout 两种扰动的行为指标取算术均值，得到每模态三个固定特征。两种扰动先在话语内部汇总，绝不扩增训练样本。视觉缺失样本的视觉行为值保留原 Pilot 的零响应；不补造视觉特征。

## 预先登记的对照

所有 E3 组使用 `logit(clip(C, 1e-6, 1-1e-6))` 作为基础输入，输出原始预测正确的概率。L2 Logistic Regression，`C=1.0`、`solver=lbfgs`、`max_iter=1000`、默认类权重；每折仅在训练 Dialogue 拟合 `StandardScaler` 和回归器。折内标准化等价于将权重换算回原始特征后的 `sigmoid(w0 + w1 logit(C) + w2 μ + w3 δ + w4 JS)`。不使用温度缩放，不改变分类标签。

| ID | 输入特征 | 目的 |
|---|---|---|
| E0 | 原始 C | 原始基线 |
| E1 | 原 Pilot 的 raw-C Logistic Regression OOF | 公平基线，复算并逐行核对 |
| L0 | logit(C) | 排除输入变换本身带来的收益 |
| E3-A | logit(C) + joint μ、δ | CaliDist 核心行为增量 |
| E3-B | E3-A + joint JS | JS 增量 |
| E3-C | E3-B + text/audio/visual 各自 μ、δ、JS | 模态互补 |
| E3-D | E3-C + 原始预测类别 one-hot | 探索性类别消融；不使用真实类别选择特征 |

额外固定消融：`A_mu`、`A_delta`、`B_js` 分别只加 joint 对应指标；`C_text`、`C_audio`、`C_visual` 分别只加该模态三指标；`C_no_text`、`C_no_audio`、`C_no_visual` 从 E3-C 中删去该模态三指标。这些对照只诊断贡献，不据其 OOF 分数重新选择 E3 配置。没有超参数搜索。类别 one-hot 使用全部预测类别和统一 L2 收缩，少样本类别不会独立训练模型。

## OOF 与评价

IEMOCAP 用 `LeaveOneGroupOut`，MELD 用 `GroupKFold(n_splits=5)`，均按原始 Dialogue ID 分组，与 Pilot E1 的生成顺序一致。每折训练/留出 Dialogue 不交叉，标准化器只用训练折。E1 使用原 Pilot 的 OOF 并由相同折的原始配置复算验证。

主要指标：Top-label ECE15（15 个固定区间）、二元 Brier、二元 NLL。辅助指标：Accuracy、Weighted-F1、Macro-F1、正确性 ROC-AUC、均值及正确/错误条件均值、相对原始 C 的上调/下调比例。只报告 OOF 结果；五 Seed、独立 Test、E3 部署与原始数据干扰均不在本次范围。对主要 E3-A/B/C/D 与 E1 的差异使用 2,000 次 Dialogue 成组配对 bootstrap，固定随机种子 1475；区间仅描述本 Seed 的 Dialogue 采样不确定性。
