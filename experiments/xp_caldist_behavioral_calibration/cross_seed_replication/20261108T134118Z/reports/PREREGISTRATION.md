# 跨 Seed 模态证据敏感性复现预注册

本实验只使用 XP04 的真正 Validation Split 与 validation-selected `best.pt`，不训练、不访问 Test、不重新选择 checkpoint。IEMOCAP 和 MELD 各五个模型 Seed 使用同一组验证话语，因此每个 Seed 独立分析；同一话语的五个模型输出不会拼接为扩大的训练样本。

## 冻结条件

- 复用而不重跑 Anchor：IEMOCAP seed1475、MELD seed67137。
- 新运行：IEMOCAP seed67137/2024/3407/42；MELD seed1475/2024/3407/42。
- 训练均值收缩：T-All、A、V 各 `rho=0.5,1.0`；T1–T4 各 `rho=1.0`，共 10 个条件。
- Random Training Donor：T-All、A、V，`rho=1.0`，每条验证话语 5 个确定性 donor。donor 只取同数据集 Training Split，不读取或使用 donor 标签；T-All 的四个数组来自同一训练话语。同一数据集、验证话语、目标模态和 repeat 在五个模型 Seed 上使用同一 donor，避免 donor 差异与模型初始化效应混杂。
- Donor 池保留训练集中实际存储的全零向量，不依据实验结果事后筛选。每次替换保存 donor 是否为全零；Visual Donor 同时报告全池、有效视觉 donor 和全零视觉 donor 三种统计。MELD 训练集全零视觉比例较高，因此该分层是解释边界的一部分。
- T1–T4 仅表示 dataloader 中四个已保存文本数组的顺序，不宣称对应具体 RoBERTa 编码层。
- 全零视觉输入保留在样本集中；IEMOCAP 1 条、MELD 95 条不执行 V 干预，保存原始概率并单独统计。

## 正确性增量诊断

每个数据集、模型 Seed 分别执行 Leave-One-Dialogue-Out。每折只在其余对话拟合标准化与固定 `L2 C=1` Logistic Regression，不搜索超参数。主行为特征来自训练均值参考的 `rho=0.5`：prediction flip、absolute confidence change、JS divergence。

比较两类基础预测器：最大概率单特征；最大概率、预测熵和 Top-1/Top-2 margin 的严格概率基线。扩展模型分别加入 Text、Audio、Visual 或三者行为特征。Donor 行为独立评价，不与主特征混合。指标为 AUROC Success、AUPRC Success、Binary Brier 和 Binary NLL。

对固定 OOF 分数执行 2,000 次 dialogue-level paired bootstrap。该区间不在每次 bootstrap 中重新拟合 Logistic Regression，因此不覆盖完整拟合不确定性；IEMOCAP 仅 12 个对话，区间预计较宽。

## 解释边界与决策

训练均值与随机 donor 都只是表征参考，不是情绪中性输入。敏感性表示模型对存储表征的依赖，不等同于预测错误、因果情绪证据或真实物理扰动。跨 Seed 排序、方向一致性和控制概率后的 OOF 增量共同决定是否值得另行预注册 Reliability Score 或校准实验；本阶段不会自动进入后续实验。

机器可读的完整冻结协议见 `configs/preregistered_config.json`。
