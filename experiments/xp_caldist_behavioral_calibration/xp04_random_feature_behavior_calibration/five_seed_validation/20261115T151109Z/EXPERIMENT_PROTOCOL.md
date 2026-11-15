# XP04五Seed固定方法验证

本次承接统一模块Pilot，固定五Seed1475/67137/2024/3407/42，IEMOCAP真正Validation527条/12 Dialogue、MELD1109条/114 Dialogue。主比较仍为XP04+E1_DA对XP04，同时保留E1/L0基础映射与μ/δ行为消融。结构和λ选择规则在两个数据集一致，分别拟合模型参数。

第一批独立实验复用真正Validation原始概率，完成XP04、E1、L0五Seed校准；核验后上传报告、完整OOF及折参数。第二批复用Seed1475既有joint扰动结果，为其余4 Seed补齐固定joint Gaussian σ=.05和精确5%训练均值替换、各10重复，再运行被冻结的嵌套校准，完成后上传数据报告。

不同模型Seed使用相同的特征扰动随机流，RNG中的扰动seed固定1475，模型seed另存；这与原Seed1475 Pilot相容。每个特征数组的扰动仅作用于目标话语。Padding、umask无效话语不参与；qmask、其他话语、对话长度及图结构均保持原样。缺失视觉不补造，joint仍扰动可用文本和音频。T1–T4全部同时参与text部分，但不认定其具体RoBERTa层。

保持原mixed-dialogue batch16，严禁复制对话拼批。原checkpoint fresh Validation推理必须先匹配冻结的真正Validation概率和全部IDs/labels/predictions；绝不使用旧XP04 validation/中的Test预测。首次/逐batch核对图缓存与未缓存baseline，保存全部概率及log-probabilities、随机配置、mask审计和权重不变证据。

基础E1为raw-C L2 Logistic Regression，L0为训练折标准化logit-C Logistic Regression。行为offset修正仅有4权重，基础映射冻结，StandardScaler只用相应训练Dialogue。内层GroupKFold5重拟合每个组件，仅依据内层Brier选择off/λ1/.1/.01；外层IEMOCAP LODO12、MELD GroupKFold5。源码复制冻结自已核验统一模块，不据五Seed结果更换强度、模态或模型复杂度。

统计分别报告两个数据集和每个Seed，Mean、Sample SD、方向一致性；Dialogue配对bootstrap2000次的抽样Dialogue索引必须跨Seed共享，不能将5个Seed视为5个独立数据集。原Validation已用于多轮设计，五Seed结果只能提供训练随机性的稳定性证据。所有指标为Top-label ECE15、Binary Brier/NLL，Accuracy/F1仅辅助且保持类别不变。

沿用XP04本地服务器。启动前复查GPU编号、UUID、显存、算力；按容量尝试同卡多个独立作业，若总吞吐下降则降低并发。每个作业独立输出和日志，不修改batch或实验协议。没有CMERC训练、温度缩放或原始数据干扰。用户已明确授权每个完成实验核验后上传GitHub；不上传checkpoint或原始特征。
