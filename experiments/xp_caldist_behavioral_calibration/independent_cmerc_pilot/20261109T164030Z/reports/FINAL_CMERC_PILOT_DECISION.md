# Final CMERC Pilot Decision

结论：**两个 Fold 1 Pilot 均完成，具备在研究者审核后启动正式五折五 Seed 训练的技术条件。** 本报告不授权自动启动剩余 48 个模型。

1. 两个 CMERC 均从随机初始化开始，初始化 state hash 已保存，未加载历史 CMERC checkpoint。
2. Difficulty 与批准 Manifest 绑定正确，Teacher 只作为 Difficulty 来源，未用于 student 初始化。
3. CapacitySafeBatchSampler、Curriculum schedule、Unknown-Label Mask 和全部 11 条标签依赖损失路径通过真实训练验证。
4. IEMOCAP 完成 80 epochs，best epoch 79，Development Weighted-F1 67.94%；MELD 完成 40 epochs，best epoch 14，Development Weighted-F1 70.97%。
5. best/latest checkpoint 均可 strict reload，optimizer/RNG state 和完整哈希已保存。
6. Calibration 成功。IEMOCAP grouped OOF 更支持 Global TS/强收缩 E2，不支持不受约束类别温度；MELD grouped OOF 中 E2 优于 E1，但仍需独立 Evaluation 验证。
7. 未发现训练过拟合导致的数值异常；IEMOCAP Development 指标在后期波动，best checkpoint 选择按预注册规则完成。Calibration 内部拟合收益不能解释为 Evaluation 改善。
8. Evaluation DataLoader 未构建，标签未读取，未运行前向或指标。

正式 50 模型阶段需保持：同一批准 Manifest、每 Fold 独立 Difficulty、每模型随机初始化、Development 严格选模、Calibration 后拟合、Evaluation 统一封存，以及 IEMOCAP 每张 12 GiB GPU 至多一个训练任务。
