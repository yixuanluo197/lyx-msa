# MELD CMERC Pilot Training

状态：**完成并通过验证**。Fold 1、seed 1475 的 CMERC 从随机初始化训练，未加载历史 CMERC checkpoint。模型使用批准的 Protocol B Manifest、当前 Fold 的 Difficulty 和 XP04 冻结源码。

- 训练预算：40 epochs；最佳 checkpoint：epoch 14。
- 最佳 Development：Accuracy 71.1665%，Weighted-F1 70.9729%，Macro-F1 59.1693%，NLL 1.079731。
- checkpoint 选择：严格最大化四舍五入到 0.01 个百分点的 Development Weighted-F1；并列保留最早 epoch。
- Unknown Label：0。
- 训练时间：426.18 秒（0.1184 GPU-hours）。
- 峰值显存：allocated 0.958 GiB，reserved 1.215 GiB。
- 最大 batch 容量：343 edges / 885 nodes。
- best.pt 与 latest.pt 均已 strict reload，state hash 完全一致。

所有 epoch 的损失和梯度均为有限值，未出现 NaN/Inf、容量越界或非法角色访问。Development 只负责选模；Calibration 在完整训练结束并冻结 best.pt 后才运行。Evaluation 未构建、未前向、未计算指标。
