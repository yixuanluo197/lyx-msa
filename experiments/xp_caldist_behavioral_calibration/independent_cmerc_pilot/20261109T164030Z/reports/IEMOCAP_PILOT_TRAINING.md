# IEMOCAP CMERC Pilot Training

状态：**完成并通过验证**。Fold 1、seed 1475 的 CMERC 从随机初始化训练，未加载历史 CMERC checkpoint。模型使用批准的 Protocol B Manifest、当前 Fold 的 Difficulty 和 XP04 冻结源码。

- 训练预算：80 epochs；最佳 checkpoint：epoch 79。
- 最佳 Development：Accuracy 68.7764%，Weighted-F1 67.9416%，Macro-F1 64.5589%，NLL 0.766410。
- checkpoint 选择：严格最大化四舍五入到 0.01 个百分点的 Development Weighted-F1；并列保留最早 epoch。
- Unknown Label：Training 23、Development 14；Calibration 另有 8 条按同一规则屏蔽。
- 训练时间：1600.31 秒（0.4445 GPU-hours）。
- 峰值显存：allocated 7.348 GiB，reserved 11.447 GiB。
- 最大 batch 容量：1000 edges / 2856 nodes。
- best.pt 与 latest.pt 均已 strict reload，state hash 完全一致。

所有 epoch 的损失和梯度均为有限值，未出现 NaN/Inf、容量越界或非法角色访问。Development 只负责选模；Calibration 在完整训练结束并冻结 best.pt 后才运行。Evaluation 未构建、未前向、未计算指标。
