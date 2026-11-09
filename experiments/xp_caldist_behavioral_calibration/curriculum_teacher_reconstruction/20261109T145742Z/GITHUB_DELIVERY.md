# CMERC Fold-local Curriculum Teacher Reconstruction 交付

本目录记录 Protocol B 的 Curriculum Teacher 来源审计、Difficulty 公式恢复、Unknown-Label Mask 扩展和正式协议修订。

## 核心结论

- CMERC 文档规定的 Teacher 训练入口是 `CMERC/train_cm.py`，其结构属于 M³Net 模型家族。
- 外部 M³Net 固定审计版本为 `4d0fa88cd4f92c72b053527e877763f54e29ccb2`；CPU schema 检查确认 IEMOCAP 85 个、MELD 83 个 tensor 可严格加载到 CMERC `model_cm.Model(causal=False)`。
- 作者没有公开历史 Teacher checkpoint 的保存与选择规则。Protocol B 因此采用 Training-only、固定最终 epoch 的新规则：IEMOCAP 80 epochs，MELD 40 epochs。
- 每个 Dataset × Fold 使用一个 Teacher，并供该 Fold 的五个 CMERC Seed 共享：正式设计共10个 Teacher和50个 CMERC。
- Unknown utterance 保留完整上下文，仅在 true-class gap gather 前屏蔽；主 Difficulty 使用有效标签数分母，总长度公式仅作预注册敏感性对照。
- 真实模型 batch1/batch16 尚未证明等价，正式 Difficulty 生成固定使用 batch size 1。

## 验证状态

- 新增 CPU 测试 20/20 通过，历史 mask helper 5/5 通过。
- Approved Manifest SHA-256 保持为 `6d43928865915fe22a913c98b1165a05ba6e3e8f21d4d4c896cf51eb9c9b62be`。
- 所有 proposed Teacher job 仅包含对应 Fold 的 Training IDs。
- 本轮没有启动 Teacher 或 CMERC 训练，没有生成正式 Difficulty，没有访问 Calibration/Evaluation 标签。

## 执行门槛

当前结果是经验证的协议修订和实现准备，不是 Teacher Pilot 结果。正式启动前仍需使用真实 fold-local Teacher checkpoint 和 role-scoped Training artifact 完成固定 Training 对话的真实 batch 检查，并通过完整训练 Runner 的所有 mask 与角色访问测试。

最终决策见 `FINAL_PROTOCOL_AMENDMENT.md`，完整验证状态见 `verification.json`。
