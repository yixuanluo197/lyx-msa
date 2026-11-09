# CMERC Independent Training Pilot 门控交付

本目录记录 Protocol B 正式 Manifest 冻结、Unknown-Label Mask 核心验证，以及 fold-local curriculum difficulty 忠实性审计。

实验按预注册停止条件停在训练前。XP04 冻结快照不包含 difficulty 生成器；上游生成流程要求一个训练完成但来源与选择协议未公开的 M³Net teacher，而且公开 batch16 命令不能复现完整官方 difficulty artifact。因此没有使用旧 difficulty、旧 checkpoint、随机模型或替代算法继续训练。

交付事实：

- Protocol B 五折分配已按候选清单原样批准和冻结；
- 53 条 `xxx` 样本的独立 supervision mask 已与冻结特征精确对齐；
- mask 核心和现有 difficulty helper 共9项 CPU 测试通过；
- 两个 Pilot 的模型初始化、optimizer step、checkpoint、Calibration 和 Evaluation 前向均为零；
- Evaluation 标签仅用于任务明确要求的 Manifest 类别支持审计，没有用于训练、校准、模型选择或性能评价。

最终状态与后续所需协议修订见 `FINAL_PILOT_DECISION.md` 和 `verification.json`。
