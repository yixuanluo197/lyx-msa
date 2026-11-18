# XP04 双数据集 RoBERTa 特征蒸馏（20261118T125439Z）

**两套学生均完成10轮训练，均选中第8轮；IEMOCAP、MELD 的四层特征均未通过冻结兼容性门槛（0/4层）。XP04概率验证按协议未执行，不能直接开展基于原Checkpoint的文本干扰。**

| 数据集 | 有效 Train（拟合/内部验证） | 真正 Validation | 四层 MAE 范围 |
|---|---:|---:|---:|
| IEMOCAP / Seed1475 | 5283（4747/536） | 527 | 0.067008–0.219918 |
| MELD / Seed67137 | 9989（8989/1000） | 1109 | 0.069392–0.172344 |

MAE门槛为1e-5；均未达标。训练执行成功与重建兼容性成功是两个不同结论。本次否定的是本轮固定协议的兼容目标，不能外推为所有特征蒸馏均不可行。

- [最终判断](FINAL_DISTILLATION_DECISION.md)、[完整报告](DISTILLATION_REPORT.md)
- [IEMOCAP逐层验证](IEMOCAP_TEXT_CLEAN_PARITY.md)、[MELD逐层验证](MELD_TEXT_CLEAN_PARITY.md)
- [训练曲线](results/loss_curves.png)、[PDF](results/loss_curves.pdf)
- [本地实验核验](verification.json)、[发布核验](publication_verification.json)
- [发布说明](PUBLICATION_NOTE.md)、[仅本地资产](LOCAL_ONLY.md)、[发布文件哈希](publication_manifest.json)

scripts/ 为蒸馏、数据准备、单次验证及核验实现；configs/ 保留原固定协议和实际配置；training_records/ 保存两套学生全部10轮、步骤事件、最佳选择与显存/梯度检查记录；results/ 保存完整逐层/逐坐标统计和失败状态。

这是研究者明确授权后整理的公开副本。原始实验报告中的“未Commit/Push”及Git字段对应实验冻结时状态，发布发生在其后。代码中的 /LOCAL/、/LOCAL_HOME 和 LOCAL_HOST 是环境占位符，需要本地授权数据、模型与XP04资产，不能把公开副本当作已验证的即装即跑包。历史SHA对应本地字节，公开文件以publication_manifest.json为准。

未重新训练CMERC，未使用Test话语或标签，未进行原始数据干扰，未根据真正Validation反馈调参。学生编码器不是作者原始RoBERTa权重。IEMOCAP上游缓存可能存在历史划分监督重叠风险，见完整报告。
