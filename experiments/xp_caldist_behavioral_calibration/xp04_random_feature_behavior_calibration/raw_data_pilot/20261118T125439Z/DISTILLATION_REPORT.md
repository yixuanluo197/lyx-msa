# XP04 双数据集 RoBERTa 特征蒸馏报告

生成时间：2026-11-18T14:17:41.016008+00:00。

两套独立 RoBERTa Large 学生均从公开 roberta-large 的缓存快照 722cf37b1afa9454edce342e7895e588b6ff1d59 初始化（权重 SHA-256 047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38），仅使用各自有效 Train 文本及四层缓存目标进行全编码器蒸馏。损失为四层 BOS 原尺度 MSE 的平均；没有情绪标签、分类头、回归转换器或目标归一化。

每个数据集按固定种子将约10%的 Train Dialogue 留作内部验证，最小 FP32 MSE 选择最佳 epoch，patience=2、最多10轮；不再以全部 Train 重拟合。官方 Validation 未参与早停或任何超参数选择。

| 数据集 | 有效 Train | 实际拟合 / 内部验证 | 完成 / 最佳 epoch | 最佳内部 MSE | 训练耗时（小时） | 特征门 |
|---|---:|---:|---:|---:|---:|---|
| IEMOCAP | 5283 | 4747 / 536 | 10 / 8 | 0.042795573 | 0.682 | 未通过 |
| MELD | 9989 | 8989 / 1000 | 10 / 8 | 0.035225588 | 1.150 | 未通过 |

硬件为本地四张 TITAN Xp 12GB：IEMOCAP 使用宿主 GPU0/1，MELD 使用 GPU2/3。每套模型双卡 DDP；训练 FP16 autocast/动态缩放，模型和 AdamW 状态保持 FP32，内部与最终提取均为 FP32；开启梯度检查点。有效 batch、显存峰值、AMP 跳步及完整命令保存在训练配置、摘要和日志。DistributedSampler 为双卡均分，每个 epoch 的拟合部分补齐重复1条 Train 样本；内部验证使用无重复分片，逐样本加权汇总。

Tokenizer 使用本地公开 RoBERTa 资产，经 COSMIC byte-BPE 与 Fairseq 字典 ID 全词表核对；IEMOCAP 原句直接输入，MELD 按 COSMIC 参考进行 ASCII 忽略清理。BOS=0、PAD=1、EOS=2；最多512 token（含特殊 token），固定右侧截断，四层顺序为 [-1,-2,-3,-4]，每层1024维 BOS。

特征兼容性阈值完整沿用历史 TEXT_RECOVERY_PROTOCOL.json：MAE≤1e-5、RMSE≤1e-4、元素 allclose(atol=rtol=1e-4)、平均余弦≥0.99999、最小余弦≥0.9999、有效逐维相关 p05≥0.999，且所有目标非恒定坐标相关性可定义。当前授权仅替代旧协议中禁止新微调的来源前提，未放宽数值门槛。

IEMOCAP 风险：上游文本缓存的历史划分与 XP04 Seed1475 effective split 不同；此前审计发现部分当前 Validation Dialogue 属于缓存自己的历史 Train。上游编码器监督是否实际覆盖这些话语仍不确定，不能将本轮划分隔离解释为已经排除缓存的历史重叠风险。

共享 pkl 的反序列化包含多个划分，但本轮只按明确的有效 Train 和最终 Validation ID 索引文本及四层目标；未索引标签槽或 Test 话语。没有处理音频/视觉，没有修改 XP04 pkl、Checkpoint，没有执行干扰，没有 Git commit/push。交互监控进程在会话中断后消失，已通过 tmux 接管原有分离训练工作进程；没有重启或重复训练。接管后无法取得原父进程的子进程退出码，完成依据同时核对完成日志、训练摘要、最佳模型锁、完整断点和最终验证产物，事件及 PID 记录见 results/execution_status.json。

产物：local_only/{dataset}/training/best_student 保存各学生编码器、Tokenizer 与配置；latest_training.pt 保存 epoch 边界模型/优化器/Scaler/各 rank RNG，history.jsonl 和 logs 保存全部训练记录；results 保存曲线和逐层/逐坐标指标。数值不理想的条件完整保留。Train-only 相同分词输入的目标一致性诊断见 results/{dataset}/train_duplicate_consistency.json；两套拟合数据的重复输入均未出现超出 allclose 容差的目标冲突。这一诊断不参与选模型或改变训练。
