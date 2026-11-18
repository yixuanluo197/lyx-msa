# XP04 双数据集 RoBERTa 特征蒸馏交付索引

完成状态与科学结论以 [verification.json](verification.json) 和 [最终判断](FINAL_DISTILLATION_DECISION.md) 为准。verified=true 只表示实验与产物完整核验，不代表特征兼容性通过。

- [蒸馏报告](DISTILLATION_REPORT.md)：方法、训练结果、数据边界、来源风险与结论。
- [IEMOCAP 四层特征比较](IEMOCAP_TEXT_CLEAN_PARITY.md) / [MELD 四层特征比较](MELD_TEXT_CLEAN_PARITY.md)。
- [训练曲线 PDF](results/loss_curves.pdf) / [PNG](results/loss_curves.png)。
- configs/DISTILLATION_PROTOCOL.json：正式训练协议；configs/TOKENIZATION_PROTOCOL.json：Tokenizer/BPE 与文本处理；configs/TEXT_RECOVERY_PROTOCOL.json：原冻结数值门槛副本。
- results/{dataset}/：逐层与逐坐标 CSV、特征门 JSON、概率阶段状态、训练曲线 CSV、Train 重复输入目标一致性诊断。
- logs/{dataset}_formal.log 与 local_only/{dataset}/training/history.jsonl：完整步骤与每轮记录。
- results/execution_status.json：实际命令、宿主 GPU、PID、监控恢复和最终阶段状态。
- provenance/：输入哈希、源码版本、启动脚本哈希、环境和 GPU UUID；各学生完整断点的数值与结构核验。

## 仅本地资产

这些文件不得直接加入公开仓库：

- local_only/IEMOCAP/training/best_student/：IEMOCAP 学生编码器、配置和 Tokenizer。
- local_only/MELD/training/best_student/：MELD 学生编码器、配置和 Tokenizer。
- local_only/{dataset}/training/latest_training.pt：最终完成 epoch 的模型、AdamW、Scaler、各 rank 随机状态；可能与最优 epoch 不同。
- local_only/{dataset}/train_prepared.pt 与 train_texts.jsonl：原始训练文本及目标特征。
- local_only/{dataset}/validation/：一次性最终验证的学生特征和本地索引。

## 执行入口

scripts/prepare_distillation_data.py 只准备有效 Train；scripts/train_student.py 执行纯特征 MSE 蒸馏；scripts/evaluate_student.py 在完成锁存在时才能进行一次真正 Validation；scripts/verify_training_checkpoint.py 与 scripts/verify_distillation.py 核验交付。已存在的目录与最终验证不会被静默覆盖。

scripts/complete_existing_runs.py 只接管已经存在的训练进程，不重新提交训练。当前 run 的训练进程未因监控恢复而重启。

本轮没有训练 CMERC、使用 Test、执行文本/音频干扰或 Git commit/push。学生模型不是作者原始 CMERC RoBERTa 权重。
