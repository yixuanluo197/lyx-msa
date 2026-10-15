# CMERC 从头训练报告

运行日期：2026-10-13（UTC）。服务器：`biggpu`；宿主机健康 GPU 5（Tesla V100-SXM2 32 GiB），GPU 4 明确排除。环境和数据哈希见 `02_ENVIRONMENT.md`、`03_DATA_AUDIT.md`。

## 实验矩阵

每个数据集均运行 M³Net baseline 与完整 CMERC，并分开保存两种协议：

| 协议 | 训练集合 | 选模依据 | 测试使用方式 |
|---|---|---|---|
| `official_test` | IEMOCAP 官方 train；MELD 官方代码的 train+valid | 每 epoch test W-F1 | 忠实保留官方测试集选模行为 |
| `validation` | IEMOCAP train 扣除 seed 固定的 10% 对话；MELD 仅 1,038 个 official train | validation W-F1 | 最佳 checkpoint 确定后只测试一次 |

超参数由 `scripts/run_training.sh` 固化为 README 配置。baseline 关闭 CL/HSCL/CC；CMERC 同时开启 `--courselearning --contrastlearning --calibrate`。未通过修改 batch size、模型结构、损失函数或标签追求指标。

## 训练闭环

- 每 epoch 写入 `metrics.jsonl`：分类、CC、MSCL、CSCL、总损失，train/validation/test 指标，课程比例、实际训练对话数、耗时和峰值显存。
- `latest.pt` 与 `best.pt` 保存模型、Adam、epoch、完整参数、Python/NumPy/Torch/CUDA RNG 和最佳指标，可使用 `--resume` 恢复。
- checkpoint 采用临时文件后原子替换，避免中断留下半写文件。
- IEMOCAP CL 比例为 0.15、0.30、0.45、0.60、0.75、0.90、1.00；MELD 为 0.40、0.80、1.00，此后保持 1.00。
- 规范协议的课程前缀只从排除 validation 后的训练对话中选择；difficulty 缺 ID 会直接报错。

## 运行资源与基准

正式训练前的真实 1-epoch 基准：

| 任务 | train 对话 | epoch 时间 | 峰值显存 |
|---|---:|---:|---:|
| IEMOCAP baseline / official | 120 | 9.11 s | 8,022 MiB |
| IEMOCAP CMERC / official，课程比例 0.15 | 18 | 21.53 s | 8,975.23 MiB |
| IEMOCAP baseline / validation | 108 + 12 valid | 8.46 s（另有最终 test） | 约 8 GiB |

GPU 5 采用两条 CMERC 加一条顺序 baseline 队列；增加到三进程后实测峰值约 31.3/32 GiB，因此不再增加并发。所有 run 使用独立目录和日志。

## 正式结果

正式训练仍在执行；本节只会在进程结束、checkpoint 与独立评估产物核验后填写。当前状态以 `reports/00_TASK_STATUS.md` 和 `logs/training/` 为准，不能将上述 1-epoch 基准当作复现结果。
