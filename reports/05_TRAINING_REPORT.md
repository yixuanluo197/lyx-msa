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

GPU 5 先顺序完成 baseline 队列，随后并发承载 4 条 CMERC run；并发阶段实测总占用约 24.8/32 GiB。调度器曾在重载窗口尝试启动一条重复 MELD validation 进程，约 2 秒内即在首个 epoch/文件写入前终止；随后加入“先检查 live PID、再检查 metrics”的防重逻辑并确认全程仅保留一个正式进程。所有 run 使用独立目录和日志，正式产物未受该事件污染。

## 正式训练结果

下表的 Accuracy/W-F1/Macro-F1/CEL 均来自训练结束后重新加载 `best.pt` 的独立评估，不是训练循环中提前抄取的中途值。epoch 为一基编号。

| 数据集 | 模型 | 协议 | 最佳 epoch | 选模 W-F1 | Test Acc | Test W-F1 | Macro-F1 | CEL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP | Baseline | official_test | 48 | 69.63 | 69.8706 | 69.6256 | 68.5606 | 1290.3643 |
| IEMOCAP | CMERC | official_test | 55 | 70.38 | 70.3635 | 70.3753 | 69.4909 | 1074.8478 |
| IEMOCAP | Baseline | validation | 77 | 63.47 | 68.1454 | 67.6603 | 66.9487 | 2044.3696 |
| IEMOCAP | CMERC | validation | 79 | 63.08 | 68.6999 | 68.8129 | 67.5839 | 1304.0601 |
| MELD | Baseline | official_test | 4 | 66.06 | 67.5479 | 66.0644 | 49.3229 | 537.3760 |
| MELD | CMERC | official_test | 5 | 66.18 | 67.3946 | 66.1767 | 49.6217 | 616.4127 |
| MELD | Baseline | validation | 6 | 64.62 | 66.8582 | 65.5530 | 48.0479 | 579.2167 |
| MELD | CMERC | validation | 16 | 65.02 | 65.5556 | 64.7363 | 50.0695 | 864.9562 |

`official_test` 的“选模 W-F1”就是每轮 test W-F1，存在官方代码固有的测试集选模偏差；`validation` 的选模列来自 validation，表中 Test 指标只在最佳 checkpoint 确定后计算一次。

## 训练资源与损失证据

| 数据集 | 模型 | 协议 | 训练耗时合计 | 峰值显存 | 最佳 epoch 总损失 | 分类 / CC / MSCL / CSCL |
|---|---|---|---:|---:|---:|---|
| IEMOCAP | Baseline | official_test | 12.34 min | 8022.00 MiB | 0.2402 | 0.2402 / 0 / 0 / 0 |
| IEMOCAP | Baseline | validation | 11.41 min | 4826.57 MiB | 0.0729 | 0.0729 / 0 / 0 / 0 |
| IEMOCAP | CMERC | official_test | 145.67 min | 8978.78 MiB | 1.1787 | 0.3270 / 108.5222 / 3.7099 / 2.4721 |
| IEMOCAP | CMERC | validation | 118.46 min | 5987.97 MiB | 0.9909 | 0.2576 / 86.1972 / 3.7329 / 2.3131 |
| MELD | Baseline | official_test | 7.26 min | 784.05 MiB | 3.4128 | 3.4128 / 0 / 0 / 0 |
| MELD | Baseline | validation | 6.59 min | 781.89 MiB | 3.1936 | 3.1936 / 0 / 0 / 0 |
| MELD | CMERC | official_test | 113.17 min | 903.26 MiB | 4.5788 | 3.4330 / 49.6539 / 4.1147 / 2.8618 |
| MELD | CMERC | validation | 96.76 min | 906.29 MiB | 4.0133 | 2.8401 / 68.2795 / 4.2048 / 2.7064 |

耗时为各 run 的 epoch wall time 求和；CMERC 四路曾并发运行，因此不能相加解释为整批墙钟时间。总损失使用官方权重后的实际反传值；分量列保留未加权原始损失。

课程比例/实际训练对话数按日志验证为：IEMOCAP official `0.15/18 → 0.30/36 → 0.45/54 → 0.60/72 → 0.75/90 → 0.90/108 → 1.00/120`；IEMOCAP validation 对应 `16/32/48/64/81/97/108`。MELD official 为 `0.40/460 → 0.80/921 → 1.00/1152`；validation 为 `415 → 830 → 1038`。

## Checkpoint 闭环

- 8 个 run 的 `metrics.jsonl` 行数严格为 IEMOCAP 80、MELD 40；`latest.pt` 的零基 epoch 分别为 79、39。
- 8 个 `best.pt` 均含模型、Adam、完整参数、RNG 与最佳指标；约 1.6 GiB 的 run 产物已同步到本地。
- 两个 validation 数据集的所有逐 epoch test 字段均为 `null`；最终 test 只在训练结束后运行一次。
- 全部训练日志未检出 Traceback、OOM 或 RuntimeError；正式评估预测也未发现 NaN/Inf。
