# XP 多 seed 复现

批次：`20261014T1627Z`；服务器：本地 `user23`，TITAN Xp GPU 0/1/3。原产物目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/xp_multiseed/20261014T1627Z`。

全部 12 项正式任务及独立评估已于 2026-10-14 20:24:47 UTC 完成，退出码均为 0：两个数据集各 5 个 CMERC seed，另各 1 个 baseline 诊断。最终汇总与原有独立复核记录均保留。

## 文件入口

- `code/`：该批次冻结的可审查源码、脚本、测试和文档。
- [最终报告](REPORT.md)、[结果表](results.csv)、[五 seed 汇总](summary.json)、[状态表](status.csv)。
- `batch_manifest.json`、`snapshot.json`：预注册任务、代码与输入哈希、环境和资源策略。
- `runs/`：12 个 run 的最终生效配置和逐 epoch 指标。
- `results/`：12 个最佳 checkpoint 的独立评估聚合指标。
- `preflight/`：环境、显存、并发、官方 checkpoint 和差距审计证据。
- `verification.json`、`independent_verification.json`：原实验完成时的验证记录。
- 批次根目录的 Python 文件：运行观察器、流水线及独立复核工具，保留原文件；完整复核需要服务器上未公开的权重和预测。
- `export_manifest.json`：公开文件来源与哈希。

## 科学协议

使用非因果模型、固定作者 curriculum 和 `official_test` 测试集逐轮选模。样本标准差采用 ddof=1。

| 数据集 | CMERC seed | 五 seed W-F1 Mean ± Std |
| --- | --- | ---: |
| IEMOCAP | 1475, 42, 3407, 2025, 2026 | 69.6150 ± 0.8254 |
| MELD | 67137, 42, 3407, 2025, 2026 | 65.5253 ± 0.4451 |

两个均值均未达到论文 ±0.5 点。不得挑最高 seed 或混入旧 V100 分数；两个 baseline 只是训练链诊断，不构成五 seed 配对对照。

如需重跑，进入 `code/`，按 `code/scripts/xp_multiseed.py` 和原计划准备新的批次目录。外部数据与 checkpoint 不在本公开包中；两份作者难度文件可从仓库根目录 `caldiff/` 获取并核对 `snapshot.json`。环境中的 UUID 计数修复和显存回收配置需要按原环境说明验证，不要盲目应用到其他 PyTorch 版本。

目录中的验证 JSON 是原实验保留的证据。本次上传只检查文件、哈希、聚合数值与相关 CPU 测试，没有重新训练或加载全部模型复算预测。
