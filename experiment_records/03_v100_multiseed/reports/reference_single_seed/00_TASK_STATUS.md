# CMERC 复现任务状态

更新时间：2026-10-13（UTC）

官方源码：`HITSZ-HLT/CMERC@815c7c060914f5b55db816c0ba3662b11d08552f`

| 阶段 | 状态 | 完成标准 / 当前证据 |
|---|---|---|
| 源码与论文协议审计 | 已完成 | 见 `01_CODE_AUDIT.md`；论文公式、代码路径、协议差异和兼容问题均已定位 |
| 独立环境 | 已完成 | 专用 CPython 3.10.13 / torch 1.13.1+cu117 / PyG 2.4.0 环境通过 GPU forward/backward |
| 数据与 checkpoint | 已完成 | 4 个真实特征、2 个 difficulty、2 个官方 checkpoint 已按大小和 SHA-256 核验 |
| 真实数据 Smoke Test | 已完成 | IEMOCAP/MELD 均通过 forward/loss/backward/更新/完整 checkpoint 回载 |
| 官方 checkpoint 评测 | 已完成 | W-F1 71.9838/66.8484，CEL 1143.4928/605.0503，与论文 CMERC 值对齐 |
| M3Net baseline 完整训练 | 已完成 | IEMOCAP 2×80 epoch、MELD 2×40 epoch；official_test 与 validation 协议分别保存 |
| CMERC 完整训练 | 已完成 | IEMOCAP 2×80 epoch、MELD 2×40 epoch；CL/HSCL/CC 全开启并保存分项损失 |
| 指标与论文结果对比 | 已完成 | 8 个最佳 checkpoint 均经独立评估；预测、CEL、校准指标和汇总已保存 |
| 可选消融 / 五随机种子 | 未执行 | 主实验已完成；额外运行成本较高且论文未公开五个 seed，本轮不虚构或缩减替代 |

## 执行资源

- 本机：Ubuntu Linux，NVIDIA 驱动 575.51.03，4 张 TITAN Xp 12 GiB；检查时 GPU 0/1/3 空闲，GPU 2 满载。
- `biggpu`：8 张 Tesla V100-SXM2 32 GiB；全部正式实验固定使用健康宿主机 GPU 5，宿主机 GPU 4 按项目规则始终禁用。
- 特征复用已有可信同哈希副本；正式 run、日志和结果隔离保存，并已同步回当前工作区。
- 当前工作区属于上层 `/data2/yb/paper` Git 仓库，已有大量与本任务无关的用户修改；本任务只在 `lyx-msa/CMERC` 及其复现产物范围内工作。

## 科学协议约束

- 官方协议与验证集选模的规范协议分开记录。
- Smoke Test 不计入正式复现结果。
- 不使用随机数据代替缺失的 IEMOCAP/MELD 特征。
- 不在测试集上调参；论文值、官方 checkpoint 实测值、从头训练值分别标注。

## 最终产物

- `runs/formal/`：8 组 `best.pt`、`latest.pt`、`metrics.jsonl`，validation 协议另含 `final_test.json`。
- `results/formal/`：8 组独立评估结果及 `summary.json`。
- `logs/training/`、`logs/formal_evaluation/`：原始训练与评估日志。
- 正式结果与论文差异见 `05_TRAINING_REPORT.md`、`06_EVALUATION_REPORT.md`、`07_REPRODUCTION_COMPARISON.md`。
