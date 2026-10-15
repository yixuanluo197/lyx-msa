# CMERC（XP）：三张 TITAN Xp 上的多 seed 复现提示词

生成日期：2026-10-14（UTC）。此为历史任务提示词的公开归档副本；实验已完成，最终状态见 [XP 实验档案](../../experiment_records/02_xp_multiseed/README.md)。本文件是交给后续执行会话的任务提示词；本次只完成证据核对、资源检查和文件生成，尚未启动 XP 训练。

## 已核对的事实

1. [论文](https://doi.org/10.1145/3664647.3681515)第 4 页、§4.2 报告 **5 次随机运行的测试集均值**，并称超参数使用 hold-out validation 调整；正文未找到 seed 数值列表。[核对的论文 PDF](https://www.atailab.cn/seminar2024Fall/pdf/2024_MM_Multimodal%20Emotion%20Recognition%20Calibration%20in%20Conversations.pdf)。
2. [官方 README](https://github.com/HITSZ-HLT/CMERC)明确给出 **IEMOCAP seed=1475、MELD seed=67137**，并允许使用已提供的难度文件跳过 baseline／难度计算步骤。不能说作者完全没有给 seed；当前核对的正文和 README 没有列出完整五次 seed 集。
3. 当前正式实验来自 V100。下面数值摘自 [本地结果对比](../../reports/07_REPRODUCTION_COMPARISON.md)，单位为 W-F1 百分点，不能把 checkpoint 评测与从头训练混为一谈。

| 数据集 | 论文 CMERC | 官方 checkpoint 实测 | 从头训练 official_test | 从头训练 validation |
| --- | ---: | ---: | ---: | ---: |
| IEMOCAP | 71.98 | 71.9838 | 70.3753（差 -1.6047） | 68.8129（差 -3.1671） |
| MELD | 66.85 | 66.8484 | 66.1767（差 -0.6733） | 64.7363（差 -2.1137） |

4. 本地机器 `user23` 有四张 TITAN Xp，每张 12,288 MiB。2026-10-14 15:52 UTC 检查时 **0、1、3 空闲，2 正在运行其他用户任务**。这只是资源快照，后续每次启动前必须重新检查。

| 本地宿主机 GPU | UUID | 本方案用途 |
| --- | --- | --- |
| 0 | `GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248` | 健康卡候选 |
| 1 | `GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a` | 健康卡候选 |
| 3 | `GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456` | 健康卡候选 |

5. 既有 V100 完整 CMERC / official_test 日志的峰值 allocated 显存为 IEMOCAP **8,978.78 MiB**、MELD **903.26 MiB**。这些只支持 XP 12 GB 初步可行的判断，不等于已测得 XP 的 reserved 显存、吞吐或最重 batch。
6. 本地四份数据特征均存在，SHA-256 与 [数据审计](../../reports/03_DATA_AUDIT.md)一致。历史专用环境 `/data2/yb/conda_envs/cmerc-repro` **在本机不存在**。现有 `multimodalerc310` 可导入 torch/PyG/CUDA，识别 XP，但版本是 torch 2.2.2+cu121、scatter 2.1.2、sparse 0.6.18，与既有 V100 复现的 torch 1.13.1+cu117 不同；未在该环境验证完整 CMERC GPU 前反向。

## 可直接交给执行会话的任务

你是本项目的实验执行代理。请按现有 AGENTS.md 与 `research-experiment-driver` 执行以下 **本地 TITAN Xp 多 seed 复现批次**，直至计划内运行、独立评估和统计交付完成。先审计和验证，再启动完整训练；不要仅给出建议或启动命令。

### 目标与判定规则

- 调查从头训练与论文 W-F1 的差距是否可能来自随机运行波动，同时定位环境、默认非因果路径和复现协议差异。
- 目标是解释并如实缩小差距。多 seed 估计稳定性，不能承诺提升均值。
- 预先规定达标条件：两个数据集分别满足 `abs(五次 W-F1 均值 - 论文 W-F1) <= 0.5` 个百分点。IEMOCAP 区间为 `[71.48, 72.48]`，MELD 为 `[66.35, 67.35]`。
- 必须按预先固定的五 seed 完整报告均值与样本标准差；不选最高分 seed，不随测试结果追加 seed、修改超参数或提前结束批次。失败运行也必须进入状态清单。
- 即使数值接近，也只能声称当前协议下的数值接近；不能抹去论文声明与公开源码的协议差异。未达到 0.5 点时，交付真实差距和原因证据，不继续无限搜索。

### 范围、服务器与代码隔离

- 源码：`/data2/yb/paper/lyx-msa/CMERC`；当前公开提交 `91a1bffa763d57015896dee1014f0d11c5d09029`。
- 用户为本批次明确选择本地三张 XP。将该批次命名为 `xp_multiseed`，记录服务器归属为本地 `user23`；其 CMERC、baseline、所有 seed、checkpoint 与评估都在本地 XP 完成。
- 保留旧 V100 批次，不迁移或覆盖其运行目录。V100 结果只作单独历史对照，不混入 XP 五 seed 均值。不要因为本地资源忙而把剩余任务自动交给 `biggpu`。
- 本次复现原论文 **默认非因果 CMERC**，确保 `causal=False`，不传 `--causal`。已有因果代码继续保留，不训练因果变体，不把因果模型的指标与论文非因果模型比较。
- 为本批次创建固定代码快照／独立 checkout，并连接只读数据、作者难度文件与官方 checkpoint。先记录 commit、未提交 diff 和文件哈希；正式 run 始终引用固定版本，后续编辑不能污染运行中的代码。
- 使用全新的批次目录，例如 `CMERC/experiments/xp_multiseed/<batch_id>/`，将代码、run、日志、结果与状态分开。不覆盖 `runs/formal/`、`results/formal/`。

### 阶段 A：环境与协议对齐

1. 阅读 `docs/UPSTREAM_README.md`、`REPRODUCTION_GUIDE.md`、`reports/01_CODE_AUDIT.md`、`reports/03_DATA_AUDIT.md`、`reports/05_TRAINING_REPORT.md`、`reports/07_REPRODUCTION_COMPARISON.md`，并检查实际训练与评测源码。
2. 检查是否有相同任务运行／完成，检查三张候选卡的 UUID、占用与健康状态。不要终止 GPU 2 或任何无关任务。
3. 优先在本地建立独立的匹配环境，例如 `/data2/yb/conda_envs/cmerc-xp-repro`。沿用已有 `scripts/bootstrap_environment.sh`；本地 `CONDA_SH` 可设为 `/home/yangbin/miniconda3/etc/profile.d/conda.sh`，不得照抄旧远端用户路径。安装前确认目标路径未占用，不覆写已有环境。
4. 对齐 Python 3.10、torch 1.13.1+cu117、PyG 2.4.0、scatter 2.1.0+pt113cu117、sparse 0.6.15+pt113cu117 及 NumPy 等历史依赖。上游 README 的 sparse 0.5.15 不存在，已有复现记录使用 0.6.15。不要将 PyTorch 2.2 的因果单元测试环境直接当作原论文训练环境。
5. 在健康 XP 上验证 CUDA、PyG/scatter/sparse 的真实前反向，运行现有测试与两个数据集的 Smoke Test。Smoke Test 用独立目录，不算正式分数。
6. 独立评测两个官方 checkpoint，检查 W-F1 是否与既有 71.9838 / 66.8484 对齐。允许的小数舍入／数值偏差预先设为 0.05 个百分点；CEL 对照 1143.4928 / 605.0503，并单独检查五路信息移除定义。超出阈值时先定位环境、输入、模型或指标差异，不盲目启动全部种子训练。
7. 测量完整课程比例、长对话和验证／测试阶段的峰值 allocated/reserved 显存及耗时。仅测初期 15% 课程或 batch_size=1 不足以批准正式 batch_size=16。预留 CUDA context、缓存与峰值余量。
8. 记录 GPU 型号与软件差异：论文使用单张 RTX 4090，旧复现使用 V100，本批使用 XP。只做数值与协议可比性分析，不把旧并发 V100 的耗时直接当作 XP 工时预测。根据 XP 实测完整阶段耗时估计剩余 GPU-hours 和三卡墙钟时间。

### 阶段 B：冻结实验矩阵

首批正式范围限定为 **10 个 CMERC run + 2 个 baseline 诊断 run**，均为当前公开代码的 `official_test` 协议。此协议每 epoch 使用测试集 W-F1 选 checkpoint，属于忠实公开代码的诊断复现，有已知测试集选模偏差；它不是无偏验证集选模。保留并显式披露该事实，不再用测试分数选择 seed／参数。

| 数据集 | 模型 | seed 列表 | epoch |
| --- | --- | --- | ---: |
| IEMOCAP | CMERC | `1475, 42, 3407, 2025, 2026` | 80 |
| MELD | CMERC | `67137, 42, 3407, 2025, 2026` | 40 |
| IEMOCAP | baseline | `1475` | 80 |
| MELD | baseline | `67137` | 40 |

第一项 seed 来自作者公开命令，其他四项是本批次预先指定的复现种子，**不是声称找到了作者完整五 seed 列表**。不得根据结果替换种子。

- 使用 `scripts/run_training.sh` 的既有官方配置：batch_size=16、lr=0.0001、GRU、hyper、concat_DHT、avl、BN、num_L=3；IEMOCAP dropout=0.5／num_K=4，MELD dropout=0.4／num_K=3；CMERC 同时开启 CL、HSCL、CC，权重与课程比例沿用脚本。
- 固定上游 `caldiff/IEMOCAP_diff.pkl`、`caldiff/MELD_diff.pkl` 并记录哈希；两个数据集的所有种子共用只读官方难度文件。因此本实验估计的是固定作者 curriculum 条件下的训练随机性，不是每 seed 重建 curriculum 的完整端到端方差。
- 检查默认非因果路径相对上游的修改是否意外改变数学定义、随机数消耗、辅助模型同步、课程样本顺序、优化器或指标。必要修复以可复现失败为依据，不能为了追分重写算法。影响训练含义的修复要用新批次 ID，不能混合修复前后 seed。
- 两个 baseline 仅用于公开 seed 下的 XP/V100 训练链诊断；单 seed baseline 不能支持五 seed 配对提升或显著性结论。
- 本批次不自动扩展为消融、超参数搜索、每 seed 难度生成、validation 五 seed 或因果训练。

### 阶段 C：三卡调度与持久运行

- 三张卡承载独立 run，不采用三卡 DDP／DataParallel 来改变单次训练语义。每个进程明确绑定一个健康 GPU UUID，内部设备为 `cuda:0`。
- 先在三张卡各启动一个任务，错开启动并观察实际峰值、算力与总吞吐。根据测量决定同卡并发：IEMOCAP 历史峰值约 9 GB，不应在 12 GB 卡上盲目并发两条 IEMOCAP；MELD 较小，可在保留足够显存且实测总吞吐改善时同卡增加任务。遇到 OOM／吞吐下降时降低并发或排队，保持 batch_size、精度与实验协议。
- 先运行官方 seed 的 CMERC 和 baseline 作为锚点；资源释放后从固定队列补位剩余种子。不得因为锚点分数不高就取消有效后续种子。
- 采用项目可用的持久方式（tmux、已有调度器或可靠后台 supervisor）。记录 batch ID、GPU UUID、PID／任务 ID、命令、状态、退出码和日志；避免会话断开导致训练停止。
- 每次启动检查 run 目录、状态表和 live PID，用锁避免重复提交。当前训练脚本会 append 指标，不能依赖它自动防止覆盖。
- 恢复中断任务前比较所有有效参数、数据与版本，完整恢复模型、优化器、epoch 和 RNG。非因果路径的恢复校验不够严格，执行者需自行核对 seed、损失、batch、协议等。核对 JSONL 最后 epoch 与 latest.pt，处理异常中断产生的重复／不完整记录。

### 已核对的命令模板

以下均在冻结的 CMERC 代码目录中执行；变量由调度器填入，需先创建批次日志目录。模板是独立任务示例，不是把同一 seed 重复提交到三张卡。

```bash
CMERC_DATASET=IEMOCAP
CMERC_SEED=42
CMERC_VARIANT=cmerc
CMERC_PROTOCOL=official_test
CMERC_GPU_UUID=GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248
CMERC_BATCH_ROOT=/data2/yb/paper/lyx-msa/CMERC/experiments/xp_multiseed/REPLACE_BATCH_ID
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro
CMERC_PYTHON="$CMERC_ENV_PREFIX/bin/python"

mkdir -p "$CMERC_BATCH_ROOT/logs"

# 执行前由调度器检查 live PID、已有产物与 GPU 资源。
CUDA_VISIBLE_DEVICES="$CMERC_GPU_UUID" \
CMERC_ENV_PREFIX="$CMERC_ENV_PREFIX" \
CMERC_PYTHON="$CMERC_PYTHON" \
CMERC_OUTPUT_DIR="$CMERC_BATCH_ROOT/runs" \
  bash scripts/with_cmerc_env.sh bash scripts/run_training.sh \
  "$CMERC_DATASET" "$CMERC_VARIANT" "$CMERC_PROTOCOL" \
  --seed "$CMERC_SEED" --num-workers 0 \
  >"$CMERC_BATCH_ROOT/logs/${CMERC_DATASET}_${CMERC_VARIANT}_${CMERC_PROTOCOL}_seed${CMERC_SEED}.log" 2>&1
```

`run_training.sh` 将追加参数放在末尾，`--seed` 能覆盖脚本默认 seed。实际 checkpoint 位于：

```text
<batch_root>/runs/<variant>/<Dataset>_<protocol>_seed<seed>/best.pt
<batch_root>/runs/<variant>/<Dataset>_<protocol>_seed<seed>/latest.pt
<batch_root>/runs/<variant>/<Dataset>_<protocol>_seed<seed>/metrics.jsonl
```

直接逐 run 独立评测，并在输出路径中包含 seed：

```bash
CUDA_VISIBLE_DEVICES="$CMERC_GPU_UUID" \
CMERC_ENV_PREFIX="$CMERC_ENV_PREFIX" \
  bash scripts/with_cmerc_env.sh "$CMERC_PYTHON" scripts/evaluate_checkpoint.py \
  --dataset "$CMERC_DATASET" \
  --checkpoint "$CMERC_BATCH_ROOT/runs/$CMERC_VARIANT/${CMERC_DATASET}_${CMERC_PROTOCOL}_seed${CMERC_SEED}/best.pt" \
  --output-dir "$CMERC_BATCH_ROOT/results/$CMERC_VARIANT/$CMERC_DATASET/$CMERC_PROTOCOL/seed$CMERC_SEED" \
  --device cuda:0 --batch-size 16 --num-workers 0
```

注意：`run_job_queue.sh`、`evaluate_training_matrix.sh`、`wait_then_evaluate_all.sh` 和 `summarize_results.py` 硬编码旧单 seed，其中评测路径也不含 seed。**不能直接用它们执行或汇总本批次**。优先新增小型、清晰的 seed-aware 启动／汇总入口，并验证 run 识别、重复检测和均值计算，不改训练算法；新辅助脚本不引入额外依赖。

### 阶段 D：统计、差距定位与交付

1. 每个正式 run 完成足量 epoch；重新加载 best.pt 独立评估，保留逐 seed 的 W-F1、Accuracy、Macro-F1、CEL、选中 epoch、最终配置、耗时与峰值显存。
2. 按数据集分别统计五个 CMERC run 的 Mean ± 样本 Std（ddof=1）、最小／最大值、均值与论文差、是否落入 0.5 点区间。可以给近似 t 区间，但明确 n=5 的限制；不把非显著差异写成复现等价。
3. 既有 V100 单 seed、XP 单 seed、XP 五 seed、官方 checkpoint 与论文五次均值分别列列，不混成同一统计组。不得把五个模型预测集成的分数当作五次单模型均值。
4. 两个数据集完整五次均值均达到阈值时，报告“当前公开代码协议下五 seed 均值数值接近论文”，仍披露 seed 集未知与测试集选模偏差。仅一个 seed／一个数据集达到时不能宣布整体达标。
5. 若均值仍明显落后，核对官方 seed 的 XP/V100 差异、baseline 诊断、难度文件、随机状态、辅助模型同步、数据 split、公开脚本与论文声明。报告有证据的原因与未排除因素；不要直接归咎于显卡或随机种子。
6. 若今后另做 validation 多 seed，要单独建组。当前 IEMOCAP 的 `--seed` 同时控制训练随机性和 10% validation 对话划分；其 Std 混合训练与 split 变异。固定划分的纯 seed 实验需要先独立配置 split seed 并做回归验证，不能悄悄改变；MELD 的官方 validation split 固定。原作者具体 split／五 seed 未公开时如实注明。
7. 交付 `batch_manifest.json`、状态表、逐 seed `results.csv`、统计 `summary.json`、中文 `REPORT.md` 和可复用启动／汇总命令。记录全部 planned/running/failed/completed，不遗漏失败和负结果。
8. 原始日志、数据、checkpoint 与预测留在批次目录。任何后续 GitHub 交付仅包含适当的源码、配置和小型摘要，遵守现有忽略规则和 Lore 提交协议。

### 完成条件

环境与 checkpoint 评测验证有证据；12 个正式任务有明确终态；10 个 CMERC run 全部成功时才报告两个五 seed 统计；独立评测产物齐全；差距、协议和失败项均如实记录。结果未达 0.5 点不等于程序失败，不自动增加实验预算。
