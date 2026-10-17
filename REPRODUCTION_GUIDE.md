# CMERC 可复现运行指南

本工程以官方 `HITSZ-HLT/CMERC@815c7c0` 为主体。默认非因果路径保留官方网络和损失定义；另增加显式 `--causal` 时间因果变体，改动与使用边界见 [CAUSAL_README.md](CAUSAL_README.md)。原版与因果版 checkpoint、课程难度和结果不能混用。

## 1. 环境

推荐在 Linux + NVIDIA GPU 上使用独立环境：

```bash
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-repro \
  bash scripts/bootstrap_environment.sh
```

此后所有命令经包装器运行，以加载 PyTorch 1.13.1 所需的 CUDA 11.7 动态库：

```bash
CUDA_VISIBLE_DEVICES=5 \
CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
  bash scripts/with_cmerc_env.sh python -c \
  'import torch, torch_geometric, torch_scatter, torch_sparse; print(torch.__version__)'
```

`biggpu` 的宿主机 GPU 4 是坏卡，禁止放入 `CUDA_VISIBLE_DEVICES`。进程内的 `cuda:0` 是上述白名单中的第一张卡，不等同于宿主机 GPU 0。

## 2. 数据与 checkpoint

将官方文件放在源码实际读取的位置：

```text
data/IEMOCAP_features.pkl
data/iemocap_features_roberta.pkl
data/MELD_features_raw1.pkl
data/meld_features_roberta.pkl
caldiff/IEMOCAP_diff.pkl
caldiff/MELD_diff.pkl
checkpoints/IEMOCAP_checkpoint.pkl
checkpoints/MELD_checkpoint.pkl
```

下载来源、大小和 SHA-256 见 `reports/03_DATA_AUDIT.md`。pickle 只应从官方链接或已核验为同哈希的可信归档加载。大文件、特征与权重均被 `.gitignore` 排除。

## 3. 验证工程

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
  bash scripts/with_cmerc_env.sh python -m pytest -q tests

CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
  bash scripts/with_cmerc_env.sh python scripts/smoke_test.py \
  --dataset IEMOCAP --device cuda:0 --output-dir results/smoke --batch-size 1
```

MELD smoke test 只需把 `--dataset` 改为 `MELD`。Smoke Test 不是正式论文结果。

## 4. 评估官方 checkpoint

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
  bash scripts/with_cmerc_env.sh python scripts/evaluate_checkpoint.py \
  --dataset IEMOCAP \
  --checkpoint checkpoints/IEMOCAP_checkpoint.pkl \
  --output-dir results/official_checkpoint/IEMOCAP \
  --device cuda:0
```

评估器保存 Accuracy、Weighted/Macro F1、逐类指标、混淆矩阵、显式假设下的 ECE/MCE/RMSCE/AUROC/AUPRC，以及按官方五路信息移除定义计算的 CEL。MELD 使用对应数据集、checkpoint 和输出目录。

## 5. 一条命令训练

统一入口：

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
CMERC_OUTPUT_DIR="$PWD/runs/formal" \
  bash scripts/with_cmerc_env.sh bash scripts/run_training.sh \
  IEMOCAP cmerc validation --num-workers 0
```

三个位置参数为：

- 数据集：`IEMOCAP` 或 `MELD`；
- 模型：`baseline` 或 `cmerc`；
- 选模协议：`official_test` 或 `validation`。

`official_test` 忠实保留官方源码用测试集 W-F1 选最佳 epoch 的行为；`validation` 对 IEMOCAP 做固定 seed 的对话级 10% holdout，对 MELD 使用官方 114 个 valid 对话，训练时排除验证集，且仅在验证集选出最佳 checkpoint 后测试一次。两套结果不可混报。

每个 run 生成：

```text
metrics.jsonl      # 每 epoch 分项损失、比例、对话数、耗时、显存和分类指标
latest.pt          # 模型、优化器、epoch、参数、RNG 和当前最佳信息
best.pt            # 对应选模协议的最佳 checkpoint
final_test.json    # 仅 validation 协议；验证选模后的单次测试结果
```

从中断点继续时，复用相同配置与输出目录，并把 checkpoint 作为附加参数传入：

```bash
bash scripts/run_training.sh IEMOCAP cmerc validation \
  --resume runs/formal/cmerc/IEMOCAP_validation_seed1475/latest.pt
```

`--epochs` 表示目标总 epoch 数，而不是额外训练的 epoch 数。

## 6. 顺序运行实验矩阵

显存不足以同时容纳全部任务时可使用队列：

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
CMERC_OUTPUT_DIR="$PWD/runs/formal" CMERC_LOG_DIR="$PWD/logs/training" \
  bash scripts/with_cmerc_env.sh bash scripts/run_job_queue.sh \
  IEMOCAP baseline official_test \
  IEMOCAP baseline validation \
  IEMOCAP cmerc official_test \
  IEMOCAP cmerc validation \
  MELD baseline official_test \
  MELD baseline validation \
  MELD cmerc official_test \
  MELD cmerc validation
```

以上一条命令顺序重跑完整 8-run 主矩阵。队列在发现同名 run 已存在时会拒绝覆盖，防止日志和 checkpoint 被无意混写；复跑时应给 `CMERC_OUTPUT_DIR` 和 `CMERC_LOG_DIR` 指定新的目录。正式结果应始终以 `results/`、`runs/` 中的产物和 `logs/` 原始日志为依据，不以终端截屏或论文参考值代替。

训练矩阵完成后，可一条命令独立评估 8 个最佳 checkpoint 并生成汇总：

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
CMERC_RUN_DIR="$PWD/runs/formal" CMERC_RESULT_DIR="$PWD/results/formal" \
  bash scripts/with_cmerc_env.sh bash scripts/evaluate_training_matrix.sh \
  IEMOCAP baseline official_test IEMOCAP baseline validation \
  IEMOCAP cmerc official_test IEMOCAP cmerc validation \
  MELD baseline official_test MELD baseline validation \
  MELD cmerc official_test MELD cmerc validation

python scripts/summarize_results.py \
  --run-root runs/formal --result-root results/formal \
  --output results/formal/summary.json
```

评估脚本为防误覆盖会拒绝已有 `metrics.json`；重复评估时应使用新的 `CMERC_RESULT_DIR`。

## 7. 复现实验边界

- 官方公开 checkpoint 可严格复现论文 CMERC W-F1 与 CEL；细节见 `reports/06_EVALUATION_REPORT.md`。
- 从头训练已分别完成官方测试集选模与规范验证集选模；真实结果见 `reports/07_REPRODUCTION_COMPARISON.md`。
- 五随机种子与消融未在本轮执行；缩小 epoch、修改 batch size 或 Smoke Test 不能标记为其替代结果。
- 论文没有充分公开 AUROC/AUPRC 的正类与聚合定义，因此当前实现明确标记评估假设，不声称严格复现论文 Table 5。

## 8. 因果正式配对实验

2026-10-17 的 [IEMOCAP 第二轮归档](reports/causal_formal_round2/20261017T_round2_1937/README.md)完成了训练 seed1937、split seed1475 的 Causal Baseline 80轮 → 真实 Causal Difficulty → Causal CMERC 80轮，按 validation W-F1 选模，并在主要比较冻结后各 best 一次正式测试。单 seed 不作统计显著性声明；验证 MCE 变差和固定位置分段的负向结果一并报告。

因果配对训练需显式 `--causal --split-seed 1475 --seed 1937 --defer-final-test --fail-on-nonfinite --diagnostic-batches 1`。`--defer-final-test` 避免训练结束时提前看到测试结果；默认旧自动最终测试行为仍保留。完整模型还需要对应真实教师生成并严格验收的 `--difficulty-path`，不能使用第2节的旧非因果 difficulty。

参见 [CAUSAL_README.md](CAUSAL_README.md)及[完整冻结/验收/后续命令](reports/causal_formal_round2/20261017T_round2_1937/FUTURE_EXPERIMENTS.md)。MELD、其他配对种子、组件消融和辅助策略对照未在第二轮自动启动。权重、特征、真实ID/预测和原始训练日志留服务器，GitHub仅保存公开安全的报告、配置、聚合结果与曲线。
