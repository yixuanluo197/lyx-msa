# 原始非因果 CMERC 多 seed 实验

本工程位于 `/data2/yb/paper/lyx-msa/CMERC（seed）`，与 `../CMERC` 的单 seed 成果分离。V2正式运行服务器为 biggpu，执行目录 `/data2/yb/paper/cmerc_seed_v2_20261014`，专用环境 `/data2/yb/conda_envs/cmerc-repro`。旧服务器目录 `cmerc_seed_20261014` 及本地无V2后缀的run/results保留V1诊断，不能进入V2正式统计。

## 固定矩阵

首轮训练 seeds=1475,67137,2024；稳定性门控后追加3407,42。每seed含 IEMOCAP/MELD 的 Baseline 与完整CMERC，validation选模，共20项；IEMOCAP80epoch、MELD40epoch，batch16等官方超参数不变。IEMOCAP固定split-seed1475；MELD使用官方train/valid/test。禁止causal。

严格组使用continuous RNG；仅GRU/LSTM前向使用ATen后端，解决cuDNN内部dropout状态未进入普通checkpoint的问题。其余BN与图运算保持原路径。旧单seed分数作为历史参考；不同RNG/后端策略的差异需单独解释。

V2额外统一使用 `capacity_safe` 批次排列：不改变batch16/样本/损失/参数，只根据训练对话长度拒绝超过模型固定1000边/5200节点的整轮随机排列。首次候选与legacy采样顺序相同；重抽只使用独立DataLoader generator，并记录次数。此条件采样协议必须与旧单seed/V1分开报告。

## 执行命令

在 biggpu 执行目录运行。预览完整任务矩阵：

```bash
bash scripts/with_cmerc_env.sh python scripts/multiseed_scheduler.py \
  --run-root runs/multiseed_validation_v2 --log-root logs/multiseed_validation_v2 \
  --result-root results/multiseed_validation_v2 \
  --state state/multiseed_validation_v2_queue.json --dry-run
```

调度短测使用独立目录，首seed两数据集两模型各1epoch，不计入正式统计：

```bash
bash scripts/with_cmerc_env.sh python -u scripts/multiseed_scheduler.py \
  --smoke --smoke-id noncausal_seed_preflight_v2
```

正式启动一条命令（必须先通过已有测试与smoke）：

```bash
nohup bash scripts/with_cmerc_env.sh python -u scripts/multiseed_scheduler.py \
  --run-root runs/multiseed_validation_v2 --log-root logs/multiseed_validation_v2 \
  --result-root results/multiseed_validation_v2 \
  --state state/multiseed_validation_v2_queue.json \
  >logs/multiseed_scheduler_v2.log 2>&1 </dev/null &
```

调度器自身选择空闲健康宿主GPU1/2/3/5，禁用GPU4；每卡只运行一项训练/独立评估，实时检查其他进程、ECC、退休页和余量。重启同一命令会读取同一状态、附着仍在运行的PID，不重复提交；同一队列锁阻止第二调度器。错误任务保留产物、不无限重试。

单项新实验或恢复入口：

```bash
CUDA_VISIBLE_DEVICES=1 CMERC_OUTPUT_DIR="$PWD/runs/manual_recovery" \
  bash scripts/with_cmerc_env.sh bash scripts/run_training.sh \
  IEMOCAP cmerc validation --seed 2024 --split-seed 1475 \
  --rng-policy continuous --batch-policy capacity_safe \
  --edge-capacity 1000 --node-capacity 5200 --batch-max-retries 10000 \
  --require-noncausal --num-workers 0
```

恢复时必须使用原outputroot与完整latest.pt，追加 `--resume .../latest.pt`，且训练配置、source/data/split哈希均相符。已完成并写final_test的run不能作为中断任务恢复；需要完整复跑时使用新的输出根目录。

## 产物

- `runs/multiseed_validation_v2/{variant}/{dataset}_validation_seed{seed}/`：run_manifest、逐epoch记录、best/latest、最终test。
- `logs/multiseed_validation_v2/{variant}/{dataset}/validation/seed{seed}.*.log`：训练和独立评估日志。
- `results/multiseed_validation_v2/{variant}/{dataset}/validation/seed{seed}/`：逐样本预测、CEL五路移除与指标。
- `state/multiseed_validation_v2_queue.json`：计划/运行/失败/完成状态，命令、PID、宿主GPU及UUID。
- `snapshots/`：源码归档、完整文件哈希和环境记录，调度启动及每次派发会校验冻结文件。
- `results/multiseed_validation_v2/summary.json`、CSV与SVG：预注册全体seed注册表、每seed、均值/样本SD、成对差值与收敛/最佳epoch分布。
- `reports/10_MULTI_SEED_REPRODUCTION.md`：中文实测报告；未完成时明确标记缺项。

手动更新实际统计：

```bash
bash scripts/with_cmerc_env.sh python scripts/summarize_results.py --multiseed \
  --run-root runs/multiseed_validation_v2 --result-root results/multiseed_validation_v2 \
  --output results/multiseed_validation_v2/summary.json \
  --historical reports/reference_single_seed/summary.json \
  --report reports/10_MULTI_SEED_REPRODUCTION.md
```

只有完整80/40epoch、相同split、有效checkpoint、非因果validation协议和最终独立评估的run进入统计。三seed检查不会根据test高低筛选；五seed全部结束前不宣称五seedMean±Std完成。详细边界见 `reports/PROTOCOL_AND_RESOURCE_AUDIT.md`。

本地同步使用独立tmux会话 `cmerc_seed_v2_artifact_sync_20261014`，每60秒同步状态/日志/评估/报告。可用 `tmux capture-pane -pt cmerc_seed_v2_artifact_sync_20261014` 查看同步输出，日志在 `logs/artifact_sync_v2.log`。恢复同步可在本地执行 `bash operations/sync_artifacts.sh`，文件锁会拒绝同时启动第二份同步程序；完整checkpoint在队列结束后复制。
