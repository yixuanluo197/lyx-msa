# 20项预注册任务命令

命令由继承V2的启动器生成，调用时调度器显式设置本地XP的UUID与专用环境。资源阈值待完整GPU预检后冻结。

## IEMOCAP_baseline_validation_seed1475（stage1）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "baseline",
    "validation",
    "--seed",
    "1475",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed1475/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed1475",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed1475",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed1475",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## IEMOCAP_cmerc_validation_seed1475（stage1）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "cmerc",
    "validation",
    "--seed",
    "1475",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed1475/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed1475",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed1475",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed1475",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## MELD_baseline_validation_seed67137（stage1）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "baseline",
    "validation",
    "--seed",
    "67137",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed67137/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed67137",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed67137",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed67137",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## MELD_cmerc_validation_seed67137（stage1）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "cmerc",
    "validation",
    "--seed",
    "67137",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed67137/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed67137",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed67137",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed67137",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## MELD_baseline_validation_seed1475（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "baseline",
    "validation",
    "--seed",
    "1475",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed1475/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed1475",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed1475",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed1475",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## MELD_cmerc_validation_seed1475（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "cmerc",
    "validation",
    "--seed",
    "1475",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed1475/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed1475",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed1475",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed1475",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## IEMOCAP_baseline_validation_seed67137（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "baseline",
    "validation",
    "--seed",
    "67137",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed67137/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed67137",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed67137",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed67137",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## IEMOCAP_cmerc_validation_seed67137（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "cmerc",
    "validation",
    "--seed",
    "67137",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed67137/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed67137",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed67137",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed67137",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## IEMOCAP_baseline_validation_seed2024（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "baseline",
    "validation",
    "--seed",
    "2024",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed2024/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed2024",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed2024",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed2024",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## IEMOCAP_cmerc_validation_seed2024（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "cmerc",
    "validation",
    "--seed",
    "2024",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed2024/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed2024",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed2024",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed2024",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## MELD_baseline_validation_seed2024（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "baseline",
    "validation",
    "--seed",
    "2024",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed2024/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed2024",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed2024",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed2024",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## MELD_cmerc_validation_seed2024（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "cmerc",
    "validation",
    "--seed",
    "2024",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed2024/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed2024",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed2024",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed2024",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## IEMOCAP_baseline_validation_seed3407（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "baseline",
    "validation",
    "--seed",
    "3407",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed3407/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed3407",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed3407",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed3407",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## IEMOCAP_cmerc_validation_seed3407（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "cmerc",
    "validation",
    "--seed",
    "3407",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed3407/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed3407",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed3407",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed3407",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## MELD_baseline_validation_seed3407（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "baseline",
    "validation",
    "--seed",
    "3407",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed3407/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed3407",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed3407",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed3407",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## MELD_cmerc_validation_seed3407（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "cmerc",
    "validation",
    "--seed",
    "3407",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed3407/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed3407",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed3407",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed3407",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## IEMOCAP_baseline_validation_seed42（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "baseline",
    "validation",
    "--seed",
    "42",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed42/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed42",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed42",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/IEMOCAP/validation/seed42",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## IEMOCAP_cmerc_validation_seed42（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "IEMOCAP",
    "cmerc",
    "validation",
    "--seed",
    "42",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "80",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "IEMOCAP",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed42/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed42",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed42",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/IEMOCAP/validation/seed42",
  "split_hash": "d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62"
}
```

## MELD_baseline_validation_seed42（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "baseline",
    "validation",
    "--seed",
    "42",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed42/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed42",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed42",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/baseline/MELD/validation/seed42",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```

## MELD_cmerc_validation_seed42（stage2）

```json
{
  "training": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/run_training.sh",
    "MELD",
    "cmerc",
    "validation",
    "--seed",
    "42",
    "--split-seed",
    "1475",
    "--rng-policy",
    "continuous",
    "--epochs",
    "40",
    "--batch-policy",
    "capacity_safe",
    "--edge-capacity",
    "1000",
    "--node-capacity",
    "5200",
    "--batch-max-retries",
    "10000",
    "--num-workers",
    "0",
    "--require-noncausal",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc"
  ],
  "evaluation": [
    "bash",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/with_cmerc_env.sh",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/tools/xp_python",
    "-u",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/scripts/evaluate_checkpoint.py",
    "--dataset",
    "MELD",
    "--checkpoint",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed42/best.pt",
    "--output-dir",
    "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed42",
    "--device",
    "cuda:0",
    "--batch-size",
    "16",
    "--num-workers",
    "0",
    "--require-noncausal"
  ],
  "run_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed42",
  "result_dir": "/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/results/cmerc/MELD/validation/seed42",
  "split_hash": "5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8"
}
```
