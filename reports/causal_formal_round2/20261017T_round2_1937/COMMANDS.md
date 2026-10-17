# 可重跑命令与原始证据

本批训练运行源码锁定在 `41ffd6d8d56237e7d658660fccc42d70a567f548`；流程验收工具锁定在 `892c7d6`。源码快照、精确 argv/cwd/环境覆盖、每阶段起止时间和原始日志保留在服务器 `experiments/causal_formal_round2/20261017T_round2_1937/`，见 RUN_MANIFEST、ORCHESTRATION_MANIFEST、BASELINE_PROCESS、PIPELINE_STATE。请用完整哈希核对实际文件；模型/特征/真实ID不公开。

正式80轮 Baseline实际命令（在冻结code目录，CMERC_PYTHON指向已有环境，CUDA_VISIBLE_DEVICES通过当时核实的GPU UUID设置，CMERC_OUTPUT_DIR指向本批runs）：

```bash
bash scripts/run_training.sh IEMOCAP baseline validation --causal --seed 1937 --split-seed 1475 --norm LN --num-workers 0 --defer-final-test --diagnostic-batches 1 --fail-on-nonfinite
```

完整模型使用同一命令公共参数，variant改cmerc，追加 `--difficulty-path "$VERIFIED_REAL_DIFFICULTY"`，绑定本批B阶段的SHA。真实difficulty生成、教师/配对验收、验证比较冻结与正式测试的全部命令保存在原始PIPELINE_STATE。后续可移植、按各数据集/seed给出的完整CLI命令见FUTURE_EXPERIMENTS.md和FUTURE_MATRIX.json；它们全部planned，不会自动执行。已有目录拒绝覆盖。

检查/分析工具：

```bash
"$CMERC_PYTHON" scripts/round2_preflight.py --mode inventory --output-dir "$NEW_BATCH"
"$CMERC_PYTHON" scripts/round2_preflight.py --mode baseline --output-dir "$NEW_BATCH"
"$CMERC_PYTHON" scripts/round2_preflight.py --mode cmerc --output-dir "$NEW_BATCH"
"$CMERC_PYTHON" scripts/round2_verify_run.py --dataset IEMOCAP --seed 1937 --split-seed 1475 --run-dir "$BASELINE_RUN" --expected-provenance "$DATA_ENVIRONMENT" --variant baseline --epochs 80 --output "$NEW_VERIFICATION_JSON"
"$CMERC_PYTHON" scripts/round2_analysis.py --baseline-eval "$VALIDATION_BASELINE_EVAL" --cmerc-eval "$VALIDATION_CMERC_EVAL" --baseline-training "$BASELINE_RUN/metrics.jsonl" --cmerc-training "$CMERC_RUN/metrics.jsonl" --output-dir "$NEW_VALIDATION_COMPARISON"
"$CMERC_PYTHON" scripts/round2_plot.py --batch-dir "$COMPLETED_BATCH"
"$CMERC_PYTHON" scripts/round2_future_matrix.py --completed-batch "$COMPLETED_BATCH"
```

preflight是有限工程检查，不是正式结果；两条正式训练均实际完成80轮。已有完整输出勿直接重跑；使用新独立输出目录。测试推理只允许在对应验证分析及两best哈希冻结后执行一次，不对现有正式测试重复执行。

本批正式训练没有中断。若未来同配置运行在epoch<80中断，可复用相同冻结code、输出根目录及参数并加 `--resume "$RUN/latest.pt" --epochs 80`；入口验证manifest、数据/difficulty/配置、Adam/RNG及历史best/日志后继续。完成80轮的本批不能用这个命令再执行同一预算；不把加载best权重称为完整续训。

新运行控制/相关GPU回归58项通过；最初IEMOCAP专用完整run验收回归18项通过。后续泛化/分析工具回归见postrun_tool_regression.log。第一轮大规模合成因果矩阵本轮未重复。
