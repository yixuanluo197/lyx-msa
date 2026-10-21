# CMERC 官方 README 单 Seed 对照实验

本批只登记4项从头训练：IEMOCAP seed1475、MELD seed67137，各含 Baseline 与完整 CMERC。官方固定 commit：`815c7c060914f5b55db816c0ba3662b11d08552f`。实际选模协议：`official_test_selection`。

**已完成并验收：4/4正式训练、4/4固定最佳checkpoint独立进程重评、2项不更新参数的梯度诊断。** 六份中文报告及所需CSV/JSON已齐备。最终核验见 [final_verification.json](provenance/final_verification.json)，交付文件哈希见 [delivery_manifest.json](provenance/delivery_manifest.json)。

`official_code/` 是官方 git 对象的只读独立副本。`runtime_code/` 仅含两处已实证的 PyG 兼容修复；外部观测器保存分项损失及 checkpoint。运行分类为 `compatibility_patched_with_external_observer`，不称完全未修改官方运行。作者课程难度文件来源固定、生成历史未知。四项正式训练不加载历史权重。

阅读顺序：

1. [官方源码审计](reports/01_OFFICIAL_SOURCE_AUDIT.md)
2. [与 XP04 的协议差异](reports/02_README_PROTOCOL_COMPARISON.md)
3. [环境与资源预检](reports/03_ENVIRONMENT_PREFLIGHT.md)
4. [四项训练状态及验收](reports/04_FOUR_RUN_TRAINING_REPORT.md)
5. [实测结果比较](reports/05_XP04_VS_README_COMPARISON.md)
6. [差距诊断与最小后续实验](reports/06_F1_GAP_DIAGNOSIS_AND_NEXT_STEPS.md)

运行过程中以 `state/status.json` 和 `logs/supervisor.log` 为准；只有进程退出、完整80/40轮及best/final产物均核验后才记为完成。`logs/preflight_*` 全是诊断，不计正式训练数量。正式日志为 `logs/<run_id>.log`，checkpoint 与逐轮/逐 Batch 机器记录位于 `checkpoints/<run_id>/`。正式曲线原始数据为 `epochs.jsonl` 与 `batches.jsonl`，汇总后额外导出 CSV。

实际完整启动 argv、环境变量、PID及GPU UUID记录在 `state/status.json`，监督器持久启动命令在 `provenance/launch.json`；源码、数据、难度及配置哈希在 `provenance/official_source.json`、`provenance/inputs.json`。不得在已有正式输出上重新运行监督器；它会拒绝覆盖或重复训练。

单独重放已固定的最佳 checkpoint（不会重新选模型或训练）：

```bash
CUDA_VISIBLE_DEVICES=GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456 \
LD_LIBRARY_PATH=/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -u \
scripts/evaluate_saved.py --run-id IEMOCAP_baseline_seed1475
```

从本目录运行上述命令。评估器对已存在的输出拒绝覆盖。独立进程重放不等于独立于测试选模；不能将本批测试选模分数与 XP04 验证选模分数直接解释为同协议真实性能变化。每数据集只有一个预先指定 Seed，不代表五 Seed 平均表现，也不进行单 Seed 显著性检验。
