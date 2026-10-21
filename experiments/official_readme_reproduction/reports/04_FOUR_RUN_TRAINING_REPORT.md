# 四组正式训练报告

更新：2026-10-21T14:06:43.520933+00:00。状态：`completed`；启动4/4，完整训练完成4/4。

4个固定 run 均按 README 参数从随机初始化训练；只使用 IEMOCAP seed1475、MELD seed67137。预检与正式训练分开。源码两处兼容修复和外部观测见报告03，不属于完全未修改源码运行。

| 数据集 | 模型 | Seed | 状态 | 完成/计划 Epoch | 最佳 Epoch | W-F1 % |
|---|---|---:|---|---:|---:|---:|
| IEMOCAP | baseline | 1475 | completed | 80/80 | 72 | 70.0432 |
| IEMOCAP | cmerc | 1475 | completed | 80/80 | 79 | 70.5108 |
| MELD | baseline | 67137 | completed | 40/40 | 5 | 66.2257 |
| MELD | cmerc | 67137 | completed | 40/40 | 8 | 66.1301 |

## 可追溯证据

完整启动 argv、覆盖环境变量、PID、GPU UUID、启动前显存与退出码见 `state/status.json`；监督器启动见 `provenance/launch.json`。全部 epoch 标准输出见 `logs/<run_id>.log`。实际解析参数在 `checkpoints/<run_id>/effective_args.json`，训练完成后另存 `configs/*.effective.json.snapshot`。

各 run 的 `checkpoints/<run_id>/` 包含 `best.pt`、`latest.pt`、`final.pt`、`best_test.json`、逐轮 `epochs.jsonl` 和逐 Batch `batches.jsonl`。checkpoint保存model/Adam/进度及Python、NumPy、Torch/CUDA RNG；cuDNN内部dropout缓存不保证可序列化，未发生续训。

`official_test_selection`：按官方四舍五入到两位的测试W-F1、严格大于时记录首次最佳轮。观察器将该轮模型固定保存，避免原脚本best_label/pred被最低test_loss覆盖所造成的打印混淆；原始日志仍原样保留。

`results/<run_id>/metrics.json` 是固定最佳checkpoint的独立进程重放，显式不再选epoch。它不独立于此前的测试选模，不能称为无测试泄漏评估。CEL为五种移除的正真类log概率差总和，ECE为15等宽bin百分比；Baseline也只在训练结束后启用移除前向做诊断。

## 验收与资源

机器验收见 `results/metrics_summary.json`：完整轮数、首次最高rounded test-F1、best/final产物、所有Batch有限loss/gradient、每轮完整训练课程ID覆盖、完整固定测试顺序和checkpoint重放一致性。当前资源快照554条，其中发现外部GPU进程的快照0条；离散观测不能证明所有时刻独占。

显存峰值和实际耗时逐run在 `results/per_run.csv`。同卡并发影响速度，耗时不是严谨性能benchmark。`results/epoch_curves.csv` 包含训练/空验证/测试完整曲线，`epoch_loss_curves.csv` 包含各Loss逐轮batch均值及实际对话/句数，`xp04_epoch_curves.csv` 提供对应XP04训练/验证曲线与来源；分项原始值在batches.jsonl，未把分类loss误称总loss。

单面板诊断曲线在 `results/figures/`，每run各有W-F1和Loss的PNG/SVG。W-F1图明确区分README test与XP04 validation，两条曲线不是同一评估协议；Loss图中的正则项乘以原固定README系数，便于与总Loss对照。绘图仅使用既有CPU环境，版本及CSV来源哈希见 `results/figures/manifest.json`，不修改训练环境。

未因分数改变seed、超参数、训练轮数或反复重训。`logs/preflight_*` 保留2次原始兼容失败及4项成功单Batch诊断，均不计正式训练。最终以状态文件实际记录为准。
