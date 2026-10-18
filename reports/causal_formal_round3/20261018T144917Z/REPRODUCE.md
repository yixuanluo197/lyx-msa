# 只读重算与命令证据

以下命令只重算已经完成的预测与日志，不训练模型、不重新评估测试集。需要原实验服务器保留的本地批次；数据/权重/预测不随Git发布。使用现有multimodalerc310环境，不安装或升级依赖。输出目录必须不存在。

```bash
conda activate multimodalerc310
BATCH=experiments/causal_formal_round3/20261018T144917Z
python scripts/round3_statistics.py --matrix "$BATCH/ANALYSIS_MATRIX.json" --output-dir analysis_recheck_round3
python scripts/round3_reports.py --statistics-dir analysis_recheck_round3 --batch-dir "$BATCH" --matrix "$BATCH/ANALYSIS_MATRIX.json" --output-dir reports_recheck_round3
python scripts/round3_figures.py --statistics analysis_recheck_round3/paired_statistics.json --output-dir figures_recheck_round3
```

正式运行的每一条实际训练/difficulty/评估命令及输出日志SHA都在RAW_LOG_INDEX.json；完整未脱敏命令在服务器每对实验的STATE.json。每run最终生效配置在REPORT_EVIDENCE.json，完整source SHA在RUN_PROVENANCE.json。模型源码冻结在提交5e5d7dd6cb7811f3c0d7c64c4afcbcb4aed9f69d，新工具的源码SHA单独记录。

新训练必须创建新的独占目录和协议，不应把已完成批次当作可覆盖输出目录。本轮没有需要恢复的中断；若将来恢复，只能使用经manifest/difficulty/配置/历史best校验的latest.pt和round3_runner.py显式--resume-baseline或--resume-cmerc，不能只加载权重冒充续训。

图中收敛曲线保留所有epoch，星标为validation选中的best；加权梯度图只展示每epoch首batch，log轴省略零值，所有零值和原始记录仍保留在JSON/CSV。测试配对差图显示三个seed与描述性均值±样本SD，没有显著性标记。

统计脚本：scripts/round3_statistics.py；报告脚本：scripts/round3_reports.py；图脚本：scripts/round3_figures.py；正式分阶段运行器：scripts/round3_runner.py；实际训练与损失仍调用原train_cm.py。
