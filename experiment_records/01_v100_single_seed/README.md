# 原版 V100 单 seed 复现

原服务器目录：`biggpu:/data2/yb/paper/cmerc_reproduction_20261013`。正式训练使用健康宿主机 V100 GPU 5，GPU 4 禁用；该批次已完成。

共 8 个完整 run：IEMOCAP/MELD × baseline/CMERC × `official_test`/`validation`。IEMOCAP seed=1475、80 epoch；MELD seed=67137、40 epoch。两套协议独立报告。

## 文件入口

- `code/`：从原服务器留存工作目录导出的模型、训练、评测、脚本、测试及依赖配置。
- `provenance/README.source.md`：该工作目录原 README。
- `provenance/checkpoint_configuration.json`：本次从可信本地训练 checkpoint 以 CPU 加载方式提取的配置与进度元数据，未发布权重。
- [训练报告](reports/05_TRAINING_REPORT.md)、[评测报告](reports/06_EVALUATION_REPORT.md)、[结果对比](reports/07_REPRODUCTION_COMPARISON.md)：原服务器报告，保留原文与原记录日期。
- [完整汇总](results/formal/summary.json)：8 个最佳 checkpoint 独立评估的聚合结果。
- `runs/formal/`：逐轮 `metrics.jsonl`，validation 另含 `final_test.json`。
- `results/formal/`、`results/official_checkpoint/`：逐 run 和官方 checkpoint 的聚合指标。
- `export_manifest.json`：源位置、导出时间和逐文件哈希。

根项目的 `reports/05_TRAINING_REPORT.md` 等报告后续补充了更多说明，可结合阅读。这里的服务器原文不会被后来的总结替换。

## 复现与版本边界

`official_test` 忠实保留每轮按测试集 W-F1 选模；`validation` 使用验证集选择最佳 checkpoint，最终测试单独保存。不能将两套结果混成同一估计。

本组缺少与每个 run 绑定的启动时源码哈希／未提交 diff。`code/` 是本次从留存的服务器工作目录导出的审查材料，其导出哈希只证明当前公开文件与留存文件一致，不能证明它就是当时未经变化的训练版本。

如需重跑，先进入 `code/`，按其 `REPRODUCTION_GUIDE.md` 配置环境。外部特征放入 `code/data/`；官方权重放入 `code/checkpoints/`。课程难度的两份小型 `.pkl` 可从本仓库根目录 `caldiff/` 获取，复制到本组 `code/caldiff/` 并核对原审计哈希。使用新的输出目录。

当前公开档案不含模型权重、数据特征、逐样本预测和原始运行日志；不能仅凭此目录执行完整 checkpoint 复核。
