# 交付终态：已完成，不要重新训练

本批已完成且仅完成4项正式从头训练：IEMOCAP Baseline/CMERC seed1475各80轮；MELD Baseline/CMERC seed67137各40轮。4项固定最佳checkpoint独立进程重评、2项不更新参数的单Batch梯度诊断、目标分布诊断和最终验收均完成。没有追加Seed、超参数调整、正式失败重试或断点续训。

根目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/official_readme_reproduction`。`state/status.json`、`state/finalization.json` 已为 inactive/completed；交付终态另见 `state/delivery.json`。监督器与自动后处理均已退出。GPU3已释放，不得停止其他GPU任务。后续不需要重新运行监督器、评估器或训练。

入口为 `README.md` 和 `reports/01` 至 `06`。`results/per_run.csv`、`comparison.csv`、`metrics_summary.json` 为正式数值交付；独立重评完整精度在各run的 `metrics.json`。`provenance/final_verification.json` 已通过，验证原始日志、全部Batch的Loss加权、完整轮数/ID覆盖、最佳轮、latest/final模型一致、数据哈希和重评一致。16张PNG/SVG曲线的输入与输出SHA在 `results/figures/manifest.json`，formal_training_complete=true；IEMOCAP CMERC最终W-F1/Loss图已人工视觉检查。所有脚本末态SHA及交付文件SHA分别在 `provenance/tool_hashes.json`、`delivery_manifest.json`。

官方固定commit为 `815c7c060914f5b55db816c0ba3662b11d08552f`。`official_code` 为未改副本，`runtime_code` 仅含移除未使用topk导入和HypergraphConv第二次propagate尺寸轴修复；`train_cm.py` 未改。外部trace观测器保存参数、Loss、预测与checkpoint。运行分类为 `compatibility_patched_with_external_observer`。兼容补丁和2次原始预检失败均保留，4项成功单Batch预检不计正式训练。

服务器归属是本地user23，使用GPU3 UUID `GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456`；环境为 `/data2/yb/conda_envs/cmerc-xp-repro`。绘图仅复用另一个CPU环境，未改训练依赖。没有跨服务器分流、修改历史XP04或触及无关CODEX提示文件。未要求提交/推送。

正式W-F1：IEMOCAP Baseline70.0432%、CMERC70.5108%；MELD Baseline66.2257%、CMERC66.1301%。CMERC相对Baseline增量为+0.4676/−0.0956个百分点，未恢复论文增量。官方为测试集选轮 `official_test_selection`，XP04为验证集选轮，训练池/RNG/GRU/batch也不同，不作同协议泛化或单因素因果解释。每数据集一个Seed，不代表五Seed均值。作者课程文件与官方及XP04相同、训练ID覆盖完整且不含测试ID，但生成checkpoint/历史未知。

CEL为五种移除后正真类log概率差总和，不是分类交叉熵；ECE为15个等宽bin的百分比。梯度为固定checkpoint上单训练Batch，辅助快照来自当前权重而非历史Epoch开始，诊断进程BN缓冲可变但保存checkpoint未改；不代表纵向根因。HSCL目标分布不是伪标签准确率。独立重评不消除此前测试选轮偏差。四项对应XP04 sampler_retries均为0，不能归因容量拒绝重采样。

报告06的最小后续对照仅是建议，本批未执行额外训练。训练池对照需双方共同独立验证集；不能把已参与官方训练的XP04验证IDs回称独立验证。
