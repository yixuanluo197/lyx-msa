# CMERC · V100 多 seed V2 公开审查包

本包来自 `CMERC（seed）` 的本地同步材料，正式训练与评估服务器为 **biggpu**（记录中的主机名 `user22`），服务器工作目录 `/data2/yb/paper/cmerc_seed_v2_20261014`。V2 队列 **20/20 completed**，最后结束时间 **2026-10-14T21:34:23Z**。本次仅导出材料，未启动训练。

## 优先阅读

1. `reports/10_MULTI_SEED_REPRODUCTION.md`：五 seed 最终结果。
2. `reports/10_MULTI_SEED_REPRODUCTION_ERRATA.md`：历史容量错误 epoch 计数勘误。
3. `reports/PROTOCOL_AND_RESOURCE_AUDIT.md` 与 `reports/01_CODE_AUDIT.md`：协议差异与资源审计。
4. `results/multiseed_validation_v2/summary.json`、`per_seed.csv`、`aggregate.csv`、`paired.csv`：统计、逐 seed 和配对差异。
5. `code/train_cm.py`、`code/dataloader.py`、`code/scripts/multiseed_scheduler.py`、`code/scripts/evaluate_checkpoint.py`、`code/scripts/summarize_results.py`、`code/scripts/audit_batch_capacity.py`：训练、划分、队列、独立评测、汇总和容量检查入口。

## 实验协议与边界

- 两数据集 IEMOCAP / MELD × baseline / cmerc × 五个预注册训练 seed（1475、67137、2024、3407、42），共 20 run。
- IEMOCAP 固定 split-seed=1475；MELD 官方划分；按 validation 选 checkpoint，未按 test 挑 seed。IEMOCAP 80 epoch，MELD 40 epoch。
- **非因果**原模型路径；冻结包内仍包含因果模块文件，不能将其存在理解为本批开启 `--causal`。配置中 `causal=false`。
- V2 使用 `capacity_safe` epoch 排列：batch=16，训练样本完整保留，仅按对话长度拒绝越过原模型 1000 超边 / 5200 节点容量的整轮排列，最多 10000 次重排。其批次协议与原始无约束 shuffle 存在差异。
- continuous RNG，RNN 使用 ATen 后端；与历史官方 cuDNN / legacy RNG 的结果差异不能全归因于 seed。
- 正式任务宿主 GPU 白名单 1/2/3/5，禁用 biggpu GPU4；宿主编号和 UUID 在队列中留存。

## 文件布局与历史材料

`code/` 为 `snapshots/code_snapshot.tar.gz` 安全展开的冻结源码（31 个源成员全部逐文件 SHA256 验证）；其中 AGENTS.md 仅改名移至 `provenance/AGENTS.source.md`，避免历史规则变成嵌套运行指令。`provenance/` 保存来源、源清单与环境。根 `configs/` 是方便审查的配置副本，与 code/configs/ 一致。

`runs/multiseed_validation_v2/` 每 run 仅含 run_manifest.json、逐轮聚合 metrics.jsonl、final_test.json；`results/multiseed_validation_v2/` 每 run 保存独立测试聚合 metrics.json，并附最终汇总、CSV 和 3 张 SVG。数据划分 ID 与数据文件哈希作为复现来源保留，不包含特征或逐样本预测。

`summary_first3.json` 与 `reports/10_MULTI_SEED_REPRODUCTION_FIRST3.md` 是前三 seed 的**暂态快照**，不替代最终五 seed 结果。`reports/reference_single_seed/` 是源码快照携带的**历史单 seed 参考报告**，不属于这批新运行。`attempts/v1/` 是**已退役的容量失败诊断**，与 V2 分开保留和解释，不混入五 seed 均值。`results/preflight_official/` 是官方 checkpoint 的聚合评测材料；`runs/preflight_resume*/verification.json` 仅为两轮恢复检查，不能当成完整训练验证。恢复检查允许 GPU 数值误差，并非位级相等。

`tools/current_tests/` 不在冻结运行快照中，仅供补充代码审查。报告中路径沿用原始服务器目录；在本包查找时对照以上目录。本包故意不改写历史报告正文。

## 外部数据与复现限制

运行需另行按数据授权取得 IEMOCAP / MELD 特征和 caldiff 文件，并按原布局放到代码工作副本的 data/ 与 caldiff/；文件版本可用 run_manifest.json / metrics.json 中的数据 SHA256 核验。环境以 provenance/environment.json 与 configs/requirements-lock.txt 为准。本包不提供数据、模型权重或 checkpoint，不能仅靠下载此包立即复现分数。

公开导出排除所有 checkpoint、原始训练日志、逐样本预测、NPZ、数据、缓存、凭据及其他任务进程信息。V2 完成状态由本地同步队列与预期产物校验；来源本地材料与远端状态的 SHA 一致性由发布主任务核对。所有已导出文件的来源及哈希见 export_manifest.json。清单本身不自包含其 SHA256，避免自引用。
