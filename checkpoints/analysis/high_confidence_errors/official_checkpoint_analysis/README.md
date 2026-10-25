# 作者官方 Checkpoint 可靠性诊断复现

输出目录独立于旧 Validation checkpoint 的六项实验；旧文件不变。入口总报告为 `SUMMARY.md`，Word 为 `CMERC_官方Checkpoint_可靠性诊断报告.docx`，Excel 为 `CMERC_官方Checkpoint_诊断统计.xlsx`。所有 CSV 包含全量符合条件的话语，典型案例仅在正文中摘录。

## 固定输入

- 已验证完整输入预测和冻结官方/兼容代码：`checkpoints/analysis/official_checkpoint_verification/`。本轮从 CSV 派生 `correct`，不改源 CSV。NPZ 只读加载，包含 log_probabilities/probabilities/labels/predictions。
- 权重：`checkpoints/IEMOCAP_checkpoint.pkl`、`checkpoints/MELD_checkpoint.pkl`。SHA-256、来源 Google Drive URL 和参数清单复用原 `checkpoint_inventory.json` / `preflight.json`。纯 state_dict 不含作者训练种子、轮次或选模元数据。
- 原始特征：`data/IEMOCAP_features.pkl`、`data/iemocap_features_roberta.pkl`、`data/MELD_features_raw1.pkl`、`data/meld_features_roberta.pkl`，保持原验证 SHA。它们和模型权重保留在已有服务器，不由此目录分发。
- 模型服务器仍为 `ssh biggpu`（user22），根目录 `/data2/yb/paper/cmerc_reproduction_20261013`；本地镜像 `/data2/yb/paper/lyx-msa/CMERC`。环境 `/data2/yb/conda_envs/cmerc-repro`；本地 CPU 统计沿用 `/data2/yb/conda_envs/cmerc-xp-repro` / 现有报告环境。无新增依赖。

## 运行命令

以下均从项目根目录执行。已有产物不需重复推理；模态删除入口拒绝覆盖已完成条件。若需要独立复核，指定新的输出目录，并先将已核验的完整输入统计 CSV 复制或重新 CPU 生成到新目录。

```bash
# CPU 统计完整输入：实验一、三、五、六；不调用模型。
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro scripts/with_cmerc_env.sh python scripts/analyze_official_reliability.py

# 已执行的持久 V100 启动命令，详见 launch.json / evaluation.log：
# bash checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/run_ablation.sh
# 启动器固定 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1，禁用 biggpu GPU 4。
# 推理脚本 --output-dir 应指定新的单条件目录；固定 batch 16，不按测试结果换配置。

# 对已保存三种删除预测进行 CPU 复算及严格对齐，不调用模型。
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro scripts/with_cmerc_env.sh python scripts/summarize_official_ablation.py

# 从 CSV 生成 Word、Excel、可靠性图；参数详见 --help。
python scripts/render_official_reliability.py --output-dir checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis

# 本轮数值一致性与协议单元测试。
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro scripts/with_cmerc_env.sh python -m pytest -q tests/test_official_reliability.py tests/test_official_modality_ablation.py tests/test_high_confidence_calibration.py tests/test_verify_author_checkpoints.py
# 报告工具已安装于本地现有 Python 环境，独立运行其测试。
python -m pytest -q tests/test_render_official_reliability.py
```

`source_code/` 保存执行源码和必要依赖快照。冻结模型实现来自原验证目录 `code_runtime/`（保留既有两处兼容补丁），本轮不修改原项目模型。逐条件运行均通过官方 `train.py --testing --epochs 0` 创建及严格加载模型，pre-hook 仅替换目标模态，然后沿用官方评估。

## 文件和统计口径

- 每数据集 `all_predictions.csv`：全部有效话语和原概率、真实类别 log probability；`errors_all.csv` 按最大预测置信度降序保存全部错误。
- `high_confidence_errors.csv`：每个阈值 0.80/0.90/0.95 全量筛选。同一样本可在多个阈值重复；MELD 是有表头的空表。
- `class_error_summary.csv`：真实类别支持数、错误及阈值统计。
- `modality_ablation_predictions.csv`：full 加三个替换条件、原文/ID/标签/各类别概率/full 和 masked 最大概率、真实类别概率及其差值，全部保留。
- `modality_ablation_summary.csv`：四种输入条件的所有错误、三个阈值错误和高置信预测分母、准确率、Weighted F1。
- `full_correct_to_masked_error.csv`：完整输入正确到删除后高置信错误的三个阈值数量及比例；分母固定为 full correct count。full wrong → masked wrong 单列，与新增错误分离。
- `new_masked_high_confidence_errors.csv`：新增高置信错误全量逐阈值话语。
- `error_confidence_distribution.csv`：七个固定区间，最后包含 1，分母为该模型全部错误。
- `calibration_bins.csv`：10 等宽箱，左闭右开、最后包含 1，空箱 count=0，观测值缺失；Signed Gap = mean confidence − accuracy。
- 根 `official_cmerc_summary.csv`：full 全部指标。accuracy、F1、ECE、MCE、置信度、比率均为 0–1；CSV 空字段代表未定义，在报告中显示 N/A，不能视为零。
- ECE 按箱样本数加权绝对差；MCE 是非空箱最大绝对差；Brier 是每样本各类别平方误差之和的均值，不除类别数；NLL 为自然对数、由原 log_softmax 计算，不裁剪或归一化。
- 每数据集 `inference/without_{text,audio,visual}/`：原始替换预测 CSV / NPZ、分类报告及只读运行清单，运行日志见根 `evaluation.log`。清单的 input_condition 确定此为模态替换；其中旧包装器沿用的 paper_weighted_f1/gap 仅为参考字段，不能将替换条件当作论文 full 性能验证。

## 验证与解释边界

`full_input_verification.json`、`ablation_verification.json`、`independent_verification.json`、`report_verification.json` 和 `verification.json` 记录概率/ID/原始标签/文本/顺序、mask、替换、strict load、模型不更新和文档数字核验。`preserved_inputs.json` 保护此前 156 个文件；`artifact_manifest.json` 给出本轮产物 SHA-256。

高置信错误只指预测错误且最大类别概率达到阈值；它不直接证明模态冲突。真实类别概率上升是不同变量。置信度下降也不能当作已纠错。当前属于两个固定作者 checkpoint 的描述性分析，不计算训练重复的显著性。

Baseline 来源审计见 `baseline_checkpoint_audit.md` / `baseline_checkpoint_inventory.json`。如果缺少严格匹配的公开 baseline，实验二和所有 Baseline 对比保持 pending，不训练、不以旧 Validation Baseline 替代。作者实际训练选模流程未知，不能根据纯 state_dict 推断。
