# 高置信度错误诊断：复现说明

生成时间：2026-10-21T16:30:59.560038+00:00。服务器：user23。

结果位于 checkpoints/analysis/high_confidence_errors/，对应用户要求的 checkpoint 目录内 analysis/high_confidence_errors/。

## 执行命令

在 CMERC 仓库根目录执行；先复查 GPU 资源与 UUID。下例使用本地宿主机 XP3，在进程内是 cuda:0；应根据执行时资源选择健康卡。

```bash
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data2/yb/paper/lyx-msa/.omx/tmp/high-confidence-tests CUDA_CACHE_PATH=/data2/yb/paper/lyx-msa/.omx/tmp/high-confidence-tests/cuda-cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 bash scripts/with_cmerc_env.sh python scripts/extract_high_confidence_errors.py --batch-root /data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z --output-dir checkpoints/analysis/high_confidence_errors_rerun --device cuda:0 --checkpoint-inventory-json /data2/yb/paper/lyx-msa/CMERC/checkpoints/analysis/high_confidence_errors/checkpoint_inventory.json
```

诊断拒绝覆盖已有输出目录。重跑应指定新目录。TMPDIR 如根分区已满，应设为 /data2 上的独立临时目录。模型和所有文件均只读加载，未调用训练或优化器。

## Checkpoint 来源与身份

主对比固定为同一已完成 XP validation 批次的两个公开 seed，而非搜索测试集最高分。冻结代码来自该批次 code/；batch16、BN、非因果、完整 avl、ATen RNN 后端与原评估一致。

| 数据集/模型 | checkpoint | seed | 验证集选中 epoch（1-based） |
| --- | --- | ---: | ---: |
| IEMOCAP/baseline | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed1475/best.pt` | 1475 | 70 |
| IEMOCAP/cmerc | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed1475/best.pt` | 1475 | 43 |
| MELD/baseline | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed67137/best.pt` | 67137 | 5 |
| MELD/cmerc | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed67137/best.pt` | 67137 | 8 |

全部实际审计的候选身份与排除原因见 checkpoint_inventory.json；未独立核验的其他 seed、attempt、latest/final 不宣称已审计。

## 数据和列定义

- IEMOCAP：data/IEMOCAP_features.pkl、data/iemocap_features_roberta.pkl；6 类按 happy/sad/neutral/angry/excited/frustrated。
- MELD：data/MELD_features_raw1.pkl、data/meld_features_roberta.pkl；7 类按 neutral/surprise/fear/sadness/joy/disgust/anger。
- 输入数据真实路径与 SHA-256、标签映射、代码哈希、模型参数和原评测对比均在 run_manifest.json；使用实际 DataLoader 的 test ID 顺序，不使用另一份 RoBERTa 列表补 ID。
- 每个数据集/模型下 all_predictions.csv 包含全部话语，errors_all.csv 包含全部错误，errors_ge_0p8.csv / errors_ge_0p9.csv 使用 >= 阈值，错误表按置信度降序；均含所有类别 probability_* 列。
- class_error_distribution.csv 与 statistics.json 保存每类统计；各数据集 baseline_vs_cmerc_all.csv 保留全量逐样本对比，baseline_vs_cmerc_errors_union.csv 为至少一模型预测错误的子集。
- max_probability 是预测类别概率，true_class_probability 是真实类别概率。模型输出为 log_softmax，使用 exp 还原概率，不做温度缩放或再校准。
- 高置信度错误比例 = 错误且最大概率>=阈值的条数 / 所有最大概率>=阈值的预测条数；分母为零时记 null / N/A。
- CSV 使用 UTF-8 BOM，按标准 CSV 引号保留原始逗号、引号、换行和控制字符。原文包含换行时，文件物理行数不等于样本行数，请用 CSV reader 读取。

## 只读与验证边界

本次重新加载 checkpoint，执行了真实测试推理；没有使用已有预测代替推理。已有 NPZ 仅在新推理完成后用于核对完整输入输出，不读取 removed_* 作为本分析数据。未修改 checkpoint、标签、特征或旧预测；推理前后输入 SHA256 一致。没有重新训练，也未用本次测试统计选择 checkpoint、seed 或参数。

## 独立核验

38 项相关 CPU 测试通过（32 项新增统计/CSV/对齐测试，加现有评测与导入测试）。本次真实推理后，另从原始 pickle 逐条重读标签、原文与原始话语 ID，并独立复算 CSV 的阈值分母、排序和逐样本对比，结果见 verification.json。MELD 每个模型的 742 条带 U+007F–U+009F 字符原文均完成原样 CSV 回读核验。

下列只读命令会重新检查已有诊断 CSV 与原数据/原预测的一致性，不执行 GPU 推理或训练：

```bash
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro \
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data2/yb/paper/lyx-msa/.omx/tmp/high-confidence-tests \
  bash scripts/with_cmerc_env.sh python \
  checkpoints/analysis/high_confidence_errors/verify_outputs.py
```

复现前先创建 TMPDIR 目录。checkpoint_inventory.json 列出 18 个已实际审计的候选；本次只选其中 4 个 validation checkpoint，其他 checkpoint 未被改动或拿来按测试结果挑选。
