# CMERC 实验二至五：复现与文件索引

本目录独立保存后续诊断；上级实验一的全部原始文件均未修改。仅测试推理，不训练，不选择新 checkpoint。

## 已完成内容

- 复用实验一完整输入：每模型 IEMOCAP 1623 条、MELD 2610 条，共 8466 条；实验二、三、五不重新推理。
- 实验四：四个固定 checkpoint 各执行 w/o_text、w/o_audio、w/o_visual；共 120 次 batch 前向、25398 条预测。
- 生成全部明细、统计表、SUMMARY、Word 和 Excel。CSV 中的阈值集合有重叠，跨阈值行数不能作为去重样本数。
- 未找到原始实验一 Word 母版。新 Word 的实验一部分完整转录原 SUMMARY.md，并明确标注替代来源；无法保证未提供母版的独有内容和版式。

## 运行位置和环境

- 工作目录：`/data2/yb/paper/lyx-msa/CMERC`。
- 实际服务器：本地 `user23`；宿主 XP0 / `NVIDIA TITAN Xp`；GPU UUID 见 run_manifest.json。
- 推理环境：`/data2/yb/conda_envs/cmerc-xp-repro`；PyTorch 1.13.1+cu117，CUDA 11.7。
- 文档环境：`/home/yangbin/miniconda3/bin/python`，使用现有 python-docx、xlsxwriter、lxml；未安装依赖。
- 冻结源代码目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/`。模型、加载器、预处理均复用该批次冻结版本。

## checkpoint 身份与来源

沿用实验一 run_manifest.json 的相同四个 best.pt，不扫描测试分数挑模型。Baseline 的 calibrate/contrastlearning/courselearning 均为 False，CMERC 均为 True；两个模型 causal=False。严格加载权重、原模型结构不变。来自本项目已完成的 validation 训练批次，并非作者发布的预训练权重。

| 数据集/模型 | checkpoint 完整路径 | SHA-256 |
| --- | --- | --- |
| IEMOCAP/baseline | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed1475/best.pt` | `bef27e03c3e72ba32036de0ae425b03a5826b27181d0812f4028c8cb062b0c31` |
| IEMOCAP/cmerc | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/IEMOCAP_validation_seed1475/best.pt` | `2c0d1a7a39627e12d1244d23a225ad3da03bbce071a118909921a547a2bfd106` |
| MELD/baseline | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed67137/best.pt` | `22a44516e58fc6f613879085d902cfb7e572c05fb2b50c06d16f4f1cf20bb96e` |
| MELD/cmerc | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/cmerc/MELD_validation_seed67137/best.pt` | `3f79f9eb3547e35ac15ad7ff4345d9b873d0c884a6f3a1ce60a7a7e47077e701` |

IEMOCAP seed1475：Baseline/CMERC 验证集选中 epoch70/43；MELD seed67137：epoch5/8。详见上级实验一 README 与清单。

## 数据与输入协议

| 数据集 | 原始标签/文本/音频/视觉文件 | 文本特征文件 |
| --- | --- | --- |
| IEMOCAP | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/data/IEMOCAP_features.pkl` | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/data/iemocap_features_roberta.pkl` |
| MELD | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/data/MELD_features_raw1.pkl` | `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/data/meld_features_roberta.pkl` |

IEMOCAP 标签顺序：happy/sad/neutral/angry/excited/frustrated；MELD：neutral/surprise/fear/sadness/joy/disgust/anger。`utterance_index` 从0开始。MELD 对话ID沿用真实 DataLoader 的原pickle ID，不能与其他CSV的对话编号直接混用。

原始特征 [T,B,D]；文本4个RoBERTa层各独立替换。官方 caldiff/cal_diff.py 的 noncausal 方式为 `feature.mean(dim=0).repeat(T,1,1)`，包含batch中的Padding。文本替换不动音频/视觉，其他条件同理。mask、说话人、长度和标签均不改。有效预测以对话为主序，再按话语索引展平，显式排除Padding。固定 batch16、shuffle=False、ATen RNN，以保持与原完整输入结果一致。

输入及新输出概率均检查有限性；概率由原 log_softmax 输出 exp 还原，不作额外温度缩放。所有类别概率保存；跨数据集不存在的类别字段标 N/A。均值替换各模态的实际张量哈希、改变元素数和跨模型输入相等性见清单。

## 运行命令

以下推理命令必须使用全新输出目录，拒绝覆盖已有结果；仅当确实需要复现新诊断时执行，不会训练或重跑完整输入。

```bash
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro \
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
bash scripts/with_cmerc_env.sh python scripts/run_high_confidence_followups.py \
  --exp1-root checkpoints/analysis/high_confidence_errors \
  --output-dir checkpoints/analysis/high_confidence_errors/experiments_2_to_5_rerun
```

正式本次使用默认输出子目录 experiments_2_to_5，其余参数相同。复跑前必须核对宿主 XP0 空闲；CUDA_VISIBLE_DEVICES 后的 cuda:0 是进程内编号。

只重新渲染报告：将本目录所有CSV与SUMMARY复制到一个新空报告目录后运行（不需要重新推理）。若找到母版，用 --mother-docx 取代 --allow-missing-mother。

```bash
/home/yangbin/miniconda3/bin/python scripts/render_high_confidence_followups.py \
  --exp1-root checkpoints/analysis/high_confidence_errors \
  --output-dir checkpoints/analysis/high_confidence_errors/experiments_2_to_5 \
  --allow-missing-mother
```

本目录的报告已经生成，渲染器会拒绝覆盖。只读核验命令：

```bash
/home/yangbin/miniconda3/bin/python scripts/verify_high_confidence_followups.py \
  --output-dir checkpoints/analysis/high_confidence_errors/experiments_2_to_5
```

## 文件清单

| 文件 | 内容或行数（不含表头） |
| --- | --- |
| `exp2_paired_outcomes.csv` | 499 |
| `exp2_summary.csv` | 6 |
| `exp2_new_high_confidence_errors.csv` | 13 |
| `exp3_cm_remaining_high_confidence_errors.csv` | 146 |
| `exp3_class_summary.csv` | 39 |
| `exp4_modality_ablation_predictions.csv` | 25398 |
| `exp4_modality_ablation_summary.csv` | 16 |
| `exp4_full_correct_to_masked_error.csv` | 36 |
| `exp4_new_high_confidence_errors.csv` | 239 |
| `exp4_existing_errors.csv` | 7457 |
| `exp5_error_confidence_distribution.csv` | 14 |
| `exp5_overall_summary.csv` | 4 |
| `full_predictions_reused.csv` | 8466；与原实验一每类别概率完全一致 |
| `SUMMARY.md` | 原实验一摘要完整前缀＋全部后续汇总、主要发现、样本和风险 |
| `CMERC_高置信度错误诊断报告_完整版.docx` | 实验一原Markdown转录＋实验二至五真实统计表；母版缺失已标注 |
| `CMERC_高置信度错误诊断结果_完整版.xlsx` | 按实验与两个数据集保存全部CSV行，零样本表保留表头 |
| `run_manifest.json` | checkpoint身份/哈希、原始输入哈希、运行环境、逐batch替换审计 |
| `report_manifest.json` | CSV→Excel/Word的来源行、列、哈希与文档输出哈希 |
| `verification.json` | 独立重算和文档逐单元格核验 |
| `artifact_manifest.json` | 本目录全部交付文件路径、字节数与SHA-256（不含自身） |
| `README.md` | 本说明 |

CSV 所有比例、Accuracy、Weighted F1 以0–1表示；文中带%为百分数。高置信错误数/达到阈值的全部预测数，与错误区间数/全部错误数必须区分。分母0为N/A；有正分母且零错误才是0。实验四新增高置信错以full正确话语为分母；full本已错误仍错误单独另表。

## 验证与局限

独立脚本逐行核对原概率、ID/原文/标签、掩码预测概率范围/归一化/argmax、统计分母和逐样本集合、输入文件原哈希；Excel逐单元格与CSV核对，Word摘要表逐单元格与CSV核对。实验二、三、五另以不调用统计实现的标准库脚本重算，8439项数值比较一致。新增与既有单测的执行证据见 verification.json。

官方原始训练脚本存在按测试集指标选轮次风险；本次固定 validation checkpoint，不重选。原批次continuous RNG/capacity_safe、ATen后端及包含padding的替换协议限制对论文分数的直接比较。结论仅针对本批固定checkpoint与测试集，不代表多seed或全面校准结论；不将高置信错当成模态冲突，不将置信度降低当成纠错，不混用真实类概率上升与错误最大概率。

## 代码快照

`source_code/` 保存本次诊断与测试脚本的最终完整副本，SHA-256见run_manifest.json。推理使用的官方模型与数据加载冻结源码未改动，其哈希沿用实验一并已复查。Git HEAD记录父仓库版本；新诊断脚本尚未提交，本任务未创建提交或推送。
