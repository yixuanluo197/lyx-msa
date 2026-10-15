# CMERC 评估报告

评估日期：2026-10-13（UTC）。运行环境：`biggpu` 宿主机 GPU 5（V100 32 GiB），CPython 3.10.13、PyTorch 1.13.1+cu117、PyG 2.4.0。数据和 checkpoint 哈希见 `03_DATA_AUDIT.md`。

## 官方 Checkpoint 结果

| 数据集 | 样本 | Accuracy | Weighted F1 | Macro F1 | CEL |
|---|---:|---:|---:|---:|---:|
| IEMOCAP | 1,623 | 71.9039 | 71.9838 | 71.3747 | 1143.4928 |
| MELD | 2,610 | 67.7778 | 66.8484 | 51.0770 | 605.0503 |

W-F1 与论文 CMERC 的 71.98 / 66.85 一致；CEL 四舍五入后与论文 1143.50 / 605.05 一致。这两项已完成严格 checkpoint 复现。

### IEMOCAP 逐类结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| happy | 57.8616 | 63.8889 | 60.7261 | 144 |
| sad | 82.5726 | 81.2245 | 81.8930 | 245 |
| neutral | 69.7044 | 73.6979 | 71.6456 | 384 |
| angry | 72.1519 | 67.0588 | 69.5122 | 170 |
| excited | 79.7872 | 75.2508 | 77.4527 | 299 |
| frustrated | 67.3740 | 66.6667 | 67.0185 | 381 |

混淆矩阵（行是真值，列是预测，类别顺序同上）：

```text
[[ 92,   3,   8,   0,  39,   2],
 [  4, 199,  11,   1,   2,  28],
 [ 20,  24, 283,   6,   9,  42],
 [  0,   0,   9, 114,   0,  47],
 [ 43,   1,  26,   0, 225,   4],
 [  0,  14,  69,  37,   7, 254]]
```

### MELD 逐类结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 77.0360 | 83.5987 | 80.1833 | 1,256 |
| surprise | 60.0000 | 60.8541 | 60.4240 | 281 |
| fear | 32.2581 | 20.0000 | 24.6914 | 50 |
| sadness | 53.1250 | 32.6923 | 40.4762 | 208 |
| joy | 63.5294 | 67.1642 | 65.2963 | 402 |
| disgust | 33.8710 | 30.8824 | 32.3077 | 68 |
| anger | 56.6456 | 51.8841 | 54.1604 | 345 |

混淆矩阵（行是真值，列是预测，类别顺序同上）：

```text
[[1050,  39,  6, 30, 79,  8, 44],
 [  36, 171,  2,  6, 33,  5, 28],
 [  20,   4, 10,  4,  3,  3,  6],
 [  79,  13,  7, 68, 11,  4, 26],
 [  83,  20,  1,  6,270,  4, 18],
 [  21,   4,  1,  4,  2, 21, 15],
 [  74,  34,  4, 10, 27, 17,179]]
```

## CEL 复现

评估器对每个 test utterance 保存 full 输出，并按官方代码执行 5 种信息移除：

- text / audio / visual：用当前 padded batch 在时间维的均值特征替换；
- intra / inter context：使用 `Model_Intra` / `Model_Inter` 删除对应超图边；
- 对真实类别使用官方实现的 log-probability 正差并跨全部样本求和。

| 数据集 | Text | Audio | Visual | Intra context | Inter context | Total |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 311.0491 | 347.7716 | 59.9012 | 204.6824 | 220.0886 | 1143.4928 |
| MELD | 244.7213 | 44.1507 | 15.0943 | 174.5589 | 126.5251 | 605.0503 |

CEL 与论文完全对齐，证实 checkpoint、数据顺序、移除语义和官方 log-probability 尺度均正确。

## 可靠性指标

以下实现假设来自 CMERC 的引用论文，而非 CMERC 仓库公开评估脚本：最大 softmax 作为 confidence；15 个等宽 bins；ECE/MCE/RMSCE 以百分数报告；AUROC/AUPRC 以“预测正确”为正类、confidence 为 score（AUPR-Success）。

| 数据集 | ECE | MCE | RMSCE | AUROC-Success | AUPRC-Success |
|---|---:|---:|---:|---:|---:|
| IEMOCAP 实测 | 12.0684 | 18.1434 | 12.6223 | 70.3122 | 85.5461 |
| IEMOCAP 论文 | 11.27 | 13.33 | 11.97 | 93.02 | 77.01 |
| MELD 实测 | 22.8176 | 28.8641 | 23.4458 | 73.0079 | 84.2160 |
| MELD 论文 | 22.13 | 26.18 | 22.74 | 87.56 | 70.07 |

ECE/RMSCE 接近但不完全一致；AUROC/AUPRC 与论文差异很大且方向性组合不同。这支持代码审计结论：论文未公开 AUROC/AUPRC 的正类、score 和聚合定义，当前结果只能称为“显式假设下的实现”，不能声称严格复现 Table 5。CEL 不存在这一歧义，已严格复现。

## 产物

每个数据集在 `results/official_checkpoint/{DATASET}/` 保存：

- `metrics.json`：全部分类、校准、CEL 和分箱指标；
- `predictions.csv`：dialogue ID、utterance index、标签、预测和置信度；
- `predictions.npz`：完整 labels/preds/log-probs/probs，以及 5 路 removal log-probs。

对应原始终端日志位于 `logs/evaluation/` 和 `logs/official_checkpoint/`。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/with_cmerc_env.sh \
  python scripts/evaluate_checkpoint.py \
  --dataset IEMOCAP \
  --checkpoint checkpoints/IEMOCAP_checkpoint.pkl \
  --output-dir results/official_checkpoint/IEMOCAP \
  --device cuda:0
```

MELD 只需将 dataset、checkpoint 和 output-dir 改为对应名称。

## 从头训练最佳 Checkpoint 独立评估

8 个正式最佳 checkpoint 均使用同一评估器重新加载并推理。IEMOCAP 每项 1,623 个 utterance，MELD 每项 2,610 个 utterance；所有预测和五路 removal log-probability 均为有限值。

| 数据集 | 模型 | 协议 | Accuracy | Weighted F1 | Macro F1 | CEL | ECE | MCE | RMSCE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | Baseline | official_test | 69.8706 | 69.6256 | 68.5606 | 1290.3643 | 12.1581 | 73.3053 | 13.0823 |
| IEMOCAP | CMERC | official_test | 70.3635 | 70.3753 | 69.4909 | 1074.8478 | 11.8548 | 18.8577 | 12.6416 |
| IEMOCAP | Baseline | validation | 68.1454 | 67.6603 | 66.9487 | 2044.3696 | 18.7199 | 31.1153 | 19.5417 |
| IEMOCAP | CMERC | validation | 68.6999 | 68.8129 | 67.5839 | 1304.0601 | 15.6809 | 30.6557 | 15.9766 |
| MELD | Baseline | official_test | 67.5479 | 66.0644 | 49.3229 | 537.3760 | 26.7041 | 35.0349 | 27.7685 |
| MELD | CMERC | official_test | 67.3946 | 66.1767 | 49.6217 | 616.4127 | 22.9522 | 31.6342 | 23.8865 |
| MELD | Baseline | validation | 66.8582 | 65.5530 | 48.0479 | 579.2167 | 22.9670 | 32.8164 | 24.2337 |
| MELD | CMERC | validation | 65.5556 | 64.7363 | 50.0695 | 864.9562 | 11.7450 | 18.9821 | 12.4203 |

逐类 Precision/Recall/F1、support、混淆矩阵、AUROC-Success、AUPRC-Success 和 15-bin 明细均保存在 `results/formal/{baseline,cmerc}/{DATASET}/{PROTOCOL}/metrics.json`；逐样本输出位于同目录的 `predictions.csv/.npz`。`results/formal/summary.json` 汇总论文差值和同协议增益。

结果表明：CMERC 在四组对照中都降低 ECE/RMSCE；但论文定义的 CEL 只在 IEMOCAP 两协议中下降，MELD 从头训练反而升高。该现象必须与官方 checkpoint 严格复现 CEL 的结论分开：它说明单 seed 从头训练的优化结果未稳定重现论文对 MELD CEL 的改善，并非评估器失配。
