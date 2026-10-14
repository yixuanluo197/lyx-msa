# CMERC 复现最终总结

状态：**单 seed 主复现闭环已完成；五随机种子与可选消融未执行。**

## 已完成

- 固定并审计官方 CMERC 源码版本和 M³Net 参考版本；论文 PDF 已归档。
- 建立独立 CPython 3.10.13 / PyTorch 1.13.1+cu117 / PyG 2.4.0 环境，并通过 V100 GPU 图网络 forward/backward。
- 从官方来源核验 4 个真实特征文件、2 个 difficulty 文件和 2 个 checkpoint；完成哈希、维度、类别、speaker、对话和 split 审计。
- IEMOCAP 与 MELD 真实数据 Smoke Test 完成 forward、loss、backward、Adam 更新、有限值检查和完整 checkpoint 恢复。
- 官方 checkpoint 的 W-F1 与 CEL 均复现到论文报告精度；预测、置信度、混淆矩阵和五路信息移除输出已保存。
- 修复 PyG 2.4 API 兼容、difficulty 批聚合、显式 validation 拆分、分项损失日志、checkpoint/恢复和两种选模协议；未改变模型数学定义。
- baseline/CMERC × official/validation 的 8 个正式 run 均完成：IEMOCAP 每项 80 epoch、MELD 每项 40 epoch。
- 8 个最佳 checkpoint 均完成独立分类、CEL 和校准评估；预测、标签、置信度、混淆矩阵及五路移除输出已保存。
- 约 1.6 GiB 的正式 checkpoint、逐 epoch 记录、结果和日志已从 `biggpu` 同步回当前工作区。

## 实际主结果

| 数据集 | 模型 | official_test W-F1 / CEL | validation W-F1 / CEL |
|---|---|---:|---:|
| IEMOCAP | Baseline | 69.6256 / 1290.3643 | 67.6603 / 2044.3696 |
| IEMOCAP | CMERC | 70.3753 / 1074.8478 | 68.8129 / 1304.0601 |
| MELD | Baseline | 66.0644 / 537.3760 | 65.5530 / 579.2167 |
| MELD | CMERC | 66.1767 / 616.4127 | 64.7363 / 864.9562 |

IEMOCAP 的 CMERC 在两协议下均优于同协议 baseline；MELD official 的 W-F1 仅提升 0.1123、CEL 变差 79.0367，validation 的 W-F1 和 CEL 都未改善。完整论文差值见 `07_REPRODUCTION_COMPARISON.md`，不能据此宣称从头训练全面复现论文结论。

## 未执行 / 仍有定义边界

- 五随机种子：论文未公开完整 seed 集，本轮只执行官方 README 给出的 seed，不能报告 Mean ± Std。
- w/o CL / w/o HSCL / w/o CC：属于可选消融，本轮未执行，也未用缩小训练替代。
- AUROC/AUPRC：已按明确的 success-positive 假设计算，但论文未公开正类和 score 聚合细节，不能称为严格复现 Table 5。
- 官方核心图与对比学习代码仍含无条件 `.cuda()`；本轮 V100 路径已完整验证，但纯 CPU CMERC 训练仍不受支持。

## 可复现入口

环境、数据、Smoke Test、官方 checkpoint 评估、单 run 训练、队列训练和恢复命令见仓库根目录 `REPRODUCTION_GUIDE.md`。正式训练一条命令示例：

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
CMERC_OUTPUT_DIR="$PWD/runs/formal" \
  bash scripts/with_cmerc_env.sh bash scripts/run_training.sh \
  IEMOCAP cmerc validation --num-workers 0
```

已有正式产物位于 `runs/formal/`、`results/formal/` 和 `logs/`。同一命令会使用固定配置生成可恢复 checkpoint；为防止覆盖既有结果，若需重跑应设置新的 `CMERC_OUTPUT_DIR`。
