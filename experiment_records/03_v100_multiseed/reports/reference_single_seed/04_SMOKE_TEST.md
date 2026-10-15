# CMERC 真实数据 Smoke Test

运行日期：2026-10-13（UTC）。服务器：`biggpu`；宿主机 GPU 5（进程内 `cuda:0`，Tesla V100-SXM2 32 GiB）。本测试只验证工程闭环，不作为论文性能结果。

## 结果

| 数据集 | 真实对话 | Utterance | 输出 | 官方分类损失 | 峰值显存 | 状态 |
|---|---|---:|---|---:|---:|---|
| IEMOCAP | `Ses02F_script01_3` | 30 | `[30,6]` | weighted NLL = 1.9617015 | 273,207,296 B | 通过 |
| MELD | `0` | 14 | `[14,7]` | FocalLoss = 6.0179400 | 606,134,784 B | 通过 |

两套数据均实际完成：

- Dataset 初始化与真实 DataLoader batch；
- 4 × RoBERTa、视觉、音频、speaker mask、utterance mask 和标签维度核对；
- M³Net hypergraph forward；
- 官方分类损失；
- backward 与 Adam 参数更新；
- 输入、输出、损失和梯度 NaN/Inf 检查；
- `linear_a`、`linear_v`、`gru_l`、`graph_model.fc1`、`smax_fc` 的有限梯度及实际更新；
- 包含模型、优化器、Python/NumPy/Torch/CUDA RNG 的 checkpoint 保存；
- strict checkpoint 加载、optimizer 恢复和逐 tensor 精确一致性；
- 回载后推理输出一致性。

IEMOCAP 有 73 个可训练参数 tensor，其中 48 个在该 batch 路径获得有限梯度并更新；MELD 为 71 / 44。两者最大绝对参数更新约 `1.00017e-4`。回载后输出最大绝对差均为 `2.38419e-7`，在 float32 的 `rtol=1e-5, atol=1e-7` 下相等。

## 产物

```text
results/smoke/iemocap_smoke.json
results/smoke/iemocap_smoke_checkpoint.pt
results/smoke/meld_smoke.json
results/smoke/meld_smoke_checkpoint.pt
```

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/with_cmerc_env.sh \
  python scripts/smoke_test.py \
  --dataset IEMOCAP --device cuda:0 \
  --output-dir results/smoke --batch-size 1

CUDA_VISIBLE_DEVICES=5 bash scripts/with_cmerc_env.sh \
  python scripts/smoke_test.py \
  --dataset MELD --device cuda:0 \
  --output-dir results/smoke --batch-size 1
```

官方 hypergraph 内仍有无条件 `.cuda()`，因此脚本遇到 CPU 请求会明确失败，不会把不完整路径报告为通过。
