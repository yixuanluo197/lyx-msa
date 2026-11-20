# Fairseq 与 HuggingFace 输出一致性诊断

**结论：PASS。** 在相同公开预训练实际权重、相同输入 Token ID、FP32、eval 条件下，两框架在 IEMOCAP 与 MELD 固定 Train Fit 样本上最后四层 BOS 表示数值一致。观测到的约 1e-7 级 MAE 不支持框架实现差异是上一轮蒸馏大误差的主要原因。

## 冻结设计与数据边界

- 每个数据集选择已有 `fit_indices` 顺序下前32个不同 Token ID 序列；不按教师特征、误差或标签选样本。只读取上一轮 `train_prepared.pt`，不打开原始共享 pkl，不访问内部验证、官方 Validation、Test 或标签。
- 两边直接接受同一个右侧 padding 的 Token ID Tensor，BOS=0、PAD=1、EOS=2；不重新 Tokenize。HF 显式非 PAD attention mask，Fairseq 根据相同 PAD ID 构造 mask。
- 模型均 eval、FP32；关闭 TF32；批量4；提取最后、倒数第二、倒数第三、倒数第四层的 BOS。
- 输出计算前冻结判据：每层所有元素 `allclose(atol=1e-4, rtol=1e-4)`。这是框架一致性诊断判据，未改动原学生特征兼容性门槛。

## 同一份实际权重的证据

公开预训练源：`roberta-large` snapshot `722cf37b1afa9454edce342e7895e588b6ff1d59`；既有本地 `model.safetensors` SHA-256：

`047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`

不是独立下载两份 checkpoint 比较。先用 HuggingFace 加载此文件，再将同一 backbone 的权重逐张量无损复制进原生 Fairseq RoBERTa 编码器；加载后对每一对张量执行 `torch.equal`。

- Fairseq backbone 全部 **388 个张量、354,309,120 个元素**均与 HF 对应张量逐值完全相等；双方参与特征输出的编码器参数覆盖完整。
- Q/K/V 使用双方独立投影矩阵逐一映射，无训练、回归或数值变换。
- HF 额外的 `[1,1024]` token-type embedding 经实测全为零，Fairseq 无此项不改变语义；若非零，脚本会阻断。
- MLM head 与 pooler 都不参与隐藏状态计算；未用随机分类头产生特征。
- 两边24层、1024隐藏维、4096 FFN、16 attention heads、exact GELU、LayerNorm epsilon=1e-5、514个位置 embedding；相同有效 token 位置索引。

完整映射、张量形状及逐张量 SHA-256：`provenance/framework_weight_mapping.json`。

## 分数据集、逐层结果

| 数据集 | 层（距末尾） | MAE | RMSE | 最大绝对误差 | 平均余弦 | allclose |
|---|---:|---:|---:|---:|---:|---|
| IEMOCAP | 1 | 4.06453529e-07 | 5.74569931e-07 | 1.33514404e-05 | 0.9999999999998 | PASS |
| IEMOCAP | 2 | 5.09323167e-07 | 6.81717484e-07 | 7.62939453e-06 | 0.9999999999998 | PASS |
| IEMOCAP | 3 | 3.6523915e-07 | 4.87176178e-07 | 3.81469727e-06 | 0.9999999999999 | PASS |
| IEMOCAP | 4 | 2.97069204e-07 | 3.95030651e-07 | 4.76837158e-06 | 0.9999999999999 | PASS |
| MELD | 1 | 1.86307563e-07 | 2.9510906e-07 | 6.67572021e-06 | 1.0000000000000 | PASS |
| MELD | 2 | 2.35536517e-07 | 4.44129566e-07 | 7.62939453e-06 | 0.9999999999999 | PASS |
| MELD | 3 | 1.74672359e-07 | 3.16023741e-07 | 3.81469727e-06 | 1.0000000000000 | PASS |
| MELD | 4 | 1.55852176e-07 | 2.71656437e-07 | 4.23192978e-06 | 1.0000000000000 | PASS |

两个数据集均32条，形状均为 `[32,4,1024]`，所有输出有限。最大逐层 MAE 为 `5.093231674e-7`；全局最大绝对差为 `1.335144043e-5`。误差远小于上一轮教师对齐误差，并符合单精度不同 attention 运算路径的数值舍入量级；本诊断未进一步归因到单个算子。

## 实际框架版本与限制

- 使用 [Fairseq 官方 v0.9.0](https://github.com/facebookresearch/fairseq/tree/v0.9.0)，tag commit `df2f84ce619edddb80e720a56abc74d5490fed99`；实际执行官方 `RobertaModel` / `RobertaEncoder` / `TransformerSentenceEncoder`，没有以 HF 包装器仿冒 Fairseq。
- 官方源码压缩包 SHA-256：`04d2009a522b9a4d8f9e4aa1724f78b38e488edbe3047b9c7d09ca60a8e2c697`。已核对6个参与编码的关键源码与压缩包内容相同，未修改 forward。
- 仅恢复 NumPy 已移除的 `float/int/bool` 旧别名以兼容导入；源码通过隔离 PYTHONPATH 使用，未安装包、未修改已有环境。
- 实测环境：transformers `4.42.3`，torch `2.2.0+cu118`，numpy `1.24.4`。HF 此已安装版本的 RoBERTa attention 类表仅包含 eager；原训练代码无 attention implementation 覆盖，本次显式 eager 与其默认一致。
- GPU：宿主机1，`GPU-REDACTED`，TITAN Xp；进程只见 `CUDA_VISIBLE_DEVICES=1` 的 `cuda:0`。
- **没有证据确认 COSMIC 实际历史运行恰为 v0.9.0。** 因而结论限定为所测试的官方参考实现与上次 HF 环境的一致性；不宣称已经恢复 CMERC 原始微调权重或全部历史生产流程。
- 此测试不证明教师特征本身来自公开预训练权重，也不判断学生蒸馏是否充分拟合；后两项应结合其他两个诊断。

## 产物与复核

- `scripts/framework_parity.py`：完整可执行诊断，含严格权重覆盖、零 token-type、同 Token ID 和形状断言。
- `results/framework_metrics.csv`、`results/framework_sample_manifest.csv`、`results/framework_summary.json`：逐层指标、脱敏索引与结果。
- `provenance/framework_protocol.json`、`provenance/framework_weight_mapping.json`、`provenance/framework_runtime.json`、`results/framework_verification.json`：冻结判据、来源/输入哈希和验证记录。
- `logs/framework_parity.log`：原始运行日志；`local_only/framework_outputs.npz` 与隔离官方源码保留本地。

**未执行任何模型训练，未修改上一轮学生、原始 pkl 或 XP04 Checkpoint。**
