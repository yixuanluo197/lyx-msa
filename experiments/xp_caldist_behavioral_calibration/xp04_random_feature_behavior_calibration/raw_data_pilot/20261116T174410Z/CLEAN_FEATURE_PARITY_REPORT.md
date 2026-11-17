# Clean 特征与冻结 CMERC 预测 Parity 状态

> 公开报告副本：本地绝对路径已替换；文中 `sources/`、`provenance/` 和 `local_only/` 详细记录仅保留本地。公开证据摘要见 `source_summary.json`。

**NOT RUN / BLOCKED_MISSING_PRODUCER_ASSETS。** 本轮已做发布文件身份验证，没有做原始输入重提取，不能把文件哈希一致写成 clean 特征重提取成功。

## 原冻结模型与参考输入

沿用原始 XP04：`experiments/xp_mainline_validation/20261016T050823Z/`，IEMOCAP seed1475 和 MELD seed67137，分别使用对应 effective config 与 `runs/cmerc/{dataset}_validation_seed{seed}/best.pt`。模型既有服务器归属为本地 user23；没有启动 GPU 或转移服务器。

| 数据集 | checkpoint SHA-256 | 真正 Validation 池 |
|---|---|---:|
| IEMOCAP seed1475 | `2c0d1a7a39627e12d1244d23a225ad3da03bbce071a118909921a547a2bfd106` | 12 Dialogue / 527 条 |
| MELD seed67137 | `3f79f9eb3547e35ac15ad7ff4345d9b873d0c884a6f3a1ce60a7a7e47077e701` | 114 Dialogue / 1109 条 |

两个 checkpoint、effective config 和已有真正 Validation baseline NPZ 共 6 项已保存并复核哈希。特征文件另有本轮官方包比对。未加载 checkpoint 执行新前向。共享 pickle 只能整体反序列化，本轮不索引 Test 标签；总量/划分统计只取 ID 与形状。

## 用户要求的 Parity 指标

| 数据集 / 模态 | 原始每话语特征形状 | 已重提取样本数 | 新特征形状 | 最大绝对误差 | 平均绝对误差 | 余弦相似度 | 原/新 CMERC 概率最大差 / 平均差 |
|---|---|---:|---|---|---|---|---|
| IEMOCAP 文本 | 4 × (1024,) | 0 | NA | NA | NA | NA | NA / NA |
| IEMOCAP 音频 | (1582,) | 0 | NA | NA | NA | NA | NA / NA |
| MELD 文本 | 4 × (1024,) | 0 | NA | NA | NA | NA | NA / NA |
| MELD 音频 | (300,) | 0 | NA | NA | NA | NA | NA / NA |

NA 表示没有计算对象；不是零误差，也不是不相等。没有先抽样再换编码器，没有运行预训练 RoBERTa/OpenSMILE 近似替代来填表。

## 分模态阻断

- 文本：COSMIC 的 6 个指定脚本与辅助处理代码均已找到；参考 tokenizer/BPE/dictionary 也已核对，但实际两数据集 fine-tuned encoder `checkpoint_best.pt`、Fairseq 版本及其与 CMERC 发布特征的关系尚缺。下游 ERC/CMERC checkpoint 不能替代文本 encoder。IEMOCAP 还须核对 encoder 微调 split 与 XP04 split 的重叠风险。
- IEMOCAP 音频：代码命名 IS10/1582 与 M3NET 补充材料 IS13 的说法冲突，缺生产工具版本、精确 `.conf`、特征顺序、切段/采样率/声道/归一化依据。
- MELD 音频：300 维不是已确认的 IS10/IS13 原生输出；尚缺从原始音频到该 300 维向量的完整学习/选择/池化链及必要权重。不能用同宽向量代替。
- 原始媒体：上一轮确定的 MELD dev Dialogue 110 / Utterance 7 媒体缺失仍未被解决，本轮不拿同名其他 split 媒体补替；即使对少量其他完整样本做验证，提取器缺口仍须先解决。

## 恢复后的验证要求

取得原生产材料后，只从 effective config 指定的真正 Validation 池确定少量样本，固定 ID 与 token/音频处理结果；冻结代码、环境和输入哈希。每个文本层和音频向量分别记录形状、max/mean absolute error、cosine，零范数 cosine 记 NA；明确 dtype、容差和生产归一化，不能为了缩小误差临时重标定。

在原始 CMERC checkpoint 的 FP32/eval/ATen、原 batch16、原 padding、完整 Dialogue 上下文下生成 fresh 原始概率，再仅替换目标话语的目标模态 clean 重提取向量。保持其他话语、模态、speaker、mask、label、图结构和模型 state 不变，保存完整概率及逐类绝对差。不同历史批次已有约 1e-7 级浮点差，不把它当原始输入效应。以上是恢复门槛，本轮没有执行。
