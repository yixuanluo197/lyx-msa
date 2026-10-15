# CMERC 数据与 Checkpoint 审计

审计日期：2026-10-13（UTC）。只对官方 GitHub、官方 README 指向的 Dropbox / Google Drive，以及已有官方归档的同哈希文件进行反序列化。

## 来源与可用性

- 官方 Dropbox 特征链接可匿名访问，响应文件名为 `ERC features.zip`，长度 514,504,220 B。
- 工作区已有官方归档 `/data2/yb/paper/05_reproduction/external_repos/CMERC/ERC_features.zip`，长度与在线响应一致，SHA-256 为 `716e24a9852d2babfcd7bab977b94fe0a59b78aa3f8a9ab568215e1236f8b627`。当前工程以只读软链接复用其 4 个文件，避免重复占用空间。
- 官方 Google Drive checkpoint 文件夹可匿名访问。已按官方文件 ID 下载到 `checkpoints/`，大小与 Drive 元数据一致。
- 官方仓库自带 `caldiff/IEMOCAP_diff.pkl` 和 `caldiff/MELD_diff.pkl`。

## 文件清单与校验

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `data/IEMOCAP_features.pkl` | 72,216,498 | `ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3` |
| `data/iemocap_features_roberta.pkl` | 124,438,886 | `4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06` |
| `data/MELD_features_raw1.pkl` | 88,193,675 | `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0` |
| `data/meld_features_roberta.pkl` | 229,654,413 | `58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951` |
| `caldiff/IEMOCAP_diff.pkl` | 3,308 | `db4c7b96f169948a3615a62b29d2a793bd64cccd8a0387b1de9d7ceab348a150` |
| `caldiff/MELD_diff.pkl` | 13,587 | `db005ceb0e4847caffc8571eaa94ab7c5b4a310abd615b12c995c9a22ff060e0` |
| `checkpoints/IEMOCAP_checkpoint.pkl` | 26,278,739 | `f3098fed2734c18171c02a59d014ee27320d845bc1ad5dfa17162b8dcfb49925` |
| `checkpoints/MELD_checkpoint.pkl` | 56,500,755 | `1c72e29c173b749675885bdd4f477bd5d3702ab7cca7eb03cb145850bf0beba7` |

两个 checkpoint 均可由 PyTorch 加载为 `OrderedDict`；IEMOCAP 有 85 个 tensor，MELD 有 83 个 tensor。它们只包含模型 state dict，不包含优化器或运行配置。

## 实际加载结果

| 数据集 | 训练池对话 | 官方 valid ID | 测试对话 | 训练池 utterance | 测试 utterance | 类别 | speaker |
|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 120 | 无预定义 | 31 | 5,810 | 1,623 | 6 | 2 |
| MELD | 1,152（含 1,038 train + 114 valid） | 114 | 280 | 11,098（train+valid） | 2,610 | 7 | 9 |

类别计数：

- IEMOCAP train：`{0:504, 1:839, 2:1324, 3:933, 4:742, 5:1468}`；test：`{0:144, 1:245, 2:384, 3:170, 4:299, 5:381}`。
- MELD train+valid：`{0:5180, 1:1355, 2:308, 3:794, 4:1906, 5:293, 6:1262}`；test：`{0:1256, 1:281, 2:50, 3:208, 4:402, 5:68, 6:345}`。

首个真实样本验证的特征维度：

- IEMOCAP：4 × `[30,1024]` 文本、`[30,342]` 视觉、`[30,1582]` 音频、`[30,2]` speaker mask。
- MELD：4 × `[14,1024]` 文本、`[14,342]` 视觉、`[14,300]` 音频、`[14,9]` speaker mask。

IEMOCAP train/test 对话 ID 交集为 0；MELD 的 train/valid/test ID 两两交集均为 0。需要注意：官方 `MELDDataset` 并未用这些 ID 拆分 `self.keys`，所以源码训练路径会把 114 个 valid 对话并入训练池。这是协议实现问题，不是原始数据泄漏。

## Difficulty 对齐

- IEMOCAP difficulty 有 120 条，与 120 个训练对话完全一一对应，值域约 `[0.0364, 0.5329]`。
- MELD difficulty 有 1,152 条，与代码的 train+valid 对话池完全一一对应，值域约 `[0.1111, 0.8018]`。
- 仓库自带 difficulty 可直接用于忠实官方 CL。若重新生成，必须修复 `cal_diff.py` 的批聚合问题或强制 `batch_size=1`，不能照 README 使用 16。

## 路径结论

源码要求 4 个特征文件直接位于 `data/`：

```text
data/IEMOCAP_features.pkl
data/iemocap_features_roberta.pkl
data/MELD_features_raw1.pkl
data/meld_features_roberta.pkl
```

README 中 `data/iemocap/...`、`data/meld/...` 是下载包内部路径描述，不是 `dataloader.py` 的运行路径。正式运行以源码路径为准。

## 未发现与限制

- 未用随机数据替代任何正式数据。
- 未修改原始 pickle 内容。
- 数据许可沿用上游数据集和作者分发条款；工程不提交大文件、特征或权重到 Git。
