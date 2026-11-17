# 官方 MELD 300 维音频与 XP04 全覆盖数值对齐

审计日期：2026-11-17 UTC。**官方 `audio_emotion.pkl` 的全部 13,708 个真实话语向量，与 XP04 `MELD_features_raw1.pkl` 中对应的 `videoAudio` 逐维精确相同。** 本报告完成发布特征之间的数值 parity；它没有从原始音频重新提取特征，没有执行 CMERC，也没有使用任何 Test 标签。

## 输入身份与范围

官方来源为已由主审计核验身份的 `MELD.Features.Models.tar.gz`。本分支先完成以下两个本地解包文件与该归档对应成员的 SHA-256 核验，再运行正式比较脚本；没有覆盖既有解包目录。

| 归档成员（省略 `MELD.Features.Models/features/`） | 字节数 | 归档成员及既有解包文件共同 SHA-256 |
|---|---:|---|
| `audio_emotion.pkl` | 56,810,678 | `1507cb7a8b78d58ece92d5e31fba85fddd219efd490407a8e566985be6eeca73` |
| `data_emotion.p` | 9,305,133 | `4c1960a6982cbcf1ab197c3f42c7bcf8499f93d7d918194ff7c4397a4dda0aca` |

XP04 对照来自冻结 `xp_mainline_validation/20261016T050823Z/code/data/MELD_features_raw1.pkl`，SHA-256 为 `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0`。归档完整 inventory 由同目录 `evidence/package_inventory.json` 保存；本分支的成员核验见 `evidence/package_inventory.json`。

官方 `audio_emotion.pkl` 是包含 train、val、test 三个 dictionary 的 list。键为原 Dialogue ID 字符串；每个值均为 float32 `(33,300)`，包含真实话语和尾部 padding。只读取其音频向量，无标签字段。`data_emotion.p` 只使用 records 的 `split/dialog/utterance` 字段，没有读取 `y` 或末尾标签字典。XP04 只使用 ID、音频、文本和 split ID 槽位，没有索引 label 槽位。原 CSV 只选取 Dialogue/Utterance ID 及文本做身份核验，没有选取 Emotion/Sentiment 列。

## ID 对齐：保留原 Utterance ID，区分数组位置

不能依据 dialogue 行数或遍历 dictionary 的顺序匹配向量。正式脚本执行以下独立核对：

1. 从官方 `data_emotion.p` 的无标签字段和原 CSV 分别重建各 split / Dialogue 的原 Utterance ID 集合，逐对话核验完全一致。
2. 以数值升序的原 Utterance ID 构成实际话语序列，核对 XP04 `videoIDs` 完全一致，再使用该序列的 rank 作为音频数组行号。官方参考代码也按数值排序话语 ID 后 post-pad：[固定版 `get_dialogue_ids` L74–80](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/data_helpers.py#L74-L80)、[音频构造 L94–115](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/data_helpers.py#L94-L115)。本轮没有运行这些会处理标签的 helper 类。
3. 原 Dialogue ID 到 XP04 键的映射为 train `+0`、dev `+1039`、test `+1153`，每个对话都通过上述 ID 列表和文本核验。偏移不是用 train 对话数量 1038 猜测：原 train ID 范围为 0–1038，但缺少 Dialogue 60。
4. XP04 effective config 的 `provenance.split` 中 train/validation/test ID 集合均与本轮映射完全一致；因此下表 Validation 确实是 Seed 67137 的真正 Validation。

共有 **466** 条话语的原 Utterance ID 不等于数组行号：train 441、Validation 10、Test 15。例：train Dialogue 1 的原 Utterance 8 是数组 row 6；不能拿数组 row 8 代替。

原 CSV 文本与 XP04 对应文本在 13,708 条中有 13,707 条完全相同。唯一不一致为 train Dialogue 556 / Utterance 6 / row 6，保留为版本异常：两边原文长度为 73 / 13，哈希不同。本报告不公开原文。其原 Dialogue/Utterance ID 仍明确一致，音频向量也逐维相同，故没有删除这条结果；若未来进行原始文本干扰，必须另行解决此文本版本差异。

## 全覆盖数值结果

逐元素差使用 float64 计算；MAE 分母为对应的 `话语数 × 300`。`array_equal` 在原 float32 数组上检验，不使用近似容差。

| 范围 | 对话数 | 已对齐 / 应有真实话语 | 比较标量数 | 逐行精确相同 | 最大绝对误差 | MAE |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1038 | 9989 / 9989 | 2,996,700 | 9989 | 0 | 0 |
| 真正 Validation | 114 | 1109 / 1109 | 332,700 | 1109 | 0 | 0 |
| Test，仅特征身份核验 | 280 | 2610 / 2610 | 783,000 | 2610 | 0 | 0 |
| 全部 | 1432 | 13708 / 13708 | 4,112,400 | 13708 | 0 | 0 |

三组缺失真实话语数、额外真实话语数均为 **0**；没有按预测、标签或误差筛除样本。Test 特征只用于来源身份核验，没有读取 Test 标签，也没有计算 accuracy、校准或可靠性指标。

余弦使用 float64 的 `dot(a,b)/(norm(a)*norm(b))`；任一向量为零时记 NA，不记 1。Train 9988 个非零行、Validation 1109 个非零行、Test 2610 个非零行的 cosine 均值均为 1，最小值均为 `0.9999999999999998`，最大值均为 `1.0000000000000002`；全部 13,707 个有定义值均满足 `abs(cosine−1)≤1e−14`。极小越界来自浮点运算，原数组精确相等不受影响。

唯一真实双零向量是 train Dialogue 490 / Utterance 0 / row 0：官方和 XP04 的 300 个元素均为零，cosine 为 NA。真正 Validation 和 Test 均无整行零向量。零向量来源尚未在本分支认定，不能自行称为缺失音频或删除它。

## Padding 独立统计

官方共 `1432 × 33 = 47,256` 个时间步，其中 13,708 个为真实话语，33,548 个为 padding。下表 padding 不进入上述真实话语 parity 分母。

| 范围 | 官方 padding 行 | 全零 padding 行 | 非零 padding 行 |
|---|---:|---:|---:|
| Train | 24,265 | 32 | 24,233 |
| Validation | 2,653 | 0 | 2,653 |
| Test | 6,630 | 0 | 6,630 |
| 全部 | 33,548 | 32 | 33,516 |

因此不能把“超过真实长度的官方音频表示”自动认作零向量，也不能按是否为零推断有效 mask。本比较严格根据元数据真实话语长度去除尾部 padding，再对齐 XP04 实际保留的行；没有修改任何特征值。32 个全零 padding 位于 train Dialogue 490，与其 1 个真实双零行合起来使该对话的 33 个官方时间步均为零。

## 可支持的结论与剩余门槛

本结果将 MELD 300 维来源关系从“同维度、可能来自官方上下文网络”升级为**官方发布 `audio_emotion.pkl` 与 XP04 实际音频特征全覆盖、逐维精确相同**。这比只有同文件名或配置名称的证据更强。

但本分支没有从 WAV 重提取 1611 维特征，没有运行官方音频网络，没有重新运行 CMERC，也没有测量原始输入到特征或预测的 parity。原声学描述符生产配置、特征选择映射以及上下文网络的可执行验证，须分别由其审计证据支撑；不能把本表的零误差传播为整个原始音频链已经零误差。官方网络属于双向上下文模型时，目标话语输入变化还可能影响同对话其他行，后续干扰协议必须明确这一边界。

## 复核、异常与文件范围

脚本为 `scripts/compare_audio_300d.py`。默认复用已验证的归档成员证据，并再次核对本地成员哈希；`--verify-archive-members` 可重新流式扫描归档，不解包或覆盖旧文件。结构、metadata 和有效 split 断言全部通过；已另以逐话语 CSV 独立聚合核对分母和误差。旧 pickle 在当前 NumPy 2.4.4 下产生 dtype 弃用警告，读取成功且没有吞掉该警告。

机器摘要为 `evidence/feature_parity_summary.json`。逐话语结果 `local_only/audio_300d_per_utterance.csv` 仅本地保存，包含身份索引、文本哈希和误差，不含原文或标签。公开交付可使用本报告与脱敏汇总，不上传此 CSV、pickle 或原媒体。特征对比分支未训练、干扰或执行模型前向。审计完成后的脱敏报告发布遵循本轮最后的上传指令。
