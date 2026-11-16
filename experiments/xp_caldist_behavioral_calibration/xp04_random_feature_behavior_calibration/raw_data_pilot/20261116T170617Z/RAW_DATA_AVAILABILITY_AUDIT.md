# 原始数据可用性审计

结论：**当前不满足真实输入干扰 Pilot 的启动条件。** IEMOCAP 的指定 Validation 原始文件覆盖完整；MELD 候选原始包缺少 1 条 Validation 媒体。两者均缺少可证明与 XP04 发布特征一致的原始特征生产链。

## 本轮现场核验

只以 XP04 各自 effective config 的 `provenance.split.validation_ids` 为准：IEMOCAP seed1475 为 12 个 Dialogue / 527 条；MELD seed67137 为 114 个 Dialogue / 1109 条。没有把目录名包含 `validation` 当作划分证据，没有打开历史 Test 预测。共享 pickle 的反序列化会载入整个历史容器；代码只索引这两个 Validation 集的行，不读取、统计或使用 Test 标签。

| 核验项 | IEMOCAP | MELD |
|---|---:|---:|
| Validation 行 / 唯一复合 ID | 527 / 527 | 1109 / 1109 |
| 原始文本逐条完全一致 | 527 / 527 | 1109 / 1109 |
| speaker 与原预测索引一致 | 527 / 527 | 1109 / 1109 |
| 原始情绪标注与冻结标签一致 | 527 / 527 | 1109 / 1109 |
| raw / RoBERTa pickle 文本、speaker、标签一致 | 527 / 527 | 1109 / 1109 |
| 候选原始媒体存在 | 527 / 527 WAV | 1108 / 1109 MP4 |
| WAV 完整 PCM 读取 | 527 / 527 | 不适用；音频嵌于 MP4 |
| 与 XP04 原始生产输入的来源证明 | 未闭环 | 未闭环 |

原始路径来自现场目录和历史来源审计：

- IEMOCAP：`$RAW_IEMOCAP/Session*/sentences/wav/`；文本为相应 `dialog/transcriptions/`，标注为 `dialog/EmoEvaluation/`。本轮 527 个 WAV 均为 16 kHz、单声道、16-bit PCM，按 header 读取全部帧，字节数一致，逐文件保存 SHA-256。这不等于已完成感知质量或历史切段一致性认证。
- MELD：`$RAW_MELD/dev_sent_emo.csv` 和 `dev_splits_complete/`。本轮未解码 MP4 音频流；`ffprobe` 不在当前 PATH。1108 个文件已保存哈希，但“文件存在”不能证明音频流完整或可重现 XP04 特征。

## 精确 ID 对齐与异常

MELD Validation：`internal_dialogue_id = raw_Dialogue_ID + 1039`。原始 `Utterance_ID` 必须从冻结 pickle 的 `videoIDs[dialogue][utterance_index]` 读取；它不总等于数组索引。dev Dialogue 49 的实际原始 ID 为 `0,1,2,3,6,7,8,9,10,11,12,13,14`，有 10 行的原始 ID 与数组索引不同。speaker 名称按原始对话中首次出现顺序重建 slot，再与冻结 one-hot 及预测 `speaker_ids` 核对。

唯一缺失媒体：内部 Dialogue **1149**、`utterance_index=7`，对应 dev **Dialogue 110 / Utterance 7**：

`$RAW_MELD/dev_splits_complete/dia110_utt7.mp4`

已检查 OpenDataLab、`msa_strict_meld` 和 staging 候选目录的同名文件；同一来源的 `dev.tar.gz` 完整成员名遍历也没有 `dia110_utt7` 条目（仅遍历目录，不提取或解码媒体）。找到的同名 Test 媒体属于另一划分，不能拿来替换。没有使用其他项目的 `.pt/.npy` 特征代替原始音频。异常保留于 `local_only/audit_anomalies.csv`，没有删除对应话语。

20261110 来源审计还记载全语料的原始标注/版本异常；它使用了 pickle 自带的 split。该审计所称 Validation `xxx` 行 `Ses03M_impro02_F001` 不在本轮 seed1475 effective config 的 12 个 Dialogue 中。本轮实际 527 条没有 `xxx` 标签，不沿用旧表的划分名称或异常计数。未重跑旧全语料审计或访问 Test 标签。

## 审计范围和限制

全语料文件覆盖率沿用 [20261110 原始来源审计](../../../../xp04_calidist_behavioral_confidence_calibration/data_provenance_audit/20261110T055604Z/RAW_AUDIO_AUDIT.md) 的历史结论；本轮新鲜核验仅覆盖指定 Validation。旧报告记载 IEMOCAP 全部 7433 个 XP04 话语均有候选 WAV，MELD 全语料缺 2 个标准媒体文件。不能将这些历史计数说成本轮完成的全语料解码认证。

完整逐行索引、原始特征哈希、媒体哈希见 `local_only/validation_index_audit.csv`；摘要见 `provenance/audit_results.json`。共享输入文件、checkpoint、配置与冻结源码的 SHA-256 见 `provenance/source_hashes_before.json` 和验证后的 `source_hashes_after.json`。没有产生干扰后的特征哈希，因为干扰阶段未开始。
