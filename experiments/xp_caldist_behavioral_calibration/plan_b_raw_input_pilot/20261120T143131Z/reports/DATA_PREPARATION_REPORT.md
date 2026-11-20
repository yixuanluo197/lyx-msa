# Plan B 数据准备报告

仅准备原始文本、原始音频路径、官方视觉特征、原标签与 Speaker；未启动训练。

| 数据集 | Train / 对话 | Validation / 对话 | Fit | Inner dev | 视觉 | Speaker | 缺失音频 Train / Val |
|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 5283 / 108 | 527 / 12 | 4747 / 97 对话 | 536 / 11 对话 | 342 | 2 | 0 / 0 |
| MELD | 9989 / 1038 | 1109 / 114 | 8989 / 934 对话 | 1000 / 104 对话 | 342 | 9 | 1 / 1 |

有效 Train/Validation 划分沿用 Seed1475 / Seed67137 的 XP04 frozen config；Train 内 holdout 完全沿用 20261118T125439Z 已固定的 Dialogue 分组 CSV，只读取索引与文本哈希，不加载旧教师特征 `.pt`。Fit、Inner dev 与官方 Validation 之间无 Dialogue 重叠。

基础官方 pkl 仅索引 0（真实话语 ID）、1（Speaker）、2（情绪标签）、5（视觉）及 6（原始文本）槽位。共享 pkl 反序列化会物化包括 Test 的容器，但代码没有选择 Test 标识或值，未索引旧文本槽位 3、旧音频槽位 4。

IEMOCAP 文本逐字符保留官方 pkl 原始字符串；MELD 全部使用原始 Train/dev CSV 按精确 Dialogue ID+真实 Utterance ID 对齐的文本，不做 ASCII 删除或情绪表达修改。发现 1 条历史文本来源冲突，即 Train dialogue556/真实utt6；按本轮新 baseline 指令采用原始 CSV 文本，旧 pkl 文本仅在本地 text_source_conflicts.json 完整留档。Validation 文本全部与旧 pkl 一致，Train/Validation 标签全部与 CSV 一致。保留原始标点/控制字符。IEMOCAP 真实话语 ID 中 M/F 与 Speaker 一致。Speaker 使用原官方 one-hot（IEMOCAP M=[1,0]、F=[0,1]；MELD 原 9 槽）。

Train dialogue556/真实utt6 还存在唯一 Speaker 元数据冲突：原始 CSV 为 Ross，在该对话其他行对应 slot1；该行官方 pkl 为 slot0。按冻结决定保留用户要求的原 Speaker ID（slot0），保留原始 CSV 文本，不删除话语、不修改标签或划分，并将这一历史异常单独披露。不能声称全部 Speaker 与 CSV 一致；该异常仅在 Train，不在 Validation pilot。MELD Speaker 槽位是 Dialogue 内编码，同一角色在不同 Dialogue 中可使用不同槽。

MELD Validation 的 XP04 Dialogue ID 使用原始 dev ID +1039。真实 Utterance ID 与数组位置分别保留，不用数组位置猜测媒体名称。IEMOCAP 音频指向原 WAV，MELD 指向同划分原 MP4 的音轨；本阶段不变换、合成或填充音频。

已知原 MELD Train dia125_utt3 损坏、dev dia110_utt7 缺失。未成功恢复的条目保留完整文本、视觉、标签及顺序，audio_paths=None，详见本地 audio_missing.json；不删除话语、不用历史音频特征替代、不从 Test 借同名媒体。有限同版本恢复检查由音频准备负责人记录。

完整现存许可媒体逐文件 SHA-256 共 16906 个，且与历史原始媒体 SHA 一致；所有输入配置、官方 pkl、历史 ID 清单、输出 dataset.pt 均已哈希留档。见 `provenance/data_preparation.json`；媒体哈希和完整数据仅保留在 `local_only/data/`。

接口：`dataset.pt={dataset,seed,class_names,splits:{train,validation},fit_dialogue_ids,inner_dev_dialogue_ids}`；每个 dialogue 包含 dialogue_id、utterance_ids、speaker、labels、texts、audio_paths、visual。所有 Dialogue ID 与 Utterance ID 输出为字符串；标签 LongTensor，Speaker/视觉 FloatTensor。

IEMOCAP 上游视觉/标签/文本源保持历史版本；先前 RoBERTa 上游训练划分重叠风险不转化为本轮旧文本特征输入，本轮没有使用任何旧文本或旧音频表示。官方视觉提取器的历史生产元数据仍存在既有来源限制。
