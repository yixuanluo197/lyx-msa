# OpenSMILE 音频生产链补充审计

> 公开报告副本：本地绝对路径已替换；文中 `sources/`、`provenance/` 和 `local_only/` 详细记录仅保留本地。公开证据摘要见 `source_summary.json`。

审计日期：2026-11-16 UTC。对象为 XP04 IEMOCAP Seed 1475 与 MELD Seed 67137 所消费的冻结音频特征。本轮仅核查来源；未重新提取、扰动、训练或执行推理，也未使用 Test 标签。公开来源的固定版本、URL、SHA-256 及本地证据位置保存在 `provenance/audio_sources.json`。

## 结论

**仍未找到可复现 XP04 音频向量的完整生产链，不能开始 Clean Feature Parity。** 本轮取得了比上轮更明确的文件来源证据，同时发现论文补充材料与消费者代码之间的配置冲突。基础 pickle 相同仅证明文件来源，不证明从 WAV 到该文件的生产过程可恢复。

- CMERC 与 M3NET 消费者代码明确以 `IS10:1582` 定义 IEMOCAP 音频宽度，以 `MELD_audio:300` 定义 MELD 宽度；它们直接读入 pickle，未提供相应 WAV 提取入口。[CMERC train.py L341–342](https://github.com/HITSZ-HLT/CMERC/blob/815c7c060914f5b55db816c0ba3662b11d08552f/train.py#L341-L342)、[CMERC dataloader.py L9–15、L61–68](https://github.com/HITSZ-HLT/CMERC/blob/815c7c060914f5b55db816c0ba3662b11d08552f/dataloader.py#L9-L15)、[M3NET train.py L340–341](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/train.py#L340-L341)。
- M3NET 的官方 CVPR 2023 **补充材料第 1 页 §A 却写 IEMOCAP 使用 IS13、MELD 使用 IS10**。已以 PDF 原页视觉核对，并由 MinerU 与 PyMuPDF 独立取文确认，非 OCR 将配置名读错。主文第 6 页 §4.2 仅说使用 OpenSMILE，并未消解这一冲突。[官方补充材料](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_Multivariate_Multi-Frequency_and_CVPR_2023_supplemental.pdf#page=1)、[主文](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Multivariate_Multi-Frequency_and_Multimodal_Rethinking_Graph_Neural_Networks_for_Emotion_CVPR_2023_paper.pdf#page=6)。
- MM-DFN 公开基础 pickle 与 XP04 文件逐字节一致，显著加强基础特征的公开文件来源关系；其仓库仍未提供原音频生产配置或 MELD 300 维形成链。详细文件哈希与结构对照见本目录 `MMD_FN_FEATURE_RELATIONSHIP.md`、`provenance/mmdfn_audit.json`。

不能把上述冲突自行判为笔误，也不能把 IS13 换成 IS10 后声称复原了原提取器。

## IS10 / IS13 的可验证差异

下面是官方工具参考配置的事实，**不是对 XP04 生产配置的认定**。本轮固定 OpenSMILE 源码 commit `055dda577c82383f8239663e5ac6c58d25b504db`；未发现证明 XP04 使用此版本的生产记录。

| 项目 | 官方参考事实 | 对 XP04 的含义 |
|---|---|---|
| IS10 Paraling | 官方文档明确为 1582 个统计特征：34 个 LLD 与对应 delta 的 functionals，加 voiced pitch 统计、onset 与 duration | 与代码宽度兼容；代码变量名和维度仍不足以定位原配置文件 |
| IS13 ComParE | 原始挑战论文第 3 页明确为 6373 个特征；其 LLD、平滑及 functionals 与 IS10 不同 | 不能直接输出消费者所需 1582 或 300；若生产时用 IS13，必有尚未闭环的后续选择/映射步骤 |
| emobase2010 | 官方文档同样给出 1582 维，但 maxPos/minPos 的位置归一化方式与 IS10 有差异 | “1582 维”不是唯一配置标识 |
| IS10 兼容模式 | `IS10_paraling.conf` L10–15 明确警告其与早于 2.1 的版本及原 IS10 数值不相同，兼容配置为 `IS10_paraling_compat.conf` | 即便确定配置家族，仍需确切版本、compat 选择及配置哈希 |

依据：[固定版官方文档 IS10](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/doc/sphinx/get-started.rst#L1373-L1389)、[emobase2010](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/doc/sphinx/get-started.rst#L1661-L1692)、[IS10 版本警告](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/is09-13/IS10_paraling.conf#L10-L15)、[IS13 原始挑战论文第 3 页 §3](https://www.isca-archive.org/interspeech_2013/schuller13_interspeech.pdf#page=3)。

## IEMOCAP 1582 维：已知与缺失

消费者读取 `IEMOCAP_features.pkl` 中 `videoAudio`，转为 FloatTensor；该入口不执行 WAV 解码、帧描述符抽取或音频特征归一化。四层 RoBERTa 文件不承担这一音频生产过程。[读取与返回位置](https://github.com/HITSZ-HLT/CMERC/blob/815c7c060914f5b55db816c0ba3662b11d08552f/dataloader.py#L9-L15)、[音频张量](https://github.com/HITSZ-HLT/CMERC/blob/815c7c060914f5b55db816c0ba3662b11d08552f/dataloader.py#L38-L49)。

上轮已核对原始 Validation WAV 完整，但“有 WAV”与“知道如何生成原 1582 维向量”之间仍缺：原 OpenSMILE 二进制版本与编译信息、配置及所有 include 的哈希、特征字段顺序、生产命令、切段/边界选择、采样率与声道处理、缺失值策略，以及任何后续标准化的拟合统计量和作用范围。

本轮对 527 条真正 Validation 的值域检查为 float32、形状 `(527,1582)`，最小值约 −28.44635、最大值约 42.16069、总体标准差约 1.01974。这只能描述原文件，不能据此认定采用 z-score、确定拟合 split 或反推具体配置；详见 `provenance/validation_audio_value_summary.json`。

## MELD 300 维：不能视为标准 OpenSMILE 输出

MELD 官方 ACL 2019 论文第 6 页 §4.1 给出的流程是 OpenSMILE 6373 个描述符/统计量，再进行基于 SVM 的 L2 特征选择；官方 README 区分了两种产物：[原始论文](https://aclanthology.org/P19-1050.pdf#page=6)、[固定版 README](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/README.md#L143-L167)。

1. `audio_embeddings_feature_selection_emotion.pkl` / sentiment：1611 / 1422 维，是 feature selection 后的输入。
2. `audio_emotion.pkl` / sentiment：300 / 600 维，被说明为 uni-modal bcLSTM 的上下文表征。

因此官方资料提供了“低维音频向量可能来自训练后的网络”的具体上游线索，而不是一个可直接运行的 300 项声学配置。**尚无文件级/数组级证据将 XP04 `MELD_features_raw1.pkl` 的音频等同于官方 `audio_emotion.pkl`。** 不将这个线索写成已证实的 CMERC 生产链。

现存官方 baseline 还暴露出另一处边界：`get_audio_model()` 使用两层 `Bidirectional(LSTM(300, activation='tanh'))`，命名 `utter` 的层输出在默认 concat 规则下应是 600 维；导出代码载入训练后的 HDF5 模型，再取该命名层输出。2018 年最早的对应文件也如此。故不能直接用当前脚本加任意权重生成所需 300 维。[现版模型 L83–100](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/baseline.py#L83-L100)、[导出 L232–248](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/baseline.py#L232-L248)、[2018 固定历史版本](https://github.com/declare-lab/MELD/blob/5401f54cef64508e648abfa5cc3f615b7f56ec48/baseline/baseline.py#L83-L100)。

本轮仅描述 XP04 真正 Validation 的 1109 个 300 维向量：float32，范围约 `[−0.99999994,1.0]`，零元素约 44.04%，不存在整行全零。这与某些有界网络表征数值相容，但不能证明 tanh、网络架构或 checkpoint 身份，也不能证明全部零元素的来源。

若原始低维表征来自有监督网络，恢复原链必须取得**准确的原模型、原权重与原预处理**，并非本轮再训练一个模型。若是固定特征选择或投影，则必须取得原字段索引或变换矩阵、拟合统计量与执行顺序。两者目前均未闭环。

## 切段、采样率、帧级聚合与归一化核查

| 环节 | 可验证的参考证据 | CMERC / XP04 实际生产状态 |
|---|---|---|
| 话语切段 | MELD 原始公开数据按 Dialogue/Utterance 提供片段；官方 README 说明由时间戳抽取视听片段 | 上轮建立原片段 ID 对齐；本轮未找到生产者实际边界、音轨抽取命令或额外裁切策略 |
| WAVE 输入 | 官方 `standard_wave_input.conf.inc` 使用文件自带音频输入，默认 start=0、end=−1，并 monoMixdown=1 | 无证据证明原生产使用该 include / 默认值；其本身不是“全部重采样到 16 kHz”的声明 |
| IS10 帧 | 固定参考 core LLD 使用 60 ms / 10 ms 与 25 ms / 10 ms 两个分支 | 未证实为 XP04 原帧设置 |
| IS13 帧 | 固定参考 core LLD 使用 60 ms / 10 ms 与 20 ms / 10 ms 分支；后者实例名虽含 frame25，数值实际为 0.020 | 不按实例名猜 25 ms，也不套用通用的 30 Hz / 100 ms 描述 |
| 帧到话语 | 参考 `FrameModeFunctionals.conf.inc` 以 `frameMode=full, frameSize=0, frameStep=0` 聚合完整输入，包含多个 functionals | 不等同于“对帧简单求均值”；未确认实际生产有无额外窗口/池化 |
| 特征内部归一化 | 参考配置中对部分位置、时间、回归等 functionals 有各自归一化选项 | 不等同于外部对全向量做 z-score 或 L2 norm |
| 外部标准化 | 消费者加载音频时仅转换 FloatTensor；模型内的 `--norm BN` 属于网络计算 | 原文件生成前是否已标准化、其统计量/拟合 split 未知；不得将 BN 配置当成音频生产标准化证据 |
| MELD 后处理 | 官方另有 SVM selection 和 bcLSTM 表征路径 | 未获 XP04 对应字段列表、低维映射、权重、版本或模型导出节点 |

参考锚点：[MELD 切段说明 L77–79](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/README.md#L77-L79)、[WAVE 输入 L13–20](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/shared/standard_wave_input.conf.inc#L13-L20)、[IS10 core LLD](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/is09-13/IS10_paraling_core.lld.conf.inc#L54-L61)、[IS10 25ms 分支](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/is09-13/IS10_paraling_core.lld.conf.inc#L170-L176)、[IS13 core LLD](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/is09-13/IS13_ComParE_core.lld.conf.inc#L16-L56)、[全输入 functionals](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/shared/FrameModeFunctionals.conf.inc#L1-L10)、[IS10 functionals 的归一化项](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/is09-13/IS10_paraling_core.func.conf.inc#L64-L119)。

另须避免误读：IS10 参考配置中 `cSpecResample targetFs=11000` 仅位于内部 LPC / LSP 支路，并不证明输入 WAV 或所有声学描述符都使用 11 kHz。[对应 L237–246](https://github.com/audeering/opensmile/blob/055dda577c82383f8239663e5ac6c58d25b504db/config/is09-13/IS10_paraling_core.lld.conf.inc#L237-L246)。

## 对目标话语单独重提取的影响

在纯逐话语 functionals 的可信原链中，可以只处理目标 WAV；但目前无法认定 MELD 300 维是逐话语映射。若它来自 bcLSTM，对目标输入的改动会自然改变同 Dialogue 的多个上下文表征；之后仅替换目标行与重新生成全部上下文是不同实验。未来必须先明确生产链是否有上下文依赖，再定义如何满足“其他话语特征不变”的 Pilot 协议，不能凭已存向量的形状认定其独立。

## Clean Parity 门槛与本轮停止点

本轮没有选择替代提取器，也未执行候选配置试跑，特征误差、余弦相似度及预测概率误差均为 **未测量（不是 0，也不是已失败）**。

恢复音频 Clean Parity 至少需：

1. IEMOCAP 的原工具版本、完整配置/include/字段顺序与命令，及任何生产后处理统计量；若最终确认为 IS13，需要解释并提供 6373 到 1582 的原步骤。
2. MELD 的声学配置、selection/projection 记录及原 300 维映射/网络权重；若有对话上下文依赖，还需原上下文构造与输出层定义。
3. 两套生产链各自对应的少量真正 Validation 原片段 ID、字节哈希和原特征行；先做 clean 逐维比对，再用原 CMERC checkpoint 比较完整概率。

本报告保留配置冲突、低维来源不明、原版本缺失和未执行 parity 的事实，不将基础文件下载成功视作原提取链完成。公开上传建议仅包含本报告与来源/哈希摘要；`sources/audio/` 中完整论文 PDF、取文与原页截图为本地审计证据，不属于本次建议公开的统计交付。
