# IEMOCAP 1582 维音频生产来源恢复

审计日期：2026-11-17 UTC。范围为 XP04 Seed 1475 的原音频特征生产谱系；未使用 Test 标签、重训、扰动或运行候选提取器。来源、固定版本、下载失败与排除项见 `provenance/iemocap_source_recovery.json`；完整公开原文仅作本地审计证据保存在 `sources/iemocap/`。

## 本轮判断

**仍未取得可执行且有来源闭环的 WAV→XP04 1582 维生产链，不能开始 Clean Feature Parity。** MMGCN 的原论文和代码都明确支持 **IS10 配置家族**，这是独立于“1582 维”的文献证据；但它们没有给出实际生产二进制版本、准确配置文件及 include、命令、字段顺序或外部标准化参数。因此，不能把现成 `IS10_paraling.conf` 当作已确认的原配置。

本轮的重要增量是查到 M3NET 作者的明确说明：他们沿用预提取特征，没有自行提取；其 IS13 记述不能独立充当原生产日志。历史资料还包含不同维度、不同归一化分支，必须保留这些差异，不能合并成一条“通用 ERC 音频流程”。

## 文件谱系与生产谱系分开判断

XP04 与 MMGCN 官方 `IEMOCAP_features.pkl` 相同是本轮输入的既有审计结论。本轮没有重写该结论，也没有通过名称相同的其他 pickle 替换 XP04。该一致性证明冻结文件的公开来源关系，不能直接证明从某个 WAV、某版 OpenSMILE 到该数组的数值生产过程。

MMGCN 官方仓库固定为 `85732d984f70c1c84dd47c81aa97f1271397b899`。`train.py` 第 333–334 行把 IEMOCAP 音频宽度命名为 `IS10:1582`；`dataloader.py` 第 9–21 行读取 pickle 中的 `videoAudio`，返回时仅转换成 `FloatTensor`。这里没有 WAV 解码、声学抽取、字段重排或外部 z-score。[固定代码](https://github.com/hujingwen6666/MMGCN/blob/85732d984f70c1c84dd47c81aa97f1271397b899/train.py#L333-L334)、[读取入口](https://github.com/hujingwen6666/MMGCN/blob/85732d984f70c1c84dd47c81aa97f1271397b899/dataloader.py#L9-L21)。

## IS10 / IS13 冲突的证据层级

| 官方资料 | 实际记述 | 能确认什么 | 不能确认什么 |
|---|---|---|---|
| MMGCN ACL 2021 终稿 PDF 第 5 页、印刷页 5670，§4.2 | 音频由 OpenSMILE 的 IS10 配置抽取 | 论文所指配置家族是 IS10 | 版本、准确 `.conf` 文件、compat 选项、标准化 |
| MMGCN arXiv `2107.06779v1` 第 5 页，§4.2 | 与终稿相同地写 IS10 | 不是终稿单独新增的配置表述 | 仍无生产日志或工具版本 |
| M3NET 主文 PDF 第 6 页、印刷页 10766，§4.2 | 沿用预提取单模态特征；音频来自 OpenSMILE | 作者没有在正文提供新的提取配方 | 无法从主文确定 IS10 / IS13 |
| M3NET 官方补充 PDF 第 1 页，§A | IEMOCAP 用 IS13，MELD 用 IS10 | 确实存在文献冲突；已读 PDF 原页并复用上轮视觉核对结果 | 没有给出 IS13 到 1582 的原映射 |
| M3NET 作者 Issue #4 回复，2023-09-18 | “we used pre-extracted features and didn't conduct feature extraction process ourselves” | 其仓库不是原声学特征生产者 | 不能自行认定补充材料为笔误或实际配置 |

来源：[MMGCN 终稿](https://aclanthology.org/2021.acl-long.440.pdf#page=5)、[arXiv v1](https://arxiv.org/pdf/2107.06779v1#page=5)、[M3NET 主文](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Multivariate_Multi-Frequency_and_Multimodal_Rethinking_Graph_Neural_Networks_for_Emotion_CVPR_2023_paper.pdf#page=6)、[补充材料](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_Multivariate_Multi-Frequency_and_CVPR_2023_supplemental.pdf#page=1)、[作者原回复](https://github.com/feiyuchen7/M3NET/issues/4#issuecomment-1722689346)。

M3NET 还有一条应保留的历史不一致：作者在 Issue #2 的 2023-06-23 回复中称 IEMOCAP 的 A/V/T 宽度为 100/512/1024，同线程用户报告 1582/342 的文件，而本轮固定现版消费者代码使用 `IS10:1582` 和 `denseface:342`。这不能用于推断哪个隐藏提取器“应当”正确；它表明必须以具体文件和数值比对确认版本，不能仅靠论文名或作者讨论中的尺寸匹配。[作者维度回复](https://github.com/feiyuchen7/M3NET/issues/2#issuecomment-1603643903)、[现版代码](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/train.py#L340-L342)。

## 仓库历史和公开讨论实际检查范围

- MMGCN：检查默认分支可达的全部 **1 个提交**、完整递归目录、branch/tag 列表、所有状态的 **11 个 Issue 记录**及所有非空评论页。仓库包含预计算 pickle 和模型/训练/加载代码，没有生产配置或原 WAV 提取脚本。未把其他同名推荐系统 MMGCN 仓库纳入来源。
- M3NET：固定现版 `4d0fa88cd4f92c72b053527e877763f54e29ccb2`，检查默认分支可达的全部 **15 个提交**及对应 **15 份未截断 tree**。历史路径并集仅 8 个代码/README 文件，未发现 `.conf`、音频预处理脚本或标准化统计；检查所有状态的 **12 个 Issue 记录**与非空评论页。
- MMGCN Issue #2（2021–2023）、#11（2024–2025）持续有用户请求原特征脚本，所捕获回复未提供作者的提取配置。Issue #3 关于 1582/100 的纠正来自非作者，不能作为生产说明。搜索中未向作者发邮件或评论。

这是对明确仓库、分支和公开记录的有界搜索结果，不等于断言互联网上不存在相关文件。[MMGCN #2](https://github.com/hujingwen6666/MMGCN/issues/2)、[#3](https://github.com/hujingwen6666/MMGCN/issues/3)、[#11](https://github.com/hujingwen6666/MMGCN/issues/11)。

## 追溯 DialogueGCN、DialogueRNN、CMN、ICON 与 COSMIC

本轮固定 `declare-lab/conv-emotion` 为 `6128ca20e9c736605cce7e99d5d95db0356c35f5`，检查目录和相关加载/预处理入口。找到的是引用关系和不同特征分支，尚无证据把以下分支连接到 XP04 的实际 1582 维数组。

| 上游资料 | 发现的声学/归一化线索 | 与本任务的关系 |
|---|---|---|
| CMN，NAACL 2018，PDF 第 4 页、印刷页 2125，§4.1.2 | IS13 ComParE 6373 维，Z-standardization，再经全连接层得到 100 维 | 明确的另一生产方法；未公开此处拟合统计量，不能套用到 1582 维 |
| DialogueRNN，arXiv `1811.00405v4` 第 2 页，§3.2 | 声学/视觉沿用 CMN 提取方法 | 可解释其 100 维分支的文献线索，不证明 MMGCN 1582 来源 |
| ICON，EMNLP 2018，PDF 第 3 页、印刷页 2596，§4.1.2 | 先转 16-bit PCM WAV；IS13 ComParE 6373，Min-Max scaling，再做 L2 特征选择 | 与 CMN 的归一化/降维不同；未找到连接 XP04 1582 的索引、权重或统计量 |
| DialogueGCN 论文 §3.2 及仓库 `preprocess.py` | 论文聚焦文本表示；所谓 preprocess 是读取已有 pickle 后划分样本 | 不是 WAV 生产脚本；类名/数据结构继承不等于声学生产继承 |
| COSMIC `feature-extraction/` | RoBERTa/COMET 文本及知识抽取代码 | 没有提供此处所需的原 IS10 声学脚本 |

来源：[CMN](https://aclanthology.org/N18-1193.pdf#page=4)、[DialogueRNN v4](https://arxiv.org/pdf/1811.00405v4#page=2)、[ICON](https://aclanthology.org/D18-1280.pdf#page=3)、[DialogueGCN 论文](https://aclanthology.org/D19-1015.pdf#page=3)、[读取已有 pickle 的 preprocess](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/DialogueGCN-mianzhang/preprocess.py#L10-L30)、[COSMIC 抽取说明](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/README.md)。

另外，`conv-emotion` 官方维护者在 Discussion #49 推荐参考 MUStARD。实际检查其固定版 `b34b5134beeec42a36cdac79330637de9cc9894e` 后发现，音频脚本使用 librosa 的 MFCC、Mel、delta 和谱质心并做时间分箱，堆叠的描述通道为 283；这是另一音频体系，不能替换 XP04 IS10。[维护者讨论](https://github.com/declare-lab/conv-emotion/discussions/49)、[MUStARD 原脚本](https://github.com/soujanyaporia/MUStARD/blob/b34b5134beeec42a36cdac79330637de9cc9894e/extract_audio_features.py#L12-L44)。

因此，目前没有可证明适用于 XP04 1582 的外部 z-score、逐 speaker 标准化、Min-Max 或 L2 变换参数。MMGCN 的模型正则项、网络归一化层，以及其他分支论文中的语音归一化，都不能替代原生产记录。

## 音频切段、采样率与声道

IEMOCAP 原始数据库论文 PDF 第 10 页描述录制阶段采用对准两位参与者的麦克风、48 kHz 采样；第 11 页 §5.1 描述人工按 speaker turn 切段。它提供数据采集/标注背景，没有规定 MMGCN 原提取输入和处理命令。[原数据库论文](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf#page=10)。

本轮项目另做了固定少量真正 Validation 原候选音频头检查：6 个候选均为 **16 kHz、单声道、PCM s16le**，见 `provenance/validation_audio_probe_summary.json`。这是当前本地候选 WAV 的实测属性，不能把它回写成原论文录制参数，也不能据此认定生产者使用这些字节、没有进一步重采样/重切段/静音处理。未重新裁切、重采样或写入 WAV。

## 配置版本与 Clean Parity 门槛

OpenSMILE 官方 `IS10_paraling.conf`、`IS10_paraling_compat.conf`、`emobase2010.conf` 及历史版本差异由同轮配置专项证据检查。官方存在可输出 1582 维的参考配置，不能反向唯一定位 XP04。特别是兼容模式和版本可能改变数值，即使形状完全一致也不足以通过来源门槛。

| 必要环节 | 本轮状态 |
|---|---|
| XP04 冻结目标文件、1582 维接口 | 已有历史文件一致性证据；未修改 |
| MMGCN 论文所指配置家族 | IS10，文献与消费者命名相互支持 |
| 原 OpenSMILE 版本/编译二进制 | 未找到 |
| 完整配置、include、字段顺序与原命令 | 未找到 |
| WAV 选择、切段、采样率和声道的原生产记录 | 未找到；当前候选的头信息不等于生产记录 |
| 外部标准化类型、拟合范围和统计量 | 未找到 |
| WAV→1582 clean 数值验证 | 未执行；误差和余弦均未测量，不是 0 或已失败 |
| 冻结 CMERC 概率 parity | 前置特征 parity 未成立，未执行 |

本轮没有用默认配置试跑、拟合归一化、反推变换或通过多种候选挑选最接近者。下一步需要取得上述生产记录及至少少量可追溯的原始声学输出（带字段名），先核验固定 Validation 音频，再比较逐维误差与冻结 XP04 概率。现阶段应停留在来源恢复和门槛报告，不能据此开始原音频扰动。
