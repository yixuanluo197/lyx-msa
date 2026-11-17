# MELD 音频前端与特征选择器恢复调查

审计日期：2026-11-17 UTC。本轮只调查来源并静态检查已存特征，不重新训练、提取原始音频、执行扰动或访问 Test 标签；不 Commit、Push 或联系作者。既有 `20261117T010744Z` 的全量300维文件对照及24条 Validation 的 GRU parity 均不重复执行。固定来源、检索日志和 SHA-256 见 `provenance/meld_sources.json`、`provenance/meld_recovery_audit.json`。

## 结论与新增证据

**MELD 的6373维声学配置家族现在有作者历史原文支持：IS13-ComParE，而不是仅凭维度猜测。完整 WAV→1611→原GRU→300 的生产链仍未恢复。** 新找到的历史资料同时留下了前处理细节缺失及 PCA/SVM 的官方表述冲突，不能据此选择默认配置或新训练选择器。

| 环节 | 本轮证据级别 | 当前判断 |
|---|---|---|
| OpenSMILE 配置家族 | 作者 arXiv v1–v5 明文；v5同时描述6373和SVM L2 | 有可信的 IS13-ComParE 家族依据 |
| 二进制版本、完整 `.conf`/includes、字段顺序 | 未找到实际生产资产 | 仍缺，不能用当代默认配置代替 |
| 语音强度阈值与 voice Z-standardization | v1–v4描述；无阈值、作用范围或统计量 | 历史方法描述，不是可执行原协议 |
| 6373→1611 / 1422 | 最终论文与README称SVM L2；作者早期Issue称PCA | 原索引、选择器、投影、训练参数均未找到；冲突未消解 |
| 已选择1611 / 1422数组 | 官方包内文件及本轮结构核查 | 是原变换的**输出**，不是变换器 |
| 1611→对话上下文300 | 原HDF5及上轮3个Validation Dialogue / 24行数值验证 | 已部分恢复，不在本轮重复 |
| 原始 MP4/WAV 解码与下混 | 本轮固定24条 MP4 metadata 是48kHz、AAC、5.1六声道 | 原生产的下混、重采样与切段处理仍未知 |

## 1. 论文版本补回 IS13 配置家族证据

本轮下载并核查 `arXiv:1810.02508` 全部六个编号版本；每个PDF的 SHA、页数和关键段落定位在 `sources/meld/paper_version_audit.json`。v1第8页、v5第6页另保存原页截图并视觉确认。

- [v1](https://arxiv.org/pdf/1810.02508v1#page=8)、v2、v3、v4第8页声学提取段：明确使用 **IS13-ComParE**，输出6373项；此前还描述 voice intensity thresholding 及 voice normalization，后者称 Z-standardization。没有给出阈值、统计量、拟合范围、声道或采样率。这里说的是提取前的 voice 处理，不能直接当作选后1611维的外部标准化公式。
- [v5第6页 §4.1](https://arxiv.org/pdf/1810.02508v5#page=6)：同时明确 IS13-ComParE、6373维和 SVM 类估计器的 L2 特征选择。这使配置家族依据独立于“6373维”这一数值。
- [v6第6页 §4.1](https://arxiv.org/pdf/1810.02508v6#page=6)及[ACL正式版](https://aclanthology.org/P19-1050.pdf#page=6)：保留6373和SVM L2描述，未保留配置名。v6原始[TeX源包](https://arxiv.org/src/1810.02508v6)的 `main.tex` 第520行仍将 IS13-ComParE 留在注释中。源包14个成员只有论文、图和样式等资产，未发现 selector、`.conf` 或另附可执行补充材料；ACL页面也未列独立补充下载。

v1第8页 §5.2 还明确使用GRU，并说明其试验中LSTM表现不佳；这与上轮 HDF5 确认的 emotion BiGRU 是相符的历史线索。它不证明具体 checkpoint、选择器或完整预处理相同。

**没有找到原生产 OpenSMILE 版本、完整配置字节或命令。** 因而不能从 IS13 家族名继续推导是任意当前 OpenSMILE release、某个 ComParE_2016 别名或其默认音轨处理。

## 2. 官方 GitHub 全历史、删除的 notebook 与公开答复

在本轮新目录内保留官方仓库只读 mirror。固定 `master=e8cedf27b5d2877e198332c957127e16eb214afe`；检查 master164个提交及全部已获取公开refs共173个可达提交，涵盖分支、PR head和merge refs。对105个不含标签CSV的唯一文本/代码blob检索声学提取、SVM、选择、标准化、WAV/ARFF等入口；历史3个notebook仅检查代码单元，未读取输出或执行 notebook。可复跑脚本为 `scripts/meld_history_audit.py`。

所有历史 `.py` 的相关命中仅是读取已选 `audio_embeddings_feature_selection_*.pkl` 的消费者；历史 notebook 也只读该文件。没有找到 OpenSMILE执行入口、6373字段表、SVM/PCA拟合或transform代码、selector、标准化器。路径列表和命中位置分别保存在 `sources/meld/historical_paths.json`、`historical_keyword_hits.json`。这是对上述可达历史的审计结果，不声称证明所有未公开或已不可达资产不存在。

GitHub API本轮取回57条 Issue/PR 记录、96条评论，均不足一页100条，完成当前公开列表检查；无release、无tag。必须保留以下差异：

1. [Issue #4 作者评论，2018-11-04](https://github.com/declare-lab/MELD/issues/4#issuecomment-435673097)：collaborator `soujanyaporia` 将指定emotion文件解释为 “feature selection using PCA”。该评论没有PCA权重或维度映射，且与后续论文/README的SVM L2说法不同。不能自判为笔误，也不能将PCA替代为已验证的实际流程。
2. [Issue #12 作者评论，2019-08-17 UTC](https://github.com/declare-lab/MELD/issues/12#issuecomment-522201886)：collaborator `devamanyu` 建议用户参考scikit的特征选择和OpenSMILE functionals，并鼓励探索新选择方法；没有提供原训练selector、参数、字段索引或统计量。**通用工具建议不等于原生产代码已公开。**
3. [Issue #11](https://github.com/declare-lab/MELD/issues/11)：2019年已直接询问6373配置、选择步骤及索引；当前快照仅有2023年另一位用户追问，没有作者提供这些资产。提问者提出 ComParE_2016 并非作者认定该配置。
4. [Issue #18](https://github.com/declare-lab/MELD/issues/18)、[#16](https://github.com/declare-lab/MELD/issues/16)同样请求提取代码或自有音频入口，没有补齐本研究所需的原前端。

相关评论的作者、关联身份、创建/更新时间及完整原文在本地 `sources/meld/relevant_issue_evidence.json`；API快照带文件哈希留档。

进一步检查作者公开[项目页](https://affective-meld.github.io/)、Hugging Face目录、`declare-lab/multimodal-deep-learning` 固定提交 `111b401d867c14a1dcd930fd2b2ab58a5eca3f70` 的未截断238项树，以及作者在 [conv-emotion Discussion #49](https://github.com/declare-lab/conv-emotion/discussions/49) 推荐的 MUStARD。项目页/HF只导向已核验原包；作者汇总仓库的MELD子目录仍为相同baseline消费者。MUStARD固定提交 `b34b5134beeec42a36cdac79330637de9cc9894e` 的[音频脚本](https://github.com/soujanyaporia/MUStARD/blob/b34b5134beeec42a36cdac79330637de9cc9894e/extract_audio_features.py)采用librosa的MFCC、mel、delta、spectral centroid及分箱，与本研究6373→1611→300链不同，不能据此恢复MELD选择器。Discussion内后续300维MELD提取追问也没有给出该映射。

检索日志还保留GitHub代码搜索及网页检索范围。宽泛的全站搜索会返回大量无关结果，不能作为“全GitHub没有selector”的证明；本结论限定于已核查官方及作者渠道。

## 3. 两个已选特征文件：能证明什么

本轮只对**既存Validation特征数值**做静态结构/列关系核查，不解码音频、不调用selector或网络，也不读取Test特征值或任何标签。外层split计数只检查字典长度。

| 文件 | SHA-256 | 结构与Validation矩阵 |
|---|---|---|
| `audio_embeddings_feature_selection_emotion.pkl` | `0cda206983988ab61de5b30ec32b8ac652e133f84cda4d3d8105a2db8fea363e` | 三个普通字典；counts9989/1109/2610；dev `(1109,1611)` float64 |
| `audio_embeddings_feature_selection_sentiment.pkl` | `a6c2c5d1ddd9fb0dcf1992702c5cd275afb227ff14b907f7d11ae3333cdf2097` | 三个普通字典；counts9989/1109/2610；dev `(1109,1422)` float64 |

内容只有按Dialogue/Utterance键保存的数组，没有原6373字段名、索引、SVM或PCA对象、`coef_`、`support_`、scaler或拟合统计量。上轮已检查6个HDF5及完整包目录：emotion音频权重输入1611，sentiment输入1422，没有6373输入/选择层；GRU/LSTM权重及优化器状态不能补回外部选择器。

按全部1109个相同dev ID对齐后，对每个任务的整列float64连续字节做SHA-256，发现 **1135个一对一完全相同列**。已排除常数或重复列抬高结果：emotion有1611个唯一列签名，sentiment有1422个；两者常数列0、全零列0、重复签名组0。这里只报告匹配计数，不公开逐列数值或映射。

这一结果说明两种任务的已存输入存在共享数值坐标；**不能从中定位原6373字段、补出其余列、认定实际选择算法或排除变换前后其他处理**。不从相似度或维度逆推selector。

两个dev矩阵均有限，最小0、最大约1.0000000000000002。每行最小值均为0，最大值近1的行数分别603、551；这与多种数值处理相容，**不证明使用MinMaxScaler，也不等于历史论文的voice Z-standardization**。没有找到任何原统计量、缩放范围、作用轴、拟合split或处理先后顺序。脚本与完整汇总见 `scripts/meld_selected_assets_audit.py`、`provenance/meld_selected_assets.json`。

## 4. 当前无法确定的具体生产参数

- 6373输入：原OpenSMILE release/二进制与编译、IS13完整配置及includes哈希、字段顺序、输出精度、缺失值/坏片段处理。
- 选择步骤：SVM具体类、线性或其他核、L2所指的正则化或权重聚合、C与class weight、多类策略、阈值/排名/选择计数、随机种子、训练split、所选索引；如实际为PCA，则缺原中心化、投影矩阵及其拟合资料。只有1611/1422宽度不能识别这些条件。
- 标准化：早期voice阈值及Z标准化范围/参数，以及选前/选后是否另有缩放。不能用当前数据重新拟合参数并称为原生产。
- 音频：字幕/Gentle边界的具体使用、MP4解码为WAV的参数、下混矩阵或保留声道、重采样率、幅度缩放和静音/强度阈值。本轮root仅ffprobe固定dev Dialogue0/1/2共24条现有MP4，均AAC48kHz、六声道5.1；这证明当前原片段的编码属性，不证明原提取过程采用mono16k或当前解码默认值。汇总见 `provenance/validation_audio_probe_summary.json`。

## 5. Clean parity 的门控结论

既有证据仍有效：官方 `audio_emotion.pkl` 与XP04全部13708条真实话语的300维逐维精确一致；原emotion HDF5为BiGRU150×2，上轮完整3个dev对话/24条真实行的1611→300最大绝对误差 `1.1920928955078125e-6`、MAE `4.268819692596379e-8`。这是此前末端验证，不能写成本轮原WAV parity；padding历史差异也保留不变。

本轮原WAV→1611的门槛未通过，因此原WAV→300数值误差和冻结CMERC预测概率误差均为**未测量**，不是0、不是已失败。没有新训练SVM、采用PCA替代、试跑默认ComParE配置、运行额外GRU/CMERC、扩大原音频样本或执行扰动。

后续只需补齐原声学配置/预处理、6373字段与准确selector/投影、全部缩放统计量，即可在已经固定的3个真正Validation对话先验证1611，再按原对话上下文执行已恢复GRU并比较300维及冻结CMERC概率。双向GRU要求完整对话重编码；不能把300维当成逐话语独立OpenSMILE输出。

**停止判断：配置家族来源得到实质补充；原selector、归一化与精确音频前端仍未恢复，本轮不具备原音频干扰执行条件。**
