# XP04 RoBERTa 文本提取来源恢复

**结果：未找到可确认实际生成 CMERC/XP04 四层文本特征的两套 encoder 权重，原始文本重提取阻断。** 这不是数值Parity失败，也不是模型不具备文本敏感性的结论。本轮没有使用替代权重、训练、提取或CMERC前向。

## 1. 论文与参考实现的证据层级

CMERC正式版《Multimodal Emotion Recognition Calibration in Conversations》，DOI [10.1145/3664647.3681515](https://doi.org/10.1145/3664647.3681515)，本地PDF SHA-256 `8b4133721eb6020eadc470ad15b6337ee2c06772404dd34f29dc4364351905e0`。PDF第3页§3.2引用Ghosal等[9]，描述最后四层、layer normalization与均值操作；第9页参考[9]是COSMIC。论文**没有明确说明直接以未微调的公开初始化权重提取，也没有给出encoder checkpoint身份**。不能把未写微调解释成没有微调。

COSMIC [正式论文PDF第3页§3.2](https://aclanthology.org/2020.findings-emnlp.224.pdf) 明确先把RoBERTa Large微调用于话语情绪分类，再取首特殊Token的最后四层。其公开脚本与此吻合。但引用关系、实现结构和特征包关联不能证明CMERC使用了哪一个微调run、Fairseq版本或当时的完整预处理。

固定代码版本：CMERC `815c7c060914f5b55db816c0ba3662b11d08552f`；COSMIC/conv-emotion `6128ca20e9c736605cce7e99d5d95db0356c35f5`。论文、源码定位、请求失败与范围见 [paper_source_evidence.json](provenance/paper_source_evidence.json)。

## 2. 权重与下载资源分类

| 对象 | 实际类型/核查结果 | 可否作为XP04原文本encoder |
|---|---|---|
| CMERC Dropbox已缓存514,504,220字节包 | 目录项加四个预提特征pkl；当前包SHA与历史下载一致 | 否，是特征文件 |
| CMERC官方Drive的IEMOCAP/MELD_checkpoint.pkl | README与模型实现指向下游CMERC/M3Net；本轮匿名文件夹列表SSL超时 | 不能当作RoBERTa encoder |
| Fairseq roberta.large.tar.gz | 官方README的微调初始化包；本轮HEAD200、655,283,069字节，无大下载 | 不能替代未知任务微调权重 |
| 本地HF roberta-large snapshot | 已有完整预训练MaskedLM，24层、1024维 | 无实际生产身份，未采用 |
| 本地IEMOCAP/MELD微调权重 | 自有2026年HF实验；包含speaker token、pooling、seed等与COSMIC不同的配置 | 无CMERC来源证明，未采用 |
| COSMIC官方Drive erc-training.zip | 实验特征资源；本轮通过HTTP Range完整列中央目录 | 未发现权重命名条目，不能据此恢复encoder |

本地重点权重哈希：

- HF预训练safetensors：`047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`。
- 自有IEMOCAP speaker-token encoder：`874ee62a61f6d4cb284957e4abfcd502a8eda26177332d4f8734a5a64655cf99`。
- 自有IEMOCAP seed2025 encoder：`6e50cf9565e855b9ea533faee4c75afaba513a3103436c327ab33044c28ba50a`。
- 自有MELD seed42 encoder：`a12a4505d09e75db1abe0e0e0def5e85fdf711f016b5956f62eee0ed6d480285`。

扫描范围、文件数、对应生成代码/配置、哈希与排除理由见 [local_weight_asset_audit.json](provenance/local_weight_asset_audit.json)。名称盘点未找到checkpoint_best.pt；9个现有Conda环境未发现Fairseq包目录。搜索不是全世界资产不存在的证明，也不能由当前安装状态倒推历史Fairseq版本。

## 3. COSMIC特征包的新增目录实查

官方ZIP共5,488,532,347字节；只用Range读取末尾131,072字节，由ZIP64记录定位4,475字节中央目录，列出36项。普通内容按命名是四个数据集的RoBERTa/COMET特征pkl及macOS元数据，没有.pt/.pth/.bin/.safetensors/.hdf5或checkpoint命名条目。未下载整个5.49GB包、未解压pkl、未读取任何原始语料或Test标签，也未加载模型。目录只证明可见条目，不声称检查了任意pickle内部的对象语义。

IEMOCAP/MELD的RoBERTa条目与CMERC已缓存官方ZIP的大小、CRC32都吻合：124,438,886字节/`c5e7cb82`；229,654,413字节/`6e6c53de`。这是较强的包元数据一致性证据，**不是本轮测得的COSMIC payload SHA-256相等，更不是实际encoder权重身份**。

[完整目录](provenance/cosmic_archive_directory.json)、[两包元数据比较](provenance/cosmic_cmerc_zip_metadata_comparison.json)、[限量网络探测](provenance/public_resource_probe.json)。

## 4. 四层表示、Tokenizer与处理规则

已核查的COSMIC参考代码加载 `checkpoints/iemocap/checkpoint_best.pt` 或 `checkpoints/meld/checkpoint_best.pt`，然后eval。预训练模型是训练初始化，不是这两条提取路径的实际权重。

| 项目 | 参考代码可确认内容 | 尚不能认定的实际CMERC信息 |
|---|---|---|
| 架构 | Fairseq roberta_large；各1024维 | 原权重SHA、run、环境 |
| 四层顺序 | 保存T1=hidden[-1]，T2=[-2]，T3=[-3]，T4=[-4] | 不能重排后再找Parity |
| Pooling | 每个话语独立encode，在每层取token index0的BOS表示 | 不用mean pooling替代 |
| batch/context | 同一Dialogue的话语作为batch，pad_idx=1；不拼接对话文本 | 真正encoder上下文配置仍待生产记录 |
| Tokenizer/BPE | GPT-2 byte BPE和Fairseq dictionary；训练init-token0、separator-token2 | 原checkpoint BPE参数、Fairseq版本及实际tokens |
| 文本 | IEMOCAP直接传存储字符串；MELD参考代码ASCII忽略非ASCII字符 | 不能未经确认就清洗或改写XP04原文 |
| 长度 | 训练max_positions512、shorten_method truncate | extraction没有显式截断/分块；不能擅自补造 |
| 保存归一化 | 四路BOS数组直接写pkl，没有可见额外zscore/L2/仿射 | 原生产可能的外部处理仍缺记录 |

三个参考tokenizer资产SHA：encoder.json `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`；vocab.bpe `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`；dict.txt `2c31c5b236661067fe8626775eb995ef5fe808ac87d2ad4a6ddbef68550f5cc6`。HF merges文件相同不等于权重或完整tokenization行为相同。

论文的LN/均值描述不能覆盖原checkpoint配置。本轮两份XP04 effective config实际都是 `norm=BN`，冻结model.py逐层归一化后求四层均值。该模型内处理与pkl生产是两个阶段；Clean Parity应逐层比较存储特征，不能先套LN/平均使差异消失，也不修改checkpoint的BN策略。

## 5. 处理失败与复用范围

新源码下载中6个raw请求出现代理502；其余成功获取字节已留档。历史来源清单的source_url/url字段差异、ZIP目录项计数以及两个README同名写出冲突也都留存失败记录/代码快照。修正后以固定commit、既有SHA核验的缓存源码补齐参考，不把缓存复用写成全部新网络请求成功。没有执行会遍历train/valid/test的上游init/extraction脚本。

ACM403、OpenReview浏览器验证、Drive SSL超时与Dropbox502保留。正式CMERC PDF复用本地归档，COSMIC论文从官方ACL下载；此类检索障碍没有用非正式摘要取代论文结论。

## 6. 阻断决定

本轮论文证据没有满足“明确允许直接用公开未微调权重”的条件，也没有找回可确认的实际任务encoder。因此不从HF缓存、其他本地微调模型或新微调模型启动Validation提取。不拟合任何转换器。需要补齐：两数据集实际生产encoder checkpoint及来源/哈希、训练与原始split记录、字典与BPE关联、Fairseq版本和抽特征文本/长度处理记录。

当前结果为BLOCKED/NOT_RUN，不能写成Clean Parity数值失败或通过。权重缺失不影响既有XP04模型使用原预提特征的历史结果，但尚不支持生成兼容的新文本特征。
