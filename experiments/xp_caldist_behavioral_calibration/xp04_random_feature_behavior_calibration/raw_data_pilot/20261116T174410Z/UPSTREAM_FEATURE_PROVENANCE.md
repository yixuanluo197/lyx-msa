# XP04 上游发布特征来源补充审计

> 公开报告副本：本地绝对路径已替换；文中 `sources/`、`provenance/` 和 `local_only/` 详细记录仅保留本地。公开证据摘要见 `source_summary.json`。

审计日期：2026-11-16 UTC。结论：**官方发布文件的身份已经闭环；原始输入到特征的完整生产配方仍未闭环。** 本轮新鲜下载 CMERC 和 M3NET 官方 README 指向的两个 Dropbox 包，四个文件均与冻结 XP04 数据逐字节身份一致。MM-DFN 官方仓库的两个基础文件也一致。该结果排除了“本地拿错发布特征”的疑问，但不能恢复微调 encoder 权重或音频生产参数。

## 1. 固定上游与直接证据

| 上游 | 固定 commit | 本轮核验角色 |
|---|---|---|
| [CMERC](https://github.com/HITSZ-HLT/CMERC/tree/815c7c060914f5b55db816c0ba3662b11d08552f) | `815c7c060914f5b55db816c0ba3662b11d08552f` | 官方四文件下载入口、消费者代码 |
| [M3NET](https://github.com/feiyuchen7/M3NET/tree/4d0fa88cd4f92c72b053527e877763f54e29ccb2) | `4d0fa88cd4f92c72b053527e877763f54e29ccb2` | 第二个官方四文件入口、主文/补充材料与代码 |
| [COSMIC](https://github.com/declare-lab/conv-emotion/tree/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction) | `6128ca20e9c736605cce7e99d5d95db0356c35f5` | 四层 RoBERTa 参考生产流程、BPE 来源 |
| [MM-DFN](https://github.com/zerohd4869/MM-DFN/tree/da970366069247e05de3b9298f1e1bbc5c77a187) | `da970366069247e05de3b9298f1e1bbc5c77a187` | 基础特征同资产身份、旧文本特征的区别 |

官方特征入口来自 [CMERC README L19–20](https://github.com/HITSZ-HLT/CMERC/blob/815c7c060914f5b55db816c0ba3662b11d08552f/README.md#L19-L20) 与 [M3NET README L15–17](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/README.md#L15-L17)。两个链接本轮均 HTTP 200，返回 `ERC features.zip`，各 **514,504,220 字节**。完整响应体下载后计算 SHA-256，两份 ZIP 都是：

`716e24a9852d2babfcd7bab977b94fe0a59b78aa3f8a9ab568215e1236f8b627`

请求时间、来源/最终 URL、响应长度和 SHA-256 见 `provenance/{cmerc,m3net}_download.json`。下载内容留在 `local_only/`，未覆盖历史包或提取覆盖 XP04。浏览器工具最初无法打开 Dropbox；随后使用独立 HTTP 下载成功，不能把早先浏览器错误当作链接已失效。

## 2. 四文件哈希与数据结构

下表的同一 SHA-256 在 CMERC 官方新包、M3NET 官方新包及 XP04 冻结 `code/data` 三处均一致。MM-DFN 官方 Git 仓库另外确认两个基础包一致。

| 文件 | 字节数 | 共同 SHA-256 |
|---|---:|---|
| `IEMOCAP_features.pkl` | 72,216,498 | `ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3` |
| `iemocap_features_roberta.pkl` | 124,438,886 | `4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06` |
| `MELD_features_raw1.pkl` | 88,193,675 | `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0` |
| `meld_features_roberta.pkl` | 229,654,413 | `58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951` |

本节四文件 schema 分支先核对新包成员哈希，随后只反序列化既有可信本地 pickle 获取结构，没有反序列化新下载的 ZIP 成员。MM-DFN 分支另行读取过官方副本，其记录见独立报告。因字节身份相同，结构比较没有依靠“名称相似”或“维度恰好相同”。完整证据为 `provenance/official_feature_comparison.json`。

| 文件 | 容器及槽位 | Dialogue / 话语总数 | 每话语特征 |
|---|---|---:|---|
| IEMOCAP 基础包 | list，9 项；ID、speaker、label、旧文本、音频、视觉、sentence、train/test IDs | 151 / 7433 | 旧文本 100；音频 1582；视觉 342 |
| IEMOCAP RoBERTa | list，10 项；speaker、label、T1–T4、sentence、train/test/valid IDs | 151 / 7433 | T1–T4 各 1024 |
| MELD 基础包 | list，10 项；前 9 项同基础模式，末项 None；train/test 为 set | 1432 / 13708 | 旧文本 600；音频 300；视觉 342 |
| MELD RoBERTa | list，11 项；speaker、emotion、sentiment、T1–T4、sentence、train/test/valid IDs | 1432 / 13708 | T1–T4 各 1024 |

维度是发布数组的实测宽度，不是由名称推断。四层文本和音频均 float32；IEMOCAP 视觉为 float64，MELD 视觉字典存在 float32/float64 两种 dtype。本轮仅记录结构，未处理或干扰视觉。

## 3. 文件内 split 与 XP04 effective split 的区别

下表为 Dialogue 数 / 话语数，仅读取 ID 列表和各 ID 的行数，不索引或使用 Test 标签。

| 来源 | Train | Validation | Test |
|---|---:|---:|---:|
| IEMOCAP 基础包 | 120 / 5810（含 XP04 Val） | 无独立槽位 | 31 / 1623 |
| IEMOCAP RoBERTa 包 | 108 / 5163 | 12 / 647 | 31 / 1623 |
| **XP04 IEMOCAP effective** | **108 / 5283** | **12 / 527** | **31 / 1623** |
| MELD 基础包 | 1152 / 11098（含 dev） | 无独立槽位 | 280 / 2610 |
| MELD RoBERTa 包 | 1038 / 9989 | 114 / 1109 | 280 / 2610 |
| **XP04 MELD effective** | **1038 / 9989** | **114 / 1109** | **280 / 2610** |

IEMOCAP 不能仅凭同为 108/12 个 Dialogue 就认定 split 一致。XP04 Validation 的 12 个 ID 中，**11 个在 RoBERTa 文件自带 train IDs，1 个在其 valid IDs**；RoBERTa 自带 valid IDs 中另有 11 个属于 XP04 effective train。MELD 的 RoBERTa 三组 ID 与 XP04 完全一致。真正 Validation 的唯一权威仍是各 seed effective config，本轮未用文件内 valid 列表替代。

**风险解释（条件性）：** 若 RoBERTa 文件中的 train IDs 对应生成特征的有监督微调训练集合，则 XP04 下游 Validation 的大部分对话并非 encoder 未见过。这是表示学习阶段可能重叠的风险，不能把下游 CMERC held-out 自动推广为整个原始输入流水线 held-out。当前缺原 encoder checkpoint、生产运行日志及 split manifest，因此不直接宣布发生标签泄漏，也不回改历史实验结果。该风险与本次二进制文件相同同时成立。

## 4. 已闭环与未闭环的证据层级

| 层级 | 当前状态 | 可支持的结论 |
|---|---|---|
| 官方发布文件身份 | 已验证 | XP04 用的是当前官方公开包中的这四个精确文件 |
| 跨项目基础资产关系 | 已验证 | MM-DFN 与 CMERC/M3NET 共享两个精确基础文件；不能据此指定最初生产者 |
| RoBERTa 参考结构与 tokenizer 来源 | 已验证参考实现 | 四层 BOS 结构匹配；参考 BPE/dictionary 与官方 Fairseq 下载一致 |
| CMERC 原实际微调权重/版本 | 未闭环 | 不能从特征 pickle 恢复生产 encoder |
| OpenSMILE 实际配置及 MELD 300 维形成过程 | 未闭环且有材料冲突 | 不能按 1582/300 维指定 IS10/IS13 或压缩方法 |
| 原始输入 clean 重提取及预测 parity | 未执行，先决条件缺失 | 不提供伪造的误差为 0 或 cosine 为 1 |

文本细节见 [RoBERTa 审计](ROBERTA_PIPELINE_AUDIT.md)，音频细节见 [OpenSMILE 审计](OPENSMILE_PIPELINE_AUDIT.md)，MM-DFN 独立证据见 [基础特征关系](MMD_FN_FEATURE_RELATIONSHIP.md)。本轮没有干扰、训练、额外 seed、Test 标签评价或修改历史资产。
