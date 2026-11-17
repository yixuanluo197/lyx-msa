# MM-DFN 与 XP04 / CMERC 基础特征关系补充审计

> 公开报告副本：本地绝对路径已替换；文中 `sources/`、`provenance/` 和 `local_only/` 详细记录仅保留本地。公开证据摘要见 `source_summary.json`。

审计时间：2026-11-16 UTC。固定上游：`zerohd4869/MM-DFN@da970366069247e05de3b9298f1e1bbc5c77a187`，`git ls-remote ... HEAD` 与本地旧副本 HEAD 一致。新目录 `sources/mmdfn/` 保存该固定版本。旧工作区存在代码修改，因此仅使用其已提交 blob 和预计算数据作比对，没有采用旧工作区修改后的代码。

## 已核实的来源关系

**MM-DFN 官方仓库分发的两个基础特征文件，与 XP04 inputs/data 及本地 CMERC 基础特征逐字节一致。** 这是共享发布资产的直接证据；它不能单独证明最初由 MM-DFN 作者生产，也不能给出生产时的工具参数。

| 文件 | 官方大小（字节） | 官方 / 旧 MM-DFN / XP04 / 本地 CMERC 的共同 SHA256 |
|---|---:|---|
| IEMOCAP_features.pkl | 72,216,498 | `ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3` |
| MELD_features_raw1.pkl | 88,193,675 | `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0` |

官方固定版本文件：[IEMOCAP 基础特征](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/data/iemocap/IEMOCAP_features.pkl)、[MELD 基础特征](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/data/meld/MELD_features_raw1.pkl)。本地比对目标为 `experiments/xp_multiseed/20261014T1627Z/inputs/data/`，以及 `$PUBLISHED_CMERC_FEATURES/`。逐项机器证据在 `provenance/mmdfn_audit.json`。

## 容器、维度与数据划分

| 数据集 | pickle 容器 | Dialogue / Utterance 总数 | 文件内 train ID 数 / 话语数 | 文件内 test ID 数 / 话语数 | 文本 / 音频 / 视觉维度 |
|---|---|---|---|---|---|
| IEMOCAP | list，9 槽 | 151 / 7433 | 120 / 5810 | 31 / 1623 | 100 / 1582 / 342 |
| MELD | list，10 槽，末槽 None | 1432 / 13708 | 1152 / 11098 | 280 / 2610 | 600 / 300 / 342 |

样本数仅通过话语 ID 与 split ID 统计。没有读取任何 Test 标签值，没有输出原始文本。特征维度检查使用 train 列表中的代表 Dialogue，IEMOCAP 对应形状为 `(30,100)/(30,1582)/(30,342)`；MELD 为 `(14,600)/(14,300)/(14,342)`。完整文件字节一致意味着全部序列化数据一致，不需要为身份核验遍历标签。

MM-DFN loader 仅解包并消费已经生成的特征：[IEMOCAP loader L11–26](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/code/dataloader.py#L11-L26)、[MELD loader L39–55](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/code/dataloader.py#L39-L55)。文件内没有单独 Validation ID 槽；不得直接把其 train/test 包装等同于 XP04 真正 Validation 协议。MELD 1432 个键与论文正文宣称的 1433 个 conversations 存在一项差异，已保留，不能通过修改历史特征“修正”。

## 文本：基础 TextCNN 与四层 RoBERTa 必须区分

README 明确按“文本、音频、视觉”顺序声明 TextCNN + GloVe、OpenSmile、DenseNet，且链接直接指向仓库 data 目录：[README L23–29](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/README.md#L23-L29)。消费者维度映射为 IEMOCAP 文本 100、MELD 文本 600：[run_train_erc.py L359–362](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/code/run_train_erc.py#L359-L362)。

因此，共享 `IEMOCAP_features.pkl` / `MELD_features_raw1.pkl` **不能补齐 CMERC 独立的四层 RoBERTa 特征生产链**。它证明基础音频等资产相同；基础包中 TextCNN 特征不能用于替换 XP04 的四层 RoBERTa 输入。该仓库不分发 `iemocap_features_roberta.pkl`、`meld_features_roberta.pkl`，也未提供对应 RoBERTa 微调权重、Tokenizer/BPE 或四层抽取脚本。

## 音频：可确认的消费者声明与尚缺生产参数

代码将 IEMOCAP 音频维度映射到名为 `IS10` 的条目，值为 1582；MELD 使用另一个名为 `MELD_audio` 的 300 维条目：[run_train_erc.py L359–360](https://github.com/zerohd4869/MM-DFN/blob/da970366069247e05de3b9298f1e1bbc5c77a187/code/run_train_erc.py#L359-L360)。

这能确认 MM-DFN 消费者对 IEMOCAP 资产采用了 IS10 名称，且其实际基础音频资产与 XP04 相同。**它不是原始 OpenSMILE 命令或配置文件，不能仅据此认定原生产时的完整 IS10 配置，更不能把 MELD 300 维反推为 IS10、IS13、PCA 或某个帧均值。** 当前固定仓库仅有模型、loader、运行脚本和两个 pickle，没有 OpenSMILE `.conf`、音频切段程序、重采样脚本、帧级聚合实现、归一化参数或原始生产日志。

尚缺：原始 OpenSMILE 版本和配置、切段边界/音轨来源/采样率及声道规则、300 维 MELD 的形成过程、归一化 fit 范围与参数、IEMOCAP 1582 维功能量排序。这些缺口意味着不能从该仓库直接重提取 Clean 音频后声称与 XP04 等价。

[MM-DFN 论文 v1](https://arxiv.org/html/2203.02385v1) §3.1 给出划分与总量，§3.2 提及沿用 MMGCN 的特征处理，但未补齐上述生产参数。该段对 acoustic/visual/textual 与提取器顺序的表述本身不清晰，本审计采用 README 的显式“textual/acoustic/visual”映射，不据论文排列猜测模态配置。

## Clean parity 的边界与结论

已完成的是**分发资产身份验证**：两个基础 pickle 最大可能的字节差异为零（SHA256 一致）；这不是“原始输入重新提取后的逐维误差为零”。本分支未找到完整生产链，因此不启动原始输入重提取，不加载替代提取器，不运行 CMERC 前向预测，也不填造 MAE、余弦相似度或概率差异。

本分支结论：基础特征发布链来源关系已从文件名相似升级为逐字节身份确认；原始文本 RoBERTa 权重链与音频生产配方仍需其他上游证据补全。MM-DFN 单独不足以解除 Raw Data Pilot 阻断。

## 异常和本地保留范围

- 历史 MM-DFN 本地副本中三个代码文件有未提交修改；未采纳这些修改。
- 旧 pickle 在 NumPy 2.4 下产生 dtype 反序列化弃用警告，两文件均成功读取，警告不影响 SHA256 身份验证。
- 新官方 clone 在取消重复大文件下载前已完成，保留为独立官方副本；没有改动原资产。
- 初步误查的 arXiv `2203.02369` 与该论文无关，已排除；所用论文始终是 README 指向的 `2203.02385v1`。
- `sources/mmdfn/data/*.pkl` 含原始文本等数据，仅本地保存，禁止把该源码镜像目录连同数据整体公开上传。可公开本报告、固定链接、哈希与脱敏 JSON。
- 审计分支未训练、未运行 GPU、未使用 Test 标签；审计阶段未 commit / push，公开报告随后按研究者要求发布。
