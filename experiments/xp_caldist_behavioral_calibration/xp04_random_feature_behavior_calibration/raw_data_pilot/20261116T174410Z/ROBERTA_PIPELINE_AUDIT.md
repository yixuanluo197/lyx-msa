# RoBERTa 文本特征生产链路补充审计

> 公开报告副本：本地绝对路径已替换；文中 `sources/`、`provenance/` 和 `local_only/` 详细记录仅保留本地。公开证据摘要见 `source_summary.json`。

审计日期：2026-11-16 UTC。结论：**找到与 XP04 四层结构吻合的公开参考生产代码，并确认本地参考代码未偏离该远端版本；尚未找到可证明实际生成 CMERC/XP04 发布文本特征的两套 fine-tuned RoBERTa encoder checkpoint，不能执行 clean feature parity。** 本轮没有训练、特征提取、CMERC 推理或读取 Test 标签。

## 1. 固定版本与本地一致性

上游为 `declare-lab/conv-emotion` 的 `master`，GitHub API 查询固定到 [6128ca20e9c736605cce7e99d5d95db0356c35f5](https://github.com/declare-lab/conv-emotion/tree/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction)。远端六个指定脚本、两份 preprocess 脚本、`multiprocessing_bpe_encoder.py`、`train.py`、feature-extraction README 及三个 tokenizer/dictionary 文件均已取回；它们与本地既有参考目录的对应文件逐字节相同。

本地参考目录：`$REFERENCE_FEATURE_EXTRACTION/`。各文件 SHA-256、来源 URL、字节数、本地 SHA-256 和一致性见 `provenance/roberta_sources.json`。这证明“本地参考代码等于此次固定远端代码”，不证明其就是 CMERC 当年实际使用的运行版本。

## 2. 六个指定脚本证明的流程

| 阶段 | IEMOCAP | MELD | 证据与边界 |
|---|---|---|---|
| 输入 pickle | `iemocap/iemocap.pkl`，6 项 | `meld/meld.pkl`，7 项，另含 sentiment labels | 初始化脚本第 3 行；本轮不执行其 train/valid/test 导出 |
| 文本清洗 | 原字符串逐话语写行 | `encode('ascii', errors='ignore').decode('utf-8')` | IEMOCAP init 13–15；MELD init 13–16；MELD extraction 22 再执行同一 ASCII 清洗 |
| 微调架构 | Fairseq `roberta_large`，6 类 | Fairseq `roberta_large`，7 类 | 两个 train 脚本 7–21 行；加载 `roberta.large/model.pt` 是微调初始化 |
| 微调主要差异 | LR `1e-6`，dropout `0.2`，4840 updates，290 warmup | LR `1e-5`，dropout `0.1`，9364 updates，562 warmup | 均最多 30 epochs，batch 8 × update-freq 4，best checkpoint metric accuracy；脚本未固定 seed |
| 实际抽特征权重 | `checkpoints/iemocap/checkpoint_best.pt` | `checkpoints/meld/checkpoint_best.pt` | 两个 extraction 8–12 行；不能用初始化预训练权重代替 |
| token 与 batch | 每个 dialogue 的话语分别 `roberta.encode(s)`，pad_idx=1 后 batch | 同左，先 ASCII 清洗 | IEMOCAP extraction 19–23；MELD 19–24；上下文不是拼成一个文本输入 |
| 特征 | 最后四个 hidden states，各取 token index 0 的 BOS 表示 | 同左 | IEMOCAP extraction 24–27；MELD 25–28；每层 1024-D 的 RoBERTa-large 表示，与 XP04 四路输入结构相符 |
| 序列化 | 10 项，四层位于索引 2–5 | 11 项，四层位于索引 3–6 | extraction 最后一条 pickle.dump；每层 dictionary 由 dialogue ID 索引 |

精确源码链接：

- [IEMOCAP init，L3–L22](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_init_iemocap.py#L3-L22)、[MELD init，L3–L23](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_init_meld.py#L3-L23)。
- [IEMOCAP train，L1–L32](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_train_iemocap.sh#L1-L32)、[MELD train，L1–L32](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_train_meld.sh#L1-L32)。
- [IEMOCAP extraction，L8–L29](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_feature_extract_iemocap.py#L8-L29)、[MELD extraction，L8–L30](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_feature_extract_meld.py#L8-L30)。

脚本中的 `HEAD_NAME` 变量没有传给训练命令；不能把这个变量当作已经验证的 checkpoint classification-head key。提取代码设 `eval()`，没有显式 GPU 调用；本轮不将“可读的源码”描述为已成功执行的提取器。

## 3. Tokenizer、BPE、dictionary 与长度

IEMOCAP [preprocess L1–L24](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_preprocess_iemocap.sh#L1-L24) 明确从 `dl.fbaipublicfiles.com/fairseq/gpt2_bpe/` 下载 `encoder.json`、`vocab.bpe` 和 `dict.txt`。MELD [preprocess L1–L17](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/roberta_preprocess_meld.sh#L1-L17) 复用这些文件。BPE helper 通过 Fairseq `get_encoder`，并在离线 preprocess 时 `.strip()` 每一行；保留空行由 `--keep-empty` 控制（[helper L15、L94–L117](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/multiprocessing_bpe_encoder.py#L94-L117)）。

本轮额外下载三个小文件作字节核对：固定 COSMIC 仓库文件、本地参考文件、当前官方 Fairseq 下载端点文件三者完全相同：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| encoder.json | 1042301 | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| vocab.bpe | 456318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| dict.txt | 603290 | `2c31c5b236661067fe8626775eb995ef5fe808ac87d2ad4a6ddbef68550f5cc6` |

这使“参考 tokenizer 来源”比前轮更明确，但 extraction 的 `RobertaModel.from_pretrained` 没有显式提供 `bpe`、`bpe_codes` 或 `bpe_vocab` 参数；它依赖 checkpoint 所存参数及对应 Fairseq hub 实现。实际生产 checkpoint 不在场，就不能证明历史运行使用的资产和行为完全相同。

两个 train 脚本均给 `--max-positions 512`、`--init-token 0 --separator-token 2`、`--shorten-method truncate`；extraction 脚本自身没有截断/分块逻辑。训练时截断不能自动推出抽特征时超长话语如何处理。离线 BPE `.strip()` 与 extraction 直接传原字符串也不是同一条可见清洗路径，是否由 `roberta.encode` 内部消除差异需核对原 Fairseq 版本。后续 parity 必须记录实际 tokens、长度、特殊 token、空串及 ASCII 清洗输出。

## 4. Fairseq 版本与归一化

`train.py` 只把调用转发给 `fairseq_cli.train.cli_main`（L7–L11）；所审 COSMIC 目录树未发现锁定 Fairseq 版本的 requirements、环境文件或 vendored Fairseq。不能把整个 conv-emotion 仓库其他模型小节的 Python/PyTorch 版本当作 COSMIC extractor 的版本记录，也不能按发布时间推断 Fairseq tag。

六个指定脚本和 BPE/preprocess 脚本中未出现抽取后 L2、z-score、StandardScaler 或 min-max 归一化。可见输出是 hidden-state BOS 数组直接 `.detach().numpy()` 后写 pickle。该结论只覆盖这些参考代码；不证明 CMERC 发布前没有另行加工。

XP04 冻结 `code/dataloader.py:110–111,134–137,161–163,190–193` 读取四层数组并转 FloatTensor；`code/model.py:722–740` 才按已存 `norm` 策略处理四路输入。模型内归一化不能被当作特征文件生产归一化，也不能在 clean parity 前额外 normalize 原始输出。

## 5. 可下载资产与 checkpoint 身份

[feature-extraction README L9–L20](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/COSMIC/feature-extraction/README.md#L9-L20) 的两个主要链接分别是：

1. `https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz`：README 明确称为 **pretrained weights**；本轮 HEAD 返回 200、Content-Length 655283069。只做 HEAD，不下载 655 MB 初始化包，因为它不能补齐 dataset fine-tuned encoder。
2. Google Drive ID `1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr`：README 明确称为 **features used in experiments**；根 README 161 行也称 RoBERTa and COMET features。HEAD 可达不能证明内部内容，更不能把这个“已提特征”链接写成 fine-tuned encoder 权重下载。本轮没有下载整个特征包。

固定仓库完整树中没有 `checkpoints/{iemocap,meld}/checkpoint_best.pt`、`roberta.large/model.pt` 或 `{iemocap,meld}-bin` 生产资产；本地参考目录的这些位置也不存在。GitHub releases 查询结果为空。GitHub repository metadata 显示 `has_issues=false`；all-issues API 返回 14 条，全部为 PR，无普通 issue；RoBERTa 搜索只返回其他目录依赖更新的 2 个 PR，未得到生产权重线索。以上是当前公开表面与已知本地参考位置的搜索范围，不声称世界上不存在该权重，也没有联系作者或发 issue。

## 6. 对 CMERC 来源关系与后续 parity 的判断

目前证据支持“COSMIC 提供结构一致的参考生产路径”，不足以支持“CMERC 四层特征一定由这一 commit、这一微调 run 生产”。即使官方 CMERC 发布 pickle 与 XP04 文件字节相同，也只能闭环发布文件的身份，不能倒推出 encoder 权重。

继续验证至少需要每数据集的实际 fine-tuned RoBERTa encoder checkpoint、对应 data dictionary/tokenizer 资产和 Fairseq 版本，以及它们与 CMERC 发布特征的来源说明/哈希。COSMIC ERC 任务 checkpoint 和 CMERC 分类 checkpoint 都是下游网络权重，不等于上述文本 encoder；本轮没有用其替代。

本轮 clean 重提取：**NOT RUN / BLOCKED**。特征逐维最大绝对误差、平均绝对误差、cosine 和 CMERC 概率差异均为 **NA（没有重新提取值，不能记作 0 或通过）**。禁止通过新微调、预训练 Hugging Face 模型或相同维度的替代 encoder 绕过缺口。

## 7. 失败与留档边界

- 请求 `COSMIC/README.md` 返回 404；COSMIC 使用目录内 feature-extraction README 和根 README 文档。
- 试探 `src/gpt2_bpe.py`、`src/README.md` 返回 404；完整树确认 `src` 为 COMET 相关代码，真实 BPE import 指向安装的 Fairseq。
- 公开接口查询只读；三个 tokenizer 文件之外没有模型/语料大下载。完整文件/请求状态、404、HTTP probe 和哈希留档在 `sources/roberta/` 与 `provenance/roberta_sources.json`；HTTP 响应只保留内容类型/长度/ETag/修改时间等必要字段，未保留 cookie。
- 本轮未导出原始话语文本、未运行默认遍历 train+valid+test 的 init/extraction 脚本，未改变历史实验或原始文件。
