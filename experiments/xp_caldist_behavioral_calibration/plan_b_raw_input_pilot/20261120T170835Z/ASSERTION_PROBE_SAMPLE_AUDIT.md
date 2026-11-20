# Assertion、Probe、Sample 状态核查

结论：三者在早期可行性清单中被点名，但在本轮检索的工作区资料和本地 Git 历史中，**没有找到可执行的原始实验定义、专用代码、运行清单或对应结果**。因此三者均保留为 `UNRESOLVED / 待确认`，本轮不自行定义或执行，也不将现有文本干扰改名为三者。

## 分项状态

| 名称 | 原始资料能够确认的内容 | 明确操作协议 | 已执行证据 | 结果文件 | 本轮处理 |
|---|---|---|---|---|---|
| Assertion | 早期清单将其列为原始文本干扰；需要固定编码器后重新编码 | 未找到 | 未找到可归属的执行记录 | 未找到 | 待确认，不补造定义 |
| Probe | 同上；仅凭名称不能确定问题、提示或构造方式 | 未找到 | 未找到研究级执行记录 | 未找到；工程 probe 文件另行排除 | 待确认，不补造定义 |
| Sample | 历史资料的确切名称为 **Sample Corruption**，不能进一步假定 corruption 的内容和强度 | 未找到 | 未找到可归属的执行记录 | 未找到 | 待确认，不与 feature dropout、改写混同 |

“未找到执行证据”是有界审计结论；不能据此证明研究者从未在其他位置设计或执行过这些实验。能够确认的是：当前交付中没有足以核验三组实验已经完成的证据。

## 原始证据

1. `experiments/xp04_calidist_behavioral_confidence_calibration/data_provenance_audit/20261110T055604Z/PERTURBATION_FEASIBILITY_MATRIX.md:6–8`：依次列出 Assertion、Probe、Sample Corruption，三个状态均为 `CONDITIONAL`；所列内容只有 required data、可用性和 `text before re-encoding`，没有变换模板、生成算法、强度、重复数或标签保持规则。该文件 `:18` 明确可行性不等于科学验证。
2. 同目录 `perturbation_feasibility.csv:3–5`：使用 `assertion`、`probe`、`sample_corruption` 三个机器标签，状态为 `FEASIBLE after rebuilding encoder provenance`。这是可行性表，不是执行 manifest 或结果表。
3. 同目录 `RAW_TEXT_AUDIT.md:35`：要求 modified strings 必须重新编码，并先固定编码器、tokenizer、checkpoint 和输出 schema；没有给出三种修改的具体操作定义。
4. `experiments/xp_caldist_behavioral_calibration/20261107T172302Z/reports/SINGLE_MODAL_PILOT_REPORT.md:5` 明确：“Text Representation 不等同于文本语义 Assertion/Probe”；对应报告生成代码 `scripts/analyze_pilot.py:258` 保留同样声明。旧随机特征 Pilot 不能填补此处缺口。
5. Plan B 首轮 `20261120T143131Z/configs/PLAN_B_PROTOCOL.json:101–113` 只注册 `neutral_append`、`semantic_paraphrase`、四种 pitch 和两种 timbre。`configs/RAW_PILOT_SCHEMA.json:57–71` 的文本注册也只有两类。该注册不是三类方法的原始定义。

## 实际执行内容对照

读取 Plan B 首轮两个 `results/pilot/<dataset>/transformation_manifest_both.csv` 的 family/status 字段：每个数据集均登记 1,200 条条件，其中 `neutral_append=240`、`semantic_paraphrase=240`、`pitch_shift=480`、`timbre_tilt=240`；`Assertion`、`Probe`、`Sample/sample_corruption` 命名的执行行均为 **0**。IEMOCAP 成功 1,189、失败 11；MELD 成功 1,182、失败 18。此处只核对执行归属，失败项没有删除，也没有根据结果重新命名。

本轮 MELD 匹配探索继续使用已经明确冻结的 `neutral_append` 和 `semantic_paraphrase`；它不构成对 Assertion、Probe、Sample 的补做。

## 本地历史与检索范围

- 审计时 Git HEAD：`b856df71e628781d341a458edbd48d645e7a3836`；本地 `--all` 可达提交共 71 个，包含当前 main、其他本地分支和已有 remote refs；未联网 fetch。
- 对 MD/TXT/PY/JSON/CSV/YAML/YML 的历史差异执行 `git log --all -G`；Assertion/Sample Corruption 的研究相关命中只有 `a284d1f...`（2026-11-08，旧特征 Pilot 的非等价声明）和 `93ee01a...`（2026-11-10，来源可行性审计）。文档中大写 Probe 的历史命中也仅这两笔。另做不区分大小写的 Assertion/Sample 历史搜索，共命中 7 笔；逐项排除 ZIP 计数、GPU 设备、torch.equal 重放、因果来源回归测试等工程 assertion 后，研究相关证据仍只有上述两笔。原可行性矩阵 `--follow` 只定位到 `93ee01a...`，未找到更早操作定义。
- 搜索了工作区 `experiments/`、`sources/`、`.omx/`、`CMERC/docs/`、当前已追踪文档与相关代码，以及本地可达历史；排除 `.git` 二进制和 `local_only` 模型/媒体，不加载 pickle、模型或 Test 数据。`sources/` 当前只有 CMERC 论文 PDF，未发现命名为 CaliDist/三类方法的原始文献或设计文档；本轮没有用外部资料替代缺失的仓库协议。
- 文件名和全文命中的 `concurrency_probe.py`、HTTP/resource probe、特征重建 candidate probe、梯度 probe 及 Python assertion 都是工程诊断；没有将它们误认作这里的研究实验。
- 精确检索参数、命中内容、引用文件 SHA-256、公开副本字节一致性和 family 计数保存在 `provenance/aps_search_evidence.json`。文件检索和历史取证完成；三类实验的科学协议仍未恢复。

## 缺失内容与后续条件

三类方法分别还需要原始定义来源及可执行构造规则：如何修改目标文本、是否使用上下文或标签、强度/版本数、标签保持边界、审核标准与指标分组。获得这些证据前保留待确认，不能仅凭名称补做，更不能声称本轮两类文本干扰等同于三类方法。
