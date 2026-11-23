# 历史暴露与Eval边界审计

**结论：本轮维持暂停。旧正式实验的同fold Eval尚未做前向，与整个数据集从未暴露是不同命题。** 元数据确认同fold角色互斥，但五折Eval并集包含历史Train、Validation及Test，且均在其他fold承担过Train/Dev角色；不能据此声称全新的研究者未见评估。上游文本生产encoder也未被证明满足折内独立。

## 1. 本次读取边界事件：必须单独保留

本次审计发生了一次过宽的复合manifest/readiness元数据输出：未先执行白名单投影，输出显示了unknown-label mask中逐项的标签相关字段，其中可能包含旧Test Session的掩码记录，以及Calibration优化元数据。该失误已立即向主代理披露；本报告不重复任何相关值、记录ID或标签。

这些过宽输出字段**没有用于本次统计、模型选择、映射、参数设置或后续实验**。随后停止宽泛读取，以下计数全部由严格白名单投影取得：JSON仅保留dataset/fold/role的Dialogue IDs、数量及分组元数据；CSV仅保留ID、原split、历史暴露布尔标记及覆盖信息，其他子树/字段直接跳过。可复核实现为`results/exposure/project_metadata.py`。

独立的Eval/Test标签数组、Eval pkl/tensor、原始Eval/Test文本均未打开，没有模型加载、前向、新Eval指标或干扰生成。本次发生的“混合元数据越界”不能被改写成“全程未见任何标签字段”，也不能与“独立Eval/Test标签数组没有打开”混为一谈。事件边界见`audit_access_boundary.json`；不以旧正式实验的sealed声明掩盖本次失误。

## 2. 旧正式实验确实停在Calibration内部

依据旧`APPROVED_PROTOCOL.md`、`EVALUATION_READINESS_REPORT.md`及冻结`formal_protocol.json`，旧正式流程记录为未构建Evaluation DataLoader、未做对应Eval前向或指标计算。白名单读取完成清单显示50个CMERC run（每数据集25个、5fold×5seed），均有strict reload元数据；Calibration结果400行，仅含200行fit与200行grouped_oof，覆盖同一50个run。这里没有读取这些表中的指标数值或逐样本结果。

以上说明“旧正式同fold Eval计算尚未执行”，**不能证明**这些Dialogue从未在其他历史实验或其他fold被研究者看到；本次也没有通过打开Eval数据来进一步验证旧sealed声明。

## 3. Eval中的原split别名：不是新的Train-only队列

以下来自历史lineage的白名单元数据，而非解析样本标签；所有Dialogue的original_split映射均得到对应记录，未验证映射为0。

| 数据集 | 五fold Eval并集D | 原Train D/话语 | 原Validation D/话语 | 原Test D/话语 |
|---|---:|---:|---:|---:|
| IEMOCAP | 151 | 108/5283 | 12/527 | 31/1623 |
| MELD | 1432 | 1038/9989 | 114/1109 | 280/2610 |

每个Dialogue仅在五fold Eval中出现一次。MELD的原test别名明确存在，不能把所有Eval文本当作原Train文本；当前提取器阻断与Eval访问边界未解除，本轮不生成这些样本的改写，也不读取其标签。

| 数据集 | Fold | Eval D | 原Train D | 原Validation D | 原Test D |
|---|---|---:|---:|---:|---:|
| IEMOCAP | fold1 | 28 | 26 | 2 | 0 |
| IEMOCAP | fold2 | 30 | 25 | 5 | 0 |
| IEMOCAP | fold3 | 32 | 28 | 4 | 0 |
| IEMOCAP | fold4 | 30 | 29 | 1 | 0 |
| IEMOCAP | fold5 | 31 | 0 | 0 | 31 |
| MELD | fold1 | 286 | 205 | 24 | 57 |
| MELD | fold2 | 294 | 219 | 27 | 48 |
| MELD | fold3 | 296 | 222 | 20 | 54 |
| MELD | fold4 | 281 | 195 | 21 | 65 |
| MELD | fold5 | 275 | 197 | 22 | 56 |

历史lineage对原Train标记为原模型训练已用，对原Validation标记为checkpoint选择及方法开发已用，对原Test标记为重复Test评价及方法开发暴露。其`not_previously_evaluated`是旧时间点的狭义标记，不代表模型未见、数据集未见或截至今天研究者未见，尤其不能覆盖之后的Plan B用途。

## 4. 五折轮转的跨fold暴露

| 数据集 | Fold | Eval D | 其他fold Train/Dev已用D | 其他fold Calibration已用D | 同fold四角色互斥 |
|---|---|---:|---:|---:|---|
| IEMOCAP | fold1 | 28 | 28 | 28 | 是 |
| IEMOCAP | fold2 | 30 | 30 | 30 | 是 |
| IEMOCAP | fold3 | 32 | 32 | 32 | 是 |
| IEMOCAP | fold4 | 30 | 30 | 30 | 是 |
| IEMOCAP | fold5 | 31 | 31 | 31 | 是 |
| MELD | fold1 | 286 | 286 | 143 | 是 |
| MELD | fold2 | 294 | 294 | 153 | 是 |
| MELD | fold3 | 296 | 296 | 153 | 是 |
| MELD | fold4 | 281 | 281 | 143 | 是 |
| MELD | fold5 | 275 | 275 | 134 | 是 |

IEMOCAP五fold Eval并集151个Dialogue全部也出现在另一fold Calibration；MELD对应为726/1432。MELD的其余706个不能误称已做过Calibration，但它们也已在其他fold的Train/Dev角色出现。

这种跨fold复用本身是交叉验证结构，**不自动等于同fold训练泄漏**；必须始终使用与该fold身份一致的模型和Calibration。问题在于历史跨fold结果已经用于研究过程，不能再把全体Eval称为研究者完全未见；使用其他fold的Calibration/Dev标签或结果来改当前fold方法，会使所谓新Eval失去研究者独立性。

## 5. 后续Plan B的重叠：仅用可核查ID桥

没有靠数字相似猜映射。MELD正式ID规则由历史构建代码明确规定，Plan B manifest提供实际feature Dialogue ID与raw Dialogue ID；两者通过原split及raw ID桥连接。IEMOCAP使用已匹配的完整Dialogue ID。未读取原始MELD CSV、任何Test标签、pkl或特征张量；全部非Test桥接均验证成功。

| 数据集 | Fold | 旧Plan B Train Fit D | 旧Plan B内部Dev D | 旧Plan B完整Validation D | 旧120条干扰Pilot覆盖D |
|---|---|---:|---:|---:|---:|
| IEMOCAP | fold1 | 20 | 6 | 2 | 2 |
| IEMOCAP | fold2 | 24 | 1 | 5 | 5 |
| IEMOCAP | fold3 | 27 | 1 | 4 | 4 |
| IEMOCAP | fold4 | 26 | 3 | 1 | 1 |
| IEMOCAP | fold5 | 0 | 0 | 0 | 0 |
| MELD | fold1 | 185 | 20 | 24 | 24 |
| MELD | fold2 | 197 | 22 | 27 | 27 |
| MELD | fold3 | 196 | 26 | 20 | 20 |
| MELD | fold4 | 178 | 17 | 21 | 21 |
| MELD | fold5 | 178 | 19 | 22 | 22 |

旧Plan B固定120条样本覆盖IEMOCAP原Validation的12个Dialogue、MELD原Validation的114个Dialogue，全部属于当前五fold Eval并集。旧Plan B官方Validation整体指标也已被分析；本轮只投影其manifest的ID列，没有读取该表的标签、置信度、类别或概率列。

新MELD Train-only Plan B四角色同样映射进此五fold Eval并集：base_fit656、base_inner_dev72、score_fit155、score_eval155，合计原Train的1038个Dialogue。原数据角色重命名不能消除已有监督训练、选择、可靠性拟合或结果查看记录；具体逐fold计数保存在`plan_b_overlap.json`。

## 6. 上游提取器是另一层独立性条件

- 旧CMERC/XP04四层RoBERTa特征的实际生产encoder、微调run及完整训练split尚未恢复。已有资料支持COSMIC参考流程先作情绪监督微调，但引用和相同特征结构不证明CMERC实际用的是哪次训练，也不证明为本五fold分别做过折内微调。
- 先前IEMOCAP纯元数据审计记录：原XP04 Validation Dialogue多数出现在文本包自带Train列表中；这只支持条件性的表示学习重叠风险，不能由pkl附带ID列表直接证明实际encoder发生标签泄漏。旧cached文本表示不能被默认视为完全fold-local。
- MELD原300维音频已被识别为带情绪输出模型中的上下文BiGRU表示，而不是单纯的原始OpenSMILE统计；精确上游选择器及训练split/处理记录仍不完整。只重新训练下游fold模型不能自动证明上游表示和标准化未见过对应Eval样本。
- 视觉等原缓存的fold-local生产范围也不能由“固定不变”推导出来。本lane不新作权重审计或加载；模型和来源哈希由主代理的兼容性审计负责。

因此，下游同fold角色互斥、旧正式Eval尚未前向、上游特征未在Eval标签上训练、研究者没有看过历史结果，是四个不同条件，不能相互替代。本轮不可借用新的不兼容提取器去声称延续旧XP04特征空间。

## 7. 交付与停止位置

- `role_counts.json`、`eval_alias_counts.json`、`exposure_summary.json`：白名单角色与原split计数。
- `fold_rotation_exposure.json`：同fold互斥及跨fold使用计数。
- `plan_b_overlap.json`、`prior_completion_metadata.json`：安全ID桥及历史完成/覆盖元数据。
- `audit_access_boundary.json`：本次metadata过宽输出的披露；没有把事件改写为完全无标签字段访问。
- `project_metadata.py`：可复核的严格字段投影，不导出原始label字段、文本、预测或模型哈希。
- 停在审计阶段。未启动5×5训练、Calibration新拟合、Eval改写、Eval推理或指标比较；本报告不构成开启这些操作的许可。
