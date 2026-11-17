# 转换器方法比较：仅 XP04 Train 的模型选择证据

本文只汇总 Train 前端筛选、Train 内 Dialogue 分组交叉验证及最终 Train 拟合，不读取或引用新一轮 Validation 目标。表中的 OOF 指标已经参与前端、方法或参数选择，不能作为无偏泛化估计。本轮方法均为重建候选；没有作者原转换器的来源证据，不能据此声明类别 A。最终是否达到 B 或 C，由根代理的一次独立 Validation 评估及门控决定。

## 1. 输入范围与完整记录

| 数据集 / 阶段 | 有效 Train 总清单 | 参与 OOF 的话语 | Dialogue | case 数量 |
|---|---:|---:|---:|---:|
| IEMOCAP / frontend_cv | 5283 | 5283 | 108 | 60 |
| IEMOCAP / model_selection | 5283 | 5283 | 108 | 11 |
| IEMOCAP / model_selection_final | 5283 | 5283 | 108 | 14 |
| MELD / frontend_cv | 9989 | 256 | 256 | 64 |
| MELD / model_selection | 9989 | 9988 | 1038 | 8 |
| MELD / model_selection_final | 9989 | 9988 | 1038 | 11 |

MELD 前端只使用固定的 256 条 Train probe，每个 Dialogue 一条；其余 9,733 条不参与该阶段，不能解释为提取失败。锁定前端的完整 Train 有 9,988/9,989 条成功，无法配对的原 Train row_index=1165（raw Dialogue 125 / Utterance 3，工作目录缺文件、原包及同划分备用媒体损坏无法解码）保留 NaN 与失败记录，未用官方特征补齐。模型在 1,038 个 Dialogue 中的成功配对上拟合，完整 Train 覆盖检查为 false。IEMOCAP 的 5,283 条 / 108 个 Dialogue 完整参与。

[CSV](converter_method_comparison.csv) 保存六个阶段 JSON 中全部 168 个 case 汇总行，包含未获选、极大误差和全秩 Ridge 失败条件。最终阶段重复引用早先的 baseline case 是扩展菜单下的重新排名，不是额外独立重复实验。CSV 不包含特征向量、媒体、原文或标签。各行包含来源 JSON 的 SHA-256，可追溯原始记录。

## 2. 固定菜单与拟合边界

- IEMOCAP 前端：`is10`、`is10_compat`、`emobase2010`；对应候选配置为 IS10_paraling、IS10_paraling_compat、emobase2010。它们是已明确版本的候选提取配置，不代表已经找到了 XP04 作者的原始配置。
- MELD 前端：`ffmpegmono_16k_none`、`ffmpegmono_16k_zscore`、`ffmpegmono_native48k_none`、`ffmpegmono_native48k_zscore`、`meanall_16k_none`、`meanall_16k_zscore`、`meanall_native48k_none`、`meanall_native48k_zscore`。包含两种声道下混、16/48 kHz、是否逐话语 waveform z-score；名称中的波形处理与下面的特征行处理是不同步骤。
- 所有前端都比较特征行变换 `none`、`row_minmax`、`row_zscore`、`row_l2`，总计 IEMOCAP 12 个前端/行变换组合、MELD 32 个组合。
- IEMOCAP 每个组合比较 `identity`、`best_column_exact`、`diagonal_affine`、`speaker_diagonal_affine`、`best_column_affine`；MELD 6373→1611 只比较两种列匹配方法，不使用同维对角方法。
- 低秩菜单：PCR rank 16/64/128；保留 rank 256 的 Ridge α=0.01/1/100。每折只拟合一次最大 rank=256 的 Train 投影，较小 rank 复用前导分量；随机种子为 424242，SVD 工作数组为 float32。不是作者原 PCA 的恢复。
- V2 另增全秩 Ridge α=0.01/1/100，保留全部输入字段。每折以 float64 构造一次标准化 X 的 Gram 与 X/Y 交叉积，三个 α 各做 Cholesky 求解，共享矩阵，不执行 SVD。它与限制在 rank 256 子空间中的 Ridge 不同。

任何列索引、相关性搜索、OLS 系数、speaker 参数、X/Y 均值和标准差、低秩投影、回归系数都只从该折的 fit-Train 学习。Ridge 的目标为 `sum((Zx @ B - Zy)^2) + α * sum(B^2)`，未除以样本量。反变换只用同一 fit-fold 的 Y 统计量。输入行归一化是预先指定的无状态变换，不跨样本拟合统计量。

IEMOCAP 的 speaker affine 使用 `raw_utterance_id[:5] + raw_utterance_id.rsplit('_', 1)[1][0]` 构造 Session+gender 全局人物键，未将模型的 Dialogue-relative speaker slot 0/1 误当作同一个人。未见 speaker 回退到该折全局仿射，并逐折记录 fallback 行数。

## 3. 3-fold Dialogue CV、评分与版本

相同阶段内的所有条件使用相同公共有限值掩码和相同 GroupKFold(3) Dialogue 分组；任何 Dialogue 都不会同时出现在该折拟合和留出部分。模型阶段与全秩扩展阶段复用相同折定义和 `cvfoldhash`。MELD probe 与完整 Train 的 CV 使用不同样本范围，二者分数不能直接视作同一个固定评测集合上的提升。

对每个留出 Dialogue，先计算 `sqrt(mean((预测误差 / fit-fold target scale)^2))`，其中 scale 是 fit-fold 的目标逐维标准差，最小值 1e-8；再对 Dialogue 等权平均。跨折按 held Dialogue 数加权，得到主指标 `score_dialogue_equal_nrmse`。辅助指标包括原量纲 MAE、RMSE、逐维相关性、余弦及所有 OOF 元素按各自 fit-fold scale 计算的 global NRMSE。`mean(column_RMSE / scale)` 与 global NRMSE 不是同一量，不用于主排序。

在最优主分数 1% 范围内，优先较低预指定复杂度，再按 candidate 和行变换名的字典序打破并列。简单方法顺序为 identity、best-column-exact、diagonal-affine、speaker-diagonal-affine、best-column-affine；其后为 PCR rank 16/64/128、低秩 Ridge α=0.01/1/100，再其后是 V2 全秩 Ridge α=0.01/1/100。CSV 同时给出纯数值排名、是否落入 1% 区间及阶段最终选择，避免把数值排名与协议选择混淆。

初始 V1 菜单不含全秩 Ridge。根代理在 Train 方法覆盖审查阶段、尚未新开 Validation 输入和目标之前，明确增补这一方法族为 V2；前端、阈值、Validation 边界不变。不能将整个最终菜单描述为起初全部预注册。V1 选择和模型均保留，最终在 `final/` 按 V2 重新拟合并记录当前协议 SHA。

- IEMOCAP final CV fold hash：`1afd5cb6e223fa2f00d7f30543513670ae21eebe2d5f49df6b3a887c1eeeca01`；V2 protocol SHA：`02377d045ceeb34932eabbb227a72712707f8c7704be483206e13aaef644a9b2`。
- MELD final CV fold hash：`faf6e93da16c68562d73794d8eda8dd580ae2e2bd1b09aeb3954157abf737a84`；V2 protocol SHA：`02377d045ceeb34932eabbb227a72712707f8c7704be483206e13aaef644a9b2`。

MELD 首次 frontend 的全部数值记录和 OOF 已写出后，日志 Tee 在解释器关闭时再次 flush，进程退出 120。原始源码、失败说明及完整数值输出都保留；随后仅修复日志关闭检查。IEMOCAP frontend、两数据集原 model 阶段和全秩扩展均退出 0。该工程异常不能隐去，也不代表前端数值结果经过了补调。

## 4. 精确列、常数列与缺失值

所有数据集、所有前端和所有 fit-fold 中，`best_column_exact` 的 `exact_train_column_count` 都为 **0**。这里真正的精确搜索先将整列 canonical float64 化、统一 signed zero，计算 SHA-256，再逐值复核；未找到精确列时的最大绝对 Pearson 匹配保留为未经符号/尺度/偏移校正的候选，不能因方法名中含 exact 就声称精确恢复。`best_column_affine` 在相同 fit-fold 的相关列上另拟合 OLS；允许不同目标维选择同一个源列。共用搜索时优先精确源列打破相关并列，本次没有找到精确源列。

原始 fit-fold 目标标准差为 0 的维数：IEMOCAP 三折均 0；MELD probe 三折均 0；MELD 完整成功 Train 三折均 0。由保存的 full-ridge fold scalers 核对，锁定 `none` 行变换下两个数据集的输入常数维也均为 0。前端各候选行变换后的输入常数维计数没有单独持久化，CSV 中留空，不用标准差 floor 后的 0 计数冒充原始常数维证据。CSV 的目标常数维计数来自相同 Train 折的原始标准差；没有接触 Validation。

代码仍为可能的常数维保留策略：零方差 X 不除以 0；常数目标由训练均值处理；零向量余弦、任一向量零方差的逐维相关性为 NA，单列计数，不当作 1。缺失/非有限值保留原总清单，在候选比较的公共成功掩码之外列出索引；本轮未从缺失输出删除不利指标，也未推测缺失媒体的官方输入。

## 5. 每个前端与行变换的最佳简单方法

以下各格为该前端/行变换组合内部按同一 1% 规则选择的方法及主分数；全部其他方法和失败值仍在 CSV。

### IEMOCAP

| 前端 | none | row_minmax | row_zscore | row_l2 |
|---|---|---|---|---|
| emobase2010 | 列仿射 / 0.38246342 | 列仿射 / 0.95588645 | 列仿射 / 0.90960169 | 列仿射 / 0.88518472 |
| is10 | 同序仿射 / 0.10158337 | 人物仿射 / 0.93874777 | 列仿射 / 0.88489554 | 列仿射 / 0.7741241 |
| is10_compat | 列仿射 / 0.33769414 | 人物仿射 / 0.93933691 | 列仿射 / 0.89294833 | 列仿射 / 0.83959194 |

### MELD

| 前端 | none | row_minmax | row_zscore | row_l2 |
|---|---|---|---|---|
| ffmpegmono_16k_none | 列仿射 / 0.49015006 | 列仿射 / 1.6349448 | 列仿射 / 0.97433702 | 列仿射 / 1.2509358 |
| ffmpegmono_16k_zscore | 列仿射 / 0.51202755 | 列仿射 / 0.97666685 | 列仿射 / 0.97415086 | 列仿射 / 1.2149818 |
| ffmpegmono_native48k_none | 列仿射 / 0.5722298 | 列仿射 / 0.9775361 | 列仿射 / 0.97333197 | 列仿射 / 1.4450073 |
| ffmpegmono_native48k_zscore | 列仿射 / 0.5972873 | 列仿射 / 0.97881118 | 列仿射 / 0.97535338 | 列仿射 / 1.3050951 |
| meanall_16k_none | 列仿射 / 0.54261823 | 列仿射 / 0.98446557 | 列仿射 / 0.97386759 | 列仿射 / 1.2811693 |
| meanall_16k_zscore | 列仿射 / 0.5344701 | 列仿射 / 0.98295388 | 列仿射 / 0.97282618 | 列仿射 / 1.2450457 |
| meanall_native48k_none | 列仿射 / 0.59317771 | 列仿射 / 0.99072598 | 列仿射 / 0.97282208 | 列仿射 / 1.41239 |
| meanall_native48k_zscore | 列仿射 / 0.59308447 | 列仿射 / 0.98725892 | 列仿射 / 0.9746597 | 列仿射 / 1.3382677 |

## 6. 锁定前端后的完整 Train 方法比较

以下只比较该数据集已锁定的前端与 `none` 行变换。全秩 Ridge 是 V2 新增，表中所有候选均列出，不按效果删除。

### IEMOCAP

| 方法 | Dialogue NRMSE | global NRMSE | MAE | RMSE | mean cosine |
|---|---:|---:|---:|---:|---:|
| identity | 193.72298 | 199.07897 | 29.718124 | 199.71884 | 0.11196107 |
| best_column_exact | 193.72298 | 199.07897 | 29.7182 | 199.71884 | 0.11196057 |
| diagonal_affine **（最终选择）** | 0.10158337 | 0.11863534 | 0.011411119 | 0.11462974 | 0.99461886 |
| speaker_diagonal_affine | 0.10309166 | 0.12045236 | 0.011839492 | 0.11631117 | 0.994492 |
| best_column_affine | 0.1028731 | 0.11993785 | 0.011495213 | 0.11551321 | 0.99456881 |
| PCR r=16 | 0.68379842 | 0.69280078 | 0.45137494 | 0.67978142 | 0.71440132 |
| PCR r=64 | 0.56729137 | 0.57430772 | 0.3701392 | 0.56399192 | 0.81357148 |
| PCR r=128 | 0.48898243 | 0.4951551 | 0.31810893 | 0.48623967 | 0.86590709 |
| 低秩 Ridge r=256, α=0.01 | 0.39210744 | 0.39814178 | 0.25482807 | 0.39110629 | 0.91664577 |
| 低秩 Ridge r=256, α=1 | 0.39210702 | 0.39814121 | 0.25482229 | 0.3911047 | 0.91664596 |
| 低秩 Ridge r=256, α=100 | 0.39216239 | 0.39818298 | 0.25432835 | 0.39104671 | 0.91663454 |
| 全秩 Ridge α=0.01 | 0.13646042 | 0.14993965 | 0.038492659 | 0.14655457 | 0.99122622 |
| 全秩 Ridge α=1 | 0.13305341 | 0.1467205 | 0.037726075 | 0.14332437 | 0.99158731 |
| 全秩 Ridge α=100 | 0.13906326 | 0.15080836 | 0.065748883 | 0.14726793 | 0.99073485 |

### MELD

| 方法 | Dialogue NRMSE | global NRMSE | MAE | RMSE | mean cosine |
|---|---:|---:|---:|---:|---:|
| best_column_exact | 7697522.7 | 7970109.6 | 48468.494 | 1015067 | 0.029344611 |
| best_column_affine **（最终选择）** | 0.48312885 | 0.49391364 | 0.040988795 | 0.092293778 | 0.98288993 |
| PCR r=16 | 0.88825641 | 1.5118615 | 0.10273326 | 0.25102745 | 0.95385755 |
| PCR r=64 | 0.84125425 | 1.7763011 | 0.092919334 | 0.30553001 | 0.96055748 |
| PCR r=128 | 0.82178553 | 1.9761986 | 0.088282052 | 0.32619214 | 0.96366068 |
| 低秩 Ridge r=256, α=0.01 | 0.80849423 | 2.2287746 | 0.082943575 | 0.37275626 | 0.96713322 |
| 低秩 Ridge r=256, α=1 | 0.80848961 | 2.2287179 | 0.082943458 | 0.37274605 | 0.96713326 |
| 低秩 Ridge r=256, α=100 | 0.80803053 | 2.2230708 | 0.082931956 | 0.37172891 | 0.96713752 |
| 全秩 Ridge α=0.01 | 5.4833154 | 64.013081 | 0.23796514 | 11.452877 | 0.89308447 |
| 全秩 Ridge α=1 | 3.0280235 | 31.261249 | 0.15460247 | 6.1337562 | 0.92957994 |
| 全秩 Ridge α=100 | 1.2433966 | 9.1049867 | 0.080829512 | 1.7780832 | 0.96877283 |

IEMOCAP 的同序仿射优于全部低秩及全秩 Ridge；MELD 的列仿射优于这些多输出回归候选。全秩 Ridge 保留了输入的全部字段，结果仍未提升，因此本轮失败不能仅解释为先前 rank=256 的截断限制。该结论只针对当前候选声学前端、固定 α 网格和数据范围，不排除未知原始提取配置、不同原始选择器或其他尚未验证变换。

两数据集最终候选的 Train OOF 严格 feature gate 均未通过：IEMOCAP 虽然逐维相关性中位数较高，仍存在明显逐值误差和不稳定样本；MELD 同时存在较大误差与一条 Train 媒体损坏且无可用副本。均不能凭相同维度、较高平均余弦或训练拟合能力认定 XP04 兼容。

## 7. 结果的适用范围

这些模型只学习配对 Train 特征之间的重建关系，不是作者原 SVM、PCA、特征字段列表、OpenSMILE 预处理或标准化对象的来源恢复。候选行变换、前端选择和回归超参数共同使用同一套 Train OOF 分数，存在模型选择偏差；分组只处理 Dialogue 依赖，不能自动去掉这种选择偏差。OOF 分数也不是情绪分类准确率或置信度校准收益。

本报告不读 Validation 特征、不调整冻结阈值、不进行 CMERC 概率推理。原始转换器精确恢复类别 A 缺少来源证据；B/C 由根代理使用锁定的 V2 最终模型，在独立 Validation 全覆盖特征检验及满足门槛后的冻结 Checkpoint 概率检验中判定。不得因为某一方法在 Train 上较好就直接进入原始音频干扰。

## 8. 证据索引

- [IEMOCAP_frontend_cv.json](provenance/IEMOCAP_frontend_cv.json)，SHA-256 `0f7b1ab2d04dd50d6e1439da4adb89c4c1f7e79ccf8815d5b2e59b1e349ef46b`。
- [IEMOCAP_model_selection.json](provenance/IEMOCAP_model_selection.json)，SHA-256 `aa228704c45767602416d88091feb96a598d3cada9c4c4754e684abb2bd898d6`。
- [IEMOCAP_model_selection_final.json](provenance/IEMOCAP_model_selection_final.json)，SHA-256 `2a754593079b0b4fff83f89a31766a49270b22ad90f9e07345fdc1b0a9d15ed4`。
- [MELD_frontend_cv.json](provenance/MELD_frontend_cv.json)，SHA-256 `214089c174ca5c10806a7362ec55c1bc1690c0a13ef4c591890cf3587ab08bd8`。
- [MELD_model_selection.json](provenance/MELD_model_selection.json)，SHA-256 `245eeeefd87549bdc4a0a468723081186dda2584365791639a1225ceb90d02e4`。
- [MELD_model_selection_final.json](provenance/MELD_model_selection_final.json)，SHA-256 `b8160f2487dca717a4bc033d8ae2b816870e69eac841c559714916eb5bef18d1`。

- 同折原始 scaler：`local_only/{dataset}/full_ridge_extension/fold_{0,1,2}_scalers.npz`，由最终选择 JSON 引用哈希；数值数组保留本地。
- [代码版本快照记录](provenance/converter_code_versions.json)、[MELD 日志关闭失败记录](provenance/MELD_frontend_shutdown_failure.json)、[runner 版本记录](provenance/cv_runner_versions.json)。
- [完整 case 汇总 CSV](converter_method_comparison.csv)。公开发布前仍应遵循整个 run 的脱敏与审核规则。
