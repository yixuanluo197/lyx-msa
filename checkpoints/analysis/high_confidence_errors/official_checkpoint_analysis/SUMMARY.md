# 作者官方 CMERC Checkpoint 可靠性诊断

本轮独立分析作者发布的 IEMOCAP / MELD 权重；不与旧本地 Validation checkpoint 的六项实验混合。复用已验证完整输入的逐话语 CSV / NPZ，仅新增三种均值替换推理；未训练、未更新权重、未选模或调参。所有结论仅适用于当前两个固定 checkpoint 与测试集，不能据此声称 CMERC 相对 Baseline 提升或跨 seed 稳健性。

## 输入来源和性能复核

完整输入来源：`../../official_checkpoint_verification/`。官方代码提交 `815c7c060914f5b55db816c0ba3662b11d08552f`，继续使用该次验证的两处已记录 PyG 兼容补丁（删除不可用且未使用的 topk 导入、HypergraphConv 反向传播尺寸顺序修正）。未切换成本地训练模型代码。原始模型文件为纯 state_dict，缺少作者训练轮次、种子、选模记录；本轮仅依据既有下载来源与固定测试命令标记为作者发布权重。

| Dataset | Valid utterances | Actual W-F1 (%) | Paper W-F1 (%) | Gap (pp) |
|---|---:|---:|---:|---:|
| IEMOCAP | 1623 | 71.9837935825 | 71.98 | +0.0037935825 |
| MELD | 2610 | 66.8483922007 | 66.85 | -0.0016077993 |

IEMOCAP checkpoint SHA-256：`f3098fed2734c18171c02a59d014ee27320d845bc1ad5dfa17162b8dcfb49925`。

MELD checkpoint SHA-256：`1c72e29c173b749675885bdd4f477bd5d3702ab7cca7eb03cb145850bf0beba7`。

全部 4233 条原预测与 NPZ、源特征中的真实标签、原文、source ID、测试对话顺序核对通过。保留原始 log_softmax 的 exp 概率，不做归一化、温度缩放或其他校准后处理。新 CSV 派生 correct，原 CSV / NPZ 不变。

## 实验一、三：完整输入错误与残余高置信错误

本报告所有 CSV 的 accuracy、F1、ECE、MCE、置信度与比率均为 0–1 分数。Word/Excel 表头按实际单位标注。阈值包含等号；没有达到阈值的预测时，高置信错误率缺失并显示 N/A。

| dataset | valid_utterances | total_errors | errors_ge_0p8 | errors_ge_0p9 | errors_ge_0p95 | mean_error_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1623 | 456 | 200 | 128 | 76 | 0.747381 |
| MELD | 2610 | 841 | 0 | 0 | 0 | 0.382656 |

高置信错误率 = 高置信错误数 / 全部达到阈值的预测数，分母不是错误总数。IEMOCAP ≥0.90 的预测 796 条，其中错误 128 条（16.0804%）；MELD ≥0.80 只有 2 条预测、均正确，≥0.90 和 ≥0.95 没有预测，错误率均为 N/A。各数据集 `errors_all.csv` 保留全部错误；`high_confidence_errors.csv` 每个样本在每个满足的阈值各占一行，不能跨阈值直接相加当作去重样本数。MELD 的高置信错误表仅表头，表示真实空集合。

### IEMOCAP 按真实情绪类别

| true_emotion | support | total_errors | error_rate | errors_ge_0p8 | errors_ge_0p9 | errors_ge_0p95 |
| --- | --- | --- | --- | --- | --- | --- |
| happy | 144 | 52 | 0.361111 | 18 | 12 | 10 |
| sad | 245 | 46 | 0.187755 | 22 | 16 | 8 |
| neutral | 384 | 101 | 0.263021 | 51 | 36 | 23 |
| angry | 170 | 56 | 0.329412 | 24 | 15 | 4 |
| excited | 299 | 74 | 0.247492 | 22 | 9 | 6 |
| frustrated | 381 | 127 | 0.333333 | 63 | 40 | 25 |

### MELD 按真实情绪类别

| true_emotion | support | total_errors | error_rate | errors_ge_0p8 | errors_ge_0p9 | errors_ge_0p95 |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | 1256 | 206 | 0.164013 | 0 | 0 | 0 |
| surprise | 281 | 110 | 0.391459 | 0 | 0 | 0 |
| fear | 50 | 40 | 0.800000 | 0 | 0 | 0 |
| sadness | 208 | 140 | 0.673077 | 0 | 0 | 0 |
| joy | 402 | 132 | 0.328358 | 0 | 0 | 0 |
| disgust | 68 | 47 | 0.691176 | 0 | 0 | 0 |
| anger | 345 | 166 | 0.481159 | 0 | 0 | 0 |

IEMOCAP 的 ≥0.90 残余错误在 frustrated（40）与 neutral（36）最多；MELD fear 类内错误率 80%、disgust 69.12%、sadness 67.31%，高总体 F1 不代表各类别均表现良好。完整典型案例与原文见 Word；全量明细见 CSV。

## 实验四：单模态均值替换

使用 `torch.mean(feature, dim=0).repeat(time, 1, 1)`，与官方 `caldiff/cal_diff.py` 第 241–247 行一致。文本四路 RoBERTa 分别替换；音频、视觉分别替换其自身特征。均值沿 padded 时间轴计算，包含填充值，**不是零向量，也不是仅有效话语均值**，因此可能依赖同 batch 的最大对话长度。保留 batch size=16、原 test order、num_workers=2、mask、对话长度、cuDNN 后端、官方测试 argv/seed=1475 和兼容补丁。替换仅通过模型 forward pre-hook 改输入，不改模型结构。

### IEMOCAP 各输入条件

| input_condition | valid_samples | total_errors | errors_ge_0p8 | errors_ge_0p9 | errors_ge_0p95 | mean_error_confidence | accuracy | weighted_f1 | high_confidence_error_rate_ge_0p9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 1623 | 456 | 200 | 128 | 76 | 0.747381 | 0.719039 | 0.719838 | 0.160804 |
| w/o_text | 1623 | 641 | 325 | 221 | 165 | 0.777124 | 0.605052 | 0.611511 | 0.316619 |
| w/o_audio | 1623 | 537 | 232 | 145 | 79 | 0.731949 | 0.669131 | 0.671066 | 0.201669 |
| w/o_visual | 1623 | 468 | 192 | 118 | 69 | 0.733989 | 0.711645 | 0.711790 | 0.154248 |

完整输入正确、删除后成为高置信错误的统计，比例分母为该模型完整输入正确数。still_wrong_count 独立记录完整输入已错且删除后仍错的数量（跨阈值重复记录，同一条件不应相加）。

| input_condition | threshold | full_correct_samples | new_high_confidence_errors | ratio | full_wrong_samples | still_wrong_count | still_wrong_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w/o_text | 0.800000 | 1167 | 104 | 0.089117 | 456 | 363 | 0.796053 |
| w/o_text | 0.900000 | 1167 | 66 | 0.056555 | 456 | 363 | 0.796053 |
| w/o_text | 0.950000 | 1167 | 44 | 0.037704 | 456 | 363 | 0.796053 |
| w/o_audio | 0.800000 | 1167 | 34 | 0.029135 | 456 | 379 | 0.831140 |
| w/o_audio | 0.900000 | 1167 | 22 | 0.018852 | 456 | 379 | 0.831140 |
| w/o_audio | 0.950000 | 1167 | 10 | 0.008569 | 456 | 379 | 0.831140 |
| w/o_visual | 0.800000 | 1167 | 0 | 0 | 456 | 446 | 0.978070 |
| w/o_visual | 0.900000 | 1167 | 0 | 0 | 456 | 446 | 0.978070 |
| w/o_visual | 0.950000 | 1167 | 0 | 0 | 456 | 446 | 0.978070 |

### MELD 各输入条件

| input_condition | valid_samples | total_errors | errors_ge_0p8 | errors_ge_0p9 | errors_ge_0p95 | mean_error_confidence | accuracy | weighted_f1 | high_confidence_error_rate_ge_0p9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 2610 | 841 | 0 | 0 | 0 | 0.382656 | 0.677778 | 0.668484 | N/A |
| w/o_text | 2610 | 1296 | 0 | 0 | 0 | 0.335564 | 0.503448 | 0.435585 | N/A |
| w/o_audio | 2610 | 867 | 0 | 0 | 0 | 0.367354 | 0.667816 | 0.662565 | N/A |
| w/o_visual | 2610 | 850 | 0 | 0 | 0 | 0.381038 | 0.674330 | 0.665199 | N/A |

完整输入正确、删除后成为高置信错误的统计，比例分母为该模型完整输入正确数。still_wrong_count 独立记录完整输入已错且删除后仍错的数量（跨阈值重复记录，同一条件不应相加）。

| input_condition | threshold | full_correct_samples | new_high_confidence_errors | ratio | full_wrong_samples | still_wrong_count | still_wrong_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w/o_text | 0.800000 | 1769 | 0 | 0 | 841 | 640 | 0.760999 |
| w/o_text | 0.900000 | 1769 | 0 | 0 | 841 | 640 | 0.760999 |
| w/o_text | 0.950000 | 1769 | 0 | 0 | 841 | 640 | 0.760999 |
| w/o_audio | 0.800000 | 1769 | 0 | 0 | 841 | 814 | 0.967895 |
| w/o_audio | 0.900000 | 1769 | 0 | 0 | 841 | 814 | 0.967895 |
| w/o_audio | 0.950000 | 1769 | 0 | 0 | 841 | 814 | 0.967895 |
| w/o_visual | 0.800000 | 1769 | 0 | 0 | 841 | 834 | 0.991677 |
| w/o_visual | 0.900000 | 1769 | 0 | 0 | 841 | 834 | 0.991677 |
| w/o_visual | 0.950000 | 1769 | 0 | 0 | 841 | 834 | 0.991677 |

删除文本的分类损害最大：IEMOCAP 错误增加 185，MELD 增加 455；IEMOCAP 中原本正确而删除文本后成为 ≥0.90 错误的有 66 条，删除音频后有 22 条。MELD 删除文本后 W-F1 降至 43.5585%，但三个阈值的高置信错误仍全为 0，表明该阈值统计并不能替代分类性能。高置信错误不直接等于模态冲突。`true_label_probability_delta` 记录真实类别概率变化，与错误预测最大概率分别保存，不混用。

## 实验五：全部错误的置信度分布

### IEMOCAP

| confidence_range | error_count | ratio_of_all_errors |
| --- | --- | --- |
| [0,0.5) | 50 | 0.109649 |
| [0.5,0.6) | 69 | 0.151316 |
| [0.6,0.7) | 64 | 0.140351 |
| [0.7,0.8) | 73 | 0.160088 |
| [0.8,0.9) | 72 | 0.157895 |
| [0.9,0.95) | 52 | 0.114035 |
| [0.95,1] | 76 | 0.166667 |

### MELD

| confidence_range | error_count | ratio_of_all_errors |
| --- | --- | --- |
| [0,0.5) | 712 | 0.846611 |
| [0.5,0.6) | 99 | 0.117717 |
| [0.6,0.7) | 29 | 0.034483 |
| [0.7,0.8) | 1 | 0.001189 |
| [0.8,0.9) | 0 | 0 |
| [0.9,0.95) | 0 | 0 |
| [0.95,1] | 0 | 0 |

区间左闭右开，最后区间包含 1；每个数据集的分箱错误数分别严格合计 456 和 841。错误均值/中位数见 `official_cmerc_summary.csv`，错误内占比的分母为全部错误，不是高置信预测数。

## 实验六：概率校准

| dataset | accuracy | mean_confidence | ece | mce | brier_score | nll |
| --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 0.719039 | 0.831707 | 0.112668 | 0.166292 | 0.429447 | 0.869046 |
| MELD | 0.677778 | 0.449601 | 0.228176 | 0.300950 | 0.529921 | 1.140703 |

沿用前六项实验指标定义：ECE 为 10 等宽置信度箱的按样本数加权绝对置信度–准确率差；MCE 为空箱之外的最大绝对差；Brier 为各类别平方误差之和的样本均值，不除类别数；NLL 使用自然对数及原真实类别 log_softmax，不裁剪、不重新归一化。原 probability 定义和 log_softmax 定义数值一致性已核验。每箱 Signed Gap = mean confidence − accuracy；正为过度自信，负为置信度不足，空箱观测值为缺失。

IEMOCAP 全部非空箱的 Signed Gap 为正，整体 ECE=11.2668%；MELD 全部非空箱为负，整体 ECE=22.8176%。MELD 平均置信度 44.9601% 低于准确率 67.7778%，主要表现为置信度不足。因此不能因为 MELD 没有高置信错误就称校准完美。两张可靠性图见 `figures/`，图示理想对角线且不为缺失箱虚构观测点；逐箱全部统计见每数据集 `calibration_bins.csv`。

## 实验二与 Baseline 配对状态

**Baseline comparison pending: matched checkpoint unavailable**。

已检查 M³Net 官方固定提交 `4d0fa88cd4f92c72b053527e877763f54e29ccb2` 的 README、文件树和 releases/tags，以及本地和 biggpu 历史产物。官方 Dropbox 的完整分享 ZIP 已下载并做 SHA-256（`92c887f02fba3e74843b2b292259aa295123e8d5e293e6eaa05dd869afa17cab`），实际包含 IEMOCAP 1/2/3 三个预训练文件；未发现公开 MELD 权重声明。

预先固定检查的 `Roberta_IEMOCAP3.pth.tar` 来源是 M³Net 官方 README，已在 biggpu CPU 用 `weights_only=True` 安全载入。SHA-256 为 `fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa`。它是 85 张量纯 state_dict，IEMOCAP keys/shapes 完整兼容；M³Net `model.py` 与 CMERC 原模型文件哈希相同，现有 M³Net 和 CMERC 的四份特征 SHA 也相同。这些是已证实的来源/结构/特征兼容证据，**没有恢复缺失的训练 seed/epoch/选模记录或与作者 CMERC 的对应关系**，因此未把它冒充严格配对 Baseline、未按测试分数挑三份权重、未启动候选测试。

8 份已鉴定本地 Baseline 重新核验 SHA，另盘点 V100 多 seed 运行记录；它们属于本地复现而非作者配对权重。旧 Validation Baseline 还有 ATen/cuDNN 协议差异。实验二，以及 Baseline–CMERC 的错误、删除、分布和校准对比，因严格匹配证据不足保持待完成。完整检索范围、来源链接、元数据缺失项及补齐所需记录见 `baseline_checkpoint_audit.md` / `baseline_checkpoint_inventory.json` 和 `baseline_audit/`。

## 执行、核验与限制

- 服务器 `ssh biggpu` / user22，宿主机 GPU 1（UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`），进程内 cuda:0；禁用宿主机 GPU 4。
- 共 60 次 masked batch forward，新增 full forward 为 0，优化器 step 为 0。每个条件 strict=True 完整载入相同作者权重；eval/no_grad、模型参数和 BN buffer 不变、原权重和特征 SHA 不变。
- 原完整输入日志与清单保持不变；原官方评估只返回末 batch IDs 的问题继续由只读 observer 逐 batch 捕获全部 IDs，未更改原评估函数。
- 作者权重缺少训练轮次、种子及选模元数据。官方训练流程中测试集选模风险详见原验证协议报告和本轮 Baseline 审计；不能推断作者实际是否采用该流程。
- 固定 checkpoint 的描述性分析，不提供训练重复的方差或显著性；无严格 Baseline 时无法隔离 CMERC 的相对收益。
- 全部旧本地 Validation 实验和原官方性能验证文件由 `preserved_inputs.json` 的 156 项 SHA 清单保护；运行及文档一致性验证见 `verification.json`、`independent_verification.json`、`report_verification.json`。
