# CMERC 官方公开权重实验二：分类纠错与概率校准对照

本实验只读取 IEMOCAP 的两份既有官方公开权重预测，共1623条有效话语。不训练、不做GPU推理、不重新选checkpoint、不修改输入。以 dataset/dialogue_id/utterance_index 唯一匹配，重新核对原文、真实标签、source utterance ID、原始顺序和六类概率映射。

**这是作者分别公开的固定权重行为对照，严格训练配对尚未确认。** 实际训练seed、轮次、选模方式及训练配对关系未知，不能将差异归因于CL、HSCL或CC；不是严格模块消融实验。

所有CSV的Accuracy、F1、置信度、ECE、MCE、差值与比率均为0–1分数；Brier/NLL按原定义。Word会明确百分数或原分数单位。CSV的N/A代表分母为零或空箱，不是0。

## 一、全部话语的分类变化

| transition | count | ratio |
| --- | --- | --- |
| both_correct | 1083 | 0.667283 |
| baseline_wrong_cmerc_correct | 84 | 0.051756 |
| baseline_correct_cmerc_wrong | 93 | 0.057301 |
| both_wrong | 363 | 0.223660 |

CMERC纠正84条原错误，同时将93条原正确话语变错，错误净增加9条。纠错数不能脱离新增错误单独解释。

## 二、高置信预测和错误率

| threshold | baseline_high_confidence_predictions | baseline_high_confidence_errors | baseline_high_confidence_error_rate | cmerc_high_confidence_predictions | cmerc_high_confidence_errors | cmerc_high_confidence_error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.800000 | 1060 | 222 | 0.209434 | 1055 | 200 | 0.189573 |
| 0.900000 | 790 | 137 | 0.173418 | 796 | 128 | 0.160804 |
| 0.950000 | 605 | 88 | 0.145455 | 619 | 76 | 0.122779 |

高置信错误率分母是该模型所有达到阈值的预测数，不是错误数。以下前三种outcome互斥，合计各阈值原Baseline高置信错误；最后“仍超过原阈值”是与两种仍错状态可能重叠的独立统计，不可作为第四类相加。

| threshold | baseline_high_confidence_errors | corrected | still_wrong_confidence_lower | still_wrong_confidence_not_lower | still_wrong_above_threshold |
| --- | --- | --- | --- | --- | --- |
| 0.800000 | 222 | 16 | 134 | 72 | 138 |
| 0.900000 | 137 | 6 | 85 | 46 | 87 |
| 0.950000 | 88 | 2 | 60 | 26 | 49 |

这些outcome各自比例见CSV，分母固定为对应Baseline高置信错误数。CMERC新增高置信错误定义为Baseline原本不是高置信错误、CMERC变为高置信错误，包含原本正确与原本低置信错误两种来源。

| threshold | new_cmerc_hc_errors | eligible_non_baseline_hc_errors | new_cmerc_hc_error_ratio_of_eligible | new_cmerc_hc_errors_from_baseline_correct | new_cmerc_hc_errors_from_baseline_wrong_below_threshold |
| --- | --- | --- | --- | --- | --- |
| 0.800000 | 62 | 1401 | 0.044254 | 16 | 46 |
| 0.900000 | 41 | 1486 | 0.027591 | 5 | 36 |
| 0.950000 | 27 | 1535 | 0.017590 | 1 | 26 |

在阈值0.90下，原137条高置信错误中，6条被纠正、44条仍然错误但降到阈值以下、87条仍为高置信错误；另外新增41条高置信错误，所以最终是87+41=128条。高置信错误数量下降与分类纠错不是同一个统计量。

## 三、按真实情绪类别

| true_emotion | support | baseline_errors | cmerc_errors | corrected | regressed | both_wrong | net_error_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| happy | 144 | 49 | 52 | 8 | 11 | 41 | 3 |
| sad | 245 | 50 | 46 | 12 | 8 | 38 | -4 |
| neutral | 384 | 94 | 101 | 12 | 19 | 82 | 7 |
| angry | 170 | 66 | 56 | 15 | 5 | 51 | -10 |
| excited | 299 | 72 | 74 | 22 | 24 | 50 | 2 |
| frustrated | 381 | 116 | 127 | 15 | 26 | 101 | 11 |

net_error_change=CMERC errors−Baseline errors=regressed−corrected，正值表示错误增加。

## 四、概率校准与分类性能

| model | valid_utterances | accuracy | weighted_f1 | total_errors | mean_confidence | mean_error_confidence | ece | mce | brier_score | nll |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Official M³Net | 1623 | 0.724584 | 0.726601 | 447 | 0.832952 | 0.766322 | 0.115123 | 0.169779 | 0.432485 | 0.883990 |
| Official CMERC | 1623 | 0.719039 | 0.719838 | 456 | 0.831707 | 0.747381 | 0.112668 | 0.166292 | 0.429447 | 0.869046 |

两份权重的W-F1分别为72.660131%和71.983794%，CMERC低0.676338个百分点；ECE、MCE、Brier、NLL的CMERC−Baseline变化依次为-0.002454688、-0.003487035、-0.003038223、-0.014943267。这组数据表现为分类略差、所列校准/概率指标略好，不能把校准改善写成整体分类纠错提高。没有跨seed重复，不能据此宣称统计显著或模块因果收益。

错误平均置信度分别基于各自447条和456条错误集合，不能单凭均值差推断同一批错误均降低了置信度；同话语变化应以配对记录为准。

定义与原实验六完全一致：ECE按10等宽箱的样本权重加权绝对置信度–准确率差；MCE是非空箱最大绝对差；Brier为各类别平方误差之和的样本均值，不除类别数或2；NLL用原真实类别概率的自然对数，不裁剪、不归一化。Signed Gap=mean confidence−accuracy；空箱不绘制观测点，可靠性图同时显示理想对角线及样本量。

## 五、典型样本与完整概率

`paired_predictions.csv`保留全部1623条与两模型全部12个概率、最大置信度差、真实类别概率差、原Baseline预测类别在CMERC中的概率差和预测类别是否改变。最大置信度下降不意味着同一错误类别的概率下降，也不等于已纠正。

`typical_cases.csv`按三个阈值保留四种案例的全部匹配样本：原高置信错误被纠正；仍错但最大置信度下降；Baseline正确而CMERC新增高置信错误；两者均高置信错误。类别与阈值重叠时可重复。Word每类在0.90阈值按selection_score降序摘录前2条，稳定ID破同分，正反案例同规则展示；全量原文和概率始终在CSV中，不以Top2代替全部样本。

## 六、复核、来源与范围

两份原始预测和权重的SHA-256、统计源码、协议见`run_manifest.json`；独立重算与参考值差异检查见`independent_verification.json`及`verification.json`。参考值只用于计算后核对，不参与生成。旧实验文件SHA由`preserved_inputs.json`保护。无MELD分析、无新的模型推理。图和Word均为本轮新增，不覆盖旧报告。
