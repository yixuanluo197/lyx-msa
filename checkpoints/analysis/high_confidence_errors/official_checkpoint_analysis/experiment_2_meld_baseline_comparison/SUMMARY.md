# MELD 实验二：本地 README Baseline 与作者官方 CMERC 对照

本轮先审计再固定首选 `experiments/official_readme_reproduction/checkpoints/MELD_baseline_seed67137/best.pt`，依据身份和技术兼容性选择，没有按测试成绩比较候选。Baseline 是本地按官方 README 复现的M³Net结构，**不是作者公开的MELD Baseline**；训练seed=67137、epoch=5、历史选模=`official_test_selection`。它有测试集选模偏差，不能称为独立Validation选择。

源权重SHA `a6ab85cb6c0c12054dade9c6bcdf01cfaa6abc417625059a927de5ba386d3738`。83张量与官方MELD结构strict=True完全匹配，calibrate/contrastlearning/courselearning均关闭。纯模型state导出SHA `1d446d95fc21d14fd550e20bed939e0510d7ac343284507dad9f40a4e2217c6d`，张量内容与源model逐值一致，原始完整权重未修改。详见`MELD_baseline_checkpoint_audit.md`。

训练发生于本地XP；本轮按用户要求统一到已成功验证官方CMERC的biggpu V100/cuDNN、原冻结代码和两处既有PyG兼容补丁。仅首选Baseline进行了18批次完整T+A+V前向；CMERC直接复用既有官方MELD预测，未训练、未改参数、未重新选权重、未删除模态。模型参数/BN缓冲区前后哈希不变，optimizer step=0。

## 一、真实性能和全量对齐

以下Accuracy和F1单位为百分数；该表对应`baseline_evaluation_results.csv`，其他统计CSV统一使用0–1分数。

| model | accuracy | weighted_f1 | macro_f1 | valid_samples |
| --- | --- | --- | --- | --- |
| Local README Baseline (test-selected) | 67.547893 | 66.225692 | 50.118559 | 2610 |
| Official CMERC | 67.777778 | 66.848392 | 51.077023 | 2610 |

两模型2610条有效话语的dataset/dialogue_id/utterance_index、原文、真实标签、source utterance ID、原始测试顺序和七类概率字段严格一致。batch16、num_workers2、原Padding/Mask、GRU、hyper、concat_DHT、dropout.4、L3/K3、BN、avl、cuDNN后端和测试argv一致；评估seed1475不等于Baseline训练seed67137。

## 二、分类纠错

| transition | count | ratio |
| --- | --- | --- |
| both_correct | 1715 | 0.657088 |
| baseline_wrong_cmerc_correct | 54 | 0.020690 |
| baseline_correct_cmerc_wrong | 48 | 0.018391 |
| both_wrong | 793 | 0.303831 |

CMERC纠正54条原错误，新增48条分类错误；错误净变化为-6条。纠错和新增错误必须同时解释。

## 三、原始高置信阈值：0.80、0.90、0.95

| threshold | baseline_high_confidence_predictions | baseline_high_confidence_errors | baseline_high_confidence_error_rate | cmerc_high_confidence_predictions | cmerc_high_confidence_errors | cmerc_high_confidence_error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.800000 | 0 | 0 | N/A | 2 | 0 | 0.000000 |
| 0.900000 | 0 | 0 | N/A | 0 | 0 | N/A |
| 0.950000 | 0 | 0 | N/A | 0 | 0 | N/A |

| threshold | corrected | still_wrong_confidence_lower | still_wrong_confidence_not_lower | still_wrong_above_threshold | new_cmerc_hc_errors | new_cmerc_hc_errors_from_baseline_correct |
| --- | --- | --- | --- | --- | --- | --- |
| 0.800000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.900000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.950000 | 0 | 0 | 0 | 0 | 0 | 0 |

达到阈值的预测数为零时，错误率是N/A，不是0%。原CMERC在MELD上≥0.80只有2条预测且均正确、≥0.90与≥0.95无预测；这些阈值没有高置信错误不能证明其校准完美或鲁棒。

## 四、预先固定辅助阈值：0.40、0.50、0.60、0.70

| threshold | baseline_high_confidence_predictions | baseline_high_confidence_errors | baseline_high_confidence_error_rate | cmerc_high_confidence_predictions | cmerc_high_confidence_errors | cmerc_high_confidence_error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400000 | 1496 | 271 | 0.181150 | 1647 | 342 | 0.207650 |
| 0.500000 | 809 | 90 | 0.111248 | 977 | 129 | 0.132037 |
| 0.600000 | 159 | 9 | 0.056604 | 324 | 30 | 0.092593 |
| 0.700000 | 0 | 0 | N/A | 43 | 1 | 0.023256 |

| threshold | corrected | still_wrong_confidence_lower | still_wrong_confidence_not_lower | still_wrong_above_threshold | new_cmerc_hc_errors | new_cmerc_hc_errors_from_baseline_correct |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400000 | 1 | 100 | 170 | 250 | 92 | 2 |
| 0.500000 | 0 | 28 | 62 | 80 | 49 | 0 |
| 0.600000 | 0 | 2 | 7 | 7 | 23 | 0 |
| 0.700000 | 0 | 0 | 0 | 0 | 1 | 0 |

辅助阈值在读取本轮Baseline预测前按用户要求固定，与主阈值分表，不替代原始三阈值。corrected/lower/not_lower是针对各阈值原Baseline高置信错误的互斥三类；still_wrong_above_threshold是独立重叠统计。高置信错误率的分母是所有达到阈值的预测，不是错误总数。新增CMERC高置信错误来自原Baseline正确或原低置信错误，两种来源与明确分母完整保留在CSV。

在阈值0.60下，原9条高置信错误中，0条被纠正、2条仍然错误但降到阈值以下、7条仍为高置信错误；另外新增23条高置信错误，所以最终是7+23=30条。高置信错误数量下降与分类纠错不是同一个统计量。

## 五、七类真实情绪

| true_emotion | support | baseline_errors | cmerc_errors | corrected | regressed | both_wrong | net_error_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1256 | 177 | 206 | 4 | 33 | 173 | 29 |
| surprise | 281 | 115 | 110 | 8 | 3 | 107 | -5 |
| fear | 50 | 40 | 40 | 0 | 0 | 40 | 0 |
| sadness | 208 | 149 | 140 | 11 | 2 | 138 | -9 |
| joy | 402 | 154 | 132 | 24 | 2 | 130 | -22 |
| disgust | 68 | 49 | 47 | 3 | 1 | 46 | -2 |
| anger | 345 | 163 | 166 | 4 | 7 | 159 | 3 |

net_error_change=CMERC errors−Baseline errors，正值为错误增加。fear/sadness/disgust均按实际数据呈现，不预设改善。

## 六、全量概率校准

| model | valid_utterances | accuracy | weighted_f1 | mean_confidence | mean_error_confidence | ece | mce | brier_score | nll |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Local README Baseline (test-selected) | 2610 | 0.675479 | 0.662257 | 0.427979 | 0.363830 | 0.247775 | 0.331598 | 0.539553 | 1.160414 |
| Official CMERC | 2610 | 0.677778 | 0.668484 | 0.449601 | 0.382656 | 0.228176 | 0.300950 | 0.529921 | 1.140703 |

CMERC−Baseline的W-F1为+0.622700个百分点；ECE/MCE/Brier/NLL的差依次为-0.019598301/-0.030647637/-0.009632222/-0.019711152。各错误均值置信度分别基于自己的847/841条错误，不能替代同话语配对分析。

沿用此前实验六：全部有效话语10等宽左闭右开箱，最后包含1；ECE按箱占比加权绝对平均置信度与准确率差，MCE取非空箱最大绝对差；Brier为所有类别平方误差之和再样本平均，不除类别数或2；NLL为原真实类别概率的自然对数负均值，不裁剪、不温度缩放、不归一化。Signed Gap=mean confidence−accuracy，空箱观测N/A，不画虚假点。

官方CMERC平均置信度44.9601%而准确率67.7778%，明显置信度不足；应结合辅助阈值与可靠性图，不能把无≥0.90预测当成完美可靠性。

## 七、典型样本与完整明细

`paired_predictions.csv`保留2610条两模型全部14个类别概率，最大置信度变化、真实类别概率变化和原预测类别概率变化独立保存；置信度下降不等于分类已纠正。`typical_cases.csv`保存四种案例、全部三个主阈值与四个辅助阈值的全部匹配；阈值及案例类型可以重叠。Word案例预先固定辅助0.60阈值，每种类型按selection_score降序前2条，稳定ID破同分；空类型如实标N/A，不以摘录代替全量数据。

## 八、可比性与综合报告范围

第一层：相同测试协议下的固定权重描述性行为对比技术可行。第二层：两份模型缺少严格训练配对，Baseline又有历史测试集选模风险，不能将分类或校准差异直接归因于CL/HSCL/CC。作者官方CMERC实际训练seed/epoch/选模历史未知，不补写成Validation选模。

IEMOCAP实验二仍使用作者公开的Roberta_IEMOCAP3.pth.tar；MELD采用本轮明确来源的本地README复现Baseline，二者来源不同，不能合称同一组严格训练配对实验。综合Word保留原官方CMERC实验一/三/四/五/六的全部核实内容，追加此前IEMOCAP实验二和本轮MELD实验二，另存新文件，不修改旧Word。

## 九、复核和文件

`verification.json`汇总原始标签/文本/ID、概率/分类指标/分箱、独立逐表复算、旧报告和原模型权重SHA不变以及Word数字一致性。`baseline_classification_report.csv`和`baseline_confusion_matrix.csv`保存七类实际性能；`baseline_predictions.csv/.npz`是本次新推理，`baseline_evaluation_results.csv`为与官方CMERC性能对照，其百分数单位独立注明。模型推理清单、真实日志和统计代码版本见`baseline_inference/`、`evaluation.log`、`run_manifest.json`与`source_code/`。
