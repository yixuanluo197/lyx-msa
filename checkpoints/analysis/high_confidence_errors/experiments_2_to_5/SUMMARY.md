# 完整三模态高置信度错误分析

仅分析测试集，未重新训练、未执行模态掩码或上下文移除。主分析使用已完成 XP validation 批次、预先固定的公开 seed；不按测试分数挑选 seed 或 checkpoint。

## 总体统计

高置信度错误比例的分母是所有达到该阈值的预测，包括正确预测；表中的概率阈值为 0–1。

| 数据集 | 模型 | 测试话语 | 错误总数 | ≥0.8 预测数 | ≥0.8 错误数 | 错误比例 | ≥0.9 预测数 | ≥0.9 错误数 | 错误比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IEMOCAP | baseline | 1623 | 538 | 1051 | 242 | 23.03% | 808 | 154 | 19.06% |
| IEMOCAP | cmerc | 1623 | 518 | 634 | 91 | 14.35% | 362 | 37 | 10.22% |
| MELD | baseline | 2610 | 874 | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | cmerc | 2610 | 869 | 4 | 0 | 0.00% | 0 | 0 | N/A |

## 主要发现

- **IEMOCAP 存在大量高置信度错误。** Baseline 的 >=0.8 / >=0.9 错误分别为 242 / 154 条，CMERC 为 91 / 37 条；对应高置信预测内错误比例由 23.03% / 19.06% 变为 14.35% / 10.22%。这些结果仅针对本次固定 checkpoint。
- **高置信错误减少不等于同等数量的错误被纠正。** IEMOCAP 总错误只从 538 降到 518；Baseline 的 154 条 >=0.9 错误中，只有 5 条被 CMERC 改判正确。需要结合逐样本对比区分错误纠正与置信度降低，不能把 154→37 解释为纠正了 117 条分类错误。
- **MELD 在这两个 checkpoint 下没有 >=0.8 或 >=0.9 的错误样本。** Baseline 没有任何 >=0.8 的预测；CMERC 只有 4 条且全部正确，>=0.9 预测为零。因此空分母记 N/A，不能解释为所有预测无错误或模型完美校准。MELD 仍有 874 / 869 条错误；典型样本部分列的是全部错误中概率最高的样本，不冒充达到阈值的高置信错误。
- 类别方面，IEMOCAP Baseline 与 CMERC 的错误都以 frustrated 为最多（181 / 161 条）；MELD Baseline 的 fear 类内错误率为 82%，CMERC 的 fear 同为 82%、disgust 为 80.88%。应同时查看类别支持数，不能只按绝对错误条数比较不同类别。

## 逐话语对比

- IEMOCAP：同一 ID/索引、相同真实标签与原文，匹配 1623 条。Baseline 错而 CMERC 对 109 条；Baseline 对而 CMERC 错 89 条；两者均错 429 条；两者均对 996 条。
  - 阈值 0.8：Baseline 高置信错 242 条，CMERC 高置信错 91 条；前者中有 12 条被 CMERC 正确分类。
  - 阈值 0.9：Baseline 高置信错 154 条，CMERC 高置信错 37 条；前者中有 5 条被 CMERC 正确分类。
- MELD：同一 ID/索引、相同真实标签与原文，匹配 2610 条。Baseline 错而 CMERC 对 81 条；Baseline 对而 CMERC 错 76 条；两者均错 793 条；两者均对 1660 条。
  - 阈值 0.8：Baseline 高置信错 0 条，CMERC 高置信错 0 条；前者中有 0 条被 CMERC 正确分类。
  - 阈值 0.9：Baseline 高置信错 0 条，CMERC 高置信错 0 条；前者中有 0 条被 CMERC 正确分类。

高置信错误数的变化同时受总错误数与置信度分布影响；不能只凭固定阈值下的数量判断整体准确率或校准性能。

## 类别错误分布与典型样本

每个模型的 class_error_distribution.csv 提供真实情绪类别分布；statistics.json 另保留混淆对。以下按错误置信度降序展示，每组最多 3 条，不用于选择模型。CSV 保存完整原文；本页仅转义控制字符并截取前 240 字符。

### IEMOCAP/baseline

| 真实情绪 | 测试支持数 | 错误数 | 类内错误率 | ≥0.8 错误数 | ≥0.9 错误数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| happy | 144 | 53 | 36.81% | 19 | 12 |
| sad | 245 | 45 | 18.37% | 19 | 10 |
| neutral | 384 | 89 | 23.18% | 38 | 30 |
| angry | 170 | 49 | 28.82% | 27 | 7 |
| excited | 299 | 121 | 40.47% | 57 | 38 |
| frustrated | 381 | 181 | 47.51% | 82 | 57 |

| 对话 ID | 索引（0-based） | 真实 → 预测 | 最大概率 | 真实类概率 | 原文 |
| --- | ---: | --- | ---: | ---: | --- |
| Ses05F_impro06 | 44 | neutral → sad | 0.999968 | 0.000032 | Do you know what, it's going to pass,eventually, I mean it'll never pass, it'll always be there, and you'll always remember and everything like that, but you know.  It'll get easier I guess. |
| Ses05F_impro06 | 42 | neutral → sad | 0.999878 | 0.000122 | yeah. |
| Ses05F_impro06 | 50 | neutral → sad | 0.999847 | 0.000152 | Do you know what, if it was me that died, I'd really want you to be crying and being really upset too any way. So, I think you should definitely do this, because it's trying what it's supposed to be. |

最常见错误混淆对：frustrated → neutral，90 条。

### IEMOCAP/cmerc

| 真实情绪 | 测试支持数 | 错误数 | 类内错误率 | ≥0.8 错误数 | ≥0.9 错误数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| happy | 144 | 56 | 38.89% | 6 | 3 |
| sad | 245 | 50 | 20.41% | 10 | 4 |
| neutral | 384 | 108 | 28.12% | 30 | 13 |
| angry | 170 | 56 | 32.94% | 5 | 0 |
| excited | 299 | 87 | 29.10% | 6 | 2 |
| frustrated | 381 | 161 | 42.26% | 34 | 15 |

| 对话 ID | 索引（0-based） | 真实 → 预测 | 最大概率 | 真实类概率 | 原文 |
| --- | ---: | --- | ---: | ---: | --- |
| Ses05F_impro06 | 44 | neutral → sad | 0.999134 | 0.000842 | Do you know what, it's going to pass,eventually, I mean it'll never pass, it'll always be there, and you'll always remember and everything like that, but you know.  It'll get easier I guess. |
| Ses05F_impro06 | 42 | neutral → sad | 0.996475 | 0.003485 | yeah. |
| Ses05F_impro06 | 50 | neutral → sad | 0.995817 | 0.003802 | Do you know what, if it was me that died, I'd really want you to be crying and being really upset too any way. So, I think you should definitely do this, because it's trying what it's supposed to be. |

最常见错误混淆对：frustrated → neutral，86 条。

### MELD/baseline

| 真实情绪 | 测试支持数 | 错误数 | 类内错误率 | ≥0.8 错误数 | ≥0.9 错误数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| neutral | 1256 | 205 | 16.32% | 0 | 0 |
| surprise | 281 | 119 | 42.35% | 0 | 0 |
| fear | 50 | 41 | 82.00% | 0 | 0 |
| sadness | 208 | 145 | 69.71% | 0 | 0 |
| joy | 402 | 154 | 38.31% | 0 | 0 |
| disgust | 68 | 51 | 75.00% | 0 | 0 |
| anger | 345 | 159 | 46.09% | 0 | 0 |

| 对话 ID | 索引（0-based） | 真实 → 预测 | 最大概率 | 真实类概率 | 原文 |
| --- | ---: | --- | ---: | ---: | --- |
| 1178 | 12 | neutral → joy | 0.628579 | 0.071658 | Oh absolutely. I like it even more on you than I did on Colonel Sanders.  Ross! Ross! I\u0092m kidding! |
| 1393 | 15 | sadness → neutral | 0.619844 | 0.060407 | Oh uh, one of the writers. |
| 1194 | 4 | anger → neutral | 0.607893 | 0.054340 | It\u0092s Joshua. |

最常见错误混淆对：joy → neutral，94 条。

### MELD/cmerc

| 真实情绪 | 测试支持数 | 错误数 | 类内错误率 | ≥0.8 错误数 | ≥0.9 错误数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| neutral | 1256 | 235 | 18.71% | 0 | 0 |
| surprise | 281 | 102 | 36.30% | 0 | 0 |
| fear | 50 | 41 | 82.00% | 0 | 0 |
| sadness | 208 | 140 | 67.31% | 0 | 0 |
| joy | 402 | 134 | 33.33% | 0 | 0 |
| disgust | 68 | 55 | 80.88% | 0 | 0 |
| anger | 345 | 162 | 46.96% | 0 | 0 |

| 对话 ID | 索引（0-based） | 真实 → 预测 | 最大概率 | 真实类概率 | 原文 |
| --- | ---: | --- | ---: | ---: | --- |
| 1178 | 12 | neutral → joy | 0.710140 | 0.051317 | Oh absolutely. I like it even more on you than I did on Colonel Sanders.  Ross! Ross! I\u0092m kidding! |
| 1194 | 4 | anger → neutral | 0.697405 | 0.035744 | It\u0092s Joshua. |
| 1342 | 6 | neutral → joy | 0.694927 | 0.113111 | I think it\u0092s great. It\u0092s great. Y\u0092know, they\u0092re thinking of changing the name of this place. |

最常见错误混淆对：neutral → joy，101 条。

## 解释与限制

- 这是固定 checkpoint 的描述性分析。高置信度错误体现 softmax 置信度与正确性不一致，不代表已证明某个训练模块的因果效果；两个模型不保证置信度校准尺度相同。
- 统计包含全部有效测试话语。只筛选错误不会改变真实标签或模型概率，不能从此表直接推断所有预测的准确率。
- 原 train.py 以及 official_test 流程存在逐轮测试集选模问题；官方 README 复现的 best 也按测试 W-F1 选择。本次排除这些 checkpoint 作为主对比，选用 metadata 明确记载 validation 的 checkpoint。
- Baseline 与 CMERC 主模型结构相同，身份以内部 calibrate/contrastlearning/courselearning 开关和运行 manifest 为据。checkpoints/ 下两份作者 plain state_dict 无内部训练配置，清单标为身份待确认，没有冒充 Baseline。
- MELD dialogue_id 是特征 pickle 内部全局编号，未经映射不能当作原始 CSV Dialogue_ID。utterance_index 从 0 计数，source_utterance_id 另保留 pickle 中的原始 ID。
- 保留原始 batch16 和测试话语排列：未 pack 的双向 RNN、padding 和图层批次位置可能影响预测，不任意调 batch 或按长度重排。
- 原文来自特征包 videoSentence，不做拼写或编码修正；MELD 部分原文带控制字符，CSV 原样保存。
- 所有 checkpoint、输入和已有预测的 SHA-256 在推理前后核对一致；完整来源、环境、逐 batch 对齐及预测一致性见 run_manifest.json。


---

# 后续实验二至五

实验一以上内容从原 SUMMARY.md 完整复制；原始文件未修改。所有比率、Accuracy 和 Weighted F1 在后续 CSV 中以 0–1 表示。
沿用实验一同一四个 validation checkpoint 和原测试话语，完整输入不重新推理。只新增三种官方均值替换推理，不训练、不选模、不调参。

## 实验二：Baseline 高置信错误的改善与新增错误

| dataset | threshold | valid_test_utterances | baseline_high_confidence_errors | new_cmerc_high_confidence_errors | new_error_eligible_count | new_error_ratio | corrected_count | corrected_ratio | still_wrong_confidence_lower_count | still_wrong_confidence_lower_ratio | still_wrong_confidence_not_lower_count | still_wrong_confidence_not_lower_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 0.800000 | 1623 | 242 | 10 | 1381 | 0.007241 | 12 | 0.049587 | 227 | 0.938017 | 3 | 0.012397 |
| IEMOCAP | 0.900000 | 1623 | 154 | 3 | 1469 | 0.002042 | 5 | 0.032468 | 147 | 0.954545 | 2 | 0.012987 |
| IEMOCAP | 0.950000 | 1623 | 103 | 0 | 1520 | 0.000000 | 1 | 0.009709 | 101 | 0.980583 | 1 | 0.009709 |
| MELD | 0.800000 | 2610 | 0 | 0 | 2610 | 0.000000 | 0 | N/A | 0 | N/A | 0 | N/A |
| MELD | 0.900000 | 2610 | 0 | 0 | 2610 | 0.000000 | 0 | N/A | 0 | N/A | 0 | N/A |
| MELD | 0.950000 | 2610 | 0 | 0 | 2610 | 0.000000 | 0 | N/A | 0 | N/A | 0 | N/A |

## 实验三：CMERC 剩余高置信错误及类别分布

| dataset | threshold | true_label | true_emotion | class_support | total_errors | class_error_rate | high_confidence_error_count | high_confidence_prediction_count | high_confidence_error_rate | high_confidence_error_fraction_of_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 0.800000 | 0 | happy | 144 | 56 | 0.388889 | 6 | 37 | 0.162162 | 0.041667 |
| IEMOCAP | 0.800000 | 1 | sad | 245 | 50 | 0.204082 | 10 | 153 | 0.065359 | 0.040816 |
| IEMOCAP | 0.800000 | 2 | neutral | 384 | 108 | 0.281250 | 30 | 183 | 0.163934 | 0.078125 |
| IEMOCAP | 0.800000 | 3 | angry | 170 | 56 | 0.329412 | 5 | 81 | 0.061728 | 0.029412 |
| IEMOCAP | 0.800000 | 4 | excited | 299 | 87 | 0.290970 | 6 | 99 | 0.060606 | 0.020067 |
| IEMOCAP | 0.800000 | 5 | frustrated | 381 | 161 | 0.422572 | 34 | 81 | 0.419753 | 0.089239 |
| IEMOCAP | 0.900000 | 0 | happy | 144 | 56 | 0.388889 | 3 | 15 | 0.200000 | 0.020833 |
| IEMOCAP | 0.900000 | 1 | sad | 245 | 50 | 0.204082 | 4 | 124 | 0.032258 | 0.016327 |
| IEMOCAP | 0.900000 | 2 | neutral | 384 | 108 | 0.281250 | 13 | 86 | 0.151163 | 0.033854 |
| IEMOCAP | 0.900000 | 3 | angry | 170 | 56 | 0.329412 | 0 | 63 | 0.000000 | 0.000000 |
| IEMOCAP | 0.900000 | 4 | excited | 299 | 87 | 0.290970 | 2 | 51 | 0.039216 | 0.006689 |
| IEMOCAP | 0.900000 | 5 | frustrated | 381 | 161 | 0.422572 | 15 | 23 | 0.652174 | 0.039370 |
| IEMOCAP | 0.950000 | 0 | happy | 144 | 56 | 0.388889 | 2 | 3 | 0.666667 | 0.013889 |
| IEMOCAP | 0.950000 | 1 | sad | 245 | 50 | 0.204082 | 2 | 97 | 0.020619 | 0.008163 |
| IEMOCAP | 0.950000 | 2 | neutral | 384 | 108 | 0.281250 | 10 | 27 | 0.370370 | 0.026042 |
| IEMOCAP | 0.950000 | 3 | angry | 170 | 56 | 0.329412 | 0 | 50 | 0.000000 | 0.000000 |
| IEMOCAP | 0.950000 | 4 | excited | 299 | 87 | 0.290970 | 0 | 26 | 0.000000 | 0.000000 |
| IEMOCAP | 0.950000 | 5 | frustrated | 381 | 161 | 0.422572 | 4 | 4 | 1.000000 | 0.010499 |
| MELD | 0.800000 | 0 | neutral | 1256 | 235 | 0.187102 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.800000 | 1 | surprise | 281 | 102 | 0.362989 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.800000 | 2 | fear | 50 | 41 | 0.820000 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.800000 | 3 | sadness | 208 | 140 | 0.673077 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.800000 | 4 | joy | 402 | 134 | 0.333333 | 0 | 4 | 0.000000 | 0.000000 |
| MELD | 0.800000 | 5 | disgust | 68 | 55 | 0.808824 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.800000 | 6 | anger | 345 | 162 | 0.469565 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 0 | neutral | 1256 | 235 | 0.187102 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 1 | surprise | 281 | 102 | 0.362989 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 2 | fear | 50 | 41 | 0.820000 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 3 | sadness | 208 | 140 | 0.673077 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 4 | joy | 402 | 134 | 0.333333 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 5 | disgust | 68 | 55 | 0.808824 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.900000 | 6 | anger | 345 | 162 | 0.469565 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 0 | neutral | 1256 | 235 | 0.187102 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 1 | surprise | 281 | 102 | 0.362989 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 2 | fear | 50 | 41 | 0.820000 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 3 | sadness | 208 | 140 | 0.673077 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 4 | joy | 402 | 134 | 0.333333 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 5 | disgust | 68 | 55 | 0.808824 | 0 | 0 | N/A | 0.000000 |
| MELD | 0.950000 | 6 | anger | 345 | 162 | 0.469565 | 0 | 0 | N/A | 0.000000 |

## 实验四：单一模态均值替换

| dataset | model | input_condition | valid_utterances | total_errors | mean_error_confidence | errors_ge_0p8 | high_confidence_predictions_ge_0p8 | high_confidence_error_rate_ge_0p8 | errors_ge_0p9 | high_confidence_predictions_ge_0p9 | high_confidence_error_rate_ge_0p9 | errors_ge_0p95 | high_confidence_predictions_ge_0p95 | high_confidence_error_rate_ge_0p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | baseline | full | 1623 | 538 | 0.750002 | 242 | 1051 | 0.230257 | 154 | 808 | 0.190594 | 103 | 630 | 0.163492 |
| IEMOCAP | baseline | w/o_text | 1623 | 666 | 0.758537 | 317 | 999 | 0.317317 | 197 | 737 | 0.267300 | 133 | 537 | 0.247672 |
| IEMOCAP | baseline | w/o_audio | 1623 | 593 | 0.719030 | 231 | 960 | 0.240625 | 119 | 661 | 0.180030 | 57 | 427 | 0.133489 |
| IEMOCAP | baseline | w/o_visual | 1623 | 535 | 0.746424 | 238 | 1044 | 0.227969 | 148 | 797 | 0.185696 | 100 | 626 | 0.159744 |
| IEMOCAP | cmerc | full | 1623 | 518 | 0.615968 | 91 | 634 | 0.143533 | 37 | 362 | 0.102210 | 18 | 207 | 0.086957 |
| IEMOCAP | cmerc | w/o_text | 1623 | 631 | 0.652079 | 132 | 601 | 0.219634 | 64 | 317 | 0.201893 | 30 | 188 | 0.159574 |
| IEMOCAP | cmerc | w/o_audio | 1623 | 596 | 0.584634 | 76 | 473 | 0.160677 | 22 | 195 | 0.112821 | 4 | 50 | 0.080000 |
| IEMOCAP | cmerc | w/o_visual | 1623 | 519 | 0.614286 | 89 | 634 | 0.140379 | 37 | 364 | 0.101648 | 17 | 210 | 0.080952 |
| MELD | baseline | full | 2610 | 874 | 0.346139 | 0 | 0 | N/A | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | baseline | w/o_text | 2610 | 1347 | 0.299560 | 0 | 0 | N/A | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | baseline | w/o_audio | 2610 | 884 | 0.338825 | 0 | 0 | N/A | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | baseline | w/o_visual | 2610 | 869 | 0.346154 | 0 | 0 | N/A | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | cmerc | full | 2610 | 869 | 0.408199 | 0 | 4 | 0.000000 | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | cmerc | w/o_text | 2610 | 1310 | 0.378305 | 0 | 0 | N/A | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | cmerc | w/o_audio | 2610 | 889 | 0.394890 | 0 | 4 | 0.000000 | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | cmerc | w/o_visual | 2610 | 880 | 0.406848 | 0 | 2 | 0.000000 | 0 | 0 | N/A | 0 | 0 | N/A |

## 实验四：完整输入正确到替换后高置信错误

| dataset | model | input_condition | removed_modality | threshold | full_correct_samples | new_high_confidence_error_count | ratio | full_wrong_samples | full_wrong_still_wrong_count | full_wrong_still_wrong_high_confidence_count | full_wrong_still_wrong_high_confidence_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | baseline | w/o_text | text | 0.800000 | 1085 | 83 | 0.076498 | 538 | 435 | 234 | 0.434944 |
| IEMOCAP | baseline | w/o_text | text | 0.900000 | 1085 | 36 | 0.033180 | 538 | 435 | 161 | 0.299257 |
| IEMOCAP | baseline | w/o_text | text | 0.950000 | 1085 | 22 | 0.020276 | 538 | 435 | 111 | 0.206320 |
| IEMOCAP | baseline | w/o_audio | audio | 0.800000 | 1085 | 38 | 0.035023 | 538 | 403 | 193 | 0.358736 |
| IEMOCAP | baseline | w/o_audio | audio | 0.900000 | 1085 | 9 | 0.008295 | 538 | 403 | 110 | 0.204461 |
| IEMOCAP | baseline | w/o_audio | audio | 0.950000 | 1085 | 3 | 0.002765 | 538 | 403 | 54 | 0.100372 |
| IEMOCAP | baseline | w/o_visual | visual | 0.800000 | 1085 | 0 | 0.000000 | 538 | 531 | 238 | 0.442379 |
| IEMOCAP | baseline | w/o_visual | visual | 0.900000 | 1085 | 0 | 0.000000 | 538 | 531 | 148 | 0.275093 |
| IEMOCAP | baseline | w/o_visual | visual | 0.950000 | 1085 | 0 | 0.000000 | 538 | 531 | 100 | 0.185874 |
| IEMOCAP | cmerc | w/o_text | text | 0.800000 | 1105 | 27 | 0.024434 | 518 | 420 | 105 | 0.202703 |
| IEMOCAP | cmerc | w/o_text | text | 0.900000 | 1105 | 13 | 0.011765 | 518 | 420 | 51 | 0.098456 |
| IEMOCAP | cmerc | w/o_text | text | 0.950000 | 1105 | 5 | 0.004525 | 518 | 420 | 25 | 0.048263 |
| IEMOCAP | cmerc | w/o_audio | audio | 0.800000 | 1105 | 3 | 0.002715 | 518 | 401 | 73 | 0.140927 |
| IEMOCAP | cmerc | w/o_audio | audio | 0.900000 | 1105 | 0 | 0.000000 | 518 | 401 | 22 | 0.042471 |
| IEMOCAP | cmerc | w/o_audio | audio | 0.950000 | 1105 | 0 | 0.000000 | 518 | 401 | 4 | 0.007722 |
| IEMOCAP | cmerc | w/o_visual | visual | 0.800000 | 1105 | 0 | 0.000000 | 518 | 514 | 89 | 0.171815 |
| IEMOCAP | cmerc | w/o_visual | visual | 0.900000 | 1105 | 0 | 0.000000 | 518 | 514 | 37 | 0.071429 |
| IEMOCAP | cmerc | w/o_visual | visual | 0.950000 | 1105 | 0 | 0.000000 | 518 | 514 | 17 | 0.032819 |
| MELD | baseline | w/o_text | text | 0.800000 | 1736 | 0 | 0.000000 | 874 | 678 | 0 | 0.000000 |
| MELD | baseline | w/o_text | text | 0.900000 | 1736 | 0 | 0.000000 | 874 | 678 | 0 | 0.000000 |
| MELD | baseline | w/o_text | text | 0.950000 | 1736 | 0 | 0.000000 | 874 | 678 | 0 | 0.000000 |
| MELD | baseline | w/o_audio | audio | 0.800000 | 1736 | 0 | 0.000000 | 874 | 850 | 0 | 0.000000 |
| MELD | baseline | w/o_audio | audio | 0.900000 | 1736 | 0 | 0.000000 | 874 | 850 | 0 | 0.000000 |
| MELD | baseline | w/o_audio | audio | 0.950000 | 1736 | 0 | 0.000000 | 874 | 850 | 0 | 0.000000 |
| MELD | baseline | w/o_visual | visual | 0.800000 | 1736 | 0 | 0.000000 | 874 | 859 | 0 | 0.000000 |
| MELD | baseline | w/o_visual | visual | 0.900000 | 1736 | 0 | 0.000000 | 874 | 859 | 0 | 0.000000 |
| MELD | baseline | w/o_visual | visual | 0.950000 | 1736 | 0 | 0.000000 | 874 | 859 | 0 | 0.000000 |
| MELD | cmerc | w/o_text | text | 0.800000 | 1741 | 0 | 0.000000 | 869 | 653 | 0 | 0.000000 |
| MELD | cmerc | w/o_text | text | 0.900000 | 1741 | 0 | 0.000000 | 869 | 653 | 0 | 0.000000 |
| MELD | cmerc | w/o_text | text | 0.950000 | 1741 | 0 | 0.000000 | 869 | 653 | 0 | 0.000000 |
| MELD | cmerc | w/o_audio | audio | 0.800000 | 1741 | 0 | 0.000000 | 869 | 850 | 0 | 0.000000 |
| MELD | cmerc | w/o_audio | audio | 0.900000 | 1741 | 0 | 0.000000 | 869 | 850 | 0 | 0.000000 |
| MELD | cmerc | w/o_audio | audio | 0.950000 | 1741 | 0 | 0.000000 | 869 | 850 | 0 | 0.000000 |
| MELD | cmerc | w/o_visual | visual | 0.800000 | 1741 | 0 | 0.000000 | 869 | 863 | 0 | 0.000000 |
| MELD | cmerc | w/o_visual | visual | 0.900000 | 1741 | 0 | 0.000000 | 869 | 863 | 0 | 0.000000 |
| MELD | cmerc | w/o_visual | visual | 0.950000 | 1741 | 0 | 0.000000 | 869 | 863 | 0 | 0.000000 |

## 实验五：错误置信度区间分布

| dataset | confidence_bin | baseline_error_count | baseline_ratio_of_errors | cmerc_error_count | cmerc_ratio_of_errors |
| --- | --- | --- | --- | --- | --- |
| IEMOCAP | [0, 0.5) | 56 | 0.104089 | 156 | 0.301158 |
| IEMOCAP | [0.5, 0.6) | 80 | 0.148699 | 110 | 0.212355 |
| IEMOCAP | [0.6, 0.7) | 78 | 0.144981 | 97 | 0.187259 |
| IEMOCAP | [0.7, 0.8) | 82 | 0.152416 | 64 | 0.123552 |
| IEMOCAP | [0.8, 0.9) | 88 | 0.163569 | 54 | 0.104247 |
| IEMOCAP | [0.9, 0.95) | 51 | 0.094796 | 19 | 0.036680 |
| IEMOCAP | [0.95, 1] | 103 | 0.191450 | 18 | 0.034749 |
| MELD | [0, 0.5) | 838 | 0.958810 | 671 | 0.772152 |
| MELD | [0.5, 0.6) | 32 | 0.036613 | 147 | 0.169160 |
| MELD | [0.6, 0.7) | 4 | 0.004577 | 50 | 0.057537 |
| MELD | [0.7, 0.8) | 0 | 0.000000 | 1 | 0.001151 |
| MELD | [0.8, 0.9) | 0 | 0.000000 | 0 | 0.000000 |
| MELD | [0.9, 0.95) | 0 | 0.000000 | 0 | 0.000000 |
| MELD | [0.95, 1] | 0 | 0.000000 | 0 | 0.000000 |

## 实验五：整体统计

| dataset | model | valid_test_utterances | total_errors | accuracy | weighted_f1 | mean_error_confidence | median_error_confidence | errors_ge_0p8 | high_confidence_predictions_ge_0p8 | high_confidence_error_rate_ge_0p8 | errors_ge_0p9 | high_confidence_predictions_ge_0p9 | high_confidence_error_rate_ge_0p9 | errors_ge_0p95 | high_confidence_predictions_ge_0p95 | high_confidence_error_rate_ge_0p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | baseline | 1623 | 538 | 0.668515 | 0.669361 | 0.750002 | 0.763675 | 242 | 1051 | 0.230257 | 154 | 808 | 0.190594 | 103 | 630 | 0.163492 |
| IEMOCAP | cmerc | 1623 | 518 | 0.680838 | 0.682802 | 0.615968 | 0.593404 | 91 | 634 | 0.143533 | 37 | 362 | 0.102210 | 18 | 207 | 0.086957 |
| MELD | baseline | 2610 | 874 | 0.665134 | 0.654386 | 0.346139 | 0.333744 | 0 | 0 | N/A | 0 | 0 | N/A | 0 | 0 | N/A |
| MELD | cmerc | 2610 | 869 | 0.667050 | 0.657461 | 0.408199 | 0.390460 | 0 | 4 | 0.000000 | 0 | 0 | N/A | 0 | 0 | N/A |

## 分母与解释边界

- 实验二三个 outcome 的分母是各阈值的 Baseline 高置信错误数；新增 CMERC 高置信错误比例的分母是 Baseline 原本不是高置信错误的有效话语数。
- 实验三的类内错误率分母为该真实情绪类全部话语；高置信预测内错误率分母为该类中所有达到阈值的预测，包含正确预测。
- 实验四新增高置信错误比例分母是该模型完整输入下全部正确话语；原本已错且仍错另表保留。
- 实验五区间占比分母为全部错误；高置信预测内错误率分母为全部达到阈值的预测。分母为零统一 N/A。
- 均值替换沿时间轴 dim=0，包含原 batch padding，repeat(T,1,1)。保留 batch16、原话语顺序和 ATen RNN；不改为零填充或有效时间均值。
- 高置信错误减少可来自置信度降低，不能当作纠正了同等数量的分类错误；错误类别变化时，最大置信度变化也不等于同一类别的概率变化。
- 删除模态后的错误类别最大概率和真实类别概率异常是两种现象；本报告不将它们混为 CEL，也不把错误直接归因为模态冲突。
- 原 train.py / official_test 存在测试集选模偏差；本次仍使用实验一已验证的 validation checkpoint，不更换权重以追求测试效果。
- 固定 checkpoint、单次测试集的描述性分析不能证明总体校准优劣或训练模块的因果效应。

## 文件与来源

逐话语明细没有 Top-N 截断，按数据集、实验和阈值保留全部符合条件的样本。
原实验一、四个 checkpoint、数据及原预测文件的哈希在运行前后验证一致；详见 run_manifest.json。
Word 母版状态：指定原 Word 母版未找到。若母版缺失，完整版使用原 Markdown 转录作为实验一来源并明确标注，不声称保留未知 Word 版式。

## 主要发现：实验二至五

- **完整输入的性能变化较小，IEMOCAP 的错误置信度下降明显。** IEMOCAP 的错误数为 538→518，Accuracy 为 66.8515%→68.0838%，Weighted F1 为 66.9361%→68.2802%；错误平均置信度为 0.750002→0.615968。MELD 错误数为 874→869，Weighted F1 为 65.4386%→65.7461%，但错误平均置信度反而由 0.346139 升至 0.408199，不能概括成两个数据集均降低错误置信度。
- **实验二的降低主要表现为仍然错误但置信度下降。** IEMOCAP 在 0.80/0.90/0.95 阈值下，Baseline 高置信错误数分别为 242/154/103；其中真正被 CMERC 纠正的仅 12/5/1，仍错且最大置信度下降的为 227/147/101，仍错且未下降的为 3/2/1。新增 CMERC 高置信错误分别为 10/3/0，资格分母分别为 1381/1469/1520。这些新增高置信错误全部来自 Baseline 原有低置信错误，并非来自 Baseline 正确样本。
- **实验三仍保留明显的高置信错误。** IEMOCAP CMERC 的三个阈值错误数为 91/37/18；其中双方都达到阈值且都错的为 81/34/18，Baseline 原本低于阈值但已错的为 10/3/0。0.80 下按真实类别计数为 happy 6、sad 10、neutral 30、angry 5、excited 6、frustrated 34。绝对计数须结合各类支持数；不能据此断言错误由模态冲突造成。
- **MELD 的阈值空集不代表低错误率或完美校准。** 两个模型及三种均值替换在 0.80/0.90/0.95 下均没有高置信错误。完整 CMERC 在 ≥0.80 有 4 条预测且全部正确，因此为 0/4=0；没有任何达到阈值的预测时错误率为 N/A。MELD 完整输入仍有 874/869 条错误。
- **实验四中替换文本的影响最大。** IEMOCAP Baseline 总错误为 538→666，CMERC 为 518→631；≥0.90 错误为 154→197、37→64。其中完整输入正确、替换文本后新增 ≥0.90 错误为 Baseline 36/1085=3.3180%，CMERC 13/1105=1.1765%；完整输入已错且替换后仍高置信错为 161/51，单独统计。MELD 替换文本使总错误从 874→1347、869→1310，虽无达到高置信阈值的错误，分类性能仍明显下降。
- **替换音频降低高置信错误数，并不表示准确率提高。** IEMOCAP Baseline / CMERC 的总错误升至 593/596，但 ≥0.90 错误降至 119/22。替换视觉后错误为 535/519，接近完整输入；MELD 替换音频与视觉的总错误分别为 884/889、869/880。这是固定 checkpoint 在指定替换协议下的观测，不能推出某模态普遍无用。
- **实验五的区间核对通过。** 按规定七区间，IEMOCAP Baseline 错误数为 [56,80,78,82,88,51,103]，CMERC 为 [156,110,97,64,54,19,18]；MELD 为 [838,32,4,0,0,0,0]、[671,147,50,1,0,0,0]。各组之和分别准确等于 538、518、874、869。IEMOCAP ≥0.90 高置信预测中的错误率为 154/808=19.0594%→37/362=10.2210%，这一分母与“错误内部区间占比”不同。

## 典型样本：仅用于解释，完整集合保留在 CSV 和 Excel

- **真正纠正**：IEMOCAP `Ses05M_script02_2`，索引 17（从 0 开始），原文 “No, of course not.”，真实标签 frustrated。Baseline 预测 angry（0.973172），CMERC 预测 frustrated（0.647713）。这是分类正确性的改变。
- **仍然高置信错误**：IEMOCAP `Ses05F_impro06`，索引 44，真实 neutral。Baseline 与 CMERC 均预测 sad，置信度分别 0.999968、0.999134；CMERC 真实类概率只有 0.000842。最大置信度略降，错误仍存在。完整原文见逐话语 CSV。
- **替换文本后新错误**：IEMOCAP CMERC `Ses05M_script02_2`，索引 52，原文 “Yes, I did notice that.  It does look really beautiful over the water.  Big old white moon.”，真实 happy。完整输入预测 happy（0.929026），替换文本后预测 sad（0.980759），真实类概率为 0.001548。这是定义内新增高置信错误；不直接证明模态冲突。

## 局限性与下一步

- 未找到 `CMERC_实验一_高置信度错误样本统计表.docx`。本次完整版 Word 明确以实验一原始 SUMMARY.md 完整转录作为实验一部分，另行追加实验二至五。无法验证或声称保留未提供母版中的额外文字、表格或版式；原实验一全部现存文件保持不变。找到母版后可复用全部 CSV，无须重新推理，在新目录重新渲染报告。
- 本次沿用实验一确定的四个本地训练 checkpoint；身份有内部训练参数和运行记录支撑，不冒充作者发布的预训练权重。不是多 seed 效果结论，未统计 ECE/Brier 或独立校准评测，不能只凭高置信错误数量声称整体校准最优。
- 官方原始 `train.py` / `official_test` 路径存在按测试集指标选轮次的风险；本次采用实验一固定的 validation checkpoint，不用本测试统计重新选择 checkpoint。既有批次的 continuous RNG/capacity_safe 及 ATen RNN 与作者环境存在协议差异，详见实验一 README，不能宣称严格重现论文分数。
- 官方均值替换包含同一批次 Padding，且未打包 RNN、BatchNorm 和超图实现可能使输出依赖批大小与顺序。因此严格沿用 batch16 和原数据加载顺序，分析仅适用于该协议；未静默改为有效话语均值或零填充。
- 后续若要比较校准方法，应在独立验证集固定校准参数后只做一次测试；若要验证模态冲突机制，需要独立定义冲突证据、人工核验与预先规定的对照。此处仅提出后续方向，本任务没有训练或按测试结果调参。
