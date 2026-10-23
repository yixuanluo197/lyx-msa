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
