# 独立性与冻结来源审计

**结论：没有可用于本轮的新独立Dialogue；只能按探索性Pilot执行。** 这限制研究结论，不表示冻结模型或数值链路损坏。新生成、模型推理及评分须只使用历史score_fit155个Dialogue，按运行前固定规则分成20工程/75 pilot_fit/60 pilot_eval；旧score_eval不进入新流程。

## 历史暴露核对

| 历史角色 | Dialogue | 已有用途 | 新独立性 |
|---|---:|---|---|
| base_fit | 656 | 文本监督训练及下游模型梯度 | 无 |
| base_inner_dev | 72 | 文本及下游模型早停/最佳轮选择 | 无 |
| score_fit | 155 | 1569条clean结果已分析，曾用于可靠性逻辑回归拟合 | 无 |
| score_eval | 155 | 1407条结果已评价，后续研究方向已受旧结果影响 | 无，本轮排除 |

原1038个Train Dialogue完整覆盖上述四组。文本training_config中的Fit/Dev ID与角色清单精确一致，旧2976条clean覆盖全部310个评分Dialogue，原可靠性模型的fit_sample_ids覆盖全部旧score_fit样本。更换本轮角色名称、重新盲化改写者或选择新目标，不会消除既有结果暴露。

旧Plan B报告已公布MELD官方Validation全部1109条指标；114个Dialogue的既有特征核验记录也已存在。本审计只读取旧报告及核验元数据，未重新打开其文本、情绪标签或执行推理；Test未触及。

## 来源与哈希链

- 已逐文件核验公开交付目录的35个SHA-256；公开清单绑定本地SHA256SUMS.csv，且与上一轮verification中的manifest哈希一致。
- 清单source_commit记录的是实验起始源版本，不能混称发布提交2bf3720。本分工未执行Git；2bf3720对象与该公开目录的对应须由root另行核验。
- 核验物理分离的base/score数据及音频缓存、文本四层特征与其model/tokenizer哈希、文本最佳权重、M3Net与CMERC最佳/末轮checkpoint、CMERC难度文件及其输入清单。checkpoint仅流式计算哈希，未加载。
- 两个下游阶段的provenance均声明且绑定base_fit/base_inner_dev，评分组未进入该冻结模型的梯度或模型选择。此“模型训练隔离”与“研究者未见过新评估结果”是不同条件；本轮仅前者成立。
- 上游旧共享Train/Validation数据容器不重新打开；其历史来源只保留已有清单引用。当前实际核验的dataset.pt属于旧物理隔离Train角色缓存。

## 旧概率的抵消复算

固定clean argmax c，逐版d_k=p_clean(c)−p_k(c)；D_pred=(d_a+d_b)/2，D_abs=(|d_a|+|d_b|)/2。抵消量=D_abs−|D_pred|；异号时独立核验其等于min(|d_a|,|d_b|)，同号时为0。相对抵消为每对抵消量/D_abs；D_abs=0时保留NA。

| 家族 | 历史Role | 有效K2 | 异号对 | 平均D_pred | 平均D_abs | 平均绝对抵消 | 平均样本相对抵消 |
|---|---|---:|---:|---:|---:|---:|---:|
| semantic | score_fit | 72 | 32 | 0.013491186 | 0.049968316 | 0.010305321 | 0.24389304 |
| semantic | score_eval | 69 | 20 | 0.013073643 | 0.056426746 | 0.0047325019 | 0.14665702 |
| corruption | score_fit | 31 | 6 | 0.0095670822 | 0.043734528 | 0.0034683967 | 0.076216873 |
| corruption | score_eval | 18 | 7 | 0.042494047 | 0.04974501 | 0.0021837668 | 0.1693618 |

两类旧记录共24组已有汇总的样本量、D_pred/D_abs均值及中位数均与完整概率复算一致；语义与corruption共同可用子集也逐组核对。所有计划样本和缺失保留在historical_pair_replay.csv。此处旧score_eval仅为历史审计来源，不用于新生成或新评分。

相对抵消的“逐样本平均”与“总抵消量/总D_abs”不是同一个量，CSV同时保留两者。不以跨样本平均接近0代替样本内部异号抵消；不将抵消现象单独当作D_abs有可靠性收益的证据。

## 交付与执行边界

- results/audit/audit_summary.json、checks.json：独立性结论及逐项核对。
- hash_bindings.csv：实际路径、原期望哈希、重新计算哈希及绑定来源。
- dialogue_exposure.csv：1038个Train Dialogue的既有用途与本轮可用池标记。
- historical_pair_replay.csv、historical_cancellation_summary.csv、historical_aggregate_replay_checks.csv：完整旧概率重算、分组统计及与旧汇总对照。
- 审计没有生成文本、读取原始媒体、加载模型、执行训练/推理、接触Test或新的官方Validation文本/标签，也没有Git提交或上传。
