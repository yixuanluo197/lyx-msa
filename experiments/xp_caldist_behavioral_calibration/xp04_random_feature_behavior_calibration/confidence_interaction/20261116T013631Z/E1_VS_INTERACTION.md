# E1与交互评分的对照

E1已经支持双向调整。交互版本检验的是在相同原始C之外，μδ随C变化的作用能否带来增量，并非首次赋予模型提高置信度的能力。

| dataset | metric | mean_difference | seed_difference_sample_sd | improved_seeds | ci_low | ci_high |
|---|---|---|---|---|---|---|
| IEMOCAP | ece15 | -0.003838 | 0.004715 | 3 | -0.007569 | 0.005123 |
| IEMOCAP | binary_brier | 0.000705 | 0.001039 | 1 | 0.000143 | 0.001326 |
| IEMOCAP | binary_nll | 0.001776 | 0.002581 | 1 | 0.000436 | 0.003222 |
| MELD | ece15 | 0.001868 | 0.001553 | 0 | -0.002834 | 0.003929 |
| MELD | binary_brier | 0.000568 | 0.000783 | 0 | 0.000227 | 0.000963 |
| MELD | binary_nll | 0.001200 | 0.001719 | 0 | 0.000388 | 0.002127 |

IEMOCAP ECE下降约0.003838（约0.384个百分点），3/5 Seed下降；两proper loss仅1/5 Seed下降。MELD三指标均无Seed改善。两个数据集都未满足固定的“ECE与至少一proper loss同时改善、另一proper loss不恶化且多数Seed同向”条件。ECE单项的积极迹象被保留，没有把它概括成完全没有变化。

E1和旧M_joint由同一数据/折/优化配置重算，与历史16360条OOF对齐，最大差1.11e-16；比较不是由基线或数值精度变化造成。独立Newton精化没有改变内层选择及主比较方向。
