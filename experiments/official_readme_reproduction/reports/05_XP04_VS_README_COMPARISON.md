# XP04 与官方 README 实训对照

更新：2026-10-21T14:06:43.520933+00:00。分类指标和 ECE 使用百分比，差值为百分点；CEL 为总和，Epoch 为轮数。新实训未完成的单元格不填推测值。

| 数据集 | 模型 | 官方 README 实训 W-F1 | XP04 同 Seed W-F1 | 论文报告 W-F1 | 新实训−XP04 |
|---|---|---:|---:|---:|---:|
| IEMOCAP | baseline | 70.0432 | 66.9361 | 69.61 | 3.1070 |
| IEMOCAP | cmerc | 70.5108 | 68.2802 | 71.98 | 2.2306 |
| MELD | baseline | 66.2257 | 65.4386 | 65.37 | 0.7871 |
| MELD | cmerc | 66.1301 | 65.7461 | 66.85 | 0.3840 |

| 数据集 | README CMERC−Baseline | XP04同Seed CMERC−Baseline | 论文报告差值 |
|---|---:|---:|---:|
| IEMOCAP | 0.4676 | 1.3440 | 2.37 |
| MELD | -0.0956 | 0.3075 | 1.48 |

官方README实训为 `official_test_selection`，XP04为 `validation`。两者训练池、batch与RNG等也不同，表中的差值只能作描述性对照，不能当作同协议下的泛化提升或特定设置的因果效应。

| 数据集/模型 | Acc 新/XP04 | Macro F1 新/XP04 | ECE % 新/XP04 | CEL总和 新/XP04 | 最佳Epoch 新/XP04 |
|---|---:|---:|---:|---:|---:|
| IEMOCAP/baseline | 70.1171/66.8515 | 69.2257/66.4031 | 15.8094/16.4560 | 1733.0513/1520.0235 | 72/70 |
| IEMOCAP/cmerc | 70.3019/68.0838 | 69.9827/67.5473 | 15.1093/3.9775 | 1449.1723/683.0297 | 79/43 |
| MELD/baseline | 67.5479/66.5134 | 50.1186/48.9124 | 24.7775/26.3121 | 509.6076/503.9679 | 5/5 |
| MELD/cmerc | 67.2414/66.7050 | 50.3806/49.2678 | 18.7171/19.0666 | 657.3488/644.5384 | 8/8 |

XP04四项均从真实归档独立评估文件、训练final_test与manifest重新核验；完整精度、loss、epoch和来源SHA见 `provenance/xp04_verified_metrics.json`。本批精度来自 `results/<run_id>/metrics.json`；官方选轮使用的两位精度另列 `results/per_run.csv`。

官方IEMOCAP训练120对话，XP04为108训练+12验证；官方MELD1152训练，XP04为1038训练+114验证。测试对话相同，但不能消除训练与选模差异。官方IEMOCAP CL前7轮对话数18/36/54/72/90/108/120，XP04为16/32/48/64/81/97/108；官方MELD前3轮460/921/1152，XP04为415/830/1038。实际逐轮覆盖在metrics_summary中核验。

官方训练分类loss为IEMOCAP加权NLL或MELD原作者FocalLoss；CEL是另一个校准指标，不是分类cross entropy，也不能与平均loss混比。分项loss、最佳轮和曲线见报告04及CSV。

每个数据集只有一个预先指定Seed。单Seed不能代表XP04五Seed均值，不估计跨Seed标准差、置信区间或显著性；论文报告值只是参考，不保证与当前单Seed具有同等统计口径。

## 训练曲线

下图文件为CPU生成的单面板诊断图，原始CSV与来源哈希同时保留。W-F1图的README测试曲线参与选轮，XP04验证曲线参与选轮；二者用途及训练数据不同。Loss图中的CC/MSCL/CSCL为乘固定系数后的batch均值。

| 数据集/模型 | W-F1曲线 | Loss曲线 |
|---|---|---|
| IEMOCAP/baseline | [PNG](../results/figures/IEMOCAP_baseline_seed1475_wf1.png) / [SVG](../results/figures/IEMOCAP_baseline_seed1475_wf1.svg) | [PNG](../results/figures/IEMOCAP_baseline_seed1475_loss.png) / [SVG](../results/figures/IEMOCAP_baseline_seed1475_loss.svg) |
| IEMOCAP/cmerc | [PNG](../results/figures/IEMOCAP_cmerc_seed1475_wf1.png) / [SVG](../results/figures/IEMOCAP_cmerc_seed1475_wf1.svg) | [PNG](../results/figures/IEMOCAP_cmerc_seed1475_loss.png) / [SVG](../results/figures/IEMOCAP_cmerc_seed1475_loss.svg) |
| MELD/baseline | [PNG](../results/figures/MELD_baseline_seed67137_wf1.png) / [SVG](../results/figures/MELD_baseline_seed67137_wf1.svg) | [PNG](../results/figures/MELD_baseline_seed67137_loss.png) / [SVG](../results/figures/MELD_baseline_seed67137_loss.svg) |
| MELD/cmerc | [PNG](../results/figures/MELD_cmerc_seed67137_wf1.png) / [SVG](../results/figures/MELD_cmerc_seed67137_wf1.svg) | [PNG](../results/figures/MELD_cmerc_seed67137_loss.png) / [SVG](../results/figures/MELD_cmerc_seed67137_loss.svg) |
