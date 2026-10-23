# CMERC 学习率诊断

生成：2026-10-23T11:39:43.288885+00:00；预登记4项已完成4/4。

## 范围与协议

本批接续XP04完整CMERC训练/验证控制，仅比较lr=5e-5、1e-4、2e-4。1e-4为既有XP04同Seed历史轨迹，新增4次从头训练：IEMOCAP seed1475两项、MELD seed67137两项。未继承第06组source_order或w/o CC设置，CC、HSCL和CL均保持开启及原系数。

所有新增任务只训练和验证；验证Weighted F1按百分比四舍五入两位、严格提高才更新Checkpoint，同分取首次最高轮。训练结束跳过测试，不进行测试前向，不解析或使用新旧测试指标筛选学习率。本报告与候选排序完全不含测试结果。原始综合pickle和测试loader元数据仍由既有代码加载，零测试前向不等于完全未加载含测试数据的文件；历史产物仅作字节哈希保全。skip_final_test为独立记录的工程开关，训练数学不变。

IEMOCAP沿用108训练/12验证对话、80轮、dropout0.5、K4、rank/MSCL/CSCL系数0.005/0.05/0.05、课程0.15；MELD1038训练/114验证、40轮、dropout0.4、K3、系数0.002/0.15/0.15、课程0.4。两者均batch16、GRU、ATen、continuous RNG、capacity_safe、BN、lr以外全部模型超参数保持历史值，split_seed1475、L2系数3e-5。完整登记见[registry.json](../experiments/learning_rate_diagnosis/20261023T084353Z/configs/registry.json)。

XP04完整模型历史IEMOCAP seed1475在GPU1、MELD seed67137在GPU0，本批统一同机TITAN Xp GPU3。跨物理卡与CUDA scatter非确定性限制严格单变量物理因果解释。固定作者课程文件包含validation键、生成史未知；相同来源保持配对，但不能称整个预处理validation-free。

## 三个学习率的训练/验证对照

W-F1为百分比，差值为百分点；训练分类Loss为在线逐Batch均值，验证Loss为固定模型eval统计。

| 数据集 | lr | 来源/状态 | 最佳验证W-F1 | Epoch | 末10轮验证均值 | Δ最佳 | Δ末10轮 |
|---|---:|---|---:|---:|---:|---:|---:|
| IEMOCAP | 0.0001 | 历史完整 | 65.0200 | 43 | 62.9670 | +0.0000 | +0.0000 |
| IEMOCAP | 5e-05 | 新训练完成 | 63.4700 | 67 | 62.2570 | -1.5500 | -0.7100 |
| IEMOCAP | 0.0002 | 新训练完成 | 66.2100 | 44 | 62.1440 | +1.1900 | -0.8230 |
| MELD | 0.0001 | 历史完整 | 65.2400 | 8 | 62.9090 | +0.0000 | +0.0000 |
| MELD | 5e-05 | 新训练完成 | 65.9400 | 17 | 63.4370 | +0.7000 | +0.5280 |
| MELD | 0.0002 | 新训练完成 | 66.0700 | 8 | 62.9830 | +0.8300 | +0.0740 |

四项新增训练均完成固定Epoch预算。末10轮为IEMOCAP71–80轮、MELD31–40轮，与最佳值属于同一验证曲线上的相关描述统计，不是两份独立证据。

## Loss、稳定性与过拟合线索

| 数据集/lr | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 训练Epoch峰值显存MiB | 梯度异常 |
|---|---:|---:|---:|---:|---:|---|
| IEMOCAP/0.0001 | 0.5218 | 0.8503 | 0.3433 | 0.9701 | 6978.6100 | 历史正式轨迹未记录 |
| IEMOCAP/5e-05 | 0.5737 | 0.8879 | 0.5382 | 0.9030 | 7075.0700 | 0 |
| IEMOCAP/0.0002 | 0.3369 | 0.9676 | 0.1465 | 1.4516 | 7075.0700 | 0 |
| MELD/0.0001 | 3.3453 | 4.1291 | 2.7067 | 5.1819 | 939.5400 | 历史正式轨迹未记录 |
| MELD/5e-05 | 3.2017 | 4.1387 | 2.8131 | 4.9662 | 946.3900 | 0 |
| MELD/0.0002 | 3.1508 | 4.2662 | 2.6640 | 5.1364 | 946.3900 | 0 |

- IEMOCAP/lr=0.0001：末10轮均值相对最佳验证轮，train分类Loss变化-0.1785、val分类Loss变化+0.1198；呈现训练Loss继续下降而验证Loss上升的轨迹线索。
- IEMOCAP/lr=5e-05：末10轮均值相对最佳验证轮，train分类Loss变化-0.0355、val分类Loss变化+0.0151；呈现训练Loss继续下降而验证Loss上升的轨迹线索。
- IEMOCAP/lr=0.0002：末10轮均值相对最佳验证轮，train分类Loss变化-0.1903、val分类Loss变化+0.4840；呈现训练Loss继续下降而验证Loss上升的轨迹线索。
- MELD/lr=0.0001：末10轮均值相对最佳验证轮，train分类Loss变化-0.6386、val分类Loss变化+1.0528；呈现训练Loss继续下降而验证Loss上升的轨迹线索。
- MELD/lr=5e-05：末10轮均值相对最佳验证轮，train分类Loss变化-0.3886、val分类Loss变化+0.8275；呈现训练Loss继续下降而验证Loss上升的轨迹线索。
- MELD/lr=0.0002：末10轮均值相对最佳验证轮，train分类Loss变化-0.4868、val分类Loss变化+0.8702；呈现训练Loss继续下降而验证Loss上升的轨迹线索。

本阶段watcher不保存完整概率，未计算验证ECE/CEL；不得把未采集值当零或声称校准表现已验证。

非有限Loss与watcher显式报告的非有限梯度会使严格汇总失败；历史XP04正式轮级轨迹无梯度norm，预检梯度不能冒充纵向对照。新watcher若只验证finite，也不能由此宣称各辅助项整段梯度方向已解决。显存为原训练器逐Epochallocated峰值，不等于整卡占用或训练进程完整生命周期峰值。

train Loss下降而val Loss上升、val最佳早于训练末段，可以提供过拟合/目标竞争线索；训练在线预测与eval固定模型的模式和参数时点不同，不能把两条曲线间距称为严格泛化gap，也不能仅凭早峰判定lr是唯一根因。原始CC/MSCL/CSCL与加权分项见loss_curves.csv；CC为未按句数归一化的原作者项，本轮不改公式。

L2系数始终3e-5；原实现是Adam(weight_decay=l2)耦合L2。改变lr会同时改变包含L2项在内的更新轨迹及有效正则化过程，不能解释成仅改变分类梯度步长，也不能声称三条轨迹的总正则化作用相同。

## 预登记候选规则与诊断结论

强候选：最佳验证W-F1高于同Seed 1e-4控制且末10轮均值不低于控制。弱候选/混合：最佳提高但末段下降。仅多个合格强候选按最佳值、再末10轮均值排序，完全相同优先5e-5。弱线索不具备第二Seed候选资格，无强候选时不推荐任何lr；详见[分析资格澄清](../experiments/learning_rate_diagnosis/20261023T084353Z/reports/ANALYSIS_RULE_CLARIFICATION.md)。两个新lr均无最佳改善则停止lr搜索，不扩大范围。

**IEMOCAP：弱候选/混合证据。**

- lr=5e-05：Δ最佳-1.5500 pp，Δ末10轮-0.7100 pp；类别`no_best_improvement`。
- lr=0.0002：Δ最佳+1.1900 pp，Δ末10轮-0.8230 pp；类别`weak_mixed_candidate`。

仅存在最佳值改善但末段下降的混合线索，没有符合原登记资格的第二Seed候选；只报告，不推荐扩展。

**MELD：满足预登记第二Seed候选资格（非统计强证据）。** 预登记排序首选lr=0.0002；仅供下一阶段评估。

- lr=5e-05：Δ最佳+0.7000 pp，Δ末10轮+0.5280 pp；类别`strong_candidate`。
- lr=0.0002：Δ最佳+0.8300 pp，Δ末10轮+0.0740 pp；类别`strong_candidate`。

先按原登记筛选最佳值改善且末段不低于控制的合格强候选；仅在此集合内按最佳验证W-F1、末10轮均值排序，完全相同优先5e-5；不启动第二Seed。
该资格门槛没有预设最小效应量或置信区间；通过门槛不等于改善稳健、统计显著或已确定最优学习率。
首选lr=0.0002相对另一个合格lr=5e-05，最佳验证值差+0.1300 pp、末10轮均值差-0.4540 pp；排序来自预登记的最佳值优先规则，不代表每项统计都更优。

**已证实事实：** 仅登记4个新增训练；完成项的完整轮数、选模规则、Loss有限性与加权身份、训练/验证轨迹由日志核验，候选依据只使用验证集。

**尚未证实推断：** 学习率变化是否稳定改善其他Seed、是否解释论文F1差距、各辅助目标梯度竞争是否缓解，均不能由本轮单Seed历史对照确定。本报告不提供单Seed p值或显著性结论。

**下一步建议：** 首先根据预登记验证证据判断是否值得开展第二Seed独立确认；2024仅预先保留，两数据集XP04同协议控制已存在，本批不自动启动第二Seed、Baseline或五Seed。弱候选先报告证据混合，不靠测试结果决定扩大训练。若两个lr都无最佳改善，停止此搜索。

本阶段不需要追加Batch Size=8。若有合格学习率候选，优先保持batch16做独立Seed确认；若两项均无改善，则停止本轮搜索并分析现有曲线。暂缓Batch8是控制变量和保证解释性的实验顺序，并非证明Batch8无益。Batch8不是本轮变量，也不能由lr结果直接证明有益。未来若要比较，必须独立登记其对BN统计、HSCL正负对、CC尺度、每Epoch更新步数及算力预算的同时变化，不把改Batch伪装成单一优化超参数改善。

## 曲线与可追溯证据

[训练前登记](../experiments/learning_rate_diagnosis/20261023T084353Z/reports/PREREGISTRATION.md)、[源码冻结清单](../experiments/learning_rate_diagnosis/20261023T084353Z/provenance/frozen_execution.json)、[跳过最终测试的独立补丁](../experiments/learning_rate_diagnosis/20261023T084353Z/provenance/skip_final_test.patch)、[首步学习率配对预检](../experiments/learning_rate_diagnosis/20261023T084353Z/preflight/LR_first_step_parity.json)及[真实长度容量预检](../experiments/learning_rate_diagnosis/20261023T084353Z/preflight/capacity_cpu.json)分别保存。首步初始化和RNG哈希完全相同，Loss/梯度仅在登记容差内匹配，不声称CUDA逐位确定性。

[per_run.csv](../experiments/learning_rate_diagnosis/20261023T084353Z/results/per_run.csv)、[comparison.csv](../experiments/learning_rate_diagnosis/20261023T084353Z/results/comparison.csv)、[curves.csv](../experiments/learning_rate_diagnosis/20261023T084353Z/results/curves.csv)、[loss_curves.csv](../experiments/learning_rate_diagnosis/20261023T084353Z/results/loss_curves.csv)、[metrics_summary.json](../experiments/learning_rate_diagnosis/20261023T084353Z/results/metrics_summary.json)。曲线仅train/validation，CSV不导出任何测试分数。

| 数据集 | W-F1轨迹 | 分类Loss | 加权训练Loss |
|---|---|---|---|
| IEMOCAP | [PNG](../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/IEMOCAP_wf1.png) | [PNG](../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/IEMOCAP_classification_loss.png) | [PNG](../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/IEMOCAP_weighted_loss.png) |
| MELD | [PNG](../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/MELD_wf1.png) | [PNG](../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/MELD_classification_loss.png) | [PNG](../experiments/learning_rate_diagnosis/20261023T084353Z/results/figures/MELD_weighted_loss.png) |

最终独立验收文件已生成：[final_verification.json](../experiments/learning_rate_diagnosis/20261023T084353Z/provenance/final_verification.json)，通过范围以文件内容为准。

实验根目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/learning_rate_diagnosis/20261023T084353Z`。仓库根报告正文相同，链接按报告位置调整。
