# CMERC 最小消融诊断结果

生成时间：2026-10-23T06:32:54.427474+00:00。已完成 3/3 项预注册正式训练。

## 目标、范围与依据

已读取XP04与官方README对照完整报告，重点接续05的 `05_XP04_VS_README_COMPARISON.md` 和 `06_F1_GAP_DIAGNOSIS_AND_NEXT_STEPS.md`。此前MELD单Batch CC梯度与分类梯度方向相反只是本次假设来源，不能视为整段训练根因证据。

本批只登记MELD seed67137关闭CC且保留HSCL/CL一次，以及IEMOCAP seed1475的Baseline/完整CMERC原始源顺序Batch各一次。每项从头初始化，历史源码、Checkpoint、日志和结果只读，不覆盖XP04或05。未追加Seed、系数搜索、w/o HSCL、w/o CL或最终五Seed训练。

## 配对条件、实现与边界

所有新运行使用验证集Weighted F1选模：保留XP04两位小数百分比口径，严格提高才换最佳Checkpoint，同分取首次最高轮。测试集在全部轮数完成后，只对已固定的最佳Checkpoint遍历一次；每次含主模型和五种移除诊断，不再运行独立GPU evaluator。保存概率后的CPU重算不增加测试访问。

| Run | Seed | 唯一登记的科学配置变化 | 轮数 | 划分SHA-256 |
|---|---:|---|---:|---|
| MELD_wo_cc_seed67137 | 67137 | `calibrate`：`True` → `False` | 40 | `5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8` |
| IEMOCAP_baseline_source_order_seed1475 | 1475 | `batch_policy`：`capacity_safe` → `source_order` | 80 | `d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62` |
| IEMOCAP_cmerc_source_order_seed1475 | 1475 | `batch_policy`：`capacity_safe` → `source_order` | 80 | `d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62` |

MELD的calibrate=False只取消总损失中的加权CC，rank_coff仍登记0.002；因contrastlearning=True，原始CC、五路辅助前向和HSCL目标仍计算，raw CC非零不代表其进入反传。保留MSCL/CSCL系数0.15/0.15与课程0.4。IEMOCAP保留原训练池108/验证12/测试31、continuous RNG、ATen GRU及各模型原损失系数，仅将随机排列改为训练池源顺序子序列；课程成员、Batch16、训练轮数80及验证选模不变。MELD为1038/114/280对话、40轮。

MELD冻结12个训练源文件与XP04逐字节一致；IEMOCAP仅train_cm.py含独立记录的source_order采样补丁，原capacity_safe分支保持。完整参数、源码和数据哈希在configs、run_manifest及provenance中。原始Loss数学、网络结构、输入数据不变。

配置层面的单因素对齐不等于严格物理条件完全一致：历史MELD完整CMERC为GPU0→本批GPU3，历史IEMOCAP完整CMERC为GPU1→GPU3；IEMOCAP Baseline历史与本批均GPU3。全部为同机TITAN Xp，但CUDA scatter等数值非确定性及跨卡差异限制历史配对的因果解释；详见[hardware_pairing.json](../provenance/hardware_pairing.json)。RNG策略保持不等于改变Batch张量形状后dropout逐元素轨迹仍相同。作者fixed_author课程文件含validation键，生成历史未知；两臂使用同文件以维持对照，不能称整个预处理validation-free。

## 预检和运行状态

预检状态 `passed`：4项真实首Batch前向/反向/Adam更新均有限且参数实际更新；无测试前向。MELD完整CC与w/o CC首Batch初始化模型、对话IDs及三项RNG hash精确相等；前向仅在容差内匹配（最大log概率差2.384e-7、raw CC差7.629e-6），不能声称逐位一致。目标核验及加权CC取消见[MELD_CC_pair_parity.json](../preflight/MELD_CC_pair_parity.json)。IEMOCAP CPU真实长度检查覆盖160个epoch条件；3项CPU观测器行为测试通过。证据：[gate.json](../preflight/gate.json)、[source_order_cpu.json](../preflight/source_order_cpu.json)。

| Run | 状态 | 已完成/计划Epoch | 最佳验证Epoch | 最佳验证W-F1 % | 末10轮验证均值 % |
|---|---|---:|---:|---:|---:|
| MELD_wo_cc_seed67137 | 完成 | 40/40 | 8 | 65.7700 | 62.7660 |
| IEMOCAP_baseline_source_order_seed1475 | 完成 | 80/80 | 62 | 65.5800 | 63.7060 |
| IEMOCAP_cmerc_source_order_seed1475 | 完成 | 80/80 | 75 | 64.9600 | 62.6580 |

完整轮数、每轮测试字段为空、首个最高验证轮、最终一次测试、保存概率、逐Batch覆盖和分项Loss重构均通过独立CPU验收。历史1154个常规文件及已登记链接核验未变，数据与执行源码哈希未变。641次资源采样未发现其他项目进程进入本批GPU；离散采样不能证明未采样时刻的全局互斥。

## 固定最佳Checkpoint的最终测试描述

分类指标和ECE为百分比，差值为百分点；CEL为五种移除正真类log概率差总和，非分类交叉熵。

| Run | W-F1 % | Macro-F1 % | Accuracy % | ECE % | CEL总和 |
|---|---:|---:|---:|---:|---:|
| MELD_wo_cc_seed67137 | 65.3096 | 48.8753 | 66.3602 | 23.8987 | 611.0441 |
| IEMOCAP_baseline_source_order_seed1475 | 66.8159 | 66.5574 | 66.9747 | 15.8622 | 1440.7065 |
| IEMOCAP_cmerc_source_order_seed1475 | 68.2501 | 67.4159 | 68.1454 | 15.7453 | 1248.4878 |

IEMOCAP Baseline训练/验证无辅助移除，在线CEL为null；仅在唯一最终测试函数入口临时设置calibrate=True且系数0，获得同次遍历中的五路移除，返回即恢复。此开关不改变训练或选轮。训练态在线ECE/CEL混合逐Batch变化的模型参数，CEL还使用epoch入口辅助快照，不能作为固定Checkpoint泛化指标与最终测试直接混比。

## MELD：CC是否限制分类性能

XP04完整CMERC的最佳验证W-F1为65.2400%（第8轮），末10轮均值62.9090%。w/o CC对应变化：最佳验证 **+0.5300 pp**，末10轮均值 **-0.1430 pp**。

最佳验证W-F1与末10轮均值方向不一致，证据混合，尚不能支持“CC在当前MELD配置下可能限制验证分类表现”的稳定收益解释。

固定最终测试W-F1：XP04 65.7461% → 本批65.3096%，变化-0.4365 pp，只作描述。不能据测试结果改系数或挑方案。

同一次固定最终测试的校准描述：ECE变化+4.8321 pp，CEL变化-33.4943。两者方向不同，不能把关闭CC称为统一改善校准。

## IEMOCAP：Batch组织是否影响CMERC相对收益

| 协议 | 验证CMERC−Baseline / pp | 末10轮验证均值差 / pp | 固定最终测试CMERC−Baseline / pp |
|---|---:|---:|---:|
| XP04随机排列 | -1.4300 | -1.2690 | 1.3440 |
| 本批source_order | -0.6200 | -1.0480 | 1.4342 |
| 增量变化 | +0.8100 | +0.2210 | +0.0902 |

- baseline逐臂变化：最佳验证-0.8700 pp，末10轮均值-0.5300 pp，固定测试-0.1202 pp。
- cmerc逐臂变化：最佳验证-0.0600 pp，末10轮均值-0.3090 pp，固定测试-0.0301 pp。

完整CMERC固定最终测试相对XP04的ECE变化为+11.7678 pp，CEL变化为+565.4581；分类分数接近不能理解为校准表现也接近。这仍是单Seed历史对照的描述，未据此改变配置或安排重测。

两类验证增量变化同向，提供“固定源顺序提高CMERC相对Baseline的增量”的单Seed线索，支持Batch排列可能影响相对收益；历史完整CMERC跨物理卡和CUDA数值非确定性仍限制严格因果解释，不能外推五Seed稳定结论。

本次最佳验证增量与末段验证增量仍都为负，属于负增量缩小。两臂绝对验证分数均下降，Baseline下降更多；不能表述为固定顺序提高CMERC绝对验证性能、恢复正分类收益或解释了论文增量。

两个模型分别在各自验证曲线上选轮；上表验证增量是选中分数的描述差，不能把模型胜负或变化直接等同于固定轮次的优化因果。末10轮指MELD第31–40轮、IEMOCAP第71–80轮的训练末段表现，不是选中Checkpoint的指标，也不是跨Seed稳定性证据。最佳值与末10轮均值来自同一验证曲线，是相关的描述统计；同向不等于两份独立证据。固定顺序若使增量下降，也只支持此Seed下方向相反的描述，不代表所有Batch组织均无关。

## 曲线、Loss与可追溯产物

[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[epoch_curves.csv](../results/epoch_curves.csv)、[epoch_loss_curves.csv](../results/epoch_loss_curves.csv)。CSV包括新旧训练/验证轨迹、原始与固定系数加权Loss；测试分数不形成逐轮曲线。

| Run | 训练/验证W-F1 | Loss分项 |
|---|---|---|
| MELD_wo_cc_seed67137 | [PNG](../results/figures/MELD_wo_cc_seed67137_wf1.png) | [PNG](../results/figures/MELD_wo_cc_seed67137_loss.png) |
| IEMOCAP_baseline_source_order_seed1475 | [PNG](../results/figures/IEMOCAP_baseline_source_order_seed1475_wf1.png) | [PNG](../results/figures/IEMOCAP_baseline_source_order_seed1475_loss.png) |
| IEMOCAP_cmerc_source_order_seed1475 | [PNG](../results/figures/IEMOCAP_cmerc_source_order_seed1475_wf1.png) | [PNG](../results/figures/IEMOCAP_cmerc_source_order_seed1475_loss.png) |

[IEMOCAP同轮验证增量曲线](../results/figures/IEMOCAP_validation_gain.png)展示各Epoch的CMERC−Baseline，与分别选取最佳Checkpoint后的增量统计口径不同。全部图同时保存SVG及输入/输出哈希，见[图表清单](../results/figures/manifest.json)。

## 已证实事实、推断与下一步

**已证实事实：** 当前3/3训练完成；完成项的数值来自保存输出，经CPU重算和逐Batch Loss身份检查。冻结参数、划分和唯一登记因素受运行时核验。此前单Batch梯度不是纵向根因证据。

**尚未证实：** 单Seed历史对照不能证明跨Seed稳定收益、唯一根因或与论文差距的定量分摊。CC与HSCL/CL可能协同或竞争，但本批只隔离一个配置开关，不证明其内部作用机制。设备和非确定性边界仍存在。

**Batch机制的待验证假设：** BN统计、padding参与的移除均值、HSCL正负对及未按句数归一化的CC都可能随Batch改变。另经源码确认，[model_hyper.py](../code_xp04/model_hyper.py)第127–128行定义可训练hyperedge_weight/EW_weight，第164–172行按Batch范围切片，第238–262行从零分配当前Batch节点/超边编号；[HypergraphConv.py](../code_xp04/HypergraphConv.py)第134–143行依该局部编号索引参数参与归一化。因此排列可改变样本与可训练参数槽位的对应关系。该路径的存在是代码事实，其对本次F1的贡献尚未隔离；初始化为全1也意味着不能直接据此断言早期差异。本批没有修改这些数学。

**后续建议：** 先按预登记的验证最佳值、末10轮均值和完整轨迹判断证据一致性。若两类验证统计同向支持机制影响，可另行登记w/o HSCL或w/o CL，每次独立一个因素；若证据混合或不支持，先做训练/验证梯度及伪目标稳定性诊断，不根据测试调系数。最终五Seed应在诊断结论和新方案明确后单独决定；本脚本不会自动启动消融或扩展Seed。

## 本轮完成后的实验决策建议（尚未执行）

- MELD：本次未形成CC限制分类性能的稳定验证证据，暂缓围绕“去CC提升F1”开展最终五Seed，也不直接扩展w/o HSCL或w/o CL。优先登记只用训练/验证数据的分阶段多Batch梯度norm/cos、辅助快照与dropout下HSCL目标稳定性诊断；如需确认开关效应，再补同卡控制。
- IEMOCAP：验证相对增量同向变化，使Batch组织成为值得继续验证的因素；先另行登记同卡的随机/固定排列控制及CUDA数值重复性检查，确认历史跨卡与数值波动不足以解释观察值，再考虑在固定Batch协议下独立消融HSCL或CL。当前不把改变Batch当作已确定的优化方案。
- 最终五Seed：本轮仅能形成定位线索，建议暂缓启动。应先冻结下一阶段的科学假设、同卡控制、最终生效配置及验证选模规则，再由新的独立预注册决定五Seed预算。本批未执行上述后续训练，也未重测测试集。

最终独立验收：[final_verification.json](../provenance/final_verification.json)通过；三项保存预测均独立CPU重算，6个最佳/最终Checkpoint的模型及Adam状态有限且含已登记恢复状态。

批次绝对路径：`/data2/yb/paper/lyx-msa/CMERC/experiments/minimal_ablation_validation/20261023T043115Z`。仓库根报告与本批版本正文一致，产物相对链接按报告所在位置调整。
