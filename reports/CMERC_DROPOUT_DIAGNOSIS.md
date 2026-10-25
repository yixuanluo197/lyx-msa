# CMERC Dropout 诊断

生成时间：2026-10-25T04:06:50.736831+00:00。仅1/1正式运行完成，MELD seed2024、dropout0.5、lr1e-4、batch16、40轮。

## 实际科学协议与Dropout路径

直接对照XP04完整CMERC同Seed、同lr、同Batch16，唯一登记变化为args.dropout 0.4→0.5。rec_dropout参数固定0.1、Adam/L2=3e-5、其余模型超参数不变；GRU、ATen、continuous RNG、capacity_safe、BN、hyper/direct、concat_DHT、L3/K3、FP32、worker0、split_seed1475、CC/MSCL/CSCL=.002/.15/.15、CL比例步长=.4/1。训练池1038、验证114对话，train/val均Batch16，划分SHA `5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8`。

**rec_dropout的实际边界：** train_cm.py:1040–1073的活跃graph Model入口传递dropout=args.dropout，未传args.rec_dropout。model_cm.py:658的文本两层双向GRU使用dropout=dropout，:713定义nn.Dropout(self.dropout)；因此本次通用dropout变化也作用于活跃GRU层间dropout。rec_dropout=.1是保持不变的CLI元数据，不能声称真实GRU dropout固定.1或本次只改变非循环层Dropout。av_using_lstm=False时音/视GRU分支不启用。hyper输出的hidden-feature Dropout在model_cm.py:892–896参与分类；辅助模型按原train_cm.py:608–613始终eval，辅助前向不因配置中的概率提高而启用Dropout。

实际训练源SHA `0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2`；来源`/data2/yb/paper/lyx-msa/CMERC/experiments/batch_size_diagnosis/20261024T131956Z/code_xp04`，Git HEAD `2b7d247fee86a83f775a3e5eb0e4b0d73e06cf3a`，12训练源保持09字节版本，仅保留既有skip-final-test工程门禁，不改网络或Loss数学。环境Python 3.10.13、PyTorch 1.13.1+cu117、PyG 2.4.0、CUDA build 11.7。

规范选模：验证Weighted F1按全验证池labels/predictions汇总，百分比四舍五入两位，严格提高才换Checkpoint，同分取首次最高轮。末10轮固定31–40；最佳值与末段来自同一验证曲线，是相关描述统计，不是独立证据。

## 验证结果与登记判据

| Dropout | 来源 | 最佳验证W-F1 % | Epoch | 末10轮W-F1 % | 最佳轮验证ACC % | 末10轮验证ACC % | 末段W-F1时间窗SD |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.4 | XP04_control | 66.0100 | 9 | 61.2230 | 66.8200 | 62.4990 | 0.8068 |
| 0.5 | 10_dropout0p5 | 66.4900 | 15 | 62.7750 | 67.0000 | 63.9300 | 0.5513 |

dropout0.5−0.4：Δ最佳验证 **+0.4800 pp**，Δ末10轮 **+1.5520 pp**；分类：**满足第二Seed候选资格（非统计强证据）**。

本Seed满足预登记资格，值得另行登记第二Seed确认；该资格不是统计强证据。本批不自动追加Seed、学习率、Dropout、Batch Size、五Seed训练或测试。

原登记资格不变：最佳提高且末段不降才是第二Seed候选；最佳升、末段降为mixed；最佳未升为no_best_improvement。辅助末段改善不能替换预登记主指标。资格无最小效应量或置信区间门槛，一个Seed不作p值、显著性、稳定复现或等价结论。

## Loss与训练稳定性事实

| Dropout | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 非有限梯度Epoch | optimizer.step实际记录数 |
|---:|---:|---:|---:|---:|---|---|
| 0.4 | 3.2980 | 4.1074 | 2.7076 | 5.0893 | 历史未记录 | 历史未记录；CPU重建2548 |
| 0.5 | 3.0874 | 4.3663 | 2.7396 | 5.3202 | 0 | 2548 |

训练原始分项（最佳验证轮→末10轮均值）：

| Dropout | 分类Loss | CC | MSCL | CSCL | 总Loss |
|---:|---|---|---|---|---|
| 0.4 | 3.2980 → 2.7076 | 57.0680 → 69.2599 | 3.8972 → 4.3891 | 2.8807 → 2.9733 | 4.4288 → 3.9505 |
| 0.5 | 3.0874 → 2.7396 | 77.7666 → 78.2139 | 4.5325 → 5.0488 | 3.1243 → 3.1265 | 4.3915 → 4.1223 |

验证原始分项（最佳验证轮→末10轮均值）：

| Dropout | 分类Loss | CC | MSCL | CSCL | 总Loss |
|---:|---|---|---|---|---|
| 0.4 | 4.1074 → 5.0893 | 未记录 → 未记录 | 未记录 → 未记录 | 未记录 → 未记录 | 未记录 → 未记录 |
| 0.5 | 4.3663 → 5.3202 | 35.6057 → 45.5610 | 4.7204 → 4.2672 | 5.2610 → 3.7861 | 5.9347 → 6.6193 |

XP04历史验证辅助Loss没有正式记录，表中缺失保持“未记录”，不补造零值；新watcher记录完整80次train/val分项。所有固定系数加权CC/MSCL/CSCL及总Loss保存在loss_curves.csv与图中。

- dropout=0.4：最佳验证轮→末段，train分类Loss -0.5903、val分类Loss +0.9819、val W-F1 -4.7870 pp。
- dropout=0.5：最佳验证轮→末段，train分类Loss -0.3478、val分类Loss +0.9539、val W-F1 -3.7150 pp。

**推断边界：** train Loss下降且val Loss上升、验证末段回落，可作为过拟合/目标竞争的轨迹线索，不能证明唯一根因。Dropout提高后若训练拟合降低，既可能是正则化作用也可能是欠拟合，必须同时看验证轨迹；仅训练分数低不能判定欠拟合。train在线训练态与val固定模型eval不同，间距不是严格泛化gap。时间窗SD不是跨Seed标准差。finite梯度和首步真实更新只验证执行数值，不排除伪目标噪声、梯度竞争或泛化问题。

## 资源、零测试与来源限制

本批实际设备为同host GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456）；XP04控制为GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456），同物理卡。历史时点、CUDA scatter非确定性和共享资源仍限制严格因果与速度解释，不宣称连续独占。登记共驻PID `[3670055, 3672750, 3673489]`；实际采样见最终验收。资源准入以registry记录为准：free>=3000MiB、util<=95%、3次/10秒。不终止外部项目，不因分数重跑。

本次零测试loader遍历/零测试前向，不读取新旧测试指标或预测数组、不据测试选配置；本阶段未计算ECE/CEL。综合pickle和测试loader元数据仍由原流程加载，零前向不等于完全未加载含测试数据的文件。固定作者课程包含validation键且生成史未知；同一输入保持配对，不称完整预处理链validation-free。

## 事实、推断和后续建议

已证实事实：登记配置、固定池选模、实际完整40轮轨迹、Loss身份和数值执行记录；推断仍限于上述拟合/验证变化线索。Dropout跨Seed效应、校准改善、泛化到测试集及论文差距唯一根因均未证实。

**当前直接建议：** 本Seed满足预登记资格，值得另行登记第二Seed确认；该资格不是统计强证据。本批不自动追加Seed、学习率、Dropout、Batch Size、五Seed训练或测试。

本批不自动开展任何新Seed、lr、Dropout、Batch Size、五Seed或最终测试。若将来需要确认，须在独立目录重新登记固定候选与预算，不能根据测试分数或临时扩大搜索。

## 产物和验收

[预登记](../experiments/dropout_diagnosis/20261025T032711Z/reports/PREREGISTRATION.md)、[配置](../experiments/dropout_diagnosis/20261025T032711Z/configs/registry.json)、[协议与Dropout路径审计](../experiments/dropout_diagnosis/20261025T032711Z/reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[环境](../experiments/dropout_diagnosis/20261025T032711Z/provenance/environment.json)、[源码来源](../experiments/dropout_diagnosis/20261025T032711Z/provenance/source_origin.json)、[冻结清单](../experiments/dropout_diagnosis/20261025T032711Z/provenance/frozen_execution.json)、[数据](../experiments/dropout_diagnosis/20261025T032711Z/provenance/data_inputs.json)、[容量CPU预检](../experiments/dropout_diagnosis/20261025T032711Z/preflight/capacity_cpu.json)、[预检门禁](../experiments/dropout_diagnosis/20261025T032711Z/preflight/gate.json)、[完整验收](../experiments/dropout_diagnosis/20261025T032711Z/provenance/final_verification.json)。

[per_run.csv](../experiments/dropout_diagnosis/20261025T032711Z/results/per_run.csv)、[comparison.csv](../experiments/dropout_diagnosis/20261025T032711Z/results/comparison.csv)、[metrics_summary.json](../experiments/dropout_diagnosis/20261025T032711Z/results/metrics_summary.json)、[curves.csv](../experiments/dropout_diagnosis/20261025T032711Z/results/curves.csv)、[loss_curves.csv](../experiments/dropout_diagnosis/20261025T032711Z/results/loss_curves.csv)。

[W-F1](../experiments/dropout_diagnosis/20261025T032711Z/results/figures/MELD_dropout_wf1.png)、[ACC](../experiments/dropout_diagnosis/20261025T032711Z/results/figures/MELD_dropout_accuracy.png)、[分类Loss](../experiments/dropout_diagnosis/20261025T032711Z/results/figures/MELD_dropout_classification_loss.png)、[总Loss](../experiments/dropout_diagnosis/20261025T032711Z/results/figures/MELD_dropout_total_loss.png)、[更新次数](../experiments/dropout_diagnosis/20261025T032711Z/results/figures/MELD_dropout_optimizer_steps.png)；完整原始和加权辅助曲线及输入/输出SHA见[图表清单](../experiments/dropout_diagnosis/20261025T032711Z/results/figures/manifest.json)。

从实际四实例只读属性记录的Dropout（无额外模型前向）：

| 实例 | 文本GRU层间Dropout | GRU层数/双向 | hidden-feature Dropout |
|---|---:|---|---:|
| model | 0.5 | 2/True | 0.5 |
| model_mm | 0.5 | 2/True | 0.5 |
| model_intra | 0.5 | 2/True | 0.5 |
| model_inter | 0.5 | 2/True | 0.5 |

[四实例module_audit.json](../experiments/dropout_diagnosis/20261025T032711Z/results/MELD_cmerc_dropout0p5_lr1e-4_seed2024/module_audit.json)仅验证真实配置属性，不证明性能改善或辅助损失作用方向。

最终验收实际状态：`passed`。
单次40train/40validation，真实训练Batch 2548，test前向0；每轮排列与历史及登记Batch16重放匹配`True`，train/val Batch均`16`，dropout/rec_dropout元数据为`0.5`/`0.1`。best/latest第15/40轮，模型/Adam有限`True`/`True`，完整RNG存在`True`/`True`；验收脚本核对完整args/provenance和Adam lr。best SHA `986a06e4edbc1727c2fd9da79c051c3a75510fa3641267b4763570b6f399cfe0`，latest SHA `390286abdaca916d810e9cd83c96b11f1b9d653526b7b6700106b1ffd2dc2cca`。冻结源/输入未变`True`，历史文件保全1881项。

资源证据`limited`：181次采样，既有共驻PID样本82次，新外部PID样本119次，查询错误0次，正式最小空闲6013MiB，最大利用率87%；离散采样不能证明连续资源隔离。

实验根目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/dropout_diagnosis/20261025T032711Z`。仓库根报告正文相同，相对链接按位置调整。

## 针对研究问题的最终判断

**已证实事实：** 最佳验证W-F1提高0.480个百分点，末10轮均值提高1.552个百分点，满足原先登记的第二Seed候选条件。最佳轮从9推迟到15，末10轮验证Accuracy从62.499%升至63.930%。最佳轮至末10轮的W-F1回落从4.787降至3.715个百分点，减轻1.072个百分点，但回落仍存在。

**分类Loss没有同时改善：** 末10轮验证分类Loss为5.32022，高于控制5.08928（+0.23094）；两组都保留训练Loss下降、验证Loss上升的后期轨迹。因此，F1维度的末段衰退较轻，并不足以证明整体过拟合或概率质量问题已解决。Loss和F1衡量的目标不同，本阶段没有ECE/CEL等额外校准评估，不据此判断校准机制已改善。

**欠拟合判断属于有边界的推断：** Dropout0.5的末10轮训练W-F1仍达到96.924%（控制97.768%），训练分类Loss持续下降到末段均值2.73957。训练拟合稍低且验证F1提高，未出现明显的整体欠拟合证据；单次在线训练曲线不能排除局部类别拟合不足，也不能证明正则化是唯一作用机制。

**下一步最小建议（本组不执行）：** 值得将Dropout0.5作为固定候选，在新的独立登记中用另一预先确定的Seed与XP04同Seed的Dropout0.4控制确认；其余Batch16、LR0.0001、split1475、rec_dropout0.1、损失和课程保持不变，仍用最佳验证W-F1及末10轮规则，不做测试筛选。如果跨Seed证据一致，再决定是否冻结后续正式配置；当前不启动第二Seed、Baseline、五Seed或测试。暂不叠加LR0.0002或Batch8，以免失去单因素含义。

后续若继续诊断，可先汇总已有训练/验证曲线的F1与分类Loss分歧，再另行设计有明确预算的类别错误/置信度检查；不要仅凭本Seed的最高分扩大Dropout搜索。07–10已反复使用相同验证划分探索超参数，候选结果属于开发阶段证据，最终泛化表现仍需在正式配置冻结后的独立评估中确认。本组没有执行这类评估。
