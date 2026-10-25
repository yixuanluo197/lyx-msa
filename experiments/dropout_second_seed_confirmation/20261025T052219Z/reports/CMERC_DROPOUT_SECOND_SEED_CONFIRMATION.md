# CMERC Dropout第二Seed确认报告

生成时间：2026-10-25T06:15:19.817468+00:00。第11组仅新增1次MELD seed67137完整CMERC训练；复用第10组已完成的seed2024 dropout0.5训练1次，以及XP04两个对应Seed的dropout0.4历史控制。四臂均为40轮，不能将复用结果称为本批新训练。

**当前判定：** `mixed_evidence`。第二Seed最佳验证升、末段降，证据混合，未获得同时满足原登记条件的一致确认。

**直接建议：** 暂缓扩大五Seed和正式配置确认；保持主指标，先解释已有曲线，不自动追加训练或测试。

## 预登记、配对及实际执行

唯一科学因素为通用Dropout参数0.4→0.5；两个Seed各自与同Seed XP04完整CMERC配对。lr=0.0001、L2=0.00003、Adam、train/validation Batch=16/16、CLI rec_dropout=0.1、40 epochs、GRU、hyper/direct/concat_DHT、BN、L3/K3、CC=0.002、MSCL/CSCL=0.15/0.15和CL=0.4/1全部保持。数据split_seed=1475、continuous RNG、ATen GRU和capacity_safe采样保持。逐项args、数据文件SHA与split精确核对；工程开关skip_final_test仅取消最后测试阶段。

使用完整固定验证池拼接话语后计算Weighted F1，保留原训练器按百分比四舍五入两位后严格提高才保存的规则；并列最高选首次。末段固定为31–40轮均值；末段时间窗SD使用ddof=1，不是跨Seed标准差，峰值和末段也不是独立证据。

train_cm.py SHA256：`0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2`；源码来源`/data2/yb/paper/lyx-msa/CMERC/experiments/dropout_diagnosis/20261025T032711Z/code_xp04`，12个训练源码与第10组字节相同标记`True`；复制来源记录Git HEAD `17e3c1eef813c1db7715a827c81aae6185c8d0e8`；本批实际登记时Git HEAD为`da8388a24e562d9059b700c4b99b38a13f8a632f`，以冻结清单为准。科学与工程来源见[源码和协议审计](../reports/SOURCE_AND_PROTOCOL_AUDIT.md)。

实际环境：Python 3.10.13、PyTorch 1.13.1+cu117、CUDA build 11.7、PyG 2.4.0；复用已验证环境，无新增依赖。

Dropout实际作用路径：train_cm.py:1040–1073传入args.dropout；model_cm.py:658的两层双向文本GRU层间Dropout和:713的hidden-feature Dropout（hyper路径约:892–896）均从0.4改到0.5。CLI rec_dropout=0.1未传给当前活跃图模型，仅保留元数据，不能解释为“实际GRU Dropout仍为0.1”。四模型实例属性已记录；辅助模型在train_cm.py:608–613设为eval，teacher前向Dropout关闭；不将其关闭状态等同于主模型训练Dropout关闭。

## 两Seed四臂结果（仅训练和验证）

| Seed | Dropout | 来源 | 最佳验证W-F1 % | 轮 | 末10轮W-F1 % | 末10轮SD | 最佳轮val ACC % | 末10轮val ACC % |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2024 | 0.4 | XP04_control | 66.01 | 9 | 61.223 | 0.8068 | 66.82 | 62.499 |
| 2024 | 0.5 | XP10_reused | 66.49 | 15 | 62.775 | 0.5513 | 67.00 | 63.930 |
| 67137 | 0.4 | XP04_control | 65.24 | 8 | 62.909 | 0.4975 | 66.28 | 64.245 |
| 67137 | 0.5 | XP11_new | 66.41 | 17 | 62.440 | 0.8664 | 67.09 | 63.678 |

| Seed | 最佳W-F1变化 pp | 末10轮均值变化 pp | 预登记分类 |
|---:|---:|---:|---|
| 2024 | +0.48 | +1.552 | `qualified` |
| 67137 | +1.17 | -0.469 | `mixed_evidence` |

分类门槛：最佳提高且末段不降为qualified；最佳提高但末段下降为mixed_evidence；最佳未提高为no_best_improvement，即使末段改善也不能替换主指标。两个Seed均qualified才是初步一致性支持。门槛满足不等于统计强证据，不给两Seed p值或统计显著结论，不推断测试泛化。

## Loss与训练稳定性

| Seed | Dropout | train分类Loss 最佳轮→末段 | val分类Loss 最佳轮→末段 | 梯度异常Epoch | optimizer.step |
|---:|---:|---|---|---|---|
| 2024 | 0.4 | 3.2980 → 2.7076 | 4.1074 → 5.0893 | 历史未记录 | CPU重建 2548 |
| 2024 | 0.5 | 3.0874 → 2.7396 | 4.3663 → 5.3202 | 0 | 2548 |
| 67137 | 0.4 | 3.3453 → 2.7067 | 4.1291 → 5.1819 | 历史未记录 | CPU重建 2548 |
| 67137 | 0.5 | 2.9882 → 2.7403 | 4.6263 → 5.2680 | 0 | 2548 |

训练原始Loss（最佳验证轮→31–40轮均值）：

| Seed | Dropout | 分类 | CC | MSCL | CSCL | 总Loss |
|---:|---:|---|---|---|---|---|
| 2024 | 0.4 | 3.2980 → 2.7076 | 57.0680 → 69.2599 | 3.8972 → 4.3891 | 2.8807 → 2.9733 | 4.4288 → 3.9505 |
| 2024 | 0.5 | 3.0874 → 2.7396 | 77.7666 → 78.2139 | 4.5325 → 5.0488 | 3.1243 → 3.1265 | 4.3915 → 4.1223 |
| 67137 | 0.4 | 3.3453 → 2.7067 | 60.8713 → 71.5387 | 4.0387 → 5.0568 | 3.2041 → 3.0096 | 4.5534 → 4.0597 |
| 67137 | 0.5 | 2.9882 → 2.7403 | 85.3098 → 79.1850 | 4.4920 → 4.5569 | 3.2757 → 3.0177 | 4.3240 → 4.0349 |

验证原始Loss（最佳验证轮→31–40轮均值）：

| Seed | Dropout | 分类 | CC | MSCL | CSCL | 总Loss |
|---:|---:|---|---|---|---|---|
| 2024 | 0.4 | 4.1074 → 5.0893 | 未记录 → 未记录 | 未记录 → 未记录 | 未记录 → 未记录 | 未记录 → 未记录 |
| 2024 | 0.5 | 4.3663 → 5.3202 | 35.6057 → 45.5610 | 4.7204 → 4.2672 | 5.2610 → 3.7861 | 5.9347 → 6.6193 |
| 67137 | 0.4 | 4.1291 → 5.1819 | 未记录 → 未记录 | 未记录 → 未记录 | 未记录 → 未记录 | 未记录 → 未记录 |
| 67137 | 0.5 | 4.6263 → 5.2680 | 42.8104 → 49.4332 | 4.3388 → 4.2362 | 5.1752 → 4.6797 | 6.1390 → 6.7043 |

固定系数的加权CC/MSCL/CSCL、完整train/val曲线与每轮更新数见CSV和图。历史验证辅助Loss/总Loss没有正式记录，保留缺失而非填零；新两臂的watcher均有80条train/validation记录。新臂分类Loss使用watcher完整Batch均值精度；旧臂validation分类Loss来自4位小数原日志，不能把微小差值归为高精度优化效应。Loss仍按原训练器Batch均值汇总，不将原始Loss数值当作独立因果证据。

- Seed2024 dropout=0.4：最佳轮→末段，train分类Loss -0.5903、val分类Loss +0.9819、val W-F1 -4.7870 pp。
- Seed2024 dropout=0.5：最佳轮→末段，train分类Loss -0.3478、val分类Loss +0.9539、val W-F1 -3.7150 pp。
- Seed67137 dropout=0.4：最佳轮→末段，train分类Loss -0.6386、val分类Loss +1.0528、val W-F1 -2.3310 pp。
- Seed67137 dropout=0.5：最佳轮→末段，train分类Loss -0.2479、val分类Loss +0.6417、val W-F1 -3.9700 pp。

**推断边界：** train分类Loss继续下降而val分类Loss上升、验证峰值后衰退，只是过拟合/目标竞争的轨迹线索；若W-F1和分类Loss方向不一致，也不能宣称过拟合已解决。训练拟合下降既可能是正则化作用也可能是欠拟合，不能只凭训练分数下降判定欠拟合。train为在线训练态不断变化的参数，val为固定epoch权重eval，二者间距不是严格泛化gap。有限梯度和真实更新验证数值执行，不能排除伪目标噪声、梯度竞争或泛化问题。


### 跨Seed的方向检查和最小下一步

Seed67137的末10轮时间标准差由0.4975增至0.8664个百分点，最佳轮至末段回落由2.331增至3.970个百分点，增加1.639个百分点；Seed2024则由4.787减至3.715个百分点。后期稳定性改善没有跨Seed一致确认。提高Dropout后，末段验证分类Loss在2024从5.08928升至5.32021924，在67137从5.18186升至5.26803213；最佳F1与分类Loss方向不同，两个指标必须分别报告。该现象可能涉及错误样本的置信度变化或分类/辅助目标竞争，现有聚合曲线没有隔离这些机制，均属于待验证假设。

当前不将Dropout0.5直接升级为最终配置，也不启动五Seed。最小后续工作建议先对已有验证集轨迹做按类别的错误率与置信度审计，预先固定Checkpoint、指标和比较规则；若需要额外前向，应另行登记，仍不据测试结果调参。本批没有进行此项额外评估。未来若决定完整五Seed确认，需单独冻结统一配置与判据，完整报告全部Seed；若要声称CMERC相对Baseline的收益，应安排同协议配对Baseline，不能以两个Seed的验证峰值代替。

## 来源、资源、测试访问与完整验收

本批实际设备：user23 GPU0，UUID `GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248`；XP04 seed67137控制为GPU0（GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248），同物理卡。seed2024的XP04控制和复用XP10均用GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456）。共享资源、历史时点和CUDA scatter非确定性仍限制严格因果及速度解释。

登记准入为free>=3000MiB、util<=95%、连续3次/间隔10秒；其他进程只披露，不终止，也不称连续独占。实际采样限制见下方验收。

本批与复用第10组均零新增测试loader遍历/零新增测试前向，不根据测试选择配置或轮次，未计算ECE/CEL。综合pickle和test loader元数据仍由原流程加载，零前向不等于完全未加载含测试数据的文件。另一次只读GPU来源审计误将混合run_supplement整份输出，偶然访问了历史测试汇总字段；本分析脚本只读取训练/验证净化档案，事件说明不含测试分数。该访问不能隐瞒，故不宣称“全程未读取新旧测试指标”。科学配置由用户事先固定，未据该历史字段改变训练/选模/后续判据，见[测试字段偶然访问事件](../provenance/test_metric_access_incident.json)。

作者固定课程难度包含validation键且生成史未知；同输入保持配对，但不能宣称完整预处理链validation-free。四臂数据、源文件和实际args可追溯，不修改XP04、第10组或其他历史产物。

独立完整验收状态`passed`：本批1次正式训练，40train/40validation/test0，真实训练Batch 2548；排列与历史和登记Batch16重放一致`True`，采样AST与历史重放通过`True`。
best/latest为第17/40轮，模型/Adam有限、完整Python/NumPy/Torch/CUDA RNG均由CPU验收确认；核对完整args/provenance和Adam lr，冻结源/输入未变`True`，历史文件保全2011项。

资源证据`limited`：采样256次，既有共驻PID样本97次、新外部PID样本192次，外部PID `[3875951, 3900060, 3939506, 3939591, 3969736, 3988595, 4006524, 4026340]`，查询错误0次，正式最小空闲5955MiB、最大利用率96%。离散采样不能证明连续资源隔离。

四实例实际Dropout属性（本批无额外模型前向）：

| 实例 | 文本GRU Dropout | 层数/双向 | hidden-feature Dropout |
|---|---:|---|---:|
| model | 0.5 | 2/True | 0.5 |
| model_mm | 0.5 | 2/True | 0.5 |
| model_intra | 0.5 | 2/True | 0.5 |
| model_inter | 0.5 | 2/True | 0.5 |

## 已证实事实、未证实推断与下一步

已证实事实：本批单次40轮的执行、源/参数/划分一致性、验证选模、四臂真实训练/验证轨迹、数值执行及完整验收；2024的dropout0.5是第10组复用，不是重训。尚未证实：Dropout是论文F1差距唯一根因、对任意Seed稳定改善、统计显著、校准性能或测试泛化改善。

**下一步直接建议：** 暂缓扩大五Seed和正式配置确认；保持主指标，先解释已有曲线，不自动追加训练或测试。 本阶段暂缓Batch8，没有对Batch因素的新隔离证据；不自动追加五Seed、其他Dropout/lr/Batch、Baseline或测试。

## 交付和可追溯输入

[预登记](../reports/PREREGISTRATION.md)、[源码/协议审计](../reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[配置](../configs/registry.json)、[环境](../provenance/environment.json)、[源码来源](../provenance/source_origin.json)、[冻结](../provenance/frozen_execution.json)、[数据输入](../provenance/data_inputs.json)、[XP04净化控制](../provenance/xp04_validation_references.json)、[复用XP10档案](../provenance/xp10_validation_reference.json)、[容量预检](../preflight/capacity_cpu.json)、[预检门禁](../preflight/gate.json)、[完整验收](../provenance/final_verification.json)。

[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)、[22图44文件及SHA清单](../results/figures/manifest.json)。

Seed2024：[W-F1](../results/figures/MELD_seed2024_dropout_wf1.png)、[ACC](../results/figures/MELD_seed2024_dropout_accuracy.png)、[分类Loss](../results/figures/MELD_seed2024_dropout_classification_loss.png)、[总Loss](../results/figures/MELD_seed2024_dropout_total_loss.png)、[更新次数](../results/figures/MELD_seed2024_dropout_optimizer_steps.png)。
Seed67137：[W-F1](../results/figures/MELD_seed67137_dropout_wf1.png)、[ACC](../results/figures/MELD_seed67137_dropout_accuracy.png)、[分类Loss](../results/figures/MELD_seed67137_dropout_classification_loss.png)、[总Loss](../results/figures/MELD_seed67137_dropout_total_loss.png)、[更新次数](../results/figures/MELD_seed67137_dropout_optimizer_steps.png)。

实验根目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/dropout_second_seed_confirmation/20261025T052219Z`。仓库根报告正文相同，相对链接按位置调整。
