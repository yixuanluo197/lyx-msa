# CMERC Batch Size 诊断

生成时间：2026-10-24T18:57:27.629590+00:00。本批仅1/1正式训练完成：MELD seed2024、lr1e-4、Batch8、40轮。

## 设计、参数与来源

本批直接对照XP04完整CMERC seed2024、lr1e-4、Batch16；不继承08的lr2e-4，不改变任何损失公式或系数。实际args只允许batch_size、skip_final_test、output_dir和provenance与历史不同；lr必须始终为0.0001。

实际模型：Adam、L2=3e-5、dropout0.4、GRU、ATen、continuous RNG、BN、capacity_safe、hyper/direct、concat_DHT、L3/K3、FP32、worker0、split_seed1475。CC/MSCL/CSCL=0.002/0.15/0.15，CL epoch_ratio0.4、scheduler_steps1，40轮、train1038/validation114对话。验证集完整池和ID顺序固定，split SHA-256为`5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8`。

实际train_cm.py SHA-256：`0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2`；12个训练源字节同08的来源记录`/data2/yb/paper/lyx-msa/CMERC/experiments/lr_second_seed_confirmation/20261023T175409Z/code_xp04`，来源Git HEAD为`506e35b46867c92c8edd9e426ca7f9bd8d68c3bc`。只继承既有skip-final-test工程门禁，训练/验证公式未改。

冻结环境：Python 3.10.13、PyTorch 1.13.1+cu117、PyG 2.4.0、CUDA build 11.7；依赖未新增。

## Batch变量的实际作用范围

原始--batch-size同时控制训练和验证loader，因此本次train与validation分组均16→8。比较对象是这一公开参数的整体影响，不能称只改变训练Batch而保持验证分组不变。验证W-F1由完整固定验证池的全部labels/predictions汇总计算，不是逐Batch F1的平均；按百分比两位小数严格提高才换Checkpoint，同分取首个最高轮。

Batch8同时改变每Epoch optimizer更新次数、BN统计、HSCL正负对组成及未按句数归一化CC的每Batch尺度。RNG策略和初始Seed相同不意味着global dropout RNG轨迹相同：Batch数量、张量形状与前向次数改变随机数消耗。L2与lr系数固定也不代表总更新次数或有效正则化轨迹相同。

源码model_cm.py:658,758–762,824显示双向GRU直接处理未pack的padded序列，BN亦包含padding；因此验证分组变化可能改变固定权重预测，不能预设它只改变Loss统计标尺。具体源码行号及mask/配对路径见[协议审计](../experiments/batch_size_diagnosis/20261024T131956Z/reports/PROTOCOL_AUDIT.md)。

原训练/验证Loss采用Batch均值汇总；不同大小及末Batch的权重随分组变化。原始CC/MSCL/CSCL均值跨Batch Size不能直接解释为同一个总样本归一化目标的增减，更不能仅凭数值减半判断校准作用增强或减弱。

## 最佳值、末段与原登记判断

| Batch Size（train/val） | 来源 | 最佳验证W-F1 % | Epoch | 末10轮均值 % | 末10轮时间窗SD |
|---:|---|---:|---:|---:|---:|
| 16 | XP04_batch16 | 66.0100 | 9 | 61.2230 | 0.8068 |
| 8 | 09_batch8 | 65.8900 | 9 | 63.0480 | 0.6756 |

Batch8−Batch16：最佳验证 **-0.1200 pp**，末10轮均值 **+1.8250 pp**；分类：**预登记主指标未改善**。

预登记主指标未改善，本Seed不支持扩大第二Seed或五Seed；保留末段与Loss辅助线索，不改主指标，不自动换学习率、Batch Size或Seed。

规则保持训练前登记：最佳值升且末10轮不降才具备第二Seed候选资格；最佳升末段降为mixed；最佳不升为no_best_improvement，不能凭辅助末段结果更换主指标。末10轮固定31–40，与最佳值来自同一曲线，是相关统计而非独立证据。一个Seed不提供p值、显著性、等价或五Seed稳定复现声明。

## 实际更新次数与Loss

| Batch | 40轮训练更新数 | 计数依据 | 40轮验证Batch总数 |
|---:|---:|---|---:|
| 16 | 2548 | 历史Batch16 CPU采样重建；不是原运行step日志观测 | 320 |
| 8 | 5096 | 每次真实optimizer.step后的watch日志计数，并与容量重放核对 | 600 |

| Batch | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 梯度异常Epoch |
|---:|---:|---:|---:|---:|---|
| 16 | 3.2980 | 4.1074 | 2.7076 | 5.0893 | 历史未记录 |
| 8 | 3.2102 | 4.0580 | 2.6544 | 4.7148 | 0 |

最佳验证轮→末10轮训练分项均值：

| Batch | 原始CC | 原始MSCL | 原始CSCL | 加权CC | 加权MSCL | 加权CSCL | 总Loss |
|---:|---|---|---|---|---|---|---|
| 16 | 57.0680 → 69.2599 | 3.8972 → 4.3891 | 2.8807 → 2.9733 | 0.1141 → 0.1385 | 0.5846 → 0.6584 | 0.4321 → 0.4460 | 4.4288 → 3.9505 |
| 8 | 36.0967 → 33.5536 | 4.2576 → 6.3008 | 3.0218 → 3.0485 | 0.0722 → 0.0671 | 0.6386 → 0.9451 | 0.4533 → 0.4573 | 4.3743 → 4.1239 |

- Batch16最佳轮→末段：train分类Loss -0.5903，val分类Loss +0.9819，val W-F1 -4.7870 pp。
- Batch8最佳轮→末段：train分类Loss -0.5558，val分类Loss +0.6568，val W-F1 -2.8420 pp。

train为在线训练态、val为固定模型eval，不同模式与参数时点下的曲线间距不是严格泛化gap。末段时间窗SD不是跨Seed标准差；有限梯度与真实参数更新只能证明所检数值和执行路径正常，不能据此排除目标竞争、伪目标噪声或泛化根因。训练辅助模型保留epoch入口快照语义。

## 零测试与配对边界

本次零测试loader遍历、零测试前向；不解析测试指标或预测数组、不据测试筛选。本阶段未计算ECE/CEL。原综合pickle及测试loader元数据仍加载，零前向不等于完全未加载含测试数据的文件。训练后原自动测试由明确skip-final-test门禁跳过，不进行新测试评估。

XP04 seed2024控制位于本地user23 GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456）；本批实际设备为同host GPU1（GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a）。本次与XP04控制为同host、同型号、不同物理卡；不能声称同物理卡配对。CUDA scatter数值非确定性、硬件/历史时点和共享资源仍限制严格因果与速度解释；不得称独占资源或连续隔离，不终止其他项目。registry当前登记的共驻PID为`[2783704, 2787561]`，实际观察到的PID另见最终资源证据。作者fixed_author课程包含validation键、生成史未知；使用同一文件保持配对，不称完整预处理链validation-free。

设备选择在任何GPU探针和正式训练前由CPU selector监控同host健康TITAN Xp候选GPU0/1/3；首个连续三次新采样满足资源v2门槛者作为唯一设备，记录实际registry和冻结清单后不再换卡。完整候选、时间及实际选择见[XP设备选择工程修订v3](../experiments/batch_size_diagnosis/20261024T131956Z/reports/XP_DEVICE_SELECTION_AMENDMENT_V3.md)与[设备选择记录](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/gpu_selection.json)。本修订不改变科学配置、候选资格或唯一训练预算。

资源准入v2按registry实际值执行：free>=3000 MiB、util<=95%、3次新采样、间隔10秒；本任务估计预算2200 MiB。新增外部PID是否暂停准入的登记值为`False`，不再以PID更替本身作硬拒绝；身份、资源和查询有效性仍需核验。

v2仅修订资源管理，原v1门禁与等待记录已原字节归档；科学配置、唯一正式预算、零测试与候选资格不变。时间和资源依据见[资源准入工程修订v2](../experiments/batch_size_diagnosis/20261024T131956Z/reports/RESOURCE_ADMISSION_AMENDMENT_V2.md)，不将等待结束或门禁放行解释为训练成功。

## 建议及证据状态

**当前直接建议：** 预登记主指标未改善，本Seed不支持扩大第二Seed或五Seed；保留末段与Loss辅助线索，不改主指标，不自动换学习率、Batch Size或Seed。

已证实事实是本次固定配置、日志轨迹、数值执行和登记主指标的比较；优化机制、校准收益、跨Seed稳定性和论文差距根因仍未证实。一个Batch参数改变产生多项相互耦合的执行变化，不能将观测差值分摊为BN、HSCL或CC的单独效应。后续仅提出独立预登记建议，不自动开展五Seed、其他Batch Size、lr、Seed或测试。

## 可追溯材料

[预登记](../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)、[配置](../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)、[环境](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/environment.json)、[数据](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/data_inputs.json)、[来源](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/source_origin.json)、[冻结清单](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/frozen_execution.json)、[工程补丁](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/skip_final_test.patch)、[容量与两种Batch的重放](../experiments/batch_size_diagnosis/20261024T131956Z/preflight/capacity_cpu.json)、[正式预检门禁](../experiments/batch_size_diagnosis/20261024T131956Z/preflight/gate.json)、[最终验收](../experiments/batch_size_diagnosis/20261024T131956Z/provenance/final_verification.json)。

[per_run.csv](../experiments/batch_size_diagnosis/20261024T131956Z/results/per_run.csv)、[comparison.csv](../experiments/batch_size_diagnosis/20261024T131956Z/results/comparison.csv)、[metrics_summary.json](../experiments/batch_size_diagnosis/20261024T131956Z/results/metrics_summary.json)、[curves.csv](../experiments/batch_size_diagnosis/20261024T131956Z/results/curves.csv)、[loss_curves.csv](../experiments/batch_size_diagnosis/20261024T131956Z/results/loss_curves.csv)。

[W-F1曲线](../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/MELD_batch_size_wf1.png)、[分类Loss](../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/MELD_batch_size_classification_loss.png)、[加权总Loss](../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/MELD_batch_size_weighted_loss.png)、[每轮更新计数](../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/MELD_batch_size_optimizer_steps.png)。其余原始及加权辅助Loss图和全部输入/输出SHA见[图表清单](../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/manifest.json)。

预检门禁实际状态：`passed`，test前向计数`0`。

最终验收实际状态：`passed`。
单run完成40次train/40次validation，记录5096个真实训练Batch，test前向0。登记Batch8排列重放匹配`True`，原sampler AST与历史Batch16重放通过`True`，实际train/val Batch均`8`、lr保持`0.0001`。

best/latest两个Checkpoint分别第9/40轮，模型和Adam有限`True`/`True`，完整RNG存在`True`/`True`；验收脚本同时核对完整args/provenance和Adam lr。best SHA `7f39a7da4935a5b8a2520c16f86ff25da13e3c8644f259ef714883d115dd1b23`，latest SHA `2c537b1dd32295b996e0b57ae1815d4d7a64239525870b1be272e031cc7a55d2`。冻结源/输入未变`True`，历史文件哈希保全合计1691项。

资源证据`limited`：采样450次、预登记共驻PID采样450次、新外部PID采样0次、查询错误0次；正式阶段最小空闲31 MiB、最大利用率100%。

最终验收记录的预登记共驻PID：`[2783704, 2787561]`；实际观察到的外部PID：`[2783704, 2787561]`。启动准入门槛（非持续运行门槛）为free>=3000 MiB、util<=95%，新增PID暂停准入值`False`。离散采样不能证明连续独占或资源隔离；运行期压力和新增PID仅记录，不终止他人任务。

本批根目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/batch_size_diagnosis/20261024T131956Z`。仓库根报告正文一致，链接按所在位置调整。

## 结果解释与下一步最小建议

**已证实：** Batch8最佳验证W-F1未超过Batch16；末10轮均值提高1.825个百分点，时间窗波动从0.8068降至0.6756。两组最佳验证轮均为第9轮，此后训练分类Loss继续下降、验证分类Loss上升，验证W-F1下降。新运行40轮Loss和已记录梯度均有限，无OOM或自动重训。

**有证据但尚未隔离的推断：** 曲线符合后期过拟合迹象，Batch8的后期衰退较轻；它没有消除衰退，也没有带来最佳验证主指标改善。不能把时间窗波动减小解释为跨Seed训练稳定性提高，不能把Loss尺度变化归因于某个单独机制。

**下一步建议（本次不执行）：** 暂不为Batch8追加第二Seed或最终五Seed，正式方案暂保留XP04的Batch16/LR0.0001参照。若后续专门验证Batch组织机制，先在独立预登记协议固定验证分组，只改变训练分组，以区分训练轨迹与验证padding分组效应；选模规则仍用验证W-F1，禁止测试集筛选。该设计需要新的明确实验预算，不属于本组补跑。不因本次末段辅助指标提高而改用末段均值作选参标准。
