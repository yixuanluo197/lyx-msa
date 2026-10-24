# CMERC 学习率第二Seed确认

生成时间：2026-10-23T18:50:44.958903+00:00。本批仅1/1正式运行完成：MELD seed2024、lr2e-4、40轮。

## 设计与零测试协议

首Seed67137在第07组依据预登记的最佳验证值优先排序选定lr2e-4；seed2024已在该阶段预留。本批只运行已选学习率的第二Seed，与XP04 seed2024完整CMERC lr1e-4作历史控制，不追加另一学习率、Baseline、Batch8、其他Seed或重训。两Seed中首Seed承担配置选择，第二Seed承担独立确认；不能把两点均值当作预先固定方案的五Seed性能估计。

训练科学配置沿用XP04完整CMERC：Adam、batch16、40轮、dropout0.4、GRU、ATen、continuous RNG、BN、capacity_safe、hyper/direct、concat_DHT、L3/K3、split_seed1475、L2=3e-5；CC/MSCL/CSCL系数0.002/0.15/0.15、CL比例步长0.4/1。唯一科学配置变量为lr1e-4→2e-4。规范验证集Weighted F1按百分比两位小数选首个最高轮，严格提高才更新Checkpoint。

实际train_cm.py SHA-256：`0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2`。源码复制自07的`/data2/yb/paper/lyx-msa/CMERC/experiments/learning_rate_diagnosis/20261023T084353Z/code_xp04`，12源字节一致审计状态为`True`，只保留07 skip-final-test补丁；来源记录中的Git HEAD为`7dc014805adfaf37de0754bca7ddbe8a7427a053`。

实际冻结环境：Python 3.10.13、PyTorch 1.13.1+cu117、PyG 2.4.0、CUDA build 11.7、cuDNN 8500；本批未安装或升级依赖。

本阶段零测试loader遍历、零测试前向，不解析新旧测试指标、不读取预测数组或据测试筛选。综合pickle及测试loader元数据仍由原流程加载，不能称完全未加载含测试数据的文件。skip-final-test仅跳过训练后的原自动测试块，训练/验证数学与保存规则不变。本阶段未计算ECE/CEL，不能以未测值声称校准改善。

## 两Seed的两个学习率

W-F1单位为百分比，差值为百分点。末10轮固定指31–40轮。

| Seed | lr | 来源 | 最佳验证W-F1 | Epoch | 末10轮均值 |
|---:|---:|---|---:|---:|---:|
| 67137 | 0.0001 | XP04_control | 65.2400 | 8 | 62.9090 |
| 67137 | 0.0002 | 07_selected_lr | 66.0700 | 8 | 62.9830 |
| 2024 | 0.0001 | XP04_control | 66.0100 | 9 | 61.2230 |
| 2024 | 0.0002 | 08_confirmation | 65.4600 | 5 | 62.0340 |

| Seed | Δ最佳验证 | Δ末10轮均值 | 按07登记的分类 |
|---:|---:|---:|---|
| 67137 | +0.8300 | +0.0740 | 满足候选资格（非统计强证据） |
| 2024 | -0.5500 | +0.8110 | 主指标未改善 |

**确认结论：** 第二Seed未提高最佳验证W-F1，未确认首Seed的主指标收益。

Seed2024的最佳验证值变化-0.5500 pp，末10轮均值变化+0.8110 pp。末段改善作为训练轨迹的辅助线索保留；本轮仍以预先确定的最佳验证W-F1为主指标，不据末段结果改变选模或确认规则。

暂缓扩大五Seed；分析已有训练/验证轨迹，不临时改试另一个学习率、换Seed或追加Batch8。

资格门槛沿用07：Δ最佳>0且Δ末10轮>=0才合格；最佳升但末段降仅为弱线索，无候选资格。该门槛没有预设最小效应量或置信区间，不等于统计强证据。最佳与末段来自同一验证曲线，是相关统计；两Seed不作p值、显著性、等价或稳定复现声明。

## 训练Loss与数值稳定性

| Seed/lr | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 末段W-F1时间窗SD | 非有限梯度Epoch |
|---|---:|---:|---:|---:|---:|---|
| 67137/0.0001 | 3.3453 | 4.1291 | 2.7067 | 5.1819 | 0.4975 | 历史未记录 |
| 67137/0.0002 | 3.1508 | 4.2662 | 2.6640 | 5.1364 | 0.4930 | 0 |
| 2024/0.0001 | 3.2980 | 4.1074 | 2.7076 | 5.0893 | 0.8068 | 历史未记录 |
| 2024/0.0002 | 3.4417 | 4.0139 | 2.6626 | 5.2636 | 0.7276 | 0 |

- Seed67137/lr=0.0001：最佳验证轮→末10轮均值，train分类Loss变化-0.6386，val分类Loss变化+1.0528，val W-F1变化-2.3310 pp。
- Seed67137/lr=0.0002：最佳验证轮→末10轮均值，train分类Loss变化-0.4868，val分类Loss变化+0.8702，val W-F1变化-3.0870 pp。
- Seed2024/lr=0.0001：最佳验证轮→末10轮均值，train分类Loss变化-0.5903，val分类Loss变化+0.9819，val W-F1变化-4.7870 pp。
- Seed2024/lr=0.0002：最佳验证轮→末10轮均值，train分类Loss变化-0.7791，val分类Loss变化+1.2497，val W-F1变化-3.4260 pp。

训练原始辅助Loss（最佳验证轮→第31–40轮均值；固定系数CC=.002、MSCL/CSCL=.15）：

| Seed/lr | CC | MSCL | CSCL | 总训练Loss |
|---|---|---|---|---|
| 67137/0.0001 | 60.8713 → 71.5387 | 4.0387 → 5.0568 | 3.2041 → 3.0096 | 4.5534 → 4.0597 |
| 67137/0.0002 | 59.8155 → 59.9490 | 3.8347 → 3.8701 | 2.7218 → 2.8232 | 4.2539 → 3.7879 |
| 2024/0.0001 | 57.0680 → 69.2599 | 3.8972 → 4.3891 | 2.8807 → 2.9733 | 4.4288 → 3.9505 |
| 2024/0.0002 | 56.1418 → 61.7781 | 3.8810 → 3.7934 | 2.7780 → 2.9592 | 4.5528 → 3.7990 |

以上是拟合继续增强或末段表现回落的描述线索。train为在线训练态逐Batch预测，val为固定权重eval，模式和参数时点不同，不能把两条Loss曲线间距当严格泛化gap。时间窗SD是相关Epoch内起伏，不是跨Seed标准差。finite梯度只证明所检数值有限，不能据此排除优化竞争、伪目标噪声或泛化问题。

保留全部分类、原始CC/MSCL/CSCL及固定系数加权曲线。在线训练辅助项使用epoch入口辅助快照，Loss数值不等于参数梯度方向或校准质量。原实现Adam(weight_decay=l2)为耦合L2；L2系数固定不意味着改变lr后有效正则化更新轨迹相同。

## 配对与资源限制

seed2024的XP04控制和本次均为GPU3、UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456，改善了同物理卡配对；67137历史控制为GPU0，而07新lr2e-4为GPU3。CUDA scatter非确定性、历史运行时点和当前其他进程共驻仍限制严格因果解释。共驻任务及资源准入见[registry](../configs/registry.json)和资源日志；不终止其他项目，不把共享资源下耗时称纯模型速度。

固定作者课程包含validation键，生成史未知；两臂同文件保持协议配对，但不能声称完整预处理链validation-free。单次验证选模与两Seed结果不能证明泛化到测试集或五Seed平均表现，更不能宣称lr是论文差距唯一根因。

## 后续实验边界

对最终五Seed的直接建议：暂缓扩大五Seed；分析已有训练/验证轨迹，不临时改试另一个学习率、换Seed或追加Batch8。

本阶段暂缓Batch8，没有独立隔离证据支持改变Batch。它同时改变更新步数、BN统计、HSCL对和未归一化CC尺度，需要另外预登记；暂缓属于控制变量与解释性安排，不是证明Batch8无益。不得通过换Seed或追加lr凑结果。

## 产物与验收

[预登记](../reports/PREREGISTRATION.md)、[配置](../configs/registry.json)、[训练环境](../provenance/environment.json)、[数据输入](../provenance/data_inputs.json)、[冻结清单](../provenance/frozen_execution.json)、[仅跳过最终测试的补丁](../provenance/skip_final_test.patch)、[预检](../preflight/gate.json)与[完整验收](../provenance/final_verification.json)。检查路径存在性不等于这些门禁已经通过；验收状态以对应JSON实际内容为准。

[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)、[metrics_summary.json](../results/metrics_summary.json)。所有CSV仅含train/validation字段。

| Seed | W-F1轨迹 | 分类Loss | 加权总Loss |
|---:|---|---|---|
| 67137 | [PNG](../results/figures/MELD_seed67137_wf1.png) | [PNG](../results/figures/MELD_seed67137_classification_loss.png) | [PNG](../results/figures/MELD_seed67137_weighted_loss.png) |
| 2024 | [PNG](../results/figures/MELD_seed2024_wf1.png) | [PNG](../results/figures/MELD_seed2024_classification_loss.png) | [PNG](../results/figures/MELD_seed2024_weighted_loss.png) |

正式预检门禁状态：`passed`；其首步探针状态`passed`，参数实际更新记录`True`，test前向计数`0`。

最终验收状态：`passed`。
单次完整验收覆盖40次train/40次validation、2548个训练Batch；唯一正式运行数1，test前向0。逐轮训练池/排列重放匹配状态`True`，Loss和梯度有限检查`True`。

best/latest共2个Checkpoint：best第5轮，latest第40轮；模型及Adam状态有限分别为`True`/`True`，完整RNG字段存在分别为`True`/`True`。验收脚本同时比对Checkpoint完整args/provenance与实际manifest，检查Adam参数组lr和完整RNG结构；通过状态不等于跨运行逐位数值复现。

best SHA-256：`81762564524e7295b3d1254a84c476ade08898f9baab494229f28626ac7bc312`；latest SHA-256：`7027986a045d567d7cf1ca902f380cb79834cc6694dd21f97168af0eec32067e`。冻结源/输入未变状态`True`；历史字节哈希保全合计1555文件，具体根目录计数见验收JSON。

资源证据状态`limited`：270次离散采样，许可共驻PID采样270次、新外部PID采样0次、查询错误0次。正式阶段最小空闲显存9867 MiB，最大利用率82%；三次样本准入政策通过`True`。

资源采样限制：许可的既有PID共驻不等于独占GPU；运行期间新外部PID或资源压力只记录、不终止其他任务；离散nvidia-smi采样不能证明连续资源隔离。

实验根目录：`/data2/yb/paper/lyx-msa/CMERC/experiments/lr_second_seed_confirmation/20261023T175409Z`。仓库根报告正文一致，相对链接按位置调整。
