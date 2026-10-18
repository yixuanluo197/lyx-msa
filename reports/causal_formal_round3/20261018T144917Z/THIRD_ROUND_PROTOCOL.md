# 第三轮：跨数据集、三配对种子预注册协议

冻结日期：2026-10-18 UTC；批次20261018T144917Z；起始commit `c1134e22ed9d6ae564b5e394de5686e111ee37ee`。实际运行源码、流程工具、数据/ID、环境和本协议SHA256在PLAN.json固定。沿用本地user23服务器归属，初始健康空闲设备为GPU0/1/3，以UUID绑定；GPU2正被其他任务使用，不竞争其资源。每次启动重新核对，资源不足排队，不跨服务器迁移或降低预算。

## 已知证据与研究范围

第二轮IEMOCAP/1937的测试结果已经观察：W-F1 63.958180→65.059289、ECE18.520027→14.550467、CEL1489.332581→966.648720；Angry F1下降8.079726个百分点，CEL下降较多来自intra/inter GNN边视图，ECE下降不保证AUROC-Success同步提高。此seed经真实SHA、权重/初始化、teacher/difficulty、数据及协议核查后原样复用，不训练、重新选模或重复测试。

本研究因此不是完全盲测或纯确认性研究。三seed汇总包含已经可见的IEMOCAP1937，属于预先固定运行矩阵后的复现/稳健性验证；另报告IEMOCAP新增2025/3407的描述性敏感性结果，不将其替换三seed主表。MELD1937作为技术先导，其测试会在其他四配对运行前可见；后续所有配置在先导之前冻结，不能据其性能变化改变矩阵。

本轮只训练Baseline与完整causal CMERC，不执行组件消融、替换策略或辅助策略新对照，不改变核心图、CC、HSCL、课程方法。保留prefix_mean_v1和legacy_epoch_snapshot，以及每次train/eval函数重设seed的既有RNG协议。prefix_mean是可逆前缀平滑，不称完全模态移除；intra/inter是GNN-edge ablation，不称完全切断上下文。上游特征提取未审计。

## 固定矩阵与超参数

两个数据集各训练seed `[1937,2025,3407]`，每seed Baseline与CMERC实际初始化state_dict SHA相等。IEMOCAP1937复用第二轮，新增正式训练共10个run；IEMOCAP新增4×80轮，MELD新增6×40轮，共560个训练epoch。

共同参数：causal=True、validation选模、GRU、hyper/direct/concat_DHT/avl、LN、num_L=3、batch16、Adam lr=.0001、weight_decay=.00003、rec_dropout=.1、use_speaker=True、use_modal=False、residue=False、av_encoder=False、num_workers=0、dropout正常启用。其余参数使用已核实官方wrapper默认，不依据结果调参。两种方法的epoch预算相同；CL早期自然减少样本/优化器步数，不声称step数相同。

| Dataset | epochs | dropout | num_K | CC系数 | MSCL | CSCL | 初始课程比例/步长 |
|---|---:|---:|---:|---:|---:|---:|---|
| IEMOCAP |80|.5|4|.005|.05|.05|.15 / 1 epoch|
| MELD |40|.4|3|.002|.15|.15|.4 / 1 epoch|

Baseline关闭CL/HSCL/CC；完整模型三者全开。IEMOCAP使用split seed1475对话级10% holdout（108train/12validation）；MELD按官方train/dev ID（1038train/114validation），训练排除dev；split seed1475仅作共同身份记录，不对MELD随机重划分。特征完整文件SHA与有序train/validation ID哈希见data/*/DATA_ENVIRONMENT.json，固定之后每阶段检查实际字节。

## 依赖与测试开放顺序

1. 有限真实train/validation GPU preflight。冻结独立源码快照后，先MELD/1937 Baseline40轮，验收best/latest、Adam/RNG、全部epoch和非有限值保护；按验证W-F1（现入口四舍五入2位、并列取首次）选best。
2. 只用对应Baseline validation-selected best生成真实difficulty，严格匹配seed、特征、train/validation IDs、teacher架构/训练/选模协议和prefix_mean_v1。全部且仅train ID，独占SHA寻址文件和sidecar，Full前后与checkpoint均绑定B阶段固定SHA。
3. MELD/1937 Full40轮，实际初始SHA与其Baseline相等；验收后两组validation评估/完整比较，冻结比较、原始评估输入、两best、数据与协议哈希，再各best一次正式test。
4. “MELD闭环正常”仅指技术验收通过：完整预算、有限损失/梯度、实际更新、数据/teacher/初始化/选模/SHA正确；不以W-F1/ECE/CEL改善作为进入剩余矩阵的条件。负结果同样进入后续固定矩阵。
5. 执行其余四配对：IEMOCAP/2025、3407，MELD/2025、3407。各自严格依序Baseline→difficulty→Full→validation。按实际资源可并发独立run；GPU容量先测，保留余量，不能改batch16/精度/预算换并发。
6. **所有剩余四配对的validation结果、选模对象与输入哈希全部冻结后**，统一写TEST_AUTHORIZATION.json，再对其8个best各一次test。先导MELD的已冻结结果及复用IEMOCAP1937作为已观察参考写入总冻结清单。不能提前评估其中某个seed测试来决定是否继续其他seed。

每run使用--defer-final-test、--fail-on-nonfinite、--diagnostic-batches 1，完整保存best/latest、模型/Adam、Python/NumPy/torch/CUDA RNG、课程进度及历史best引用。每epoch首batch记录真实分项及乘系数后梯度、loss、参数更新、辅助模式；所有train batch在更新前检查loss/活跃梯度有限。未执行旧参数无需梯度。保存全部epoch，不仅取最好轮的有利片段。

失败/中断不删除、不覆盖，记录阶段命令、PID、GPU、错误、最后epoch/checkpoint和哈希。仅按验证过的完整resume协议显式续训，不将权重加载称完整恢复；不能为性能重训同seed或替换超参数。若硬件或数据/provenance无法满足，停住下游并保留部分真实状态。

## 指标与预定统计

validation是主要过程比较，test为冻结协议后的预定最终复现评估，两者分开。固定15个等宽confidence bins，按现实现((m-1)/15,m/15]；Accuracy/W-F1/Macro-F1/各类F1、ECE/MCE/RMSCE、AUROC-Success/AUPRC-Success均用百分数，差值为百分点。AUROC-Success正类=预测正确，score=max softmax；某段全对/全错时记不可定义，不以0替代。

CEL统一prefix_mean_v1，报告总量和text/audio/visual/intra_context/inter_context五项：true-class log-prob正差求和，非百分比。每数据集内相同样本数配对比较；不把IEMOCAP/MELD总CEL直接横向比较为优劣，补充每话语CEL便于尺度诊断，不改变主定义。

预注册相对位置 `(zero_based_index+.5)/完整对话长度`：early[0,1/3)、middle[1/3,2/3)、late[2/3,1]；长度1归middle。保留三个段全部分类与校准、各类F1。资源报告逐epoch耗时和、进程墙钟、峰值allocated/reserved显存（已记录者）、完整Adam实际step数；共享服务器时间只作描述比较，并记录并发情况。

每个dataset/metric报告3个seed各自Baseline、Full、Full-Baseline绝对差、两方法均值/样本标准差(ddof=1)、配对差均值/样本SD。95%配对均值t区间df=2，明确小样本/正态性假设，仅作描述；不对话语当作独立重复来虚增n。

预定推断：两个数据集×(W-F1,ECE,CELtotal)共6个主要差值，validation/test各为独立的6检验家族；对配对差均值做**双侧精确sign-flip随机化检验**，枚举2^3=8种符号，统计量abs(mean(delta))，计入并列，不用随机抽样p。Holm校正每家族6项。n=3时两侧最小p=.25，因此不能产生通常alpha=.05的显著性证据；无论效果方向/幅度都原样报告。不将这个探索性检验包装成确认性创新证明。

其他类别、位置、MCE/RMSCE/AUROC、各CEL项与梯度为预定描述分析，不声称控制所有后验多重比较。重点核对Angry F1、context CEL减幅在总减幅中的贡献（分母非正/近零时不给误导比例）、ECE与AUROC-Success是否方向一致。只描述跨seed重复次数和不确定性，不依现象改协议。

## 交付与证据范围

THIRD_ROUND_PROTOCOL.md、MELD_BASELINE_VS_CMERC.md、MULTISEED_PAIRED_RESULTS.md、CALIBRATION_DIAGNOSTIC.md、CLASS_AND_POSITION_ANALYSIS.md、LOSS_GRADIENT_ANALYSIS.md、NEXT_RESEARCH_CANDIDATES.md、FINAL_EXECUTION_STATUS.md及可核验配置/哈希/原始日志/统计脚本。区分已测事实、待验证现象、候选研究假设；不将候选问题称已证实创新。

权重、特征、真实ID/预测和大型日志留服务器；公开交付仅去标识聚合、协议、配置、统计代码和必要小日志。保留第一/二轮所有结果，新的输出目录独占创建。
