# XP12：CMERC w/o HSCL 训练前登记

登记在任何XP12 GPU前向之前。服务器本地user23；GPU0/UUID GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248，与XP04 MELD seed67137同物理卡。仅新增1次正式训练，不追加Seed、参数、Baseline、五Seed或测试。

## 问题与唯一因素

检验关闭HSCL优化贡献是否改善MELD seed67137的验证W-F1。唯一科学变量contrastlearning True→False；calibrate=True、courselearning=True保持。Dropout0.4、CLI rec_dropout0.1、lr0.0001、Adam L2=3e-5、train/val Batch16/16、40 Epoch、split_seed1475、continuous RNG、ATen GRU、capacity_safe全部继承XP04。hyper/direct、concat_DHT、BN、num_L3/num_K3、其余解析args、数据输入及初始化流程不变。完整实际命令与配置见configs及运行observer。

rank_coff=.002；mscl_coff=.15、cscl_coff=.15保留元数据。训练目标严格为L_classification+.002*L_CC。MSCL/CSCL有效加权贡献为0，不能把系数元数据误认为启用，也不把原始MSCL/CSCL数值伪造为0。MELD分类Loss仍为原FocalLoss；不改损失数学。

当前源码外层为contrastlearning OR calibrate。保留CC时，五路无梯度/eval辅助前向、伪标签及原始HSCL损失计算仍执行；仅HSCL目标/反传贡献关闭。不额外删计算或补随机数来强行对齐RNG，不改模型构造。HSCL计算无显式随机采样；开关不影响初始化和首步前向的显式随机路径，但梯度/后续权重会变。CUDA scatter非确定性仍存在，不承诺完整数值轨迹一致。

## 输入与历史控制

固定XP04 1038训练/114验证对话，split SHA256=5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8；固定作者MELD_diff.pkl，其生成史未知且包含验证键。仅在固定训练池选课程样本，epoch_ratio=.4/scheduler_steps1，比例.4/.8/1…；不修改难度或数据。

原始XP04 manifest及40轮metrics重新核实：seed67137/dropout.4/B16/lr1e-4/CC+HSCL+CL，best65.24%/e8，末10轮62.909%。若不一致则不启动。XP06 w/o CC只作描述性背景，不是本次主要对照；不读取其final_test或据测试选择配置。初审打开XP06 README时可见旧测试列，配置与判据此前由用户明确固定；本组只提取训练/验证净化证据，不复述旧测试值，不称从未接触任何旧测试指标。

## 预检、资源与运行预算

先做CPU参数反例守卫、采样容量40轮重放及合成梯度检查；随后仅一个独立首Batch非正式GPU前后向，确认class/CC可导、CC非零梯度、HSCL不参与total及真实Adam更新。额外autograd诊断限于probe，retain_graph且不改变参数/.grad/RNG。probe模型/优化器/RNG不复用到正式；正式新进程从同Seed初始化，不载历史或作者权重。不额外做完整模型GPU配对probe，也不宣称测得与XP04全程RNG等价。

资源准入连续3次/10秒间隔：选定UUID一致、free>=3000MiB/util<=95%，自身峰值预算2200MiB并另留800MiB余量。允许与轻任务共驻，记录已有及新外部PID，不终止/暂停其他任务；查询失败不得准入。资源不足排队，CPU或probe不算完成；单次启动锁/状态机防重复。正式不中途早停、不按分数改配置。

## 指标与固定判据

每轮完整train及固定全validation池，记录Weighted F1/Accuracy、可得Macro F1、class/CC/raw MSCL/CSCL/effective贡献/total Loss、梯度有限性、课程比例与排列。验证Weighted F1按原百分比四舍五入2位严格提高选best，平局保留最早；保存latest及完整model/Adam/Python NumPy Torch CUDA RNG。唯一正式40轮；零测试loader遍历/测试前向、不计算新test F1/ECE/CEL。综合pickle/test loader元数据仍由原流程载入，不等同于测试前向。

主要比较XP12 vs XP04：best、epoch、末10轮31–40均值与样本时间SD(ddof1)、train/val分类Loss、CC贡献和F1衰退。best和末段均严格提高=本Seed积极证据，可建议另登记确认；best提高但末段降低=混合证据；best未提高=本Seed不支持主要分类指标改善。best提高且末段恰好相同不满足双升，保留不完整/混合证据。单Seed不做显著性检验，不推断测试收益、HSCL无效或确定目标竞争机制，不自动追加实验。

## 冻结、验收和交付

本独立目录code_xp04、configs、scripts、输入SHA、判据及净化历史来源启动前冻结。源码仅继承已有skip-final-test兼容补丁；逆补丁须还原XP04 SHA。观测器工程适配单独diff，不改网络/Loss。历史04～11公共档案与原始批次启动前快照、结束后重哈希。

正式40train/40validation/test0后独立CPU验收：args唯一contrast差异、40轮采样池及顺序、2548训练步、有限loss/梯度、有效CC/零HSCL贡献、best/latest及Adam/RNG、数据/源码/历史哈希。仅通过后生成reports/CMERC_HSCL_ABLATION_DIAGNOSIS.md、per_run/comparison/curves/loss_curves/summary及图，归档experiment_records/12_hscl_ablation_validation并发布GitHub。原始数据、权重、逐Batch原始记录和原始日志留服务器。
