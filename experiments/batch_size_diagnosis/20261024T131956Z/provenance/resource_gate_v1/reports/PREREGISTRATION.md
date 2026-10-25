# 第09组：MELD Batch Size独立对照训练前登记

2026-10-24 UTC登记；仅新增1次正式训练，run_id=MELD_cmerc_batch8_lr1e-4_seed2024。唯一科学配置变量为原始--batch-size从16改8，学习率固定0.0001。技术失败保留证据，不按分数重训或自动追加实验。

## 固定科学协议

完整非因果CMERC；MELD，seed2024/split_seed1475，40 Epoch、dropout0.4、Adam/l2=0.00003、GRU/ATen、continuous RNG、capacity_safe、FP32、BN、hyper/direct、concat_DHT/avl/L3/K3；CC=.002、MSCL=.15、CSCL=.15均开启，CL epoch_ratio=.4/scheduler_steps=1。其他完整args需与XP04同Seed逐项相同，LR禁止变化。实际argv及输出路径见configs/registry.json。

控制仅从XP04 MELD_validation_seed2024 的run_manifest.json与metrics.jsonl核对：batch16/lr.0001，完整40轮，最佳验证66.01%/第9轮，末10轮61.223%；不读取final_test或独立测试指标，不重复训练对照。数据为相同1038训练/114验证对话，split hash5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8，三份MELD特征/课程输入保留相同SHA。

## Batch变化的含义

原实现get_MELD_loaders使用同一batch_size构造训练及验证DataLoader，本组遵循原参数：训练batch8，验证batch8。验证对话全集、顺序和指标定义不变，W-F1按完整预测/标签集合汇总。原分类和辅助Loss按既有代码逐Batch记录，不重定义为逐样本归一化目标，不修改Loss权重、公式或CC尺度。

batch16→8同时改变每Epoch优化步数、BatchNorm训练统计、HSCL正负对和CC原始Loss尺度；连续RNG策略不变，但更多Batch/更新可能改变实际随机消耗轨迹。验证Batch分组也改变，本组不能将差异完全归因于训练Batch变化，跨Batchsize原Loss均值的分组权重也不同。容量CPU预检同时重放历史16与新8，完整核对训练池、采样实现与每Epoch容量/步数，不要求两个Batchsize的Batch数量或容量相等。

## 预检与数据隔离

训练源码12文件从08字节复制，保留07/08仅跳过最终测试的工程补丁；网络、Loss、epoch循环不改。外部观测器只将科学差异门禁改为batch_size-only并严格固定lr，保留不增加前向/反向/Adam步、RNG不受观测改变及测试调用在前向前拒绝的CPU验证。预检先CPU，再独立首Adam GPU探针；探针不计正式结果、不向正式运行提供权重或RNG。仅在Loss/梯度有限、参数确实更新、GPU释放后正式从头初始化。

每轮训练和验证，按原实现四舍五入两位百分比验证W-F1严格改善保存best，同分保留最早轮；保留best/latest、完整模型/Adam/RNG/进度。末10轮固定31–40。零测试loader迭代、零测试前向；综合pickle和测试loader元数据仍按原代码加载，不能称完全未加载含测试数据的文件。作者难度文件生成历史未知且含validation键，保持同来源配对，不将整个预处理链称为validation-free。

## 资源与唯一运行约束

沿用原服务器user23与GPU3 UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456，与XP04 seed2024同卡。当前GPU3既有其他用户PID1843864约430MiB，允许共驻但完整披露；启动需连续3次、间隔10s检查free>=5000MiB/util<=65且无新外部PID；预计本任务峰值预算2200MiB。活动期记录资源和进程，不发信号或终止其他项目，不把共享GPU耗时当独占速度。新冲突暂停后续准入，不自动重训或跨服务器迁移。

## 判定与停止规则

- 最佳验证F1提高且末10轮均值不下降：本Seed满足下一Seed确认候选资格。
- 最佳提高但末段下降：混合线索。
- 最佳值未提高：本Seed不支持主指标改善，其他辅助变化如实保留。

单Seed不证明跨Seed稳定收益或统计显著。最佳和末段来自同一曲线，不是独立统计证据；末段变化不用于事后替换主指标。本组不自动启动第二Seed、其他学习率/Batch Size、Baseline、五Seed或最终测试。04～08源码、权重、日志、结果和报告保持不变，独立哈希验收。
