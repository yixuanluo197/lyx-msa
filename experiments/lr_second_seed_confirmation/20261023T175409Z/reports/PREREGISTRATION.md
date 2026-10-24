# 第08组 MELD 学习率第二Seed独立确认：训练前登记

登记日期：2026-10-23 UTC。仅授权1次完整训练，run_id=MELD_cmerc_lr2e-4_seed2024；不因分数变化重复训练。

## 唯一科学变量与控制

从XP04原始 MELD_validation_seed2024/run_manifest.json 与 metrics.jsonl 提取训练/验证控制：lr=0.0001，最佳验证W-F1=66.01%，首次最佳Epoch=9，末10轮均值=61.223%。不读取其final_test、独评结果或测试预测。07中seed2024已于08:45:21 UTC预留。

本次lr=0.0002；其余科学args必须与该XP04同Seed逐项相同。完整非因果CMERC，seed2024/split_seed1475，1038训练/114验证对话，40轮、batch16、dropout0.4、Adam/l2=0.00003、GRU/ATen、continuous RNG、capacity_safe、FP32、BN；CC=.002、MSCL=.15、CSCL=.15均开启，CL=.4/scheduler_steps1。结构GRU/hyper/direct/concat_DHT/avl/L3/K3；实际完整argv与参数登记在configs/registry.json。

采用07已冻结的12个训练源与原watcher；只有已记录的skip-final-test工程补丁，与XP04原epoch循环AST完全相同。正式前完成5项CPUguard测试、真实长度40轮采样容量核对和独立首Adam预检；预检不是正式结果、不复用其权重或RNG，正式任务从头初始化。

## 选模与测试隔离

每轮训练和验证，按原实现四舍五入两位百分比验证W-F1严格改善保存best，同分保留最早轮。末10轮固定31–40。保留best/latest、模型/Adam/RNG状态；零测试loader迭代/零测试前向，不生成final_test。既有综合pickle仍含测试部分、代码会构造其loader元数据，不能称完全未加载含测试文件。作者难度来源fixed_author且生成史未知，仍保留相同来源和哈希。

## 判定规则（沿用07）

- 最佳验证W-F1提高且末10轮均值不下降：当前Seed支持候选继续验证。
- 最佳提高但末段下降：混合证据。
- 最佳值未提高：当前Seed不支持学习率改善。

报告两个Seed各自最佳与末段差值，不把两个相关轮统计或两个Seed宣称统计显著；不根据测试指标决定下一步。

## 资源与停止边界

同原服务器user23，限制可见GPU3 UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456。XP04 seed2024控制亦为同卡；seed67137旧控制不同卡。启动时GPU3已有其他用户PID2933763约352MiB，允许其正常共驻并记录；需连续3次资源检查满足free>=5000MiB/util<=65，未登记新进程则等待。预计本任务峰值约2200MiB，保留余量，不终止其他项目。共享算力下不作速度优劣结论。

只运行本项；不追加Baseline、Batch8、其他学习率、其他Seed或测试。技术失败先保留证据和检查可恢复状态，不自动重跑。04/05/06/07全部历史文件保持原字节，独立哈希核验；新源码、配置和启动器冻结后不随训练修改。
