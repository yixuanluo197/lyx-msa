# CMERC 最小根因验证：训练前登记

本批仅登记3项从头正式训练。MELD优先，随后为独立规划的IEMOCAP Batch配对；不追加Seed、不执行w/o HSCL或w/o CL、不开展五Seed最终实验。主比较对象是XP04同Seed验证选模结果，05组只提供问题来源，不能充当本批同协议对照。

| 新Run | Seed / split seed | Epoch / batch | 唯一科学因素 | 历史对照 |
|---|---|---|---|---|
| MELD_wo_cc_seed67137 | 67137 / 1475 | 40 / 16 | calibrate=True→False；HSCL和CL保持开启 | XP04 MELD完整CMERC seed67137 |
| IEMOCAP_baseline_source_order_seed1475 | 1475 / 1475 | 80 / 16 | capacity_safe随机排列→source_order固定源顺序 | XP04 IEMOCAP Baseline seed1475 |
| IEMOCAP_cmerc_source_order_seed1475 | 1475 / 1475 | 80 / 16 | 同上，完整CMERC三机制原样开启 | XP04 IEMOCAP完整CMERC seed1475 |

全部run沿用XP04 fixed_author课程、原始分类Loss、continuous RNG、ATen GRU、非因果BN、FP32、lr1e−4/l2=3e−5、worker0及1000/5200容量。MELD dropout .4、L3/K3、CC登记系数.002但关闭加权、MSCL/CSCL .15/.15、课程.4/步长1；IEMOCAP dropout .5、L3/K4，完整模型系数.005/.05/.05、课程.15/步长1，Baseline三机制及系数均关闭。其余完整有效参数逐项对照历史manifest，见configs及实际run_manifest，不依赖CLI默认值推断。

## 选模、测试与结果解释

每轮只训练和验证，以百分比两位小数Weighted F1严格提高时保存best，平局保留最早Epoch；规则与XP04一致。训练结束仅加载该best，完成一次测试前向评估；从该次预测计算Accuracy、Weighted/Macro F1、15-bin ECE和五路log概率正差总和CEL，不另调用独立GPU评估器。Baseline的五路移除仅在这一次最终测试中作评价诊断，不启用任何训练正则。不会以测试分数改配置、挑Seed或重跑。

MELD主要证据是w/o CC与历史完整模型的最佳验证W-F1差值、完整验证轨迹及预设末10轮验证均值；测试差值只作固定模型的描述性结果。IEMOCAP主要比较两种排列下CMERC−Baseline验证收益的变化，同时报告两臂各自结果和最终测试差值。单Seed没有跨Seed方差，不作稳定根因或统计显著性声明。后续机制消融和五Seed只提出建议，不在本批自动执行。

## 配对与已有边界

MELD划分为1038/114/280对话，split SHA `5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8`；IEMOCAP为108/12/31，SHA `d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62`。数据及外部难度SHA与历史相同。作者课程生成过程未知、包含当前验证键，故只声称本轮训练/选模遵守持出验证，不声称整个特征/难度生成链完全validation-free。

MELD使用逐字节XP04训练副本，calibrate关闭时因为contrastlearning仍为True，五路辅助前向、目标和raw CC仍计算，只从总Loss移除CC。IEMOCAP仅新增显式source_order sampler和门禁，保留末批、全样本及容量；先丢弃一次私有randperm以匹配历史本Seed零重采样的loader RNG消耗，不改变全局RNG。不同Batch形状导致dropout随机数映射和BN变化属于排列作用路径，不通过重播种隐藏。

历史MELD完整模型在GPU0运行，本次该卡被其他项目占用。模型归属仍为同一服务器user23，使用同型号TITAN Xp GPU3，UUID `GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456`；跨物理卡不保证逐位一致，不能把历史对照称为全部运行条件完全相同的严格因果证明。科学配置层面的唯一差异由自动门禁核对，设备UUID、源码路径、输出和观测工程差异单列。

## 产物与资源

本批根目录：`experiments/minimal_ablation_validation/20261023T043115Z/`。完整argv、每项唯一因素、代码目录及绝对输出在 `configs/registry.json`；实际训练目录为 `runs/<run_id>/<Dataset>_validation_seed<seed>/`，不会复用历史目录。原始源码、历史档案、权重和日志以 `provenance/history_before.json`逐文件记录，收尾再次核验；历史树只读，新数据链接指向既有特征。

GPU0/1/2已有其他项目，禁止终止。GPU3启动前要求UUID/型号匹配、无其他计算PID、free≥11500MiB且利用率≤10%。历史IEMOCAP reserved峰值约10440–11210MiB，加MELD约1738MiB及上下文余量不能安全容纳于12288MiB，故本批串行，不改变batch或精度来塞并发。逐次 admission、PID、命令、退出码及资源快照全部保存。资源竞争或错误保留证据并停止相关任务，不按分数自动重试。

CPU完整160轮容量/RNG检查、观测器不改变随机/更新的toy检查，以及真实首Batch有限Loss/反向/参数更新均须通过才启动正式训练。预检输出独立，不计入3项正式训练。正式配置和执行源码在GPU正式任务前冻结；报告分析脚本可在训练期间独立开发，不改变已启动run。
