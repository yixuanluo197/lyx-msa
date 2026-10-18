# XP主线技术预检与资源预算

状态：通过；本页只报告工程预检，不是正式五seed结果。

| 检查 | 证据与实测结果 |
| --- | --- |
| 来源语义 | 模型、训练器、dataloader、分类/CC/MSCL/CSCL损失、评估和统计源码与V100 V2逐文件相同；只改XP调度工程 |
| 输入/划分 | 六输入SHA与V2二十份manifest一致；两数据集全部seed/两模型同split |
| 真实长度采样 | 20项/1200epoch无容量删样本；IEMOCAP最大998边/2850节点，MELD315边/801节点；valid/test固定顺序安全 |
| CPU测试 | 172 passed、1 skipped、1 deselected；缺失的难度生成辅助模块相关测试不适用，初始错误保留 |
| XP调度测试 | 18 passed；UUID、实际型号/free、PID、锁、防覆盖、stale-util交接、失败保留 |
| GPU算子/更新 | PyG/scatter/sparse真实前反向与Adam参数更新通过；两数据集batch16全模型smoke通过 |
| 分项梯度 | 两数据集真实CMERC batch的分类/CC/MSCL/CSCL到主hidden均有限非零，observer仅预检启用 |
| 官方checkpoint | IEMOCAP W-F1 71.9837935825 / CEL1143.4927305033；MELD66.8483922007 / 605.0503030568，满足预定0.05点/0.1容差 |
| 同卡中断恢复 | 真实IEMOCAP完整CMERC两epoch；model/Adam预设rtol1e-3/atol1e-4通过，全部RNG/采样/选模完全相同，max参数差3.534369e-6；另fresh-process GRU逐位恢复单测通过 |
| 完整课程与评估 | IEMOCAP7轮至108全池、MELD3轮至1038全池；baseline全池；完整validation/test及五路移除batch16 |
| 失败/重试 | 隔离dummy进程实际exit17，保留失败任务/日志后同一注册ID重排；已有checkpoint先检查完整续训，不盲目重启 |

数据/课程边界：fixed_author难度含当前validation键；本批训练先排除validation再取curriculum，原作者生成评分是否排除当前holdout未知。不声称全预处理validation-free；不重建或覆盖难度。

环境沿用 /data2/yb/conda_envs/cmerc-xp-repro，未安装或修改环境。现有torch1.13.1 UUID计数补丁SHA已验证并做真实映射检查；GC0.8+fraction1.0只用于已匹配版本，保留FP32/batch16/原容量。缺matplotlib时复用V2标准库SVG后备。

GPU1在预检期间被其他项目占用，保留MELD/恢复已完成证据后，将IEMOCAP预检串行接在GPU3；没有终止任何外部任务。GPU4/ECC/V100白名单未移植，XP不支持指标记录N/A。

峰值 allocated 9104.63 MiB，reserved 10532.00 MiB，整卡实测 10867 MiB。正式每次启动需实际free≥11500MiB、无其他计算PID、idle utilization≤10%，UUID与宿主候选0/1/3一致，一卡一个train/eval任务。空卡实测free约12179MiB，阈值可满足。

正式epoch计算量估计 7.479 GPU-hours；含评估/保存/波动预估 7.85–8.97 GPU-hours。只有1张持续可用时约相同小时墙钟；若2或3张持续可用，理想均摊分别除2或3，另加任务尾部不均衡及未知排队，不能保证完成时间。来源为本批XP完整课程轮39.78s(IEM CMERC)/35.42s(MELD CMERC)，baseline7.88s/6.69s，非旧12-run耗时。

证据：preflight/status.json、resource_budget.json、data_and_capacity/、gpu3_IEMOCAP/、gpu3_MELD/及logs/。代码与运行capsule见 batch_manifest.json、code/snapshots/source_snapshot.json、provenance/immutable_inputs.json。正式任务和命令见 configs/task_registry.json 与 reports/REGISTERED_COMMANDS.md。
