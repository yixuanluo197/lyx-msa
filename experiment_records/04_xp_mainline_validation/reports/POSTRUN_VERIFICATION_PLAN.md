# 正式终态后的独立收尾核验（已执行）

仅在20项完整训练、独立评估和技术验收通过后执行，不改运行中的科学代码/配置，不新增训练。

1. 确认state/queue.json全部20 completed、first_seed_gate与final_gate通过；pipeline各阶段退出码和预期产物相符。
2. 验证results/acceptance.json、五seed四组/10配对统计、曲线与完整checkpoint/optimizer/RNG/sampler/test隔离证据；必要只读复核固定best，不重选配置。
3. 实际独立备份40个checkpoint、逐样本预测和原始日志；pipeline/export自身控制日志在导出时仍可能追加，流水线终态后再核验并补齐这些控制日志的最后副本及SHA，训练/评估日志不改写。
4. 检查公开04仅轻量源码、配置和审查材料，无权重/特征/逐样本预测/raw logs/缓存或symlink。补齐根batch_manifest、BASELINE_REGISTRY与最终队列/流水线审查状态，并同步export_manifest每文件来源/SHA；不覆盖历史01/02/03。
5. 为长期使用说明实际批准入口：state/recovery/continue_pipeline.py和强制XP3的xp3_scheduler.py，验收用finalize_mainline_v2.py；原tools/formal_pipeline.py保留初始控制器证据，不直接重启。配置仍是冻结configs/xp_mainline_validation.json及20项命令。code/内原V2的旧辅助启动脚本和旧默认配置只是继承来源，不直接运行；XP wrapper内绝对路径重用时须建立新批次重定向并重新freeze。CPU测试范围排除了V2导出缺失的caldiff生成模块及一个依赖它的因果测试，见preflight/test_scope.json；GPU-only RNN单测已实际另跑。
6. 更新本批README为实际终态；验证根BASELINE_REGISTRY及总实验索引的04入口。计算历史01/02/03树哈希仍与启动前一致；检查Git diff，无自动提交/推送，无旧模型/日志唯一副本删除。

## 资源竞争恢复补充

2026-10-16 发生两个独立调度器同时观察空卡的启动竞争：IEMOCAP CMERC seed2024已完整训练exit0，独立评估等待时原卡被占用；MELD CMERC seed3407启动时检测同卡外部进程，主动SIGTERM本方并保留不合格部分attempt。前者仅补评估同一个固定best，后者以独占策略违例为理由原seed原配置从头重试，不依据分数。科学源码/配置不变，调度恢复版本在state/recovery及provenance/resource_recovery.json留存；后续仅用GPU3，xp3_scheduler.py在分配入口硬排除0/1（不依赖guard），不影响外部进程。

最终须将这些失败/无效attempt及恢复控制记录写入审查报告，失败attempt权重和日志也单独备份；公开仅保留元数据，不含其二进制/原始日志。20项注册任务的终态统计与实际启动attempt数量分别报告，不把无效部分attempt混进均值。检查新的continuation pipeline终态及旧pipeline退出130为受控调度交接，不隐藏历史失败记录。

## 实际核验结果

- 队列20/20 completed，4个锚点及16项后续全矩阵门禁通过；40个训练/独评phase退出0，新流水线formal/acceptance/backup_export均退出0。manager、guard、observer进程均已退出。
- results/acceptance.json passed，40个checkpoint完整/有限，20份保存预测独立复算五项指标，1200个epoch课程与容量重放一致；四组n=5、10个同seed配对均由独立审查再次核算，ddof=1。
- 三个SVG可解析且含seed42；完整逐seed结果、聚合、配对CSV和summary均齐全。
- 实际独立备份251个文件，含40个正式checkpoint、逐样本NPZ/CSV、全部终态原始/控制日志，以及失效部分attempt及其权重；每个文件SHA一致且独立inode，备份清单与provenance副本一致。目录仍在同一/data2文件系统，非异盘/异地备份。
- 公开04清单289个文件，根batch_manifest、registry和终态queue/pipeline/旧pipeline及恢复审计齐全；无权重、逐样本预测、原始日志、特征/难度、symlink或缓存。恢复入口和CPU测试排除明确见公开README。
- 历史01/02/03逐文件SHA及文件集合均与启动前相同（71/128/176，共375个文件）。根索引已区分旧XP02和XP04，根BASELINE_REGISTRY已登记本批绝对目录、验收/报告、公开档案和独立副本路径。
- 未自动提交或推送；科学冻结源码、有效配置、数据/课程和环境补丁未变。原受控交接130、无效部分attempt -15及同seed重试均保留；21次正式训练启动只统计20个合格run。

本记录在实际交付后填写。其后仅定向刷新本文件在公开清单的SHA并重新核对公开文件/历史树，不重复写入科学结果或重选checkpoint。
