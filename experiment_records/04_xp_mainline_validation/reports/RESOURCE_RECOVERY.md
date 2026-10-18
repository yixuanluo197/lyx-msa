# GPU 调度竞争与同注册任务恢复

在 2026-10-16，CMERC 与另一项目在 GPU0/1 各自执行“检查空闲→启动”，彼此没有共享 GPU 互斥锁，发生 TOCTOU 竞争。两方检查空闲的机制都存在；单方文件锁不能约束另一项目。未停止、修改或迁移其他项目任务。

- IEMOCAP CMERC seed2024：80 epoch 已完整结束，训练退出码0；另一项目进程在训练完成后占用GPU0，独立评估等待超时。重新验证完整训练、配置及best/latest哈希后，仅在GPU3评估同一validation-best；不重训、不重新选epoch。
- MELD CMERC seed3407：GPU1新启动进程与另一项目同时启动，停止了本批自己的受影响进程（-15）。原始checkpoint、输出、日志及失败状态全部保留到 `attempts/`。虽然有部分完整状态checkpoint，该尝试违反本批一卡一任务约束，不能作为合格独占训练续接；同一注册任务、seed和科学配置在GPU3从头重试。重试依据为资源合规性，不使用分数。
- GPU3正在正常运行的IEMOCAP CMERC seed3407保留至训练和独评结束。guard先阻断本批0/1派发，待GPU3完整释放后停止已排空的旧调度器（130为受控交接），保留原控制器状态。
- 后续使用 `state/recovery/xp3_scheduler.py`，继承冻结调度器但在资源分配入口强制排除宿主GPU0/1。这一限制不依赖guard存活，所有模型、损失、训练/评估源码、有效配置、任务ID和矩阵哈希不变。工程控制修订单独留存hash。

恢复过程有持久管理器、阶段journal、防覆盖检查及独立日志；若恢复中断且已有不完整评估输出，将停止要求核验，不覆盖。登记仍以20项完整性及协议验收为准。预期20个合格训练、另保留1个失效部分训练尝试；以最终日志核验实际计数。GPU3串行处理余项，剩余时间随真实进度更新，不保证原并行预算。

`state/recovery/pipeline_v1.json` 与 `queue_before_recovery.json` 在排空后生成；`recovery_journal.json`、`manager.json` 和 `provenance/resource_recovery.json` 记录恢复阶段及结果。官方检查点、预检结果不计入正式统计。

## 终态审查器阶段计数修正

提前检查终态辅助工具时发现：原冻结 `tools/finalize_mainline.py` 的阶段数量断言仍是V2的12+8，而本批注册并实际执行的是4锚点+16后续。此断言会错误拒绝正确矩阵。保留原工具不动，新的 `state/recovery/finalize_mainline_v2.py` 仅修正为4+16及文件所在目录解析，diff/源码/hash保留在provenance/runtime_recovery_v2。该修正不放宽科学或数值验收，不修改训练/评估、完整20项矩阵和原阶段门禁；新控制器使用修正审查器。

## 已执行恢复（2026-10-16 08:27 UTC）

有效IEMOCAP CMERC seed3407已完成80轮及独立评估；旧调度器在无活跃任务且GPU3已被guard接管后受控退出130，旧父流水线终态已归档。IEMOCAP CMERC seed2024的best SHA保持 `80ee239060ae1afa3f8c49d7c99d38db8e91fb1d706c7f95087b9e6c025ebf96`，独立评估PID3807686退出0，本批完成数达到15。MELD seed3407原attempt与日志已移至 `attempts/MELD_cmerc_validation_seed3407/2026-10-16T082751.934532_0000/`，相同配置从头重试PID3808068，GPU UUID为GPU3。新流水线PID3807999及XP3强制分配调度器PID3808000已运行；无原始模型/训练/评估源码、有效参数或矩阵哈希变化。

历史资源区间审计详见 `RESOURCE_INTERVAL_AUDIT.md` 与 `provenance/resource_intervals_audit.json`。IEMOCAP seed2024的另一项目启动是在final_test落盘后、训练进程退出前；不能将进程尾部相交描述成计算阶段重叠。IEMOCAP seed67137独评产物也早于外部子进程启动；另有baseline2024的外部启动被admission guard拒绝。未发现已完成13项额外的模型计算重叠证据；缺少全程连续NVML记录，不能声称已证明所有时刻context独占。

## 正式终态

20/20已完成并通过科学/技术验收；新流水线formal、acceptance、backup_export均退出0。实际21次训练启动（20个合格+1个保留的失效部分尝试）、20次独立评估；IEMOCAP2024首次独评在启动前被阻止，未重复计算训练。五seed统计包含全部注册任务，不包含失效部分尝试和预检。后段XP3观测共1239条、unexpected PID为0；覆盖范围和离散采样限制见XP_RESOURCE_OUTCOME.md。
