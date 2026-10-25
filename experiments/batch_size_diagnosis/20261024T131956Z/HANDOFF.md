# 第09组已完成交接

唯一正式训练已在本地user23 GPU1（GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a）完成。2026-10-24 17:40:55—18:54:45 UTC，PID2791997，返回0。40轮train/validation，5096次真实Adam更新，0测试前向；不得重复启动。

最佳验证65.89%/第9轮，末10轮63.048%；XP04同Seed/lr Batch16为66.01%/第9轮、61.223%。最佳差-0.12个百分点、末段+1.825个百分点，按预登记主指标未改善，不自动追加第二Seed/五Seed或其他训练。

独立CPU验收passed：best/latest第9/40轮，模型/Adam有限、完整RNG，92项冻结源输入未变、1691历史文件未变。10组曲线PNG/SVG共20件；详见results和provenance/final_verification.json。资源共享，正式最小空闲31MiB，未发生本任务OOM；不能声明资源独占或速度因果对照。

后台delivery只因finalization.json尚未创建的启动竞态退出，未影响训练或CPU验收。完成后修复非冻结导航脚本的等待逻辑并执行成功，未改任何训练代码/配置/结果。原v1/v2队列均归档，实际科学源及资源冻结以最终frozen_execution.json为准。

最终中文报告位于批次reports及仓库reports/CMERC_BATCH_SIZE_DIAGNOSIS.md。GitHub先前adea816为排队快照，完成材料另行发布；训练数据、权重、原始日志及逐Batch跟踪保留服务器。
