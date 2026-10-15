本批仅在 user23 GPU 0/1/3 执行。冻结 10 个 CMERC + 2 个 baseline；非因果、official_test、固定作者难度、batch=16。
先完成独立环境、CUDA 扩展及现有测试、两个真实 smoke、官方 checkpoint 的 W-F1/CEL 校验，再用 7/3 epoch 工程 profiling 覆盖全部课程与长对话（不作正式结果）。用标准库调度器串行占用每张卡、锁和产物校验防重复，逐 seed 独立评估并统计。不改变算法，不按测试结果改 seed/参数。
