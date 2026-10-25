# 第10组已完成交接

唯一正式run MELD_cmerc_dropout0p5_lr1e-4_seed2024在本地user23 GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456）完成。PID3695637，2026-10-25 03:35:33.778686—04:04:13.066590 UTC，返回0，GPU已释放。不得重复提交。

40轮训练/验证、2548真实Adam更新、零测试；最佳66.49%/e15，末10轮62.775%。XP04控制66.01%/e9、61.223%。按原规则满足另行登记第二Seed的候选条件，验证末段分类Loss5.320219高于控制5.08928，不能称整体过拟合解决。当前没有额外训练预算。

CPU finalizer于04:06:54完成verify_lr→analyze_dropout→plot_dropout，各返回0。best/latest第15/40轮，模型与Adam有限、完整RNG/参数/来源；49冻结源输入、1881历史文件保全通过，22件PNG/SVG已输出。正式与预检模块/初始化/首步RNG一致，未加载历史或探针权重。

完整结果见results及reports/CMERC_DROPOUT_DIAGNOSIS.md，审读补充及与自动核心报告的差异见provenance/report_interpretation.json和.diff；没有修改冻结分析脚本或科学指标。发布清单及验收见experiment_records/10_dropout_diagnosis/。用户授权完成验收后同步GitHub；数据权重原始日志不上传。
