# 第10组完成材料：GitHub交付

本次仅在唯一40轮正式训练及CPU独立验收全部通过后发布，当前版本包含真实验证结果；没有将预检或排队称作完成。

[完整报告](../../reports/CMERC_DROPOUT_DIAGNOSIS.md)、[结果摘要](README.md)、[独立运行目录](../../experiments/dropout_diagnosis/20261025T032711Z/README.md)、[当前发布文件SHA](publication_manifest.json)。

已完成：40train/40validation、2548真实optimizer更新、0test；best第15轮66.49%，latest第40轮，末10轮62.775%。历史对照为66.01%/e9、61.223%。49项冻结输入与1881项04–09历史文件保全通过；两Checkpoint模型/Adam/RNG及完整args/provenance通过CPU检查。

公开材料包括：源码字节快照与工程Diff、预登记、真实完整argv和有效解析参数、输入及划分哈希、环境、30项CPU预检证据、GPU探针摘要、四模型实际Dropout属性、独立验收、逐轮Train/Validation W-F1/Accuracy及分类/CC/MSCL/CSCL/总Loss CSV、汇总JSON、11组PNG/SVG共22件图及图表输入/输出SHA。作者已公开的小型MELD课程文件沿用原hash。

源--dropout实际作用于文本GRU层间和分类头，rec_dropout0.1未被当前构造器使用。候选满足最佳+末段F1规则，但验证分类Loss恶化，不能将其解释为各项泛化指标均改善。单Seed、历史时点、共享GPU及scatter非确定性、作者课程生成史未知等边界见报告。额外中文解读与自动核心报告的可追溯差异见[report_interpretation.json](../../experiments/dropout_diagnosis/20261025T032711Z/provenance/report_interpretation.json)。

数据及其本地链接、best/latest权重、原始训练/资源日志、逐Batch/原始逐Epoch观测、实时state与锁保留服务器，不进入Git。公开CSV为聚合指标，不包含测试预测或新的测试F1。服务器绝对路径及未公开文件SHA仅用于溯源，不保证仅克隆GitHub就能重新运行全部验收。04–09历史档案保持原字节。

未启动第二Seed、Baseline、其他Dropout/LR/Batch、五Seed或测试评估。
