# C条件μδ交互评分实现计划

> 用户已批准执行。按子代理驱动开发分配一个实现者和一个独立数值验证者，主代理负责输入审计、两阶段审阅和交付；无需再请求确认。

目标：仅增加C×原μδ交互，固定全部对照与参数，完成五Seed完整Validation比较。
架构：训练专属变换→低容量L2 Logistic score→Dialogue OOF→共享Dialogue bootstrap。技术栈沿用Python/numpy/pandas/scipy/sklearn，无新依赖。新目录独立，不修改历史产物。

- [ ] 主代理输入审计：验证旧75项delivery哈希；复制joint_features.csv为本次inputs/features.csv；审计8180唯一主键、正确性、有限μδ、缺失视觉保留；记录来源Git SHA/环境/文件SHA256。
- [ ] 实现者独占scripts/interaction_core.py、run_interaction.py、test_interaction.py与results/：先实现有意义工程测试，记录初始失败/通过；原指标与求解配置复用vendor/mainline_core.py。交互的c0、B均值/尺度、H均值/尺度须每次拟合保存；off复用E1，λ仅off/1/.1/.01。
- [ ] 主代理规格审阅通过后做代码质量审阅。独立验证者独占scripts/verify_interaction.py及results/independent_verification.json；先完成脚本和合成自检，冻结后通知主代理。
- [ ] 运行命令：OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/test_interaction.py；预期训练隔离/常量/目标/分解/off/分组测试通过。
- [ ] 正式运行前核验没有重复活动任务。命令：同线程环境的python scripts/run_interaction.py --run，日志logs/run.log。预期8180输入、32720OOF、40指标、85外折、24 bootstrap；源码/输入前后哈希一致。
- [ ] 独立核验：python scripts/verify_interaction.py。逐行重构OOF和内层候选；原E1/M_joint对齐；全部指标/区间重算；凸目标精化不更改原产物。
- [ ] 主代理报告METHOD/RESULTS/CONFIDENCE_ADJUSTMENT/FINAL_DECISION、verification及交付manifest；明确相对E0与相对E1不同，所有Seed/负结果保存，完成GitHub上传和远端SHA核验。

关键边界：常量B的Z及交互严格为0；C原值未经变换；所有scaler/中心只读取训练样本；off置信度逐位相同；E0仅q有限，未定义R允许缺失。只有一个交互主模型，不根据结果挑选新组合。最终只读verify_delivery.py使用标准库，不依赖原绝对数据路径。
