# 审查后验证工作计划

范围：root CMERC；不改历史 experiment_records，不清除用户修改，不 push。核心源码与外审基准91a1bff无差异；HEAD ea84294增加公开档案。

1. 保存基线SHA、工作区patch、环境、全tests原始日志（已运行153 passed）。
2. 先为设备/架构、difficulty provenance、run隔离及恢复写失败回归并保留日志，做最小修复。
3. CPU因果数值矩阵、六类负对照、prefix_mean可逆性与首句五视图no-op。
4. 实际训练函数有限batch，四组loss开关、两种显式辅助策略、连续/恢复对照。
5. 在训练/验证真实特征上做少量生产维度前向/训练函数及CUDA检查，不用test划分调试。
6. 全套回归、独立审查、按要求汇总数值/状态/阻塞及正式实验计划（只写计划）。

并行文件归属：review_provenance负责dataloader/cal_diff/causal_provenance；review_evaluator负责评测器；review_training负责train_cm/causal_run；review_causality只增数值检查脚本与测试；leader负责真实特征/最终报告/集成验证。
