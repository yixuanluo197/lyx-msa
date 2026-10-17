# 第二轮实验验收与后续就绪状态

- IEMOCAP Causal Baseline：80/80正式epoch完成、validation best来源和完整checkpoint验收通过。
- Causal Difficulty：真实Baseline best生成，108个且仅train IDs，全部来源/seed/特征/架构/policy/SHA通过。
- Causal CMERC：80/80正式epoch完成、同初始化配对通过、difficulty固定同一阶段B哈希、有限梯度与更新验收通过。
- 验证比较冻结：已完成；两个best各执行一次正式test，顺序与哈希证据可追溯；未按test重选。
- CPU/GPU状态：本轮实际使用同一台本地服务器健康TITAN Xp、float32、batch16；有限preflight与正式160轮均执行。没有将CPU/合成通过替代真实GPU结果。第一轮时间因果结论复用，四个算法文件字节一致。
- 环境：沿用既有环境，无依赖升级。正式checkpoint保留原cuDNN/RNG协议；跨硬件任意长训练逐位确定性未承诺。
- MELD、其他训练seed、组件消融与same_step策略：本轮未启动；仅准备独立目录、真实对应教师依赖、验收与评测命令，见FUTURE_EXPERIMENTS.md/FUTURE_MATRIX.json。启动时须重新检查GPU UUID/健康/余量，不能沿用旧卡号假设。

当前单seed闭环没有未解决的执行阻塞。后续运行仍有按计划生成对应seed真实教师和difficulty的顺序依赖，以及资源preflight要求；不等于这些后续模型已训练。多seed固定1937/2025/3407，split seed独立固定；按配对汇总，不只挑最高seed。

科学边界：预提取特征之后的时间因果性；上游特征提取未审计。prefix_mean可逆平滑、GNN-edge ablation语义、legacy权重滞后/模式差异保留。单seed不能作显著性或稳健提升结论；性能好坏均完整披露。
