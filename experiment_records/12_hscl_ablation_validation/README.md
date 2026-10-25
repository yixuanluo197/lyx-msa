# 第12组：CMERC w/o HSCL（已完成并通过验收）

唯一正式训练MELD seed67137，Dropout0.4 / LR0.0001 / Batch16，40 Epoch；只关闭HSCL优化贡献，CC和CL保持。最佳验证W-F1 **65.69%（第5轮，较XP04 +0.45 pp）**，末10轮 **62.794%（−0.115 pp）**，按预登记规则为**混合证据**，暂缓第二Seed及五Seed扩展。零新增测试前向。

[完整中文报告](../../reports/CMERC_HSCL_ABLATION_DIAGNOSIS.md)、[配置](../../experiments/hscl_ablation_validation/20261025T080937Z/configs/registry.json)、[预登记](../../experiments/hscl_ablation_validation/20261025T080937Z/reports/PREREGISTRATION.md)、[比较CSV](../../experiments/hscl_ablation_validation/20261025T080937Z/results/comparison.csv)、[三臂指标](../../experiments/hscl_ablation_validation/20261025T080937Z/results/per_run.csv)、[F1曲线CSV](../../experiments/hscl_ablation_validation/20261025T080937Z/results/curves.csv)、[Loss曲线CSV](../../experiments/hscl_ablation_validation/20261025T080937Z/results/loss_curves.csv)、[9图18文件与SHA](../../experiments/hscl_ablation_validation/20261025T080937Z/results/figures/manifest.json)、[完整验收](../../experiments/hscl_ablation_validation/20261025T080937Z/provenance/final_verification.json)、[独立复核](../../experiments/hscl_ablation_validation/20261025T080937Z/provenance/independent_review.json)、[梯度预检](../../experiments/hscl_ablation_validation/20261025T080937Z/preflight/MELD_wo_hscl_seed67137/observer/gradient_audit.json)、[源码审计](../../experiments/hscl_ablation_validation/20261025T080937Z/reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[行号勘误](../../experiments/hscl_ablation_validation/20261025T080937Z/provenance/source_line_reference_corrections.json)、[历史指标访问说明](../../experiments/hscl_ablation_validation/20261025T080937Z/provenance/historical_metric_access_note.json)。

HSCL原始两项损失仍计算且2548步均非零，实际加权贡献全部0；正式目标为分类Loss+.002CC。非正式首Batch证实CC主干梯度非零，正式额外梯度诊断0次。XP04/XP06只复用历史验证记录，XP06仅描述，不作两个不同消融的单变量比较。

best/latest分别e5/e40；完整模型、Adam及Python/NumPy/Torch/CUDA RNG由CPU核验；40train/40validation、2548步/320valBatch、全部记录有限、51项执行冻结及4项分析冻结通过、2166项历史文件保全。原始数据、权重、逐Batch原始记录和日志保留服务器。单Seed不能证明HSCL无效、统计显著或测试收益。

[完整独立目录](../../experiments/hscl_ablation_validation/20261025T080937Z/README.md)、[GitHub交付说明](GITHUB_DELIVERY.md)。
