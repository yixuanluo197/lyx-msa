# IEMOCAP Causal Baseline：正式训练已完成

状态：**80/80 epoch 完成，独立验收通过**。从随机初始化训练，未加载旧非因果权重。CL/HSCL/CC全关；训练seed1937，split seed1475；其余配置见冻结协议与run manifest。

- 验证W-F1最佳epoch：57（从1开始），记录值 **62.91%**。验证指标按原入口保留2位小数，平手取首次最大；后续评测另报告未四舍五入的指标。
- 参数数：4,971,542；实际初始化state SHA256：`de943257bf89c6aa53abc205c794f22f17e79aa40ed0a1701a08b1cd6aa0752f`。
- best.pt SHA256：`1d9bbae19f0e4f8bd976bc3e4285f93c14f114cc54e6d28afcc76be8f90dd3bc`。
- latest.pt SHA256：`43f074746530571c614f281c56217dbb622ca400acdbaad7f439d72fa9a93c47`；epoch=79（零起始），表示完成80轮。
- run manifest SHA256：`739fc0c39b430d9ebf0490049e6e810a477d91c2e9928bfd7b6804318dc58ec7`。
- 80轮已记录训练耗时合计：215.83秒；各轮峰值allocated显存最大：3424.01MiB。进程起止时间另见BASELINE_PROCESS.json，不与逐轮计时混用。

训练/验证分别108/12对话、5283/527话语。训练W-F1从25.30%至99.89%；最后一轮验证W-F1为61.52%，没有用最后一轮替换验证最优模型。

全部epoch日志连续且唯一；保存真实主模型、非空Adam状态、Python/NumPy/torch/CUDA随机状态、课程进度，latest绑定可用历史best的SHA。所有训练batch启用更新前有限loss/活跃梯度检查；每轮首batch的分项梯度/更新证据均有限，分类梯度非零，主模型参数实际更新。辅助仅eval只读，未加入优化器。旧未执行参数不要求梯度。

本报告阶段未做测试推理，metrics中的test字段全为空，run中无final_test.json。完整模型必须先使用相同初始化和同一来源difficulty完成，再冻结主要validation比较，之后才按预注册进行一次正式测试。后续测试结果保存独立评估目录，不回写训练选模对象。

证据：BASELINE_VERIFICATION.json、BASELINE_PROCESS.json、runs/baseline/*/metrics.jsonl、training_diagnostics.jsonl、initial_state.json、run_manifest.json、best.pt/latest.pt，以及baseline_training.log。正式checkpoint/原始日志保留服务器。
