# 运行与复现说明

## 最终有效运行

仅 `runs/{IEMOCAP,MELD}/{text,audio,visual,joint}/complete.json` 标记完成的八个正式任务进入统计。每个任务使用旧0.05 Pilot的原始NPZ/CSV作为统一基线，完整验证集新鲜前向再按历史容差核验；两个强度使用完全相同的原始C和完整概率。

扰动随机数、特征向量可按记录逐位复现。原始CUDA计算含并行归约，相同输入的输出并非逐位一致；运行器分别保存输入字节哈希、权重精确一致性、类别一致性，以及历史 `atol=1e-6, rtol=1e-5` 下完整输出对齐。`replay_diagnostics.json` 明确保存 `bitwise_output_equal`，不把数值容差对齐称为逐位相同。

## 启动前问题与排除记录

1. 首次GPU检查遗漏了历史 `LD_LIBRARY_PATH`，随后恢复原有cuSPARSE库路径。相同输入的缓存与非缓存多次前向均测得细小FP32差异，因此单独记录逐位标志与数值容差，而非假定恢复库路径即可实现逐位一致。
2. 数据打开审计最初未将特征文件软链接解析到真实路径，已改为匹配预先哈希的实际数据文件。该修改没有改变模型、特征或随机数。
3. 更早生成的真实Validation NPZ与0.05 Pilot当时的新鲜原始输出存在约1e-7量级的GPU舍入差异。为避免把原始C的变化混入强度比较，正式任务统一采用0.05 Pilot的原始NPZ/CSV，且要求CSV基线精确一致。

中断的启动及部分输出完整保存在 `aborted_launch_v1/`、`aborted_launch_v2/`，没有拼入正式CSV，也没有用于拟合、指标或选择。旧运行器保存在provenance；首份失败独立预检JSON保留，双数据集修正后通过的预检为 `results/preflight_verification_v2.json`。

`prepared/` 是全量运行前对历史0.05结果的独立重算记录，保留当时的源码哈希。最终全量统计的源码与输入前后哈希以 `results/source_hashes.json`、`analysis_verification.json` 和最终独立审计为准。

## 环境与资源

模型推理和可靠性映射使用既有 `cmerc-xp-repro` 环境。绘图使用已安装matplotlib的系统conda环境，只读取汇总CSV；没有安装依赖或切换拟合环境，版本见 `provenance/plot_environment.json`。

任务启动时本地四卡空闲，随后出现其他用户的GPU进程。本轮保留原服务器、设备归属及科学设置，没有终止无关进程。资源日志和耗时是共享负载下的观测，不能用于受控速度比较。

## 分析边界

0.05是历史主条件，0.10是新增敏感性条件，所有模态与扰动均报告。两个强度的随机键包含strength，因此使用不同确定性随机流；配对比较发生在话语及Dialogue层面。

每个OOF拟合条件中，一条原始话语只有一条训练记录。跨模态、强度、扰动重复出现的CSV记录不构成新增独立样本。2000次Dialogue Bootstrap固定行为汇总和OOF输出，不重抽扰动repeat、不重新拟合，不涵盖Seed或全部拟合不确定性。

运行和报告完成后停止；不自动追加强度、重复数、五Seed或原始数据干扰。
