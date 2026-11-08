# 计算等价优化记录

原始GPU Test对齐和十份原Validation推理全部使用未缓存原前向。

运行IEMOCAP Audio发现每次目标前向约0.35秒。cProfile三次原前向显示0.333秒中约0.248秒为create_hyper_index、0.044秒为create_gnn_index，两者重复按固定length构造同一结构。未读取诊断指标或测试成绩决定优化。

停止尚未产出完整概率的IEMOCAP Audio初始进程，保留原日志及中止状态。MELD Audio原进程继续。后续独立入口只缓存图索引等不变结构，每次重新拼接当前L/A/V特征并重新执行全部模型运算；不改变batch、padding、图结构、参数、扰动、次数或样本。

CPU测试验证缓存不会复用旧特征。GPU检查在每个原批次的原始输入及首个目标的Gaussian、Dropout、Attenuation上，逐元素比较完整所有话语log probability、probability和argmax；任一超出atol1e-6/rtol1e-5立即失败。每次前向仍保留GraphGuard比较结构。

结果与精度证据保存cache_parity_*.json。缓存入口不作为新方法或性能提升结果。
