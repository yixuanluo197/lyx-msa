# 非因果 CMERC 多 seed 工程

- 此目录为独立多seed工程；不得改写 `../CMERC` 中的历史单seed结果、模型、日志或数据。
- 所有正式训练与评估固定 biggpu，禁用宿主GPU4；当前批次白名单1/2/3/5，使用宿主整数CUDA_VISIBLE_DEVICES，进程内cuda:0需记录宿主UUID映射。
- 此批次不允许 `--causal`；主模型为原非因果BN双向路径，结构/损失/超参数不变。
- 固定IEMOCAP split-seed=1475，MELD官方split；train seeds预注册1475,67137,2024,3407,42。不得按test指标筛seed。
- V2服务器目录 `/data2/yb/paper/cmerc_seed_v2_20261014`；`runs/multiseed_validation_v2`、`logs/multiseed_validation_v2`、`results/multiseed_validation_v2` 专用于正式multi-seed；smoke/调度验证必须使用独立目录。
- V2统一使用capacity_safe epoch排列：保持batch16/全部训练样本，仅按对话长度拒绝超过原模型1000超边/5200节点容量的整轮排列，最多10000次重排；不扩展模型参数，不按标签/测试成绩采样。与V1随机批次协议分开解释和统计。
- `multiseed_validation` 无V2后缀的产物以及旧服务器 `/data2/yb/paper/cmerc_seed_20261014` 为V1诊断尝试，全部保留，禁止与V2五seed统计混算。
- `reports/reference_single_seed` 是源码快照附带的历史报告，仅作审计参考，不能当本目录新实验完成证据。
- 运行前冻结源码；正在运行的batch内不编辑训练文件，变更需新批次。
