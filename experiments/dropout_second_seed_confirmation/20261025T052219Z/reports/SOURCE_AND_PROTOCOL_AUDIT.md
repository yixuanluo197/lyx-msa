# 第11组源码与协议审计

实际继承XP10 code_xp04所有Python源字节，train_cm SHA0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2。相对XP04仅既有skip-final-test三处工程补丁，反向补丁检查应恢复XP04 source hash；11个其余实际源文件字节一致，无模型/Loss数学改动。

沿用第10组已核路径：train_cm.py:1050/1082/1114/1146将dropout传到四实例；model_cm.py:658及824为文本两层双向GRU；713/893–898为分类前hidden Dropout。av_using_lstm=False，音/视GRU不构造；model_hyper仅保存dropout属性，不新增图层随机Dropout。rec_dropout.1参数入口train_cm.py:854，但四个构造调用未传dropout_rec且当前类未使用该形参。不得擅自修复。train_cm.py:607–613中主模型按train/eval切换，辅助实例复制入口权重并eval，验证关闭Dropout。

XP04同Seed67137：dropout.4/rec.1/B16/lr1e-4/split1475/continuous/ATen/capacity_safe，train1038/val114；best65.24/e8、tail62.909、样本SD.49753727、最佳到尾段回落2.331pp。e8 train/val分类Loss3.34527130/4.1291，尾段2.70667967/5.18186。历史实际GPU0，与本次同卡；确切来源字段和误访问事件见provenance/gpu_selection.json及test_metric_access_incident.json，禁止再读取混合测试摘要。

XP10 seed2024的固定已完成证据为best66.49/e15、tail62.775，相对XP04 +.480/+1.552pp；尾段val分类Loss5.32021924高于5.08928。第11组只验证另Seed的一致性，不能预判结果。对应四臂完整train/val曲线引用provenance/xp04_validation_references.json及xp10_validation_reference.json，绝不重新运行模型或测试。

本次实际四实例Dropout属性由只读module_audit记录，要求文本GRU两层双向p=.5、dropout_.p=.5及args.rec_dropout=.1；审计前后RNG哈希相同。初始化仅当前67137独立从头，不能拿2024初始hash当要求。预检与正式各自fresh初始化后核首步相同，不加载历史/探针权重。

验证集完整池和Batch16分组保持XP04，sampler AST、每轮课程池/顺序/私有RNG和步骤需CPU重放，再与真实运行匹配。原Loss按Batch聚合、系数保持，在线Train与eval Validation间距不是严格固定权重泛化gap。非因果路径/BN保持，选模沿原百分比两位小数严格改进、同分首轮。

资源同host同GPU不能排除历史时点、共享负载及CUDA scatter非确定性；作者课程生成史未知、含validation键，原pickle含测试部分。零新增test前向和无test驱动决策可以验证，但因上述误访问，不能声称全程从未接触历史测试字段。
