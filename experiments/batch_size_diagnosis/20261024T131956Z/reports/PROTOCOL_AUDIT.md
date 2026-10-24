# 第09组源码与配对只读审计

历史对照只解析XP04 run_manifest.json和40行训练/验证metrics；best=66.01%、epoch9、末10轮61.223%。禁止读取最终测试指标或预测。

实际12个训练源文件逐字节等于08冻结版本；train_cm.py SHA256=0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2。逆转登记的skip-final-test三个文本修改后，SHA匹配XP04。epoch训练循环AST与XP04完全一致，模型及Loss源未改。外部观测脚本的batch因素门禁Diff单独保存，不把观测脚本称为原官方源。

唯一科学CLI因素--batch-size16→8；lr=.0001、seed2024、split_seed1475、continuous、ATen、capacity_safe、全部结构及损失系数逐项门禁核对。train_cm.py:392–416,1282–1283：同一batch_size同时控制train/validation。数据对话池及其hash相同，验证完整集合W-F1计算不改变；验证分组改变，不能宣称纯训练batch对照。

model_cm.py:758–762 BN展平含padding张量；:658双向GRU，:824未pack序列直接送入GRU。因此验证分组改变可能影响固定权重的预测，不能预设只影响Loss。train_cm.py:659的mask操作沿时间维求均值，padding长度随同批对话变化。:689–707 CC五路正差求SUM，无话语数归一化。:272–284类频率权重取批内计数；clloss.py:224–259在当前批全部话语构造相似度/正负对，包含跨对话配对。train_cm.py:818,824–825按Batch均值汇总，末小批同权；:819–820 W-F1拼接完整预测/标签统一计算。不得将跨Batch的原始Loss差值解释成归一化每样本目标变化。

CPU重放40轮：相同eligible对话池、整轮扁平ID顺序；新旧sampler AST及新B8逐批结果一致，历史B16容量/retry逐轮匹配旧日志。B8预计5096步、B16重建2548步，验证600/320批，40轮retry均0。B8最大训练167边/429节点，远小于1000/5200容量；B16为300/756。数据量与池未裁剪。历史步数仅CPU重建，不伪称历史逐步遥测。

Batch变化还改变每Epoch的Adam/L2更新次数、BN统计、HSCL配对、CC总和尺度及dropout实际随机消耗；RNG策略和seed不变不代表跨Batch随机轨迹相同。只允许一次正式40轮；独立首Adam探针不向正式运行提供权重或RNG。

环境与08相同，Python3.10.13/torch1.13.1+cu117/PyG2.4.0，原UUID兼容工程补丁哈希相同。沿用本地user23 GPU3 UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456，当前共驻PID1843864已登记，离散资源采样不证明独占。

作者课程难度文件固定相同hash但生成历史未知且含validation键；保留可比性，不能声称整条预处理无验证信息。原pickle含测试部分与test-loader元数据仍会加载，但禁止test-loader迭代及测试前向；不等于完全未加载含测试字节的文件。不同边界分别披露。
