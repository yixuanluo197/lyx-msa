# 本轮运行说明

## 服务器与冻结条件

沿用XP04历史服务器local/user23，GPU宿主机编号0–3，启动器通过UUID限制可见设备。未使用biggpu。模型、特征、split、训练统计量和原始概率均沿用历史来源并哈希核验。

GPU推理只导入冻结原始CMERC路径；使用Torch1.13.1+cu117/FP32/ATen/eval，LD_LIBRARY_PATH沿用历史cuSPARSE目录。GPU数值复现按原协议atol=1e-6、rtol=1e-5及预测类别一致核验，不承诺GPU输出bitwise相同。噪声输入字节与RNG可精确复现。

## 执行顺序

1. 10checkpoint与数据身份静态审计通过。
2. IEMOCAP seed67137 block0及MELD seed1475 block1各一个目标smoke完成；独立推理预检通过后启动12job持久队列。
3. CPU历史校准复现与GPU推理解耦执行；历史两候选M_gaussian输出与之前记录精确一致，E1误差不超过3.33e-16，独立OOF/Bootstrap复核通过。
4. 全量推理完成后才执行全部候选校准、汇总及独立核验。完成状态以各complete.json、queue_complete.json及最终verification.json为准。

## 保留的问题与修正

- 初始历史CPU分析在保存fold元数据时遇到NumPy int64不能直接JSON序列化，尚未输出正式分析。修正仅将元数据转换为Python原生整数；保留failed_preflight_v1/及historical_analysis_failed_v1.log，并添加回归测试。未改公式或拟合规则。
- 初始独立smoke verifier将MELD配置的整数Dialogue IDs与CSV字符串直接比较而失败；修正仅规范ID类型。失败trace保留在independent_preflight.json的prior_verification。原推理不变。

## 比较边界

block0的Seed1475复用历史Pilot；其他Seed为新单模态推理。block1/2是Seed1475独立随机数的重复，各10次单独评价。不得将30次扰动拼成新的主设置，也不得按最终结果选择随机块。原始Pilot未保存每个扰动向量的字节哈希，旧1475可核验RNG键/算法/Training统计量；本轮四个新增模型Seed间另有向量哈希直接比较。

两个候选来自相同Validation上的既有探索。跨Seed和随机块的一致性不等于独立数据确认。所有预登记条件和负结果均须保留。未授权的原始数据实验、强度搜索和Test评价不在本轮执行。
