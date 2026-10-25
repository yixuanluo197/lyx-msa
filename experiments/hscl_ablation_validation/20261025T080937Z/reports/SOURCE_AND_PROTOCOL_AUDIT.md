# XP12 源码与协议只读审计

代码来自XP11冻结code_xp04，训练器SHA256 0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2。与XP04仅继承skip-final-test三处工程补丁，不改数学；启动前逆补丁核实SHA。原始XP04解析args与40epoch结果核实65.24/e8/tail62.909。

| 路径与行（本组code_xp04下） | 已证实行为 | 本次含义 |
|---|---|---|
| train_cm.py:249–253 | calibrate与contrastlearning独立组合目标 | 只关闭contrastlearning，total=class+.002CC |
| train_cm.py:656、669–675、689–771 | OR外层保留五路teacher、CC、伪标签和raw HSCL | 不称删除全部HSCL计算；raw值照记，有效贡献0 |
| train_cm.py:608–613 | 辅助模型复制主模型并eval | 辅助前向仍保留/no_grad，主干梯度来自主预测 |
| train_cm.py:658；model_cm.py:894–901 | ft_scl=hidden为赋值非detach，hidden在分类头Dropout前返回 | HSCL开启有可导主干路径，关闭后不进入目标 |
| train_cm.py:1207、1221 | MELD FocalLoss与Adam主模型参数 | 分类损失、优化器不变 |
| train_cm.py:1266–1269、1278–1289；dataloader.py:174–178 | courselearning独立选择固定池课程样本 | CL仍True，课程/数据池/采样Seed不改 |
| clloss.py:228–261、277–307、327–346 | 归一化/相似度/聚合无显式随机采样或可训练参数 | 不人为补随机数，不承诺CUDA数值全同 |
| clloss.py:308–309、347–348 | 原HSCL内部NaN/Inf可被替换0 | 返回有限不证明内部从未异常，不改或解释为机制无效 |
| model_cm.py:658、713、824、893–898 | dropout.4影响文本GRU层间/hidden-feature；CLI rec_dropout.1未传活跃GRU | 沿用XP04构造，不修rec_dropout |

调用/返回及Adam观测守卫不改正式图/梯度/RNG；额外autograd诊断只在独立非正式首Batch。作者难度生成史未知/含验证键、共享GPU、CUDA scatter非确定性和单Seed限制因果解释。历史XP06测试列曾可见但未据此决策；本组仅提取训练/验证净化字段。环境复用原依赖：裸Shell缺libcusparse.so.11时应用registry的既有LD_LIBRARY_PATH即可，无新增依赖。
