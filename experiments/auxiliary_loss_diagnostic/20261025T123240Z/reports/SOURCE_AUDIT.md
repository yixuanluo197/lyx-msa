# XP13冻结源码审计

训练器使用XP12字节相同副本SHA `0fb27f08bdfd6fabe176eb7b717587fb5e7025ba94c6e3c191b01dc18168d3a2`；与XP04仅已有skip-final-test工程补丁不同，不更改模型或Loss数学。诊断直接调用原eval损失路径，在原combine_training_losses返回处截停，避免执行原backward/step。

| 冻结源码位置 | 事实与诊断含义 |
|---|---|
| train_cm.py:608–613、669–675 | 主模型eval；五路teacher从当前主模型copy/eval、no_grad。诊断teacher=Checkpoint当前学生，不是历史Epoch入口陈旧快照，不能重建历史在线训练梯度 |
| train_cm.py:656–771 | CC或HSCL任一开即计算共享辅助路径与五组关系标签；XP12虽contrastFalse仍因calibrateTrue计算rawHSCL |
| train_cm.py:689–707 | CC按话语与五路移除求和，未按Batch或话语归一；保留原定义，不以raw norm大直接判有害 |
| train_cm.py:711–766 | 伪标签默认0；均错留默认0、错→对为0、对→错为1、均对再按真实类log概率比较，平局归0。五二值关系不是真实情绪标签 |
| train_cm.py:272–284 | 类别权重为Batch内N/n_c，缺类权重0；不额外均衡或更改规则 |
| clloss.py:224、254–261 | eq加1e-7后正mask含异类微小正权重、负mask在同类处可能微小负值；实际pos/neg/logits在原base返回处观测，不替换原公式 |
| clloss.py:293、338 | MSCL累加3关系、CSCL累加2关系后按anchor均值，没有再除关系数。权重放大与关系数会影响raw强度 |
| clloss.py:308–309、347–348 | NaN/Inf标量Loss原样回退0；诊断记录回退前logits与returned loss，不能只看最终finite |
| model_cm.py:894–901；train_cm.py:658 | hidden在分类头前返回；no_grad内赋值ft_scl不是detach，HSCL raw仍能对主干求梯度 |
| train_cm.py:1211、1225、1270–1293 | FocalLoss/Adam/CL为历史训练实现；本次只实例化原FocalLoss、不创建optimizer，不执行CL训练或任何step |

四Checkpoint为validation-best/latest真实快照，不包含完整早期轨迹。四目标epoch课程比例均1；固定训练Batch在此取eval梯度，只覆盖预登记训练子集。缺失梯度尤其classification head对HSCL需记N/A而非0cos。源码、数据、权重与所有04～12历史文件在前后重哈希。
