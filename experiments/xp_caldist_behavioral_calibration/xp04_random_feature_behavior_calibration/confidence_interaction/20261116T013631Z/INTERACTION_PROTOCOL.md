# C×μ、C×δ交互评分：固定协议

用户已批准先执行一个固定交互变体。冻结XP04，复用五Seed全部真正Validation和已存在的.05强度/10次joint随机特征扰动，不新增GPU推理。不执行原始数据处理或增加重复数。

## 假设与唯一结构变动

原评分的μδ系数固定。本轮允许其随原始C变化，检验是否改善原始置信度之外的样本级可靠性。B=(Gaussian μ,Gaussian δ,Dropout μ,Dropout δ)，保持原定义，不增加新行为指标。

每次拟合仅在训练Dialogue计算c0=mean(C)、Z=StandardScaler(B)。构造H=(C-c0)Z，再仅用该训练子集StandardScaler(H)得到T。标准化是数值与正则处理，模型函数空间仍为C、B及C×B的线性组合。先中心化B再乘C，保证常量B不会凭空复制一个C输入。

R=a+bC+Z·w+T·v；q=sigmoid(R)。保存base/additive/interaction/behavior/total分数，允许双向修正，不改变原预测类别。它是CaliDist-inspired扩展，不称为原始解析Reliability Score。

## 对照、拟合、评价

固定四组E0/E1/M_joint/M_interaction。重新拟合E1和M_joint并逐条核对旧OOF，不能将优化精度差当交互收益。基础C不标准化不取logit。L2、求解器和off回退与前轮完全相同；w/v共享一个λ，不扩展搜索。全部参数及内层候选完整保存。

IEMOCAP12折LODO、MELD5折GroupKFold，复用原85个外折；内5折按Dialogue分组，只使用外训练集选λ。每次inner/outer均独立拟合C中心与两个标准化器。每话语每Seed仅一条原始记录，不展开随机重复。

比较预登记四对，2000次共享Dialogue抽样，报告五Seed均值、样本标准差、方向及差值区间。主判据见JSON；ECE单独改善也记录，不能替代Brier/NLL证据。固定C分箱仅作描述诊断，不拟合分箱校准器、不用分箱挑选模型。

## 验证和停止

先工程测试：训练专属变换、常量行为零交互、分数分解、双向能力、正则换算、off精确E1、Dialogue隔离、原预测不变。实现与独立验证脚本齐备并冻结后才正式运行。独立验证不导入拟合函数，重构所有输入变换、分数、内层选择、指标和bootstrap；独立凸精化只诊断求解精度，不替换结果。

结果无论正负均保存并上传lyx GitHub；不因效果不佳扩充方法。本Validation已参与checkpoint选择与多轮方法设计，OOF仅隔离当前拟合，全部结论仍为探索性。完成后停止，再讨论是否增加重复次数。
