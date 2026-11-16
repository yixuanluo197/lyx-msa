# C条件μδ交互：方法说明

## 与E0、E1、旧M_joint的关系

E0直接使用XP04原始C；E1仅学习原C到原预测正确概率的Logistic映射，已允许双向调整。旧M_joint增加两类随机干扰各自的μ、δ，行为系数固定。新M_interaction只让这些行为的作用随C线性变化，不增加新的行为指标。

固定B=(μ_gaussian,δ_gaussian,μ_dropout,δ_dropout)。μ=10次mean[1(pred′≠pred)×C′]；δ=|C−mean(C′)|，C′始终为扰动后的最大类别概率。不能把μ改成无权翻转率，也不能把δ改成逐次绝对差均值。

## 每次inner/outer拟合都重新计算的变换

1. c0=mean(C_train)。
2. 在B_train上拟合StandardScaler，得Z=(B−mean_train)/scale_train。
3. H=(C−c0)Z；在H_train上单独拟合StandardScaler，得T=(H−meanH_train)/scaleH_train。
4. 留出Dialogue只能使用训练变换，不参与c0、均值、方差计算。

先中心化B后构造乘积，保证常量行为不凭空引入额外C通道。对真正恒定列使用其精确常量均值，避免浮点累加产生伪交互；标准差按StandardScaler的常量规则处理。

评分R_base=a+bC，R_add=Z·w，R_interaction=T·v，R_behavior=R_add+R_interaction，R=R_base+R_behavior，q=sigmoid(R)。因此原μδ在评分中的有效系数能够随C变化；模型没有预设“敏感即错误”的符号约束。q表示原预测类别正确的概率，不是重建后的完整多类概率。

旧M_joint只使用Z、没有v；off直接复用同训练子集E1对象。四组都保留原始情绪预测类别，只有正确概率发生变化。本学习式Reliability Score是CaliDist-inspired适配，不冒称原论文解析评分。

## 容量与评价约束

只增加4个交互系数，w/v共享一个L2强度。候选固定off/1/.1/.01，内层训练Dialogue GroupKFold5合并留出Brier选取，平局优先off/强正则。对当前n条训练样本，设计矩阵为[C,U/sqrt(nλ)]，U为Z或[Z,T]；sklearn LogisticRegression C=1等价于mean NLL+b²/(2n)+λ(||w||²+||v||²)/2，截距不惩罚。

IEMOCAP527条/12Dialogue，MELD1109条/114Dialogue，五Seed分别拟合；原85个外折和2000共享Dialogue抽样原样复用。报告固定15bin Top-label ECE、Binary Brier/NLL；不是多分类概率向量上的Brier/NLL。比较E0用于衡量整个模块收益，比较E1用于衡量行为增量，比较M_joint用于衡量交互结构增量。

## 范围

仅已有joint .05扰动和10重复；没有新GPU推理、Test数据、温度缩放、原始数据修改或CMERC训练。结果来自多次分析过的Validation，属于探索性验证。
