# 回归μ/δ评分主线：小幅变动与可行性验证

用户已明确本轮只在“目标话语特征随机干扰→预测不稳定性μ与置信度不稳定性δ→学习可靠性评分→双向校准”主线微调，以评估是否值得后续面部遮挡/音色处理。当前只复用已经生成的预处理特征干扰结果，继续按完成批次上传GitHub。

## 主方法与必要对照

主方法M_joint仅输入原始C及Gaussian μ/δ、Dropout μ/δ四维。相对前轮E1_MUDELTA唯一结构变动为基础C系数与行为系数联合学习，基础正则仍与E1一致，不再先冻结E1。μ=mean[1(pred′≠pred)×C′]；δ=|C−mean(C′)|，不改原定义。μ-only、δ-only、Gaussian-only、Dropout-only均是同一个评分模块的输入消融，不替代主模型。

Reliability Score为R=a+bC+z(B)·w，行为项单独记R_behavior=z(B)·w；q=sigmoid(R)表示原预测类别正确概率，允许高于或低于C。该学习式评分为CaliDist-inspired扩展，不冒称原论文解析Reliability Score。E0为XP04，E1仅用raw C。无D/A、JS、熵、间隔、类别、随机输入或上下文新方法。

## 拟合协议与公平性

仅行为列StandardScaler在当前训练Dialogue拟合；原C不标准化、不取logit。对n条当前训练样本，λ>0时设计矩阵为[C,z(B)/sqrt(nλ)]，用sklearn L2 LogisticRegression(C=1)共同拟合。有效行为系数换算后，目标等价于mean binary NLL+b²/(2n)+λ||w||²/2；截距不惩罚。off直接复用同训练子集E1，不能用极大λ代替。所有组统一lbfgs/maxiter2000/tol1e-8，核对与旧E1的数值精度差，不能把精度差冒充新方法收益。

每Seed/数据集独立：IEMOCAP外LODO12、MELD GroupKFold5，完全复用旧外折；内5折GroupKFold重拟合标准化、E1与行为模块，以合并内留出Brier选off/1/.1/.01，平局1e-12优先off再强正则。外层标签不参与训练/选择。五Seed共享旧2000次Dialogue抽样，各Seed独立算差后取均值，不视为五份独立数据。全部固定消融报告，不能事后把某单模态/某消融改成主结果。

## 第二批单模态Pilot

只用Seed1475现有text/audio/visual/joint、两种扰动各10重复、.05条件，每Dataset/模态单独拟合同一四维μδ主模型。不能把不同条件合成独立样本。joint逐条对齐第一批；原缺失视觉保留，不将跳过产生的零响应当视觉可靠性。其他4Seed缺乏单模态数据，本轮不自动增加GPU任务。

## 实现计划及文件责任

1. 主代理scripts/build_inputs.py：审计旧概率/μδ、建立joint8180与modality6544输入行、记录干扰响应与全部来源哈希。
2. 实现代理scripts/mainline_core.py、run_joint.py、run_modality.py、test_mainline.py及joint/modality产物：只实现原μδ评分和固定消融、保存完整内外层参数/分数/OOF/指标/区间。
3. 独立验证代理scripts/verify_mainline.py：独立重构训练标准化、正则换算、R及q、off、内层选择、指标与共享Dialogue bootstrap；核验R_behavior不引入其他输入。
4. 主代理先核验并上传完成的joint批次，再执行并核验单模态批次，综合报告三层证据，必要时支持小规模原始干扰技术探索，不能夸大为校准有效。

## 证据解释与停止

当前工程可行性、行为可测性、相对E1校准增量分别报告。不要求随机扰动产生巨大增益，也不能仅凭预测变化宣称校准获益。预先登记的初步校准信号是主模型ECE与至少一项proper loss均值改善、另一proper loss不恶化、改善proper loss多数Seed同向；区间不确定如实记录。即使符合，也仅为重复Validation上的探索性信号。

目前Validation曾参与XP04 checkpoint和多轮方法设计，无法建立独立确认。原始面部遮挡/音色处理属于后续新的干扰机制；本轮完成不会自动修改原始数据。保留阴性结果与全部消融，不反复增改公式获取有利数字。
