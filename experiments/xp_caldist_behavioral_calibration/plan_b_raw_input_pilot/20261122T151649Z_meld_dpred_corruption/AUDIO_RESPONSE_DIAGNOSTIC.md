# 历史音频响应离线诊断

本轮只重算历史 Plan B 真正 Validation 音频 Pilot 的已保存统计，没有打开媒体、权重或特征张量，没有新推理、训练或 Test 访问。历史两数据集各固定120条分层样本；它们不是数据集总体的随机抽样。

## 结论与可用证据

音频工程干扰的响应可以从保存概率中独立复算；但全部音频版本仍为 PENDING 标签审核，标签保持主分析资格为0。当前不建议把声学 D_pred 加入正式可靠性评分或以本轮选择最佳音频条件。以下正误区别仅按原句预测是否正确分组，不计算干扰后准确率或假定情绪标签仍成立。

## 覆盖与保留失败

| 数据集 | 固定样本/Dialogue | 成功/计划版本 | 失败 | clean重提取控制 |
|---|---:|---:|---:|---:|
| IEMOCAP | 120/12 | 712/720 | 8 | 119/120 |
| MELD | 120/114 | 702/720 | 18 | 119/120 |

1440个原计划版本全部保留；失败版本概率和响应为空。整类指标只在固定全部K版成功时计算，不混用不同K，也不以可用版本均值替代缺失版本。六个单版本、完整pitch K=4、完整timbre K=2及两个pitch强度配对各自分开报告。

## 指标口径

- 固定原句预测类 c=argmax(p_clean)。单版本 d=p_clean(c)-p_perturbed(c)，家族 D_pred=mean(d)，有正负号；D_abs=mean(abs(d))，cancellation=D_abs-abs(D_pred)。
- signed_delta_max_confidence=mean(max(p_perturbed)-max(p_clean))。这是有符号最大置信度变化，既不是固定原类概率变化，也不是既有正式δ的替代定义。
- JS为每版与原概率的自然对数Jensen–Shannon散度均值，零概率项贡献0；flip_rate只比较两次argmax，不把翻转率称为μ。
- 分布同时给出话语等权点估计及Dialogue等权点估计。正确/错误的总体差使用相同Dialogue抽样；随后单列同Dialogue配对、同Dialogue×0.1原置信度格配对。普通cluster区间本身不消除Dialogue或置信度混杂。
- 使用冻结协议的2000次Dialogue有放回Bootstrap，seed=20261121，95%百分位区间。同一数据集各条件共用固定120样本的Dialogue抽样矩阵，重复话语不视为独立抽样单位。少于2个有效Dialogue不生成区间，2–4个Dialogue仅标为极小支持；同时报告有效重采样次数。所有区间是探索性的逐项区间，不作多重比较显著性筛选。

## 六个版本的观察响应

| 数据集 | 版本 | n成功 | D_pred均值 | D_abs均值 | 有符号max置信度变化 | JS均值 | 翻转率 |
|---|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP | pitch_plus_0.5 | 119 | -0.000277842 | 0.00178126 | 0.000283816 | 9.27786e-06 | 0.00840336 |
| IEMOCAP | pitch_minus_0.5 | 119 | 0.000509461 | 0.00214675 | -0.000480729 | 2.16796e-05 | 0.00840336 |
| IEMOCAP | pitch_plus_1 | 119 | 0.000194019 | 0.00200622 | -9.13416e-05 | 9.40737e-06 | 0.0168067 |
| IEMOCAP | pitch_minus_1 | 119 | 0.000630524 | 0.00239585 | -0.000577005 | 2.21045e-05 | 0.00840336 |
| IEMOCAP | timbre_light | 118 | 2.61988e-05 | 0.000834115 | -2.61988e-05 | 1.97214e-06 | 0 |
| IEMOCAP | timbre_moderate | 118 | 5.41185e-05 | 0.00133288 | -5.41185e-05 | 5.59653e-06 | 0 |
| MELD | pitch_plus_0.5 | 118 | 0.000724297 | 0.00219108 | -0.000724297 | 1.56553e-05 | 0 |
| MELD | pitch_minus_0.5 | 119 | 0.000141758 | 0.00241346 | -0.000141758 | 1.91077e-05 | 0 |
| MELD | pitch_plus_1 | 117 | 0.000635134 | 0.00252867 | -0.000629563 | 1.58071e-05 | 0.00854701 |
| MELD | pitch_minus_1 | 118 | -4.20671e-05 | 0.00241414 | 4.20671e-05 | 1.34544e-05 | 0 |
| MELD | timbre_light | 116 | 1.01853e-05 | 0.000690638 | -1.01853e-05 | 1.65962e-06 | 0 |
| MELD | timbre_moderate | 114 | 7.86351e-05 | 0.00141054 | -7.86351e-05 | 5.93842e-06 | 0 |

## 正误区分与Dialogue支持

| 数据集 | 完整家族 | 比较口径 | 错误减正确 D_pred | 95%区间 | 支持Dialogue/格 |
|---|---|---|---:|---|---:|
| IEMOCAP | pitch_shift:ALL_K4 | cluster_global_error_minus_correct | 0.0004454 | [-0.000858108, 0.00167485] | 12/— |
| IEMOCAP | pitch_shift:ALL_K4 | within_dialogue_error_minus_correct | 0.000361283 | [-0.000850242, 0.00152111] | 12/12 |
| IEMOCAP | pitch_shift:ALL_K4 | within_dialogue_confidence_cell_error_minus_correct | 0.000539508 | [-0.000329512, 0.00142958] | 12/21 |
| IEMOCAP | timbre_tilt:ALL_K2 | cluster_global_error_minus_correct | -0.000221988 | [-0.000775443, 0.000409998] | 12/— |
| IEMOCAP | timbre_tilt:ALL_K2 | within_dialogue_error_minus_correct | -0.000170891 | [-0.000815943, 0.000545126] | 12/12 |
| IEMOCAP | timbre_tilt:ALL_K2 | within_dialogue_confidence_cell_error_minus_correct | -0.000151806 | [-0.00092971, 0.000687507] | 12/21 |
| MELD | pitch_shift:ALL_K4 | cluster_global_error_minus_correct | 0.000423053 | [-0.000288832, 0.00113505] | 112/— |
| MELD | pitch_shift:ALL_K4 | within_dialogue_error_minus_correct | NA | NA | 0/0 |
| MELD | pitch_shift:ALL_K4 | within_dialogue_confidence_cell_error_minus_correct | NA | NA | 0/0 |
| MELD | timbre_tilt:ALL_K2 | cluster_global_error_minus_correct | 1.68505e-05 | [-0.000515156, 0.000590202] | 109/— |
| MELD | timbre_tilt:ALL_K2 | within_dialogue_error_minus_correct | NA | NA | 0/0 |
| MELD | timbre_tilt:ALL_K2 | within_dialogue_confidence_cell_error_minus_correct | NA | NA | 0/0 |

MELD的120条分布在114个Dialogue，重复话语很少；正误同Dialogue及同置信度格的支持尤其有限。没有共享支持时明确保留NA，不能把总体正误差或较窄的cluster区间解释为控制混杂后的增量价值。D_pred有符号，负值意味着原预测类概率上升；较大绝对响应只说明受影响，不自动指示原预测错误。

## clean控制与特征移动

两数据集各119条现存音频clean重提取控制的特征及完整概率最大绝对误差均为0，预测均未变。每数据集另有1条缺少概率控制；单列缺失，未补成0。重提取误差为0时不计算响应/噪声比；直接列出响应是否超过保存的概率最大绝对误差界。该控制只能排除这些已观测clean重复的数值漂移，不能证明变换情绪保持，也不能证明所有潜在非目标影响都已因果排除。

特征RMSE、relative L2和1−cosine直接复用既有描述CSV；与D_pred、D_abs、JS的Spearman关系分别按数据集、家族、版本及pitch强度计算，区间按Dialogue聚类。相关说明当前编码变化和模型响应的关联，不能把它当成音频质量、情绪标签保持或对错误的独立诊断。逐项结果见 feature_response_correlations.csv，不按相关大小追选条件。

## 完整家族幅度与特征关联

单版本平均绝对原类概率变化约0.069–0.253个百分点。IEMOCAP pitch为476个成功版本中5次翻转，涉及3条不同话语；MELD pitch为472个成功版本中1次翻转，涉及1条话语。两个timbre家族均未翻转。这些版本计数只用于描述，同一话语的多个版本不视为独立样本。全部1414个成功版本的绝对原类概率变化均超过对应0误差clean控制；超出数值噪声不等于具有可靠性价值。

| 数据集 | 家族 | 完整K样本 | D_abs均值 [Dialogue聚类95% CI] | relative L2与D_abs Spearman [95% CI] | relative L2与JS Spearman [95% CI] |
|---|---|---:|---|---|---|
| IEMOCAP | pitch_shift | 119 | 0.00208252 [0.00156036, 0.00259821] | 0.457 [0.391, 0.540] | 0.602 [0.489, 0.693] |
| IEMOCAP | timbre_tilt | 118 | 0.0010835 [0.000762093, 0.00141537] | 0.257 [0.060, 0.467] | 0.448 [0.236, 0.643] |
| MELD | pitch_shift | 117 | 0.00229102 [0.00205205, 0.00256525] | 0.327 [0.134, 0.503] | 0.587 [0.440, 0.706] |
| MELD | timbre_tilt | 114 | 0.00105355 [0.000872541, 0.001266] | 0.407 [0.242, 0.551] | 0.707 [0.590, 0.796] |

IEMOCAP两个家族控制Dialogue×原置信度格后的D_pred正误差区间均覆盖0，本轮没有给出稳定分离证据，但这不证明真实效应必为0。MELD两个完整家族同Dialogue正误配对均为0，不能据其总体差宣称控制混杂后的价值。特征移动与无符号响应的关联较清楚，但与有符号D_pred的四个家族相关区间均覆盖0；可能存在正负抵消，仍不能由此推出错误可分。

26个失败版本按保存QC数值检查：24个存在绝对RMS变化超过0.5 dB，另2个MELD pitch版本原始变换时长差超过5%。这是既存QC失败的描述，没有重新处理或替换条件。

## 完整性与边界

逐行核对固定清单与全Validation原记录的身份、标签、原预测及完整概率；再核对扰动行的原概率、固定原类概率、最大置信度、argmax和目标话语身份。保存的transform清单与prediction清单共有字段逐字对应，成功行的非目标特征/上下文保持标记必须为真；特征移动行覆盖必须与成功音频行完全相等。来源CSV/JSON与历史最终产物清单、冻结样本及历史分析输入哈希交叉验证。

这里重新哈希的是允许读取的CSV/JSON证据，音频/权重/大型特征文件未打开；其中列出的媒体、特征和模型哈希只核对保存记录之间的一致性，并不宣称本轮重新验证了这些二进制文件。详细来源和断言见 source_integrity_audit.csv 与 verification.json。

下一步如单独获授权，先做独立音频标签保持审核和变换失真检查，再在不接触已查看评估结果的预先冻结新样本上检验可靠性增量。本轮停止于离线诊断，不自动执行审核、评分或新实验。
