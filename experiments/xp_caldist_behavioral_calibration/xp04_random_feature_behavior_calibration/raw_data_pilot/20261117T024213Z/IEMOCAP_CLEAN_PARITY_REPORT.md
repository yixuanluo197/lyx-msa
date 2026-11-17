# IEMOCAP 一次性 Validation Clean Feature Parity

**最终结果：FAIL，归类C。** Train-only重建方法 `IS10_paraling/none → diagonal_affine → 1582维` 已在Seed1475真正Validation的527条、12个Dialogue上完成一次锁定评价。覆盖率与有限值检查通过，误差、余弦和逐维相关性门槛均未达到；因此停止，未进行冻结CMERC概率比较。

本报告只汇总[最终数值记录](provenance/IEMOCAP_clean_feature_metrics.json)和[feature gate](provenance/IEMOCAP_feature_gate.json)，没有重提取Validation、重拟合或按结果调整门槛。唯一评价marker的 `attempt=1`、`status=COMPLETE`，起止时间为 `2026-11-17T03:16:13.646569+00:00` 至 `2026-11-17T03:16:14.687025+00:00`，见[评价marker](provenance/IEMOCAP_validation_evaluation_marker.json)。

**数据与绑定。** Train为5283条、108个Dialogue，Validation为527条、12个Dialogue，无Test特征或标签参与。转换器、scaler及参数选择只用Train；3折按Dialogue分组，每折独立拟合参数。最终配置使用当前官方v3.0.1 IS10非compat候选，原话语WAV实测为16 kHz单声道；特征前端无裁剪、幅度门限、重采样或外部归一化。已安装Python opensmile2.6.0、引擎3.0.1；详见[重建报告](IEMOCAP_FEATURE_RECONSTRUCTION.md)及[完整配置与include哈希](provenance/iemocap_extraction_configs.json)。这些是本轮候选处理事实，不能声称作者原流程已恢复。

| 绑定对象 | SHA-256 |
|---|---|
| 正式model lock | `788294fb64adc841ba0e12df66a2dd034d52a7ff743b583ea711c5d11867723f` |
| 协议v2 | `02377d045ceeb34932eabbb227a72712707f8c7704be483206e13aaef644a9b2` |
| Train拟合final converter | `5afb90cc7f71467ad8a90844a07f133a10c017fd9b58610fbd50f2aade23ca1f` |
| 锁定IS10外接I/O配置 | `cf96dcd56797dc05813cf27dbc1c9b4a7a9565971e40d2fad9871676d59218c8` |
| 候选Validation数组 | `363b9d5c73fbc74441b4c4af2ba21a6b97daa111b669be1591f72d6c56ca76dd` |
| 重建Validation数组 | `1d5a0ac94facc981dba919e02163a0982321b5a4adf422f0e8a27aff93acce74` |
| 原XP04 Validation目标数组 | `a191d3f74d2946bac097ae15de77de2f1c9c58eebb403c582ee5a2c04a822b30` |
| Train目标列scale | `0e326cc28b6436815f2742ed3e23fd6fb233a33d4146870b0be97176ac0b4368` |
| Validation manifest | `9f87f720d66ba916d4e5ee105bb326aa2867181b198ca2ebb4a7b4d4624c0080` |

唯一候选Validation提取耗时10.46秒，源特征为float32 `(527,1582)`；重建结果与XP04目标形状均为 `(527,1582)`。527条源音频全部存在且逐条哈希与manifest一致，逐行输出哈希与有序header一致。[提取独立验证](provenance/iemocap_extraction_validation_verification.json)27项通过；另有18项模型锁guard/synthetic检查，原Train脚本字节快照保留，Val代码只改锁定执行边界而未改DSP。其余两个前端没有执行Validation提取。

| 最终独立Validation指标 | 结果 |
|---|---:|
| 比较标量数 | 833714 |
| 最大绝对误差 | 17.9104099936 |
| MAE | 0.0135472936581 |
| RMSE | 0.127824909926 |
| 全局Train列scale归一化RMSE | 0.129320247109 |
| Dialogue等权NRMSE | 0.114389532442 |
| Dialogue等权MAE | 0.0134646377033 |
| 平均话语余弦相似度 | 0.993399079623 |
| 最小话语余弦相似度 | 0.296170893899 |
| 逐维相关性：min / p05 / median | 0.655283188805 / 0.956714570461 / 0.999200226038 |
| 逐维相关性：p95 / max | 0.999996830360 / 1.000000000000 |

全局Train-scale NRMSE定义为 `sqrt(mean(((prediction-target)/scale_train_j)^2))`；scale为仅由Train目标拟合的每列标准差，按协议以1e-8为下限。Dialogue等权NRMSE先在每个Dialogue内计算同口径RMSE，再对12个Dialogue求算术平均，避免将话语当作完全独立观测。没有用Validation重新估计scaler。表中NRMSE不是情绪预测或校准指标。

**冻结门槛逐项结果。** 本轮以全部条件同时通过为兼容要求；均值余弦较高不会覆盖其他失败。阈值在本轮Validation访问前冻结，协议v2仅增加Train内全秩Ridge菜单，未更改这些阈值。

| 条件 | 预定门槛 | 本轮结果 | 判定 |
|---|---|---:|---|
| 完整覆盖 | 527/527条 | 527/527 | PASS |
| 有限值 | 全部目标与输出有限 | 833714/833714联合有限 | PASS |
| 逐元素allclose | 每元素 `abs(error) ≤ 1e-4 + 1e-4×abs(target)` | 未全部满足 | FAIL |
| MAE | ≤1e-5 | 0.0135472936581 | FAIL |
| Train-scale RMSE | ≤0.001 | 0.129320247109 | FAIL |
| 平均非零话语余弦 | ≥0.99999 | 0.993399079623 | FAIL |
| 最小非零话语余弦 | ≥0.9999 | 0.296170893899 | FAIL |
| 非常数目标列相关性均可定义 | 所有目标非常数列均满足 | 1582/1582 | PASS |
| 非常数目标列相关性p05 | ≥0.999 | 0.956714570461 | FAIL |

**逐维与分布差异。** 每一维的误差、相关性、均值/标准差差、Train-scale NRMSE和Wasserstein距离均保存在本地[逐维CSV（仅本地保留）](LOCAL_ONLY.md)，SHA为 `887884b5cd7b594166607e5f75f53c1f02166e2ed1b40e80e89920d3501812dd`。CSV按 `feature_index` 对齐1582维；候选字段名位于[有序header（仅本地保留）](LOCAL_ONLY.md)。没有原作者有序header，不能把这些名称强行认作XP04每维原字段来源。

| 逐维分布量 | min | p05 | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| MAE | 0.0000317488802 | 0.000353512478 | 0.002563854441 | 0.06717801106 | 0.4826652320 |
| RMSE | 0.0000341975321 | 0.002929107201 | 0.04108679212 | 0.2910347774 | 0.8262615969 |
| 绝对列均值差 | 0.00000208546181 | 0.000120417433 | 0.001651751406 | 0.01267358339 | 0.06656688998 |
| 绝对列标准差差 | 0.00000120198349 | 0.000109798815 | 0.001707473228 | 0.04764639206 | 0.8244744130 |
| Wasserstein距离 | 0.0000317488802 | 0.000353482765 | 0.002438532104 | 0.03146450466 | 0.2122880914 |
| Train-scale NRMSE | 0.0000339416818 | 0.002925166851 | 0.04112030074 | 0.2934115470 | 0.9714261262 |

目标与重建数组均无非有限元素、无全零话语，527行全部可计算余弦。目标和输出常数列均为0（以Validation列总体标准差≤协议scale_floor 1e-8界定）；1582个相关性全部可定义，无“目标非常数但输出常数”维度，也没有通过排除未定义列抬高相关性。源候选数组另无全零行或全零列。所有行保留，未按大误差删样本；没有缺失值补零或官方目标回填。

**Train选择与一次验证的解释。** Train OOF选中逐维仿射，Dialogue等权NRMSE0.1015833654、MAE0.0114111190、最大误差62.78680786，同样未通过OOF兼容门槛。v2额外全秩Ridge alpha0.01/1/100分数0.1364604215/0.1330534107/0.1390632610，更差，最终方法未变。此前已看过项目Validation摘要，因此不声称是项目历史上完全未见数据；独立性指本轮Train-only选择和正式锁后，唯一一次新Validation结果不反馈选择。扩展菜单、先前版本与哈希均见[协议修订记录](provenance/protocol_revisions.json)，不是依据本轮Validation反复改映射。

失败说明仍有影响兼容性的数值残差，无法由相同维度、较高平均余弦或多数列相关性接近1来消除。现有证据不能唯一定位残差来自引擎版本、Jitter/Shimmer修复、原采样处理或未恢复的后处理，故不将猜测写成实证原因。**本轮结论C，未恢复作者原算法，也未建立通过门槛的功能等价转换器。** CMERC未执行，Seed1475权重未训练或改动，Softmax误差与预测一致率均无结果；不进入原始音频干扰。
