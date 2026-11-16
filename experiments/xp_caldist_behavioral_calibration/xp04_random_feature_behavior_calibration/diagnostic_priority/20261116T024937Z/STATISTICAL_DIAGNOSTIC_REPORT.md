# 特征扰动行为：诊断统计报告

已有joint .05强度、10次扰动，两个数据集完整Validation、五Seed8180条模型记录。相同话语的不同Seed不是独立数据。本轮没有推理、扰动生成、校准拟合或Test使用。

## 稀疏性与重复一致性

| dataset | corruption | mu_zero_fraction | mu_split_rho | delta_split_rho | delta_split_seed_sd | mean_seed_median_delta_over_se |
|---|---|---|---|---|---|---|
| IEMOCAP | gaussian | 0.985958 | 0.721708 | 0.521465 | 0.166421 | 0.693611 |
| IEMOCAP | dropout | 0.949146 | 0.743468 | 0.635846 | 0.102857 | 1.767397 |
| MELD | gaussian | 0.991344 | 0.640682 | 0.071732 | 0.032840 | 0.744550 |
| MELD | dropout | 0.961226 | 0.705252 | 0.580707 | 0.098354 | 2.555843 |

μ是带扰动置信度权重的翻转均值，与翻转率零模式相同、数值不同。MELD Gaussian μ约99.13%为0，Dropout约96.12%为0；高度相等值使μ的秩相关必须谨慎解释。拆分统计先平均每Seed的三个5/5拆分，再汇总五Seed均值/样本标准差；不是15份独立数据。常量向量记NA。

源标准差为10次重复的ddof0：sample_sd=std_pop×sqrt(10/9)，signed_mean_SE=std_pop/sqrt9。δ=abs(mean signed change)，δ/SE仅是幅度/噪声描述，不是δ置信区间或p值。MELD Gaussian的逐Seed比值中位数均值0.745，Dropout2.556；IEMOCAP分别0.694/1.767。Gaussian δ的典型幅度与估计噪声同量级，结合低一致性，支持检查测量噪声限制；不能据此断言δ完全由噪声组成。零均值采样波动经绝对值也会产生正δ。

## 未控制原C的错误风险AUC

| dataset | corruption | measure | auc_larger_is_error_mean | auc_larger_is_error_sample_sd |
|---|---|---|---|---|
| IEMOCAP | dropout | delta | 0.544935 | 0.039001 |
| IEMOCAP | dropout | mu | 0.519549 | 0.019753 |
| IEMOCAP | gaussian | delta | 0.568546 | 0.035322 |
| IEMOCAP | gaussian | mu | 0.508945 | 0.008245 |
| MELD | dropout | delta | 0.483399 | 0.026995 |
| MELD | dropout | mu | 0.522926 | 0.007146 |
| MELD | gaussian | delta | 0.541906 | 0.017738 |
| MELD | gaussian | mu | 0.506598 | 0.002379 |

方向固定为数值越大越容易错误，不事后翻转。AUC接近.5表示该单一方向区分弱；小于.5不等于没有信息。上述未控制C，不代表相对E1增量。

## 同Dialogue、近似原C配对

| dataset | corruption | measure | point_estimate | ci_low | ci_high | min_seed_pairs | min_seed_dialogues |
|---|---|---|---|---|---|---|---|
| IEMOCAP | gaussian | mu | 0.00026483 | -0.00104548 | 0.00186775 | 110 | 11 |
| IEMOCAP | gaussian | delta | -0.00001467 | -0.00009128 | 0.00005435 | 110 | 11 |
| IEMOCAP | dropout | mu | 0.00049047 | -0.00325297 | 0.00431574 | 110 | 11 |
| IEMOCAP | dropout | delta | 0.00015617 | -0.00077424 | 0.00100927 | 110 | 11 |
| MELD | gaussian | mu | 0.00068852 | -0.00038478 | 0.00187678 | 133 | 63 |
| MELD | gaussian | delta | 0.00001811 | -0.00001941 | 0.00005511 | 133 | 63 |
| MELD | dropout | mu | 0.00068180 | -0.00174698 | 0.00340177 | 133 | 63 |
| MELD | dropout | delta | -0.00037903 | -0.00103885 | 0.00027864 | 133 | 63 |

差值为wrong−correct。8个探索性95%区间全部跨0：没有清晰方向证据，不是证明组差为零。2000次Dialogue共享抽样，各Seed单独计算再平均；每组2000次均有效，没有丢弃Seed。配对只控制同一Dialogue和|ΔC|≤.02，仍有C残差、类别及语境混杂；覆盖限制见配对报告。

## 对已有校准结果的解释

E1已修正MELD大部分整体欠自信。当前稀疏性、Gaussian δ不稳定及近似同C下组差不明确，与行为模型未超过E1的结果相一致，尚未定位唯一原因。Dropout δ较稳定却也未出现明确配对组差，说明“测得稳定”和“能改善校准”必须分别检验。

## 验证与边界

276项历史hash通过、正式输入/源码前后hash一致、11项工程测试通过。独立V2审计329项通过，复算16000条抽样统计及全部配对最优目标。V1误将μ等同翻转率，已版本化勘误并保留失败记录；15个原统计CSV未改，详见provenance/verifier_erratum.json。所有区间未做多重比较修正，数据为反复分析的Validation。本轮没有新增ECE/Brier/NLL结果，不能把诊断关联当校准收益。
