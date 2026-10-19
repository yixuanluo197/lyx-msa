# XP04 CMERC 异常置信度上升样本占比

主表模型：**CMERC**。XP04另有配对模型；本表只统计所列模型，未将不同模型合并。

直接复用五个正式seed的validation-best对应测试集predictions.npz，不进行训练、推理或checkpoint重选。

## 统计口径

- 样本单位为测试话语；IEMOCAP每seed为1623条，MELD每seed为2610条。分母包含预测正确和错误的全部样本。
- 比较真实类别y的置信度，不使用argmax类别或最大概率代替。
- Modality：text/audio/visual任一路移除后p(y)>full即计入，对三路取并集；一个样本最多一次。
- Context：intra-context/inter-context任一路移除后p(y)>full即计入，对两路取并集；一个样本最多一次。
- 两类别分别计数，一个样本可以同时属于Modality和Context。相等不计入；严格>，不设额外epsilon或幅度门槛。
- 以保存的float32 log-probs转换为float64后直接比较；log为严格单调函数。另用float64 exp后的概率独立复核，所有判断一致。
- 每seed百分比=并集样本数/该seed测试样本总数×100；五seed等权均值，样本标准差ddof=1，不是标准误。
- 这是用户定义的置信度上升事件比例，不是CEL幅度、分类错误率或统计显著异常比例。

## 逐seed结果

| Dataset | Seed | Modality (%) | Context (%) |
|---------|------|--------------|-------------|
| IEMOCAP | 1475 | 81.700555 | 70.117067 |
| IEMOCAP | 67137 | 80.036969 | 80.160197 |
| IEMOCAP | 2024 | 80.653112 | 62.784966 |
| IEMOCAP | 3407 | 79.297597 | 74.676525 |
| IEMOCAP | 42 | 78.126925 | 82.193469 |
| MELD | 1475 | 67.662835 | 84.444444 |
| MELD | 67137 | 67.892720 | 84.022989 |
| MELD | 2024 | 62.643678 | 88.045977 |
| MELD | 3407 | 67.931034 | 88.390805 |
| MELD | 42 | 61.034483 | 89.731801 |

## 五seed汇总

| Dataset | Mean Modality (%) | Std | Mean Context (%) | Std |
|---------|-------------------|-----|------------------|-----|
| IEMOCAP | 79.963031 | 1.352290 | 73.986445 | 7.848437 |
| MELD | 65.432950 | 3.331285 | 86.927203 | 2.542543 |

## 绘图四个主值（百分比均值）

- IEMOCAP modality：**79.963031%**（std 1.352290个百分点）
- IEMOCAP context：**73.986445%**（std 7.848437个百分点）
- MELD modality：**65.432950%**（std 3.331285个百分点）
- MELD context：**86.927203%**（std 2.542543个百分点）

## 整数计数及所用checkpoint

| Dataset | Seed | N | Modality union | Context union | Selected epoch |
|---------|------|---|----------------|---------------|----------------|
| IEMOCAP | 1475 | 1623 | 1326 | 1138 | 43 |
| IEMOCAP | 67137 | 1623 | 1299 | 1301 | 34 |
| IEMOCAP | 2024 | 1623 | 1309 | 1019 | 71 |
| IEMOCAP | 3407 | 1623 | 1287 | 1212 | 75 |
| IEMOCAP | 42 | 1623 | 1268 | 1334 | 24 |
| MELD | 1475 | 2610 | 1766 | 2204 | 11 |
| MELD | 67137 | 2610 | 1772 | 2193 | 8 |
| MELD | 2024 | 2610 | 1635 | 2298 | 9 |
| MELD | 3407 | 2610 | 1773 | 2307 | 14 |
| MELD | 42 | 2610 | 1593 | 2342 | 5 |

## 来源与复算

十份NPZ及独立metrics.json的SHA-256均与XP04原验收记录一致；checkpoint路径和validation选中epoch一致。跨seed测试ID、话语位置及真实标签逐项相同，五路数组完整、有限、维度及概率归一化通过检查。

原始逐样本文件仍保留在本机原批次/独立备份，不复制进公开档案。本次结果是已完成XP04的派生分析，不改原训练、验收或输入文件。输入路径、哈希、各路计数及checkpoint哈希记录见 [分析来源](../provenance/uncertain_percentage_analysis.json)。

在CMERC工程根目录运行（需要原始NPZ可访问及NumPy）：

```bash
/data2/yb/conda_envs/cmerc-xp-repro/bin/python experiment_records/04_xp_mainline_validation/tools/compute_uncertain_percentage.py --archive experiment_records/04_xp_mainline_validation --output-root /tmp/xp04_uncertain_recompute
```

复算CSV的列名区分两类Std；报告和CSV保留六位及十二位小数，统计使用未舍入值。
