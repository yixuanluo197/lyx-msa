# Causal CMERC 第三轮：真实数据三配对 seed 验证

**状态：全部完成。** 新增10个正式run、560个epoch；另复用第二轮IEMOCAP/1937的2个run、160个epoch。完整矩阵为两个数据集×三个seed×两种模型，共12个模型；validation/test共12项配对比较。没有失败、中断、恢复或因指标退化而重训。36项专项测试通过；独立终验676项通过、0失败、0待完成，独立复算804条逐seed指标和268条统计记录无误。

训练源码冻结提交：`5e5d7dd6cb7811f3c0d7c64c4afcbcb4aed9f69d`。所有训练采用validation选模、真实配对初始化、匹配seed的真实causal teacher/difficulty；IEMOCAP80轮、MELD40轮，没有改模型、损失或课程算法。全部剩余validation最晚于16:14:22 UTC完成，总冻结16:14:25，剩余测试最早16:14:32。完整时序和哈希见[终验](ROUND3_FINAL_AUDIT.json)及[执行记录](FINAL_EXECUTION_STATUS.md)。

## 主要比较：validation

| 数据集 | 指标 | Baseline 均值±SD | CMERC 均值±SD | 配对差均值±SD |
|---|---|---:|---:|---:|
| IEMOCAP | weighted_f1 | 61.7609 ± 2.0892 | 62.0381 ± 2.2506 | 0.2772 ± 0.1998 |
| IEMOCAP | ece | 21.9592 ± 2.8955 | 16.7137 ± 1.3639 | -5.2454 ± 3.3644 |
| IEMOCAP | cel.total | 562.0804 ± 137.5265 | 364.5651 ± 39.9072 | -197.5154 ± 128.6393 |
| MELD | weighted_f1 | 64.3612 ± 0.3545 | 64.6287 ± 0.9272 | 0.2675 ± 0.6802 |
| MELD | ece | 22.1167 ± 2.5074 | 18.6893 ± 2.5325 | -3.4274 ± 4.9265 |
| MELD | cel.total | 262.0242 ± 37.7387 | 272.1355 ± 10.9075 | 10.1113 ± 46.1359 |

## 冻结协议后的test比较

| 数据集 | 指标 | Baseline 均值±SD | CMERC 均值±SD | 配对差均值±SD |
|---|---|---:|---:|---:|
| IEMOCAP | weighted_f1 | 64.5209 ± 1.5535 | 65.1745 ± 0.5839 | 0.6537 ± 2.0867 |
| IEMOCAP | ece | 18.1846 ± 3.9559 | 15.5042 ± 3.3705 | -2.6804 ± 7.2988 |
| IEMOCAP | cel.total | 1526.7176 ± 290.6307 | 1008.2627 ± 68.9892 | -518.4549 ± 347.6689 |
| MELD | weighted_f1 | 64.5976 ± 0.8148 | 64.8555 ± 0.4775 | 0.2580 ± 0.8211 |
| MELD | ece | 21.5639 ± 2.9880 | 17.9390 ± 1.5299 | -3.6250 ± 4.4799 |
| MELD | cel.total | 676.7711 ± 104.6482 | 703.6284 ± 44.1996 | 26.8574 ± 125.3900 |

所有差值为CMERC−Baseline；分类与校准差值单位为百分点，CEL为原定义的求和量。SD是三个训练seed的样本标准差(ddof=1)，不是标准误。CEL不能直接跨不同数据集或不同replacement policy比较优劣。

### 每个seed的测试差值

| 数据集 | seed | ΔW-F1 | ΔECE | ΔCEL | ΔAngry/anger F1 | ΔAUROC-Success |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 1937 | +1.1011 | -3.9696 | -522.6839 | -8.0797 | -0.1434 |
| IEMOCAP | 2025 | +2.4803 | -9.2488 | -863.9900 | -2.3933 | -1.0647 |
| IEMOCAP | 3407 | -1.6205 | +5.1771 | -168.6909 | -7.8892 | +2.5489 |
| MELD | 1937 | +1.1475 | -0.6982 | -117.0839 | +4.7066 | -1.8615 |
| MELD | 2025 | -0.4712 | -1.3944 | +112.3676 | -1.1491 | -0.0637 |
| MELD | 3407 | +0.0976 | -8.7823 | +85.2884 | +1.8214 | -0.8124 |

## 已证实的本轮结果

- **W-F1只有小幅平均增益，未证明稳定提升。** 两个数据集测试均为2/3 seed改善；IEMOCAP/3407下降1.6205个百分点，MELD/2025下降0.4712。不能只报告1937或最好seed。
- **CEL改善没有稳定推广到MELD。** IEMOCAP测试3/3下降；MELD只有1937下降，2025/3407上升，MELD的validation和test平均CEL均变差。
- **ECE与置信度排序必须分开解释。** IEMOCAP测试ECE为2/3改善；MELD测试ECE为3/3改善，但MELD三个seed的AUROC-Success全部下降，平均下降0.9125个百分点。
- **IEMOCAP Angry F1下降重复出现。** 三个seed在validation和test均下降；测试配对差均值−6.1207、样本SD3.2294个百分点。它是需要进一步验证机制的类别权衡，不能归因于某一单独组件。
- **上下文CEL的作用有明显边界。** IEMOCAP测试平均intra/inter项合计下降330.5070，约占总净减幅63.75%；但2025仅48.43%，3407为140.33%，后者表示其他模态项合计变差抵消了一部分减幅。MELD测试text CEL平均增加43.0700，超过context两项合计下降17.1196，不能宣称所有视图鲁棒性同步提高。

## 统计与科学解释边界

全部六个主要假设在validation/test各自的Holm校正p值均为1.0，所有主要配对95% t区间跨0。只有三个固定划分seed，双侧精确sign-flip最小p=.25；这轮不能支持通常阈值下的统计显著性或广泛泛化结论。完整均值/SD、区间和精确p见[配对统计](MULTISEED_PAIRED_RESULTS.md)。

IEMOCAP/1937测试此前已观察，本轮核验后原样复用；MELD/1937是预定技术先导，其测试在其余训练前可见。所有配置在先导之前本地冻结；这是有时间与哈希证据的内部协议冻结，不冒充外部平台注册的完全盲测。

`prefix_mean_v1`是可逆前缀平滑视图，不是完全模态移除；intra/inter是GNN-edge ablation，编码器和超图仍可保留历史信息。`legacy_epoch_snapshot`的训练CC包含辅助权重滞后及train/eval dropout差异；CEL评估则使用相同权重、相同eval模式。预提取特征之后的时间因果性沿用第一轮审查，原始特征提取过程仍未审计。

实际加权梯度、全部epoch损失和首batch诊断见[损失与梯度](LOSS_GRADIENT_ANALYSIS.md)。IEMOCAP三个seed的CC加权梯度均值为14.6645/21.3787/17.8825，分类项为6.7467/3.4985/4.4178；范数不可相加成总梯度，也不等同Adam更新份额，不能仅凭这些数值证明类别退化机制。

共享服务器耗时仅作描述比较。IEMOCAP每run实际Adam步数Baseline560、CMERC541；MELD为2600/2548。CL自然减少早期更新次数。最大记录allocated显存为IEMOCAP4900.42 MiB、MELD896.06 MiB，不能与CUDA缓存reserved或宿主机nvidia-smi占用混淆。

## 下一轮值得验证的候选问题（本轮未启动）

1. 优先做同一causal backbone的w/o CC、w/o HSCL、w/o CL配对消融，检验Angry权衡和CC梯度方向是否相关；当前证据不证明任何单项因果机制。
2. 针对MELD的text CEL反向变化与context项改善并存，检查视图、置信度尺度和真实模态缺失诊断的区别；新增严格缺失策略必须另命名，不能混比CEL。
3. 分别评估校准与错误排序/选择性预测，检验ECE下降但AUROC下降的可重复性；validation拟合的校准对照可作为独立后续方案。
4. 将legacy_epoch_snapshot与same_step_same_mode作为显式独立实验策略比较，控制同权重、同模式，再讨论辅助视图语义。

这些是研究假设，不是已证明的创新。未自动启动额外消融、更多seed或超参数搜索。

## 报告与证据

- [冻结协议](THIRD_ROUND_PROTOCOL.md)
- [MELD完整对照](MELD_BASELINE_VS_CMERC.md)
- [多seed配对统计](MULTISEED_PAIRED_RESULTS.md)
- [校准诊断](CALIBRATION_DIAGNOSTIC.md)
- [类别与前中后位置分析](CLASS_AND_POSITION_ANALYSIS.md)
- [实际损失与加权梯度](LOSS_GRADIENT_ANALYSIS.md)
- [后续研究候选](NEXT_RESEARCH_CANDIDATES.md)
- [最终执行、选模与checkpoint哈希](FINAL_EXECUTION_STATUS.md)
- [源码/环境/特征/划分哈希](RUN_PROVENANCE.json)、[teacher/difficulty来源](DIFFICULTY_PROVENANCE.json)、[原始日志索引与脱敏命令](RAW_LOG_INDEX.json)
- [收敛曲线](figures/convergence_IEMOCAP.png)、[MELD曲线](figures/convergence_MELD.png)、[配对测试差值](figures/paired_deltas_test.png)

模型权重、特征、真实ID、逐话语预测与大型原始日志全部保留服务器独立批次；公开文件只有聚合、配置、哈希、小型测试日志和图。原始日志索引中的`$BATCH`/`$REPO`是脱敏路径占位，原始精确命令和运行环境保存在服务器STATE记录中。
