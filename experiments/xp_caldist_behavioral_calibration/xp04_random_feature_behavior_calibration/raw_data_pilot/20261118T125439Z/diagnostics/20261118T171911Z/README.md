# XP04 RoBERTa 特征蒸馏失败原因诊断

基于原实验提交 `13389ee`，IEMOCAP 与 MELD 的三个 Train-only 诊断全部完成。

| 数据集 | 128条、500步 MSE：初始→最终 | 原Epoch 8 Fit / 内部验证 MSE |
|---|---|---|
| IEMOCAP | 0.763013→0.00131968 | 0.0282514 / 0.0427956 |
| MELD | 1.043726→0.00106211 | 0.0229422 / 0.0352256 |

学生具有较强近似拟合能力，但仍未达到原特征兼容精度。相同实际权重、同Token ID、FP32的Fairseq/HF比较通过，最大逐层MAE约5.1e-7。当前证据支持拟合精度不足并伴随泛化差距，不能单独归因于框架、某个超参数或证明第二轮必然成功。

- [综合判断](DIAGNOSTIC_SUMMARY.md)
- [小样本拟合](SMALL_SAMPLE_OVERFIT.md)、[框架对比](FRAMEWORK_PARITY.md)、[Fit/内部验证](TRAIN_DEV_GAP.md)
- [拟合曲线](results/figures/small_sample_overfit.png)、[误差对比](results/figures/train_dev_gap.png)
- [实验核验：855项](verification.json)、[发布核验](publication_verification.json)、[发布清单](publication_manifest.json)

仅使用已有Train Fit/内部验证缓存，未访问官方Validation/Test，未重新训练CMERC，未进行文本干扰或XP04概率验证。两个临时学生均在固定500步后停止。本次发布仅整理已有结果，没有重新训练或评估。

`training_records/` 是临时过拟合实验的完整500步标量日志、8个评估点、配置和完成记录；`results/` 保留原始统计结果；`scripts/` 为实际诊断代码。原模型权重、缓存、原始文本、Token ID、目标向量、逐话语清单、Fairseq源码包和完整特征全部留在本地。

本地路径、主机和GPU UUID已脱敏。代码中的 `/LOCAL/`、`/LOCAL_HOME`、`LOCAL_HOST` 为占位符，执行需要本地授权资产；公开代码未在新环境重新执行。原验证记录中的输入/脚本SHA对应本地原字节；公开副本SHA以发布清单为准。原报告的“未commit/push”描述实验冻结时状态，本次发布在研究者另行授权后进行。诊断交付PASS不代表特征兼容PASS。公开核验脚本只做静态检查，不读取媒体、特征或模型。
