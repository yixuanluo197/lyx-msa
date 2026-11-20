# XP04 RoBERTa 小样本特征蒸馏延长实验

基于诊断提交5957f29，从原500步学生权重及相同128条Train Fit文本继续训练。旧优化器未保存，故新建AdamW状态并重置采样器；本轮不是原状态的精确断点续训。

| 数据集 | 总步数 | 500步MSE | 最终MSE | 降幅 |
|---|---:|---:|---:|---:|
| IEMOCAP | 2000 | 0.00131968 | 0.00115999 | 12.1% |
| MELD | 5000 | 0.00106211 | 0.00042710 | 59.8% |

IEMOCAP在1500→2000步回退40.2%，按预定规则停止；MELD同窗口下降56.8%，触发延至5000。MELD最终窗口仍下降61.8%，但全程明显波动。两套最终逐层RMSE仍是1e-4门槛的约119–508倍，均未通过必要条件，不建议直接按相同配置启动第二轮完整蒸馏。

- [完整报告](TRAINING_EXTENSION_REPORT.md)、[最终判断](FINAL_EXTENSION_DECISION.md)
- [损失与RMSE曲线](results/training_extension.png)、[PDF](results/training_extension.pdf)
- [四层比较CSV](results/per_layer_comparison.csv)、[阶段比较CSV](results/stage_comparison.csv)
- [实验核验：2244项](verification.json)、[发布核验](publication_verification.json)、[发布清单](publication_manifest.json)

`training_records/`保留正式与隔离smoke的标量记录、全部评估点、连续步骤、配置、决策及checkpoint哈希；不包含任何权重、优化器张量或逐话语清单。正式新增更新为IEMOCAP1500次、MELD4500次，总步数均计入原500步。

未访问官方Validation/Test，未重训CMERC，未做文本干扰、完整数据集蒸馏或XP04概率验证。本次发布没有重新训练或评估。模型、完整训练状态、原始文本、Token ID、教师特征及逐话语文件全部保留本地。

本地路径、主机和设备UUID已脱敏；`/LOCAL/`、`/LOCAL_HOME`和`LOCAL_HOST`是占位符。代码需要本地授权资产，未在新环境重新运行。历史输入/代码/产物SHA对应本地原字节，公开副本SHA以publication_manifest.json为准。原报告“未commit/push”描述实验冻结状态，发布由研究者随后另行授权。发布核验为静态检查，交付PASS不代表特征兼容PASS。
