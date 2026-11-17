# 本轮执行范围与 Clean Parity 门控

本轮沿用 `research-experiment-driver` 与 `github-research-workflow`，在当前工作区恢复来源；GitHub仅做公开来源只读查询。本轮最新要求明确禁止自动Git Commit/Push，因此此前的上传授权不在本轮使用。

## 分工与证据层级

三个独立只读研究分支负责：MELD官方仓库/论文历史及选择器；IEMOCAP生产来源及IS10/IS13冲突；OpenSMILE官方配置/版本及本地候选的静态比较。主执行分支负责历史与输入哈希、固定Validation候选、现有媒体header、门控、综合报告和最终验证。

每个结论分别标记为：官方原始资料、公开可执行资产、推断、实际数值验证、缺失。来源相同、维度相同、配置名称相同均不能替代原WAV到特征的数值验证。

## 固定小样本

候选在新提取前固定，选择不访问标签或预测：IEMOCAP按真正Validation的Dialogue ID字典序取前六个对话，各取首个话语，共6条；MELD保留上轮已验证的完整原dev Dialogue 0/1/2（XP04 1039/1040/1041），共24条话语，保持上下文。清单与媒体字节哈希只在本地 `local_only/fixed_validation_candidates.json`。成员均与指定seed的effective config `provenance.split.validation_ids`核对。

这些是条件满足后的候选，不代表已经运行提取、也不是原120条Pilot抽样。本轮不扩大到完整数据集。

## 允许执行的条件

1. 找到能追溯到原发布特征的OpenSMILE工具/配置/include/字段顺序与音轨切段、采样率、声道规则。
2. 找到实际存在的前后处理参数或统计量。MELD还必须取得实际原selector/字段映射或其他原变换，不用新训练选择器替代。
3. MELD使用已找回的原BiGRU权重，在完整同一Dialogue上下文及真实顺序下重算，再按真实长度取输出；padding单独报告。
4. 只有上述门控通过才运行少量raw Clean Parity，并记录形状、maxabs、MAE、余弦和分布差异。任一未运行项记 `null / NOT_RUN`，不能填0或写parity失败。
5. 只有特征先通过，才允许使用冻结XP04 checkpoint检查Softmax概率。无训练、无Test标签、无扰动。

## 本轮只读现场检查

新鲜哈希锚定两个checkpoint、两个effective config、两个音频基础pickle、两个历史Validation基准，共8项；历史130份报告/JSON/脚本另作哈希快照。逐媒体只检查固定候选存在、SHA-256和音轨metadata，未解码为PCM、未重采样或提特征。当前环境发现可用ffprobe；安装的Python opensmile候选交由配置分支静态记录，不能据其可运行认定原生产版本。

MELD 24条现有MP4：AAC、48kHz、6声道5.1；IEMOCAP 6条WAV：16kHz、单声道、16-bit PCM。此为本地候选现状，不是历史生产采样率/混音方式的证明。完整记录仅本地保留，汇总见 `provenance/validation_audio_probe_summary.json`。
