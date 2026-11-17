# IEMOCAP 原始音频 Clean Feature Parity

**本轮WAV→OpenSMILE→XP04 1582维的Clean Parity未运行：未找到足以唯一确定原生产配置和后处理的记录。** IS10家族有文献支持，但它不等于已经找回原工具及配置；本报告不以默认配置试跑冒充复现。

## 已确认与尚未确认

XP04音频基础文件与MMGCN官方文件一致属于历史文件来源证据，本轮未重复下载或全量数值比对。MMGCN原论文及代码指向IS10；M3NET补充材料写IS13，但其作者又明确说明使用预提取特征、没有自行开展提取。原生产脚本未被这条引用链补回，不能自行裁决为笔误。

官方IS10_paraling、IS10_paraling_compat与emobase2010均为1582维，但兼容选项、FFT/Jitter及位置归一化不同。已缓存官方v3.0.0/1/2源码并静态比较；未找到与XP04历史生产匹配的2.x原包或完整运行记录。当前本地Python opensmile 2.6.0默认IS10指向compat；其共享库静态banner为引擎3.0.1。这是本地工具现状，不能当作原提取器。

详见[原生产来源核查](IEMOCAP_IS10_RECOVERY.md)及[官方配置比较](OPENSMILE_CONFIG_VERSION_COMPARISON.md)。

## 固定小样本准备

本轮使用seed1475 effective config的真正Validation ID，不读取Test标签。按Dialogue ID字典序取前六个对话，各选最早话语，共6条，不按情绪、预测正确性或置信度挑选。候选清单、数组行索引、原特征行哈希及WAV字节哈希仅存本地 `local_only/fixed_validation_candidates.json`；6个Dialogue都属于原12个Validation Dialogue。

6条WAV均存在，SHA-256与历史Validation审计一致；本轮header检查均为16kHz、单声道、16-bit PCM。IEMOCAP原论文的录制采样率、对话切段描述与当前release候选WAV属于不同阶段；当前header不能证明MMGCN实际使用这些字节或未再次裁切/变换。

## 数值验证状态

| 要求 | 状态 |
|---|---|
| 目标音频宽度 | 原XP04为1582；固定候选6条，尚无重提取矩阵 |
| 原WAV/对话/话语对应 | 固定候选通过真正Validation成员、媒体存在与SHA核验 |
| 原引擎/构建、完整配置/include | 未找回对应生产记录 |
| 有序1582字段header、切段/采样率/声道实际处理 | 未找回对应生产记录 |
| 外部scaler、统计量/拟合范围、缺失值处理 | 未找回；不从既有值域反推或重新拟合 |
| 重提取形状、maxabs、MAE、cosine、特征分布差异 | **NOT_RUN / null** |
| 冻结seed1475 checkpoint概率parity | **NOT_RUN / null**，等待特征先通过 |

未运行的误差不是0，也不是判定失败。旧审计中527条Validation音频的统计值只能描述发布文件，不是原特征提取验证，也不能证明做过某种z-score。

原生产链若补齐，应先在上述6条上保存有序header、完整参数/二进制哈希和原始输出，再逐维对齐已冻结的1582维目标，分别报告数值误差与分布差异。只有特征对齐后才比较冻结CMERC的完整Softmax；没有自动扩展到527条或其它split的授权。本轮不训练、不扰动、不修改XP04、不Git Commit/Push。
