# MELD 原始音频 Clean Feature Parity

**本轮原始音频→1611→原BiGRU→300的 Clean Parity 未运行：前端生产链门控未通过。** 这是未满足前置条件，不是数值验证失败；未测量的误差和概率差异不能填0。

## 历史已实测结果：复用，不重复运行

上轮 `20261117T010744Z`（公开提交 `8bf2de0`）已完成：

| 检查范围 | 已实测结果 |
|---|---|
| 官方audio_emotion与XP04 | 13708条真实话语×300逐维精确一致，maxabs=MAE=0；Test只参与无标签特征身份核验 |
| 已选择1611维→原官方BiGRU→300 | 3个真正Validation完整Dialogue、24条真实话语，maxabs=`1.1920928955078125e-6`，MAE=`4.268819692596379e-8`，平均cosine=`0.9999999999999897`；`atol=rtol=1e-5`通过 |
| 含padding的整个33步张量 | 不完全一致：当前Keras将padding输出置零，旧包尾部保留最终forward状态，误差可接近1 |
| 原始WAV→1611 | 未执行、未验证 |
| CMERC完整Softmax概率 | 未执行、未测量 |

数值来源为[上轮恢复机器证据](../20261117T010744Z/evidence/encoder_recovery_summary.json)；本轮以哈希验证证据未变，不重复进行原网络前向。原网络是每方向150单元的BiGRU，不能按报告文件名bcLSTM重建为两层LSTM。

## 本轮固定候选与现场核验

若原链补齐，固定采用原dev Dialogue 0、1、2，即XP04 1039、1040、1041，共2+16+6=24条真正Validation话语。必须送入完整同一对话，按原Utterance ID数值排序而不是字典遍历或直接用ID作为数组下标；右侧补零至33步、Masking(0)、float32、关闭dropout。只对真实行做主parity，padding单列。修改目标输入可能影响同对话多个300维输出，这些输出不能解释为独立话语的声学统计。

30项固定候选清单中的这24条媒体全部存在，SHA与旧审计一致。本轮使用已有ffprobe仅检查媒体音轨metadata：24条均为AAC、48kHz、6声道5.1。未导出PCM、未下混、未重采样、未调用声学提取器。这些header只说明现有候选媒体，不能证明历史选择了哪些声道或如何下混。

## 门控结果与未测量项

| 原链环节 | 本轮证据 / 缺失 |
|---|---|
| OpenSMILE配置家族 | 早期论文直接写IS13-ComParE；准确引擎/配置/include与构建仍缺 |
| 音轨预处理 | 早期论文提到强度阈值与Z-standardization；阈值、作用位置、拟合范围/统计量及声道/采样率命令缺失 |
| 6373→1611 | 无原selector、索引或变换矩阵；论文/README的SVM L2与作者Issue中PCA说明存在冲突 |
| 1611→300 | 原模型/权重及历史24行parity已具备；不替代上面缺失环节 |

因此本轮新生成特征形状、maxabs、MAE、cosine、分布差异与CMERC预测概率误差均为 **NOT_RUN / null**。既有1611数组不能用来伪装“从原始音频重新提取的Clean输入”。当前安装OpenSMILE或来自其它项目的选择器也不能补替。

本轮新鲜静态资产对照只比较已发布emotion/sentiment特征列，见[选择器来源报告](MELD_AUDIO_SELECTOR_RECOVERY.md)；它没有生成任何原始音频特征，也不构成原始音频parity。全Validation仍有上轮记录的原dev110/utt7媒体缺失，不能从Test同名媒体替换；本轮固定24条不受该缺失影响。

完成来源核查后停止，不运行原始音频干扰、不训练、不使用Test标签、不提交或推送。
