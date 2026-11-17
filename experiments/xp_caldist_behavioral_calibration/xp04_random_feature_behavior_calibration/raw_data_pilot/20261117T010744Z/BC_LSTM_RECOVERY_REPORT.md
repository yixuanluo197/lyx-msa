# MELD 官方低维音频编码器恢复报告

审计时间：2026-11-17 UTC。使用已核验的官方 `MELD.Features.Models.tar.gz`，SHA-256 为 `59131c6904ab1272912c6ebec7873524e6f90bce842b333bc27f02411718b2c3`，922694594 bytes。证据、模型 JSON、权重形状及数值检验保存在 `sources/bclstm/` 与 `evidence/encoder_recovery_summary.json`。

## 本轮结论

**官方 emotion 音频权重已找到，并且“官方已选择的 1611 维音频 → 原模型 `utter` 层 300 维”的 CPU Clean Parity 在 3 个真正 Validation Dialogue、24 条真实话语上通过。** 这补齐了低维上下文编码器这一段；原始 WAV 到该 1611 维输入的生产链仍未闭环，因此属于部分恢复，不能据此启动原始音频干扰 Pilot。

上轮根据当前 GitHub baseline 未找到 300 维的确切网络；本轮官方 HDF5 给出了直接证据：emotion 模型实际采用**双向 GRU，每方向 150 单元**。报告文件名沿用任务要求的 `BC_LSTM_RECOVERY_REPORT.md`；这里不把实际 GRU 写成 LSTM。

## 原资产与结构

两个原 HDF5 文件均保留在原数据目录，没有复制到公开交付或修改。静态解析时逐一核对了 tar 成员与现有解包文件的 SHA-256，相等。

| 资产 | SHA-256 | 原文件内的实际结构 |
|---|---|---|
| `models/audio_weights_emotion.hdf5` | `2f5a006d853ea6b2e2a7b40750005963e6dd59f16b1072c468e702c84632b2da` | 输入 `(None,33,1611)`；Masking(0)；一层 BiGRU150×2 concat，名为 `utter`，输出 `(None,33,300)`；逐时刻 Dense150/ReLU；Dropout0.1；Dense7/Softmax |
| `models/audio_weights_sentiment.hdf5` | `c6cd9672507b23ae210eafe3d431879b31f9897227cfc5cdcb8134dabffd26da` | 输入 `(None,33,1422)`；Masking(0)；两层 BiLSTM300×2；第二层名为 `utter`，输出 `(None,33,600)`；Dense3/Softmax |

两者文件属性均记录 `keras_version=2.2.0`、`backend=tensorflow`；文件没有记录原 TensorFlow 的准确版本。emotion 的 GRU 配置明确包含 `activation=tanh`、`recurrent_activation=hard_sigmoid`、`reset_after=False`、`implementation=1`、`return_sequences=True`、输入 dropout0.5。推理使用 `training=False`。

emotion 正向与反向每个方向的参数形状均为 kernel `(1611,450)`、recurrent kernel `(150,450)`、bias `(450,)`，与原 JSON 的三门 GRU 配置一致。sentiment 的输入1422和600维输出不能替代本研究的 emotion 1611→300链。

静态完整配置与参数形状分别见 `sources/bclstm/audio_weights_emotion_metadata.json`、`audio_weights_sentiment_metadata.json`。只导出配置、形状和哈希，没有导出权重数值、原文或标签值。

## 为什么与现有 GitHub 脚本不同

固定版本 [`baseline.py` L83–100](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/baseline.py#L83-L100) 定义的是两层每方向300的 BiLSTM，而不是该 emotion HDF5 的单层 BiGRU150。`test_model()` 则实际载入 HDF5，再导出名为 `utter` 的层，因此 **导出流程可以适用于这个原文件，重建模型时却不能照抄现有 `get_audio_model()`**。[原导出代码 L232–248](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/baseline.py#L232-L248)。

本轮直接载入原 HDF5 内的完整模型定义和权重，没有按消费者维度新建编码器，没有替换或重新训练权重。由原文件而非推断确认：本官方 emotion300 是上下文 GRU 隐状态，不是 OpenSMILE 原始300项统计，也不是分类器 Softmax 或150维ReLU层。

## 1611 输入的对话和 padding 恢复

官方 `audio_embeddings_feature_selection_emotion.pkl` 的 Validation 字典含逐话语键 `Dialogue_ID_Utterance_ID`；官方 `audio_emotion.pkl` 则是 Dialogue 键到 `(33,300)` 的表征。dict 插入次序不保证从0开始，不能把第一个键当作 Dialogue0。

原数据加载代码先将同 Dialogue 的 `Utterance_ID` 数值排序，再拼成对话，并在右侧补零到33。[排序 L74–80](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/data_helpers.py#L74-L80)、[音频拼接 L94–115](https://github.com/declare-lab/MELD/blob/e8cedf27b5d2877e198332c957127e16eb214afe/baseline/data_helpers.py#L94-L115)。

本轮只选真正 Validation 原 Dialogue0、1、2，分别2、16、6条话语；对应 XP04 Dialogue1039、1040、1041。依据原 ID 对齐后的每个完整对话构造 `(33,1611)` float32 输入；保留 Masking(0)，不读取情绪标签，也不调用原脚本中附带的分类评估函数。输入、参考真实行均保存哈希。完整映射见 `sources/bclstm/dialogue_map.json`。

## 旧 Keras 语义与当前环境检查

使用现有环境 `$AUDIT_ENV/bin/python`，TensorFlow2.14.0、Keras2.14.0、h5py3.16.0。没有安装依赖；设定 `CUDA_VISIBLE_DEVICES=''`，TensorFlow确认没有可见GPU。

- `load_model(original_hdf5, compile=False)` 一次直接成功，未进行手工架构/权重适配。
- 载入后 `utter` 的 GRU 仍是 reset-before（`reset_after=False`）、implementation1。逐个比对正向/反向的 kernel、recurrent_kernel、bias，六组已载入张量与 HDF5 原值全部 `array_equal=True`。
- 实测当前 `hard_sigmoid` 在输入 `[-4,-2.5,-1,0,1,2.5,4]` 上得到 `[0,0,0.3,0.5,0.7,1,1]`，逐值等于旧式 `clip(0.2*x+0.5,0,1)`，没有套用 Keras3 的另一种斜率。
- `training=False` 关闭 dropout；双向顺序和 mask 对真实行的兼容性通过下面的原输出 parity 数值验证。

## 小规模 Clean Parity 结果

对照目标为官方 `audio_emotion.pkl` 的 Validation 真实话语行。预先记录两档容差 `atol=rtol=1e-5` 和 `atol=rtol=1e-4`，不根据测试标签或分类表现选择容差。

| 原 dev Dialogue / XP04 Dialogue | 真实行数 | 最大绝对误差 | 平均绝对误差 | 平均余弦相似度 |
|---|---:|---:|---:|---:|
| 0 / 1039 | 2 | 6.55651e−7 | 3.95222e−8 | 0.9999999999999883 |
| 1 / 1040 | 16 | 1.19209e−6 | 4.32496e−8 | 0.9999999999999898 |
| 2 / 1041 | 6 | 7.30157e−7 | 4.22463e−8 | 0.9999999999999901 |
| 合计 | 24 | **1.19209e−6** | **4.26882e−8** | **0.9999999999999897** |

每个 Dialogue 的全部真实话语行均通过两档容差；总体真实行形状为 `(24,300)`。

同轮独立文件对照已证明：官方 `audio_emotion.pkl` 的真实 Validation1109行与 XP04 音频300维逐项完全一致（maxabs=MAE=0），且上述24行的 Dialogue/行映射相同。因此重算值对 XP04 真实行的误差与表中相同；完整文件对照见 `evidence/feature_parity_summary.json`。本报告不重复全量特征对照，也不运行 CMERC；预测概率差异为**未测量**。

### 保留的异常：padding 不能声称完全一致

现代 Keras2.14 的 Bidirectional 对被 mask 的尾部输出置零；这三个 Dialogue 的历史官方输出在 padding 位置保留了“最后真实正向状态的150维重复＋反向150维全零”。因此：

- 真实话语300维 parity 通过。
- **整个 `(33,300)` 含padding张量的严格 parity 未通过**；padding最大误差约1，完整数值已保留。
- 本轮没有改写官方目标或隐藏异常，没有用padding误差替代真实话语误差，也没有为了全张量通过而静默填回历史状态。

XP04按真实话语长度消费音频行；上述异常不会改变本轮真实行通过的结论，但后续恢复历史导出代码时必须明确其padding输出语义。

## 原始 OpenSMILE 与 selector 仍缺哪些内容

已核查官方包的全部目录：20个文件仅含 README、6个HDF5和13个pickle/p，没有独立 `.conf`、特征字段顺序、选择索引、SVM selector、标准化器或音频生产脚本。包内 `data_emotion.p` / `data_sentiment.p` 顶层均为6项：原记录列表、形状 `(6336,300)` 的矩阵、两个字典结构、最大文本长度标量、标签映射字典。原记录字段名仅为 `dialog, num_words, split, text, utterance, y`；本轮只检查类型/形状/字段名，没有读取或导出 Test 标签值或原文。

其结构与官方说明的文本词向量/索引及标签元数据一致，不能当作音频selector。HDF5只接收1611/1422维输入，也没有6373维选择层；优化器状态并不能补回外部特征选择过程。包内旧README解释数据集及原始片段，没有记录 OpenSMILE生产配置。

因此仍缺：原WAV解码与切段/声道/采样率规则，OpenSMILE版本与配置/include哈希，6373个原字段及顺序，6373→1611原选择索引或selection模型，以及其前后归一化统计量、拟合范围和缺失策略。前轮发现的M3NET IS10/IS13论文/代码矛盾仍不能由这个HDF5消解。

## 后续研究边界

已能恢复原有“已选择1611输入→对话上下文300输出”的末端，并验证它确实生成了XP04所消费的真实行。尚不能从受干扰WAV获得对应1611输入，所以本轮不开展原始音频干扰。

该300维编码器为双向对话模型：修改一条1611输入会影响同Dialogue多个话语的上下文表征。未来即使补齐前端，也要在实验协议中显式区分“完整对话重编码后仅替换目标300维行”与“替换整个对话表征”，才能判断是否满足此前只改变目标话语特征的约束；不能继续假定300维为各话语独立提取结果。

本轮未训练、未访问Test标签、未运行CMERC或音频扰动。完成后停止于**末端编码器部分恢复且小规模真实行parity通过，原WAV完整链仍阻断**。
