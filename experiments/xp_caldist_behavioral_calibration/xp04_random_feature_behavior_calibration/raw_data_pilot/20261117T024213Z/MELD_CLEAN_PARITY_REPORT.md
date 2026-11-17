# MELD 最终单次Clean Feature Parity

**结论C：1611维和真实对话上下文300维的兼容门均失败。** 本轮已经执行候选WAV前端→Train拟合转换器→原BiGRU的端到端Clean数值检验；这不是作者原前端恢复成功，也不满足功能等价转换器B。冻结CMERC概率验证未执行，不能把“未测量”写成概率误差0。

本报告只读取已完成的 `provenance/MELD_clean_feature_metrics.json`、锁定记录及提取验证，不重复提取、拟合或评估Validation。最终评估marker为attempt1，2026-11-17 03:17:21至03:18:16 UTC完成；模型、前端及参数在这之前锁定。

## 1. 输入、边界与完整覆盖

锁定配置为 `ffmpegmono_16k_none`、逐行输入变换none、Train拟合 `best_column_affine`。候选6373维由官方v3.0.1 IS13配置core、原cWaveSource及external sink产生，未知强度阈值固定none。方法和DSP细节见 [MELD_FEATURE_RECONSTRUCTION.md](MELD_FEATURE_RECONSTRUCTION.md)。

Train原清单9989行，因dia125/utt3没有可用媒体（原包成员及同划分备用文件均损坏无法解码），仅9988真实配对拟合；所有转换器/标准化/模型选择只用这9988条及Dialogue分组Train CV。Validation固定1109行、114个Dialogue，候选提取1108行有限，raw dev Dialogue110/utt7缺媒体，完整行1084保留NaN及false mask。没有从Train/Test或官方目标补齐该行。

Validation提取前，独立核对model lock、converter、Train尺度、protocol、manifest、Train特征、IS13配置、字段及DSP脚本哈希并确认无已有Validation输出；结束后所有锁定输入哈希不变。提取只执行一次，56.96秒，stderr空。候选矩阵SHA-256 `59cfdaea810e6b0f316c17cc922068e017d5b9af6ec985afad2277f86c163015`；valid mask SHA-256 `b4d3b8aa72b37c13fb897149c006b65ad2c7e1beddc93d08fa73bf801efdad24`。

历史审计接触过Validation摘要和24条末端GRU检验。本轮最终Validation在Train选型/锁定后才打开且未回流，故“独立”专指本轮模型选择和拟合边界，不声称研究者从未见过这套Validation。

## 2. 6373→重建1611维

目标是官方选择后1611维。保存完整 `(1109,1611)` 矩阵，其中1108行有限；错误指标在1,784,988个联合有限元素上计算，1行/1611个非有限预测元素单独计入覆盖失败。

| 指标 | 最终Validation结果 |
|---|---:|
| 最大绝对误差 | 2.31946768 |
| MAE | 0.06832477 |
| RMSE | 0.11538499 |
| 行cosine均值 / 最小值 | 0.97412033 / 0.79782264 |
| 逐维相关min / p05 / median / p95 / max | 0.06301753 / 0.52761816 / 0.90732178 / 0.99412830 / 0.99964266 |
| 有定义相关维度 | 1611 / 1611 |
| 全局Train尺度NRMSE | 1.00677900 |
| 逐维Train尺度NRMSE median / p95 | 0.57659992 / 1.82507788 |
| Dialogue等权MAE / NRMSE | 0.06793652 / 1.00093301 |
| 逐维MAE median / p95 / max | 0.05319456 / 0.16837739 / 0.39497659 |

分布指标同样只使用1108条整行有限配对，按列计算，未用Validation拟合任何缩放参数：

| 分布差异 | median | p95 | max |
|---|---:|---:|---:|
| 列均值绝对差 | 0.01700321 | 0.12333031 | 0.39472952 |
| 列总体标准差绝对差 | 0.03001036 | 0.10599047 | 0.31279268 |
| 列一阶Wasserstein距离 | 0.03570286 | 0.13602563 | 0.39497659 |

Train尺度来自9988成功Train目标的逐列总体标准差，floor1e-8；全局NRMSE为联合有限元素上 `sqrt(mean((prediction-target)^2 / train_scale^2))`。Dialogue等权NRMSE先在每个可评估Dialogue内做同一归一化RMSE，再等权平均，防止长Dialogue支配。114个Dialogue仍记录完整覆盖状态；缺失行不会被隐藏或当0。

## 3. 原BiGRU与对话上下文300维

使用已恢复官方HDF5的 `utter` 编码层：双向GRU150×2拼接300维。TensorFlow/Keras均2.14.0，CPU，`compile=False`，`training=False`。forward/backward的kernel、recurrent_kernel、bias共6个权重张量与原HDF5逐元素精确相等，未训练或修改权重。

每个完整Dialogue按真实utterance ID排序及有效长度组装，右侧补零到33步，使用原Masking(0)和双向上下文，之后只取真实话语行。没有将300维当作独立逐话语OpenSMILE输出。一个缺失输入会影响双向GRU整个Dialogue，故raw dev110（XP04 Dialogue1149）的10条真实话语全部保留NaN：既不以官方1611补洞，也不编码剩余9条伪称完整上下文。

因此113个完整Dialogue完成GRU，1109行中1099行300维可评估、10行/3000元素因上下文缺失非有限。完整特征门的 `full_dialogue_bigru_context=false` 表示114个Dialogue未全部覆盖，不表示113个已完成Dialogue采用了无上下文近似。

| 指标 | 真实话语300维结果 |
|---|---:|
| 完整形状 / 有限行 | `(1109,300)` / 1099 |
| 最大绝对误差 | 1.50055301 |
| MAE / RMSE | 0.02218877 / 0.06852870 |
| 行cosine均值 / 最小值 | 0.99427403 / 0.95339844 |
| 有定义相关维度 | 177 / 300 |
| 非常数目标维度相关min / p05 / median / p95 / max | 0.19049560 / 0.65388053 / 0.96683741 / 0.99650866 / 0.99867059 |
| 全局Train尺度NRMSE | 0.44332567 |
| 逐维Train尺度NRMSE median / p95 / max | 0.02243484 / 0.58717843 / 6.23721699 |
| Dialogue等权MAE / NRMSE | 0.02248074 / 0.26940730 |
| 逐维MAE median / p95 / max | 0.00007810 / 0.13013776 / 0.29823468 |

300维的目标有123个常数维，其相关系数未定义，保留为未定义而不填1或0。其余177个目标非常数维相关全部有定义，表中相关分位数仅在这些必要维度计算。固定门的“必要相关均有定义”子项通过，但p05远低于阈值。大量饱和或近常数GRU坐标使整体cosine较高、逐维误差中位数较低，不能据此认定数值兼容。

| 300维分布差异 | median | p95 | max |
|---|---:|---:|---:|
| 列均值绝对差 | 0.00002790 | 0.09110661 | 0.29685808 |
| 列总体标准差绝对差 | 0.00014382 | 0.06868283 | 0.14970839 |
| 列一阶Wasserstein距离 | 0.00005202 | 0.10879524 | 0.29686166 |

300维Train尺度只来自XP04有效Train9989条的原300维特征；没有取Validation目标拟合尺度。尺度及常数列记录在 `provenance/meld_train_scale300.json`。

## 4. 33步padding独立报告

在113个可计算Dialogue中，真实行之外另有2630个padding行。本地Keras输出全部为0；官方历史 `audio_emotion.pkl` 相应padding存在非零值。它与先前官方输入1611→300局部检验发现的padding语义差异一致，不能混入真实话语误差或归因于全部前端重建误差。

padding单独形状为 `(2630,300)`，最大绝对误差0.99999994、MAE0.23265383、RMSE0.46035681；本地2630行零向量导致cosine均未定义。300个padding维度相关亦均未定义，全部保留null。这不是2630条额外话语，也不是全114个Dialogue的padding parity；缺失上下文Dialogue的23个padding步没有进入此数值比较。真实长度、排序、33步右补零及Masking规则已保留，但不能声称整33步与原存档输出一致。

## 5. 冻结门槛与最终判断

| 冻结条件 | 阈值 | 1611维 | 真实300维 |
|---|---|---|---|
| 全覆盖、全有限 | 固定Validation全部行 | 失败：1行缺失 | 失败：10行上下文缺失 |
| 所有元素allclose | atol=rtol=1e-4 | 失败 | 失败 |
| MAE | ≤1e-5 | 失败 | 失败 |
| 全局Train尺度NRMSE | ≤1e-3 | 失败 | 失败 |
| 平均非零行cosine | ≥0.99999 | 失败 | 失败 |
| 最小非零行cosine | ≥0.9999 | 失败 | 失败 |
| 目标非常数维相关有定义 | 全部必要维度 | 通过 | 通过 |
| 必要维度相关p05 | ≥0.999 | 失败 | 失败 |

即使只看可评估行，数值误差仍显著超过冻结阈值，故失败不能仅归结为一条缺媒体。原选择器/缩放信息仍缺，当前IS13前端、音轨预处理或逐列仿射不足以产生XP04原特征。仅凭结果无法唯一定位某一个历史缺失参数，也不能证明所有未来转换器都不可行。

本轮保留分类 **C**。没有通过特征门，依协议不执行冻结XP04 Seed67137概率验证；Softmax误差、argmax一致率和置信度误差均为未测量。不得以本次Validation结果继续调整候选、再次拟合或第二次Validation后仍称独立验证；不得进入原始音频干扰。没有训练CMERC、访问Test、修改原始特征/Checkpoint或Git提交/推送。

完整逐维CSV及1611维矩阵位于 `local_only/MELD/final_validation/`，300维结果为 `local_only/MELD/rebuilt_validation_audio.npy`，源缺失mask为 `local_only/MELD/candidates/ffmpegmono_16k_none/validation_valid.npy`，只保留本地。来源与执行异常的完整披露，包括原probe启动SHA缺口及8条件逐bit重核、前端日志exit120、Train阶段v2全秩Ridge扩充，见配套重建报告及相应provenance；失败结果均保留。
