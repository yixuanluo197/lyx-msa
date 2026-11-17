# MELD 6373→1611 音频转换器反向重建

本轮结论为 **C：未建立通过独立 Validation 的 XP04 兼容转换器**。在真实有效 Train 中学习的逐列仿射候选能近似部分官方坐标，但1611维和原BiGRU的300维均未通过冻结的特征门。没有恢复作者原SVM/PCA或原标准化器，没有执行冻结CMERC，也不进入音频干扰。

## 1. 来源与重建方法的边界

沿用已核验的官方 `audio_embeddings_feature_selection_emotion.pkl`，SHA-256 `0cda206983988ab61de5b30ec32b8ac652e133f84cda4d3d8105a2db8fea363e`。它给出1611维目标，是作者变换的输出，不是6373列索引、SVM、PCA或scaler。原HDF5音频编码器为BiGRU150×2，权重SHA-256 `2f5a006d853ea6b2e2a7b40750005963e6dd59f16b1072c468e702c84632b2da`；XP04基础特征SHA-256为 `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0`。

历史官方资料支持IS13-ComParE配置家族，论文SVM L2与作者早期PCA答复的冲突仍在。本轮资产核查没有找回原6373数组、选择器、投影、精确索引、原OpenSMILE二进制或标准化统计量；详见 `provenance/local_original_converter_asset_search.json` 与前轮 `20261117T014409Z/MELD_AUDIO_SELECTOR_RECOVERY.md`。本轮全部新提取和拟合必须标为**候选前端与Train功能重建**，不能升级为作者原流程恢复。

## 2. 预先固定的8个声学候选

使用既存官方OpenSMILE v3.0.1的 `IS13_ComParE.conf` 及core/shared includes。保持原 `cWaveSource`、LLD及functionals，仅将输出sink改为external；没有用ComParE_2016替代IS13。6373个有序字段全部唯一，主配置、wrapper diff、includes和字段留存在 `configs/meld/`。

| 项目 | 固定候选或实际实现 |
|---|---|
| 引擎 | Python opensmile 2.6.0；底层库自报OpenSMILE 3.0.1 |
| 底层库SHA-256 | `7d708bd3e02ec29d1fdc94e8b60595a65052d6f955b96bb7c88b181d65d81231` |
| 音轨 | 已有官方逐话语MP4，ffmpeg映射首条音轨，去视频；不重新裁切对话 |
| 下混 | 按每个文件实际声道数等权平均，或ffmpeg标准mono下混 |
| 采样率 | 原生48kHz，或ffmpeg转16kHz |
| waveform处理 | none，或每话语减均值/除总体标准差的z-score |
| 强度阈值 | 固定none；历史阈值未知，不构造原参数 |
| 数值接口 | ffmpeg float32解码→scipy IEEE float32 WAV→官方cWaveSource；不截幅 |
| 计算 | 至多6个提取worker；OMP/BLAS线程1；CPU提取 |

两种下混×两种采样率×两种waveform处理共8条件。ffmpeg版本、二进制SHA、逐行完整解码命令、原媒体SHA、采样数、处理前后均值/标准差/峰值和特征SHA均有记录。采样率转换采用本地ffmpeg的libswresample默认行为；这也是公开的候选选择，不声称复原原解码器。

原先仅凭历史24条dev音轨误设6声道的手工预检在Train dia0/utt0失败，未产生特征；失败保留为 `meld_extraction_initial_failed_preflight.json`。正式实现改为动态实际声道数，与冻结的equal_all_channels协议一致。完整可用Train中1117条为2声道、8871条为6声道；可用Validation中203条为2声道、905条为6声道，全部48kHz。

Python opensmile的 `process_signal` 会乘32768转int16，直接用于单位方差waveform会溢出。本轮统一用float WAV接口。合成幅度控制包含±7.7，cWaveSource逐样点精确保留，最大误差0；没有截幅或int16 wrap。3条Train的额外PCM16接口控制发现量化和接口数值会影响functionals，其差异完整保留于 `meld_extraction_float_interface.json`，没有据目标选择接口。

## 3. Train配对、前端选择与正式拟合

只使用Seed67137 effective Train9989行。按照真实Dialogue ID、原Utterance ID和原行序对齐；不按连续utterance编号推断数组位置。`manifest_train.csv`保留9989行，其中dia125/utt3没有可用媒体（工作目录缺文件，原包成员及同划分备用文件均损坏无法解码）：完整行1165候选为NaN、valid=false，不能拿其他split或官方目标补齐。因此正式配对拟合为 **9988条、1038个Dialogue**，不是9989条全覆盖。备用文件的容器QC失败亦保留。

前端probe不读取目标进行抽样：排序Train Dialogue ID后，seed67137随机排列，每组随机取1条，再按完整Train行号排序，固定256条/256个不同Dialogue。3条smoke×8条件均6373维有限；256×8正式probe均有限、零失败。三折GroupKFold下，8前端×4个固定逐行输入变换×2个列映射方法共64个Train前端比较。前端锁定为 **`ffmpegmono_16k_none` + stateless `none`**，其probe Dialogue等权NRMSE为0.4901501。之后仅扩展这个前端的完整Train与最终Validation，其他7条件原probe不删除。

所有列匹配、仿射系数、均值/标准差、投影和回归均在各折fit Train中确定。选择指标为Dialogue等权、fit-fold目标标准差归一化的RMSE，三折共用分组。CV分数用于选择，不称为独立泛化估计。最终模型以9988个成功Train配对重新拟合。

| 正式Train方法 | 参数 | Dialogue等权OOF NRMSE |
|---|---|---:|
| best_column_exact | 全列哈希/精确核查优先，未匹配列用相关列作未校正候选 | 7697522.653152 |
| **best_column_affine** | 每目标选最大绝对Train Pearson输入列，再OLS缩放/截距 | **0.483129** |
| PCR | rank16 / 64 / 128 | 0.888256 / 0.841254 / 0.821786 |
| 投影Ridge | rank256，alpha .01 / 1 / 100 | 0.808494 / 0.808490 / 0.808031 |
| 全秩多输出Ridge | alpha .01 / 1 / 100 | 5.483315 / 3.028024 / 1.243397 |

最终锁定 `best_column_affine`。该方法允许不同目标复用输入列，按fit Train的绝对Pearson关联选列并拟合斜率/截距；不是SVM选择器，也不是恢复的原列顺序或原PCA。最终前端的三个fit fold中精确列匹配计数均为0；`best_column_exact`名称中的fallback只是保留未校正相关列，不能将其巨大误差解释为“精确映射已找到”。不同维度的6373→1611不使用同维speaker-diagonal方法。

锁定方法的Train OOF为 `(9988,1611)`：MAE0.04098879、RMSE0.09229378、均值cosine0.98288993、相关中位数0.91408661、相关p05为0.52702531；3个全零目标行cosine未定义并保留。Train OOF门已失败。CV均值接近目标均值不能抵消逐样本误差。

## 4. 协议、执行异常与可复现性

v2在本轮任何Validation提取/目标打开前增加全秩Ridge三档alpha，防止仅低秩rank256结果被误写成对全部线性映射的否定。此菜单扩充发生在初轮Train CV后，**不是最初完全预注册**；原协议、原结果和变更理由保留于 `protocol_revisions.json`。新增候选仍使用相同Train组折、折内scaler与原门槛，未改变最终Validation结果。

首次probe启动瞬间未记录脚本SHA，此缺口如实保留。完整Train前冻结了新脚本SHA `0bbca3adeb5acc691d4dd6517cb6f1b4a8703ffa439f0bb94f0cd79afada4e67`。随后在独立 `local_only/MELD/probe_recheck/` 预先冻结代码副本及配置，再重核完全相同的256Train×8条件：indices一致，8个npy文件SHA和逐bit内容均与原probe相同，最大误差0，配置未变。原probe没有被覆盖，不伪造旧启动SHA，重核也没有扩大数据或候选。

前端CV解释器退出码120亦保留：64个case、OOF、选择JSON已全部完成，Tee在日志关闭后最终flush触发退出异常。`MELD_frontend_shutdown_failure.json`记录原源代码、exit120、数值阶段已完成和只修日志生命周期的事实；不能将该进程写成exit0。后续正式Train提取、独立重核及单次Validation提取正常完成。

历史审计已看过Validation汇总、emotion/sentiment共享列诊断和24条官方1611→300局部验证。因此这是**本轮协议后未反馈调参的最终验证**，不是从未接触过的数据集。模型选择和参数拟合仍只用Train；锁定文件、协议、scaler、媒体manifest、候选配置/字段和Train特征哈希在新Validation提取前独立核对，结束后均未改变。`MELD_validation_evaluation_marker.json`确认最终评估attempt1完成，无失败后重调。

## 5. 最终结果与可交付范围

最终Validation1611维MAE为0.06832477、RMSE0.11538499；完整可计算Dialogue经原BiGRU后的真实300维MAE为0.02218877、RMSE0.06852870。两级特征门均失败，具体误差、分布、缺失上下文及padding差异见 [MELD_CLEAN_PARITY_REPORT.md](MELD_CLEAN_PARITY_REPORT.md)。高cosine不等于XP04数值兼容，不能凭300维压缩后误差变小宣告B。

转换器 `local_only/MELD/final/converter.pkl` 的SHA-256为 `51af500ea1c62860eb0b1e7191cdd6b13896d2be4c5902ab9e6b19b166ea3bb6`。代码、配置、完整候选特征、Train CV/OOF、失败mask、拟合参数、原权重和逐维CSV全部保留本地；敏感数据不上传。可执行性已实现，但兼容性未实现。最终分类C；不进行第二次Validation、参数调整、CMERC概率试跑或原始音频干扰。下一轮若要继续，须另立新协议并补前端来源或独立验证资料，不能复用本轮Validation反馈仍称其独立。

关键证据：`provenance/MELD_frontend_cv.json`、`MELD_model_selection_final.json`、`MELD_model_lock.json`、`meld_extraction_probe_recheck.json`、`meld_extraction_train_verification.json`、`meld_extraction_validation_verification.json`、`MELD_clean_feature_metrics.json`。本轮未用Test特征/标签，未训练CMERC、未修改原资产或Git提交/推送。
