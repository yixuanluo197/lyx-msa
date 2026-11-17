# IEMOCAP 1582维音频转换器重建

结论为 **C：当前候选转换器重建失败**。本轮完成了三个官方配置候选的有效 Train 音频提取、Dialogue 分组模型选择，以及锁定方法后唯一一次真正 Validation 特征评价。`IS10_paraling + diagonal_affine` 明显优于其他已测候选，但未通过预先冻结的 Clean Feature Parity 门槛，不能供 XP04 兼容输入使用，也未进入 CMERC 概率验证。

**资料与来源边界。** 上轮已确认 XP04 基础特征与 MMGCN 官方文件一致，MMGCN 文献指向 IS10；M3NET 的 IS13 描述与所沿用预提取资产存在冲突。本轮复用了[上轮来源审计](../20261117T014409Z/IEMOCAP_IS10_RECOVERY.md)及[配置版本比较](../20261117T014409Z/OPENSMILE_CONFIG_VERSION_COMPARISON.md)。没有找到作者实际使用的原转换器、原外部 scaler、完整生产命令或字段 header。[本轮有界本地检索](provenance/local_original_converter_asset_search.json)也没有得到作者原对象；这不证明其在所有位置均不存在。本轮所有拟合结果均明确标为重建方法，不能称为恢复作者原标准化或原配置。

**固定数据边界。** Seed 1475 effective split 对应 Train **5283条、108个Dialogue**，Validation **527条、12个Dialogue**。Train 清单按 effective Train Dialogue 原序和每个 Dialogue 的 `videoIDs` 原序构建，没有将基础 pickle 的其他 Dialogue 纳入 Train。候选提取、列匹配、仿射参数、scaler、投影及正则化选择均只用有效 Train。完整 manifest 和目标数组留在本地，统计与哈希见[数据审计](provenance/data_manifest_audit.json)。本轮不索引 Test 特征或标签，不读取情绪标签用于转换器拟合。

**实际执行的前端。** 使用既有 `msa_vision` 环境：Python opensmile **2.6.0**，`libSMILEapi.so` 静态自报引擎 **3.0.1**。引擎 SHA-256 为 `7d708bd3e02ec29d1fdc94e8b60595a65052d6f955b96bb7c88b181d65d81231`。三个候选来自此前缓存的 audEERING 官方 `v3.0.1`，没有下载替代特征或安装新依赖。

| 候选 | 官方原配置 SHA-256 | 本轮外接 I/O wrapper SHA-256 |
|---|---|---|
| IS10_paraling | `797eb7c3b912540cb5f8f8d5963151a706a60c00b50de5706a77f1976b96e0f5` | `cf96dcd56797dc05813cf27dbc1c9b4a7a9565971e40d2fad9871676d59218c8` |
| IS10_paraling_compat | `75e28f7ad5de545fa68b9e392aa3def37f10aa8e8c4e736883ab8a14dd107973` | `7fd33ff3ffe9ceb7d8806b5da35282b608e9e1aab6f97b5a7a9e5aa52e0167bb` |
| emobase2010 | `e8550e5b49f7ed26457bafddd1752cd978f0d04685d4f6fdb14257ef72fe4fe7` | `c816984f04c53d3ae4f28f2140e6f08dfeff3ed6f54de2bab256033274c14473` |

wrapper 只更换 source/sink 接口。IS10 非 compat 更换文件 include；compat 去除内嵌 `cWaveSource` 并改用 external source；emobase 去除 WAV/ARFF I/O，按原 reader 顺序 `func;funcNz;functOnsets` 拼接后输出。DSP、LLD、functionals 参数保留原样。完整配置树、递归 include、字段顺序和逐文件哈希见[配置记录](provenance/iemocap_extraction_configs.json)与[配置说明](configs/iemocap/README.md)。三套有序 header 各含1582个唯一字段，未假设同维度意味着同字段或同数值。

输入均为已发布的话语级 WAV，没有重新切段。逐条实测并强制要求16 kHz单声道；`scipy.io.wavfile.read` 解码，整数 PCM 按位宽转换成浮点幅度后以 float32 输入。无重采样、声道下混、强度门限、静音裁剪、每段增益或外部特征归一化。这里的16 kHz单声道是本轮实际候选输入事实，不能据此追认作者当年的前处理。

Train 首3条 smoke 三候选均通过，随后完整5283条均提取成功，输出各为 `(5283,1582)` float32，全部有限，无全零行或全零列，没有失败行被零值替代。完整提取用6个CPU进程、每进程OMP/BLAS单线程，耗时316.32秒。[Train独立检查](provenance/iemocap_extraction_train_verification.json)48项通过，包含每条媒体与manifest哈希匹配、逐行特征哈希、前3条smoke与完整输出一致。初始 compat wrapper 静态初始化失败和首次 `nohup` 启动未存活均有记录；修复后才执行音频 smoke，失败未冒充成功结果，见[环境与失败记录](provenance/iemocap_extraction_execution_environment.json)。

**Train 内部选择。** 使用同一组3折 `GroupKFold(Dialogue ID)`，每折36个 held-out Dialogue，拟合分别为72个Dialogue；held-out行数1756、1764、1763。每折独立拟合列匹配、仿射和标准化等参数，以 Dialogue 等权的、由该折 fit-Train 目标列标准差归一化的 RMSE 选择方法。CV分数用于选择，不能当作独立泛化估计。相对最佳分数1%内优先使用预设复杂度较低的方法。

固定前端菜单包括三配置，以及 `none`、逐行min-max、逐行z-score、逐行L2四种无拟合状态变换；简单方法为 identity、列匹配、逐维仿射、按真实说话人分组仿射、列匹配后仿射，共60个前端条件。说话人键使用Session+gender全局人员标识，未将对话内speaker槽位当作真实身份。三种 `none` 候选的逐列原值精确匹配在每折fit-Train均为 **0/1582列**；仅重排列不能解释XP04值。这里的精确列检查不是作者字段索引恢复。

| 前端候选的最佳简单方法 | 变换 | Train OOF Dialogue等权NRMSE | OOF MAE | OOF RMSE | OOF平均余弦 |
|---|---|---:|---:|---:|---:|
| IS10非compat + diagonal_affine | none | 0.1015833654 | 0.0114111190 | 0.1146297397 | 0.9946188557 |
| IS10 compat + best_column_affine | none | 0.3376941355 | 0.0818005593 | 0.3400528197 | 0.9474026343 |
| emobase2010 + best_column_affine | none | 0.3824634183 | 0.1119579782 | 0.3831861926 | 0.9283556506 |

IS10非compat的 identity OOF MAE为29.71812379、RMSE为199.71884305；逐维仿射大幅减小尺度误差。最终方法为 `y_j = a_j x_j + b_j`，每个输出只使用同序输入列，`a_j,b_j` 用Train OLS拟合。这支持“很多维度有近似仿射关系”的有限判断，不能证明恢复了原z-score统计量。其OOF最大误差62.78680786、最小话语余弦0.1098208891、逐维相关性p05为0.9672442212，已经显示残留异常和非精确对应。

随后在锁定的 `is10/none` 前端上比较PCR秩16/64/128、投影秩256的Ridge（alpha 0.01/1/100）。最好的投影Ridge Dialogue等权NRMSE为0.3921070187，仍差于逐维仿射。为覆盖用户要求的完整线性回归，在任何本轮新Validation音频提取或目标访问之前，协议v2明确加入完整秩多输出Ridge alpha 0.01/1/100；三者分数分别为 **0.1364604215、0.1330534107、0.1390632610**，全部更差，最终仍选择逐维仿射。v2不是最初完整预注册菜单；[协议修订记录](provenance/protocol_revisions.json)保留原因、时间和前版哈希，前端、折分与接受阈值未变。所有失败条件均保留于[完整最终选择记录](provenance/IEMOCAP_model_selection_final.json)、[方法CSV（仅本地保留）](LOCAL_ONLY.md)及[前端CSV（仅本地保留）](LOCAL_ONLY.md)。

**锁定与最终评价。** 当前协议SHA为 `02377d045ceeb34932eabbb227a72712707f8c7704be483206e13aaef644a9b2`。正式锁为[model lock](provenance/IEMOCAP_model_lock.json)，时间 `2026-11-17T03:12:06.885375+00:00`；final converter SHA为 `5afb90cc7f71467ad8a90844a07f133a10c017fd9b58610fbd50f2aade23ca1f`，并标为Train重建。527条Validation仅执行锁定的IS10候选一次，其他候选没有Validation输出；之后一次应用锁定转换器评价，未再调参。[详细Clean报告](IEMOCAP_CLEAN_PARITY_REPORT.md)记录全部门槛，MAE为0.01354729366、RMSE为0.12782490993、最大绝对误差17.91040999，最终失败。

历史审计已看过Validation特征统计和模态行为，因此该划分不能被描述成项目历史上从未见过的新数据。这里的独立性指自本轮协议与Train阶段起，无新Validation反馈参与拟合或选择，且v2扩展先于本轮新Validation访问。按现有记录，未通过的原因是特征数值误差，而非覆盖不足或非有限值。尚不能唯一归因于旧引擎、Jitter/Shimmer、原标准化或其他后处理；没有用新的Validation试配来猜测并删除异常。

当前归类 **C**，不是A（作者原转换器）或B（通过独立Validation门槛的功能等价转换器）。本轮没有加载Seed1475 checkpoint执行CMERC，没有产生概率一致性结果，没有干扰音频或重新训练CMERC。候选数据、拟合权重、原媒体和逐行记录均保留本地；停止后续兼容输入使用，等待研究者审核。
