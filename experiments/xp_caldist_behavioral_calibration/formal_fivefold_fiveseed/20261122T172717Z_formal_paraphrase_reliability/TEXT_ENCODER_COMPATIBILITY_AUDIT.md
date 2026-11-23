# 文本编码器兼容性审计

审计日期：2026-11-22。结论：**当前 Protocol B 正式缓存依赖旧 CMERC/XP04 四层 RoBERTa 特征，但实际生产这组特征的 encoder 尚未被确认恢复。不能把现有 Plan B encoder 或未通过兼容门的蒸馏学生直接接到这些正式模型上，将预测差解释为原始文本改写响应。**

本轮只读代码、工件 JSON sidecar、来源清单和已有诊断摘要。没有反序列化任何 pkl、role `.pt`、特征数组或模型，没有读取原始话语、Test 标签或 Protocol B Evaluation 输入，没有提取、推理、训练、网络访问或 Git 操作。其他人的权重、划分及暴露审计不由本文替代。

## 1. 正式流程实际接收什么文本表示

本次核查的既有正式根目录为 `formal_fivefold_fiveseed/20261109T174024Z/`。新审计目录尚未产生新的正式文本编码产物，不能把创建审计目录当作生产链已经替换。

实际代码链为：

1. [`build_role_artifact.py`](../20261109T174024Z/scripts/build_role_artifact.py) 的 `build_artifact()` 打开基础特征包和 RoBERTa 特征包；基础包 `source[:6]` 中的第 3 槽被 `_` 接住，正式文本取自独立 RoBERTa pkl。
2. 文本选取表达式明确为 `roberta[2:6]`（IEMOCAP）或 `roberta[3:7]`（MELD）。每个 Dialogue 的每一路数组只做 `np.asarray`、转换为 `torch.float32`、`clone()`，保存到 `record['text']`；这里不运行 tokenizer、RoBERTa、LN、z-score 或可学习转换器。
3. [`cmerc_runner.py`](../20261109T174024Z/scripts/cmerc_runner.py) 的 `validate_artifact()` / `role_loader()` 通过现有 teacher helper 接收这些角色缓存。`forward_closures()` 直接将四路 `batch['text']` 送入 CMERC。重新训练下游 CMERC 本身不会产生或恢复一个原始文本 encoder。
4. [`teacher_runner.py`](../../curriculum_teacher_pilot/20261109T154754Z/scripts/teacher_runner.py) 的形状检查要求文本四路各为 `[N,1024]`，`build_model()` 使用 `norm='BN'`、`causal=False`。角色缓存中的实际表示是四个 `[N,1024]` 张量组成的列表；概念上堆叠为 `[N,4,1024]`，并非送入一个 4096 维拼接向量。

| 数据集 | 正式基础包 | 正式文本包 | 文本零基槽位 | 路序 / 每路宽度 | 原正式音频 / 视觉宽度 |
|---|---|---|---|---|---|
| IEMOCAP | `IEMOCAP_features.pkl` | `iemocap_features_roberta.pkl` | 2、3、4、5 | T1–T4，对应参考实现 hidden `[-1,-2,-3,-4]`，各 1024 | 1582 / 342 |
| MELD | `MELD_features_raw1.pkl` | `meld_features_roberta.pkl` | 3、4、5、6 | 同上，各 1024 | 300 / 342 |

槽位来自实际 builder；hidden 对应关系来自此前已经归档的 COSMIC 提取源码核查，并非本轮重新提取测得。

下游实际源文件是 `model_cm.py`（本地记录），不是仅凭名字推断的某个 RoBERTa 模型。该文件 `Model.forward()` 约第 751–772 行先解包 `[r1,r2,r3,r4]`，按配置分别调用四个 1024 维 BatchNorm，再执行 `U=(r1+r2+r3+r4)/4`。**这段下游 BN 与层平均不是 pkl 的生成器，也不能证明新 encoder 的坐标空间与旧缓存兼容。**

## 2. 来源哈希将正式缓存连到旧官方 pkl

实际非 Evaluation sidecar 的 `provenance.source_data_sha256` 记录如下：

| 文件 | 已有来源记录 SHA-256 |
|---|---|
| `iemocap_features_roberta.pkl` | `4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06` |
| `meld_features_roberta.pkl` | `58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951` |
| `IEMOCAP_features.pkl` | `ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3` |
| `MELD_features_raw1.pkl` | `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0` |

仓库入口 `CMERC/data/{iemocap_features_roberta.pkl,meld_features_roberta.pkl}` 是指向 `/data2/yb/paper/05_reproduction/external_repos/CMERC/data/` 相应文件的符号链接；上述解析后路径也出现在 sidecar 中。本轮只查看链接元数据和已有哈希，未读取这些 pkl。

具体例证：

- IEMOCAP fold2 Training sidecar（本地记录）：工件 SHA `04ee64ac55b9de090f36cabc7ca18ae90208a4ccefb0dfc439ef442b71fda662`，记录的文本源为上表 IEMOCAP RoBERTa SHA。
- MELD fold2 Training sidecar（本地记录）：工件 SHA `1d5974c187a7746d927f2e205d902bb99006fd1698a8050eb557fa2b90d6cf19`，记录的文本源为上表 MELD RoBERTa SHA。
- 扫描既有 `runs/shared/` 下 **32 个非 Evaluation 工件 sidecar**，所有对应文本包哈希均只有上表那一个取值。它们覆盖 fold2–5、两数据集的 teacher Training、student Training、Development、Calibration；teacher Training 的 `role` 也写作 `training`，所以不能把聚合的 8 个 training 记录误解为 8 个 fold。
- 其中 24 个 sidecar 记录 builder SHA `0625c3b255edd7aa605796f080bc29d18ae9b975b57df8b7a5f2928e08741cb1`，与当前 `build_role_artifact.py` 字节哈希一致；8 个 teacher Training sidecar 记录其独立 builder SHA `c8ca8c15040433124ae69ac88036cadc41d8261f6571344cf3264f11698a1bf2`，对应文本源哈希仍完全相同。
- [共享工件核验摘要](../20261109T174024Z/results/shared_role_artifact_verification.json) 记录 `artifact_count=32`、`errors=[]`、`evaluation_artifact_built=false`。本文没有打开任何工件张量，也不据此补充未检查的 fold1/权重/暴露结论。

两份文本源 SHA 与此前官方来源及文本恢复审计中的目标 SHA 精确相同。因此这是旧文本特征生产链缺口在正式缓存上的直接延续，而不是仅由“都叫 RoBERTa”推断。

## 3. 上游来源恢复与数值验证必须分开

### 3.1 原始 encoder 恢复：阻断，数值未执行

[20261117 文本恢复决定](../../xp04_random_feature_behavior_calibration/raw_data_pilot/20261117T053929Z/FINAL_TEXT_RECOVERY_DECISION.md)、[来源报告](../../xp04_random_feature_behavior_calibration/raw_data_pilot/20261117T053929Z/ROBERTA_SOURCE_RECOVERY.md) 和 [机器状态](../../xp04_random_feature_behavior_calibration/raw_data_pilot/20261117T053929Z/provenance/text_parity_status.json) 一致：两数据集均 `actual_cmerc_encoder_found=false`，四层兼容状态为 null / `NOT_RUN_MISSING_CONFIRMED_ENCODER`，概率状态为 `NOT_RUN`。

此前固定的 XP04 真正 Validation 目标是 IEMOCAP Seed1475 的 527 条和 MELD Seed67137 的 1109 条；成功重新提取数均为 0。这里引用的是已有报告，不是本轮访问这些输入。**未执行不是数值失败，更不能写成零误差或已通过。**

COSMIC 参考脚本可定位数据集微调后的 `checkpoint_best.pt`、BOS pooling、最后四层顺序和 BPE 配置；找到参考算法不等于找到 CMERC 实际使用的微调权重、tokenizer/dictionary 关联及完整生产记录。论文也未明确支持将未知任务 encoder 当作公开未微调模型。特征包文件大小/CRC 的关联不能证明权重身份。

### 3.2 功能重建学生：真实做过，四层全部数值失败

[20261118 蒸馏决定](../../xp04_random_feature_behavior_calibration/raw_data_pilot/20261118T125439Z/FINAL_DISTILLATION_DECISION.md) 以及两份已经保存的 `results/{IEMOCAP,MELD}/feature_parity.json` 将教师目标明确绑定到第 2 节同一组 RoBERTa pkl SHA。

| 已有蒸馏验证 | 比较形状 | 四层 MAE 范围 | 四层 RMSE 范围 | 已有机器结论 |
|---|---|---|---|---|
| IEMOCAP | `[527,4,1024]` | 0.067008–0.219918 | 0.087924–0.310843 | `feature_gate_passed=false`，0/4 层通过 |
| MELD | `[1109,4,1024]` | 0.069392–0.172344 | 0.097085–0.240749 | 同上，0/4 层通过 |

原冻结必要条件包括 MAE≤`1e-5`、RMSE≤`1e-4`、逐元素 allclose 以及余弦/相关检查。已有数值与该门槛差距很大，即使部分层平均余弦较高，仍不能视为直接可替换。两数据集概率状态均为 `NOT_RUN_FEATURE_GATE_FAILED`；没有概率误差或类别一致率可供宣称。

其后 [Train 内 128 条延长诊断](../../xp04_random_feature_behavior_calibration/raw_data_pilot/20261118T125439Z/diagnostics/20261118T171911Z/20261120T090108Z/FINAL_EXTENSION_DECISION.md) 也未达必要精度：IEMOCAP 最终逐层 RMSE 0.019708–0.050839，MELD 0.011950–0.027528，均远高于 `1e-4`。它不是第二次正式 Validation 通过记录，更没有恢复作者 encoder。

### 3.3 框架一致性通过回答的是另一个问题

[Fairseq/HF 诊断](../../xp04_random_feature_behavior_calibration/raw_data_pilot/20261118T125439Z/diagnostics/20261118T171911Z/FRAMEWORK_PARITY.md) 在**同一实际公开权重、同一 Token ID、FP32/eval**条件下通过：388 个功能性 backbone 张量逐值相同，八个逐层比较全部 allclose，最大逐层 MAE 约 `5.09e-7`。

这说明测试中单纯框架实现差异不足以解释蒸馏的大残差。它不证明公开初始化就是旧教师的实际权重，也不证明对原始字符串重新分词后等价；无法据此填补旧任务 checkpoint 身份和生产日志的缺口。

## 4. Plan B 是新特征基线，不是旧缓存兼容证明

[Plan B 文本报告](../../plan_b_raw_input_pilot/20261120T143131Z/reports/TEXT_FEATURE_EXTRACTION_REPORT.md) 与 [接口合同](../../plan_b_raw_input_pilot/20261120T143131Z/configs/TEXT_ENCODER_SCHEMA.md) 明确：从公开 RoBERTa Large 重新初始化，每数据集独立使用 Train Fit 标签微调，由内部 Dialogue dev 选择，输出四层 BOS `[N,4,1024]`。这条生产链可重载、可重新编码改写文本，且没有声称恢复旧 CMERC encoder。

| 项目 | 正式旧缓存路线 | Plan B `20261120T143131Z` 路线 |
|---|---|---|
| 文本表示来源 | 旧官方 RoBERTa pkl，实际任务 encoder 未确认 | 自主监督微调 encoder，所选权重有锁 |
| 文本接口 | 四个 `[N,1024]` 张量 | `[N,4,1024]`，可拆成相同形状接口 |
| IEMOCAP / MELD 音频 | 1582 / 300 | 新冻结 wav2vec2，各 768 |
| 下游 | 已有 Protocol B teacher、difficulty、CMERC | 新 M³Net、新 difficulty、新 CMERC |
| 旧文本逐层 Clean Parity | 尚无合格生产器；蒸馏候选失败 | 未声称或验证对旧 pkl 的兼容 |

Plan B 已有 [baseline 报告](../../plan_b_raw_input_pilot/20261120T143131Z/BASELINE_REPORT.md) 是新输入空间下单 Seed 模型的真实结果，不能当成既有正式五折五 Seed模型的输入兼容证据。其原始文本干扰 clean control 检验的是“同一 Plan B encoder 与其自身 clean cache/自身 CMERC”的重复性，不是“Plan B encoder 与旧官方 pkl”的一致性。

所选新 encoder 的已有 lock 记录为：IEMOCAP `4b1227dde6aab184700890fe4d063469a7213e977c516699f0e34d103e6f24fe`；MELD `306402e9a6787ce7fccdfd935d08c13e5a01dc5b6f04cf94a5a22a685a0e1a2f`。这些是有独立训练来源的新权重；不能将权重文件 SHA 与 pkl 文件 SHA 作跨对象比较来推断数值兼容或不兼容。

同形状文本可以被旧模型前向代码接受，但“能运行”不是坐标一致；若整套 Plan B 缓存一起复制，音频 768 维还直接不符合正式旧模型的 1582/300 维输入合同。新模型方案需单独定义全套特征、teacher/difficulty/student 及各 fold 的编码器边界，不能悄悄复用旧正式权重或沿用原结果身份。

## 5. 对当前原始改写研究的判定

正式旧模型学到的是旧特征函数 `E_old` 下的响应。用另一个未验证的 `E_new` 编码改写，比较 `f(E_new(x'))` 与 `f(E_old(x))`，变化中同时混入了 encoder 更换和文本变化；它不能解释为冻结原系统的纯改写响应。即便两边都换成 `E_new`，也不是原 cache 对应的原始预测对象，且仍缺其在旧模型输入空间中有效的证据。

因此当前应将**旧正式模型上的原始文本干扰及其行为可靠性数据生成**标为 `BLOCKED_TEXT_ENCODER_COMPATIBILITY`。这不否定既有模型使用原缓存进行 clean 分类的历史结果，也不由本审计推断所有未来 encoder 恢复或蒸馏方法均不可能。

可解除这一特定阻断的证据是：确认原 encoder 与完整预处理/tokenizer 来源，或获得明确标为重建方法且通过冻结逐层兼容门的替代器；随后在预先锁定的非 Evaluation clean 样本上完成逐层特征及相应正式 checkpoint 概率一致性验证。不能先用 Evaluation 结果反复选择方案，也不能把缺失行为指标填成零、把高余弦当作通过、用 BN/均值处理掩盖逐层差异。

若选择另立新输入空间的正式基线，应明确它是另一项经研究者审核的方案及计算工作，不是当前旧特征模型的 encoder 修复。本轮不启动该方案，不扩大训练、提取或实验。

## 6. 本轮只读证据指纹

以下 SHA 为本轮读取代码/摘要文件后重新计算；pkl、工件 `.pt` 与权重 SHA 均引用上述既有来源记录，未重新读取其载荷。

| 代码或诊断记录 | SHA-256 |
|---|---|
| `20261109T174024Z/scripts/build_role_artifact.py` | `0625c3b255edd7aa605796f080bc29d18ae9b975b57df8b7a5f2928e08741cb1` |
| `20261109T174024Z/scripts/cmerc_runner.py` | `bc28a815dc2df3d085dd579e99b3bc6e66f009187f7b01d2e2296a66779ed093` |
| teacher helper `20261109T154754Z/scripts/teacher_runner.py` | `fb92aa8cb3572d45ba4e959a17e72f91f4879588c3a729efab48c14eed8b3330` |
| 实际 `model_cm.py` | `94acf11e2e9edd1bcd98049eb12d85b3604f8d71d46f64f0c3b8b9837fa12625` |
| 32 个共享工件核验摘要 | `6677d691ce5faf8bfeb6b78be3cb85f00d93b1e0c3b726de701e2df99926c333` |
| 20261117 `provenance/text_parity_status.json` | `69428b4b79901adb4f58a3db620cb7f74020b77725f51d59e56093d5199ffbc3` |
| 蒸馏 IEMOCAP `results/IEMOCAP/feature_parity.json` | `579ab32c85e25368bb502f144a98d41e2dea9b93ddcaf92e1de3a59fce74cef0` |
| 蒸馏 MELD `results/MELD/feature_parity.json` | `6dd1fda81f881d817d66e15925e62ea4cc87ac7389c0923bd5f37d7e10ba4983` |
| Fairseq/HF `FRAMEWORK_PARITY.md` | `22b700ed3bfff39f4b7ba2865f344c130cd661f87f92cb839980369ff158741f` |
| 20261120 `FINAL_EXTENSION_DECISION.md` | `f09f393a7bb0905960e6671e95e7b85f5180844eccfde6ca49782204916a3850` |
| Plan B `TEXT_FEATURE_EXTRACTION_REPORT.md` | `3c71107c70246227284cf2cd1d7af1924b65169d499c164bdd115410cc61866e` |
| Plan B `BASELINE_REPORT.md` | `e99df0d4f29b0bf70888ebb02acb1993e8c2dfcfa37f3c121fc3ad3ddeca0100` |
