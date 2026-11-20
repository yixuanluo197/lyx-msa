# Plan B 独立审查记录

当前为协议及实现审查，完整训练与Pilot的最终核验尚待实际产物；不能将本文件当作实验完成或效果通过证据。独立审查不启动训练，不修改其他负责人的实现，不新增依赖。

## 必须保留的研究与数据边界

- 两数据集分别从公开RoBERTa Large初始化，使用固定Train Fit监督微调；同一份按Dialogue固定的inner dev仅用于各阶段checkpoint选择。官方Validation只做最终评价和本次120条机制诊断，不参与任何拟合或选型。
- 冻结wav2vec2 base生成新音频，文本生成四层新特征；原官方视觉允许复用。旧预提文本/音频、旧学生或教师权重、旧difficulty不能成为新训练输入。共享旧pkl整体反序列化与实际索引旧T/A字段须分别披露。
- 新M3Net、新Train difficulty、新CMERC必须有完整来源依赖。CMERC的课程进度、五路信息移除的CC rank项、模态MSCL与上下文CSCL须保留，并检查非零项和有效梯度；不能退化为纯分类损失。
- MELD损坏或缺失媒体保留清单和固定处理策略，不能从其他划分借媒体、从旧音频特征补洞或删除不利样本。120条文本/音频必须用同一名单；排除不可处理媒体的样本池及数量须报告。
- 扰动时冻结特征提取器与CMERC，统一eval模式，保留同一Dialogue及实际话语顺序、Speaker、标签、其余模态与非目标话语；只替换目标话语的目标模态，重跑完整Dialogue。

## 已核对的正式行为定义

来源为 `xp04_random_feature_behavior_calibration/mainline_mu_delta/20261115T165025Z/MAINLINE_PROTOCOL.md` 以及其 `scripts/build_inputs.py` 的 `response()`。

原概率为p，原类别a=argmax(p)，原最大置信度C=max(p)。某一种干扰有K个版本，记p'_k、a'_k=argmax(p'_k)、C'_k=max(p'_k)。

- μ = (1/K)Σ_k 1[a'_k≠a] C'_k。
- δ = abs(C − (1/K)Σ_k C'_k)。
- flip_rate = (1/K)Σ_k 1[a'_k≠a]，单独保留，不能代替μ。
- 平均绝对confidence变化是mean(abs(C'_k−C))，不能代替δ；原类别概率p'_k[a]变化另列。
- JS(p,p') = 0.5 KL(p||m) + 0.5 KL(p'||m)，m=(p+p')/2，统一自然对数，零概率项按0处理。

中性追加、语义改写、pitch、timbre各自按本类型K汇总，不将不同类型或强度混作相同独立重复。未确认情绪标签保持的版本保留记录，但不进入标签保持主分析。

## 核心机制检查依据

`CMERC/experiments/xp_mainline_validation/20261016T050823Z/code/train_cm.py` 的 `train_or_eval_graph_model()` 会将当前主模型state复制到辅助模态、intra及inter模型，再将辅助模型设为eval。其CC采用真类log-prob被移除信息后上升的正部分，MSCL/CSCL使用原主模型hidden；不能误detach该hidden使对比项失去梯度。官方 `clloss.py` 对非有限损失有置零分支，必须披露其实际触发次数。

`CMERC/caldiff/cal_diff.py` 的 `compute_dialogue_difficulties()` 定义每话语五路移除真类log-prob正增量之和R_i。对话难度为 `(Σ_i R_i+s)/(length+s)`，其中s为该对话实际Speaker最大槽位索引加1。新difficulty不可用熵或CE平均静默替代，且不得引用旧难度文件。

## 统计与解释检查

120条若按正确/错误及confidence分层，须分层报告，整体指标不能当作完整Validation总体表现。Dialogue bootstrap应在同次重抽中对所有模态/条件共享cluster计数，并报告有效/不可计算重复数。控制原confidence的比较只在预定bin的correct/wrong共同支持区域进行，空bin和覆盖率不得隐藏。当前新特征/新训练模型与历史XP04随机特征扰动不同，历史对比只能作为背景，不能据此作严格的原始干扰优于随机噪声因果结论。

## 已发现并转交的问题

初版数据脚本曾要求MELD全部基础pkl原文等于raw CSV；历史审计已记录Train556/utterance6唯一原文冲突。根代理最终冻结策略为全部MELD采用原CSV文本，同时保留官方pkl Speaker槽位，明确保存该行的文本来源及历史Speaker不一致。该决定覆盖早期“保留pkl原文”的临时处理；当前dataset哈希及 `provenance/data_schema_verification.json` 是最终依据。首次准备失败及不一致均留档，不删除话语或静默改变Speaker。

IEMOCAP原训练代码硬编码的六个类别频率没有给出其统计划分，无法证明来自本轮TrainFit。根代理在任何正式下游训练前采纳独立下游修订 `configs/DOWNSTREAM_CLASS_WEIGHT_AMENDMENT.json`，仅用4747条Fit标签计数392/704/1015/750/661/1225、使用N/n_c，保留weightedNLL机制。独立验证已重算计数和权重一致；运行中文本master协议字节及launch哈希未改变。最终仍须在M3Net与CMERC实际有效配置/产物中核验该修订生效。

## 当前阶段核验

已实现独立 `scripts/verify_plan_b.py`，按data/features/models/pilot/final逐阶段读取证据，不执行模型前向、训练或网络请求，不反序列化历史原始pkl或模型参数。第一轮data阶段3895项检查通过；`complete=false`，两套encoder完成锁仍待运行结束，因此不构成完整交付。

验证器已加入真实新特征的shape/hash/缺失mask、选中encoder及训练配置来源、下游完成标记/模型文件哈希、新difficulty五路分量数值重算、保存Validation概率的指标重算、120名单及全部固定版本/失败分母、clean重提取阴性control、target-only与完整Dialogue预测工件、正式μ/δ/JS重算。统计部分从保存CSV独立恢复correct/error cluster bootstrap及精确Dialogue×confidence-bin固定等权配对；不只依赖分析脚本的成功标记。

分析负责人已落实显式token-identical检测、clean重提取控制、跨类型/模态共享全cohort Dialogue重抽，以及未确认音频标签保持的独立探索scope。验证器通过12项纯合成跨实现统计检查，包含空支持/缺失配对和2000次CI重算；没有读取真实Validation或据其调参。最终科学结论须等待实际结果，不能由以上工程检查推出。

最终完成状态、逐项证据和失败/待完成项将由本目录 `scripts/verify_plan_b.py` 与 `verification.json` 补充。

## 编码器与下游运行环境拆分复核

2026-11-20增加工程运行环境拆分；原TEXT_repro缺少PyG的首次下游导入失败已有独立记录。编码器仍固定TEXT_repro（Torch 2.2.0+cu118、Transformers 4.42.3），CMERC使用既有multimodalerc310（Torch 2.2.2+cu121、Transformers 4.45.1、PyG 2.4.0）。未新增依赖，未更换编码权重或算法。V2 producer保存CPU FP32目标向量，consumer只读取这些向量并运行冻结CMERC，不在下游环境重新实例化HF编码器。

只读复核确认producer先检查精确运行环境、冻结text/audio脚本SHA，绑定clean文本权重及Tokenizer文件、音频checkpoint/完整归一化协议、数据及clean特征来源。所有输入文件提取后重新哈希，全部加载encoder state与BN buffers前后指纹必须相等。clean重提取control与variant共同保留目标身份、实际截断后的token IDs与hash、CPU FP32形状/hash。consumer进一步检查sidecar与bundle元数据相等、before/after state证据、固定family→variant→modality关系、已知status、control/variant原始特征及目标身份。初次审查发现的未知modality及错family入口缺口已由各代码负责人修正。

独立执行修订后的producer纯合成测试8项全部通过；验证器自身纯合成测试通过。以上命令未读取真实数据、Validation目标、媒体或模型。自有验证器已支持V2实际产物的逐项复核，包括两类encoder与clean缓存一致、全部固定版本/失败记录、原始cache对应行、source/state hash、CPU FP32和actual token hash及TrainFit工程smoke。尚未执行真实pilot阶段核验，完整训练、完成锁和后续正式Validation仍须遵守既定先后顺序，测试通过不代表科学结果已完成。

资源恢复记录应区分已完成M3Net的正常复用与未保存的非训练difficulty步骤中断后重算；后者不构成新增seed或训练续跑。当前独立审查未运行任何训练或真实扰动。

## 已完成smoke的I/O与provenance勘误

模型阶段由根代理实际执行独立验证器，10156项通过；此时尚无扰动推理结果。随后发现两处I/O/provenance问题，已只读比较封存源码与修订源码，差异仅限以下两函数。

`read_jsonl()`原先使用`splitlines()`，会将JSON字符串内合法的U+0085、U+2028、U+2029误当记录边界。修订按物理LF分记录，显式UTF-8读取，保留原文字符；不清洗或改写原文。独立执行合成回归，ensure_ascii真/假与LF/CRLF四种组合均保持两条原记录及输入文件字节，并复现旧解析器失败。

`verify_baseline_complete()`原先循环变量遮蔽了completion路径，实际模型完成状态、checkpoint及训练输入哈希检查使用正确completion内容，但返回的provenance路径/hash误指最后一个audio_features文件。修订分开completion_path与input_path，独立合成回归确认真实completion路径/hash和原输入拒绝门仍生效。两数据集既有TrainFit工程smoke报告、缓存及数值保持原字节，另由`provenance/SMOKE_COMPLETION_PROVENANCE_ERRATA.json`明确关联错误引用、真实completion、原smoke哈希、baseline锁及输入一致性，不重新运行smoke。

验证器兼容历史smoke时，仅接受指定归档中与原哈希完全相同的这两份旧源码，且AST确认其余函数/模块代码未改变；还要重算勘误记录中的真实完成标记、checkpoint、baseline锁和三类输入哈希。新pilot producer必须绑定当前源码及真正completion.json，不能使用历史错误引用。新增兼容逻辑的纯合成正例及错误hash/越界代码改变拒绝用例通过，未读取真实Validation或运行模型。合法QC失败、UNCERTAIN/PENDING以及未完整K的family仍保留全部固定版本分母，不因这些失败而调整样本或版本并重跑。

## 最终实际产物独立核验

两数据集全部固定Pilot与2000次Dialogue bootstrap完成后，独立实际执行`verify_plan_b.py --stage final`，最终62723项检查全部通过，失败0、待完成0，`verification.json`记录`status=PASS`、`complete=true`，完整日志为`logs/verify_final.log`。验证器最终SHA为`ba19b6fb5063573c057d22f519159f5ad4737a552ae6bfbcd908d4b43623fd8d`。该完成标记表示既定流程和交付的实际证据核验完成，不表示扰动能改善可靠性或校准。

独立重算覆盖2371个成功variant的原/扰动类别、最大confidence、原类别及真实类别概率、flip和JS；两个逐variant衍生表共4742行核验均与完整概率一致。所有保存的完整Dialogue概率均检查有限、非负、归一化与目标行一致；各完整family按固定K重算正式μ、δ、flip、JS均值/最大值及confidence变化。Dialogue配对重抽与精确Dialogue×confidence-cell配对中，504项可计算bootstrap CI已从CSV独立按同一预定seed恢复2000次cluster抽样比较，其余无支持情况按原协议保留NA及支持数。

全部2400个预定条件均留档：IEMOCAP1189成功、11失败（3个文本token QC、8个音频signal QC），MELD1182成功、18个音频signal QC失败。29条失败逐条核对为既定QC错误，均保留错误原因且不进入main；不存在忽略的unexpected失败。946个完整family、14个不完整family、344个严格标签审核main family和468个audio signal-QC-only探索family均由现有分析记录明确区分。音频标签仍为PENDING，不能计算其扰动后accuracy或校准结论；没有虚构听审。

首次final核验在记录项目外FFmpeg显示路径时发生验证器自身`relative_to()`异常，保留于`logs/verify_final_attempt1.json`及同名log；仅修正外部已绑定文件的哈希显示key，合成回归通过后重跑。中间53710项通过结果也保留于attempt2记录；最终增加所要求逐variant标量与两衍生表核验后再次完整通过。上述过程只读取既有产物，不重新提取特征、运行模型、拟合、调整统计方法或修改实验条件。

研究解释仍受单seed、120条分层样本、有限confidence/Dialogue共同支持、AI文本标签审核及音频标签未确认限制；历史随机特征实验使用不同模型/特征，不能作为严格同模型因果对照。本审查未拟合可靠性评分或校准器，也未访问Test。
