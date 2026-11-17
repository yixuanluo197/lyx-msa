"""Write blocked source-recovery reports from audited metadata; no model/data loads."""
from pathlib import Path
import csv,hashlib,json
OUT=Path(__file__).resolve().parents[1]

def load(name):return json.loads((OUT/'provenance'/name).read_text())
def write(name,text):
 with (OUT/name).open('x') as f:f.write(text)
def main():
 source=load('paper_source_evidence.json');assets=load('local_weight_asset_audit.json');index=load('input_index_audit.json')
 assert not source['decision']['paper_explicitly_authorizes_untuned_public_roberta_as_actual_extractor']
 metrics={};csvrows=[]
 for ds,n,seed,start in [('IEMOCAP',527,1475,2),('MELD',1109,67137,3)]:
  layerrows=[]
  for k in range(4):
   row={'dataset':ds,'seed':seed,'layer_slot':k+1,'reference_pickle_slot':start+k,'reference_hidden_offset':-1-k,'expected_reference_shape':[n,1024],
        'new_reference_shape_measured':False,'extracted_shape':None,'max_absolute_error':None,'MAE':None,'RMSE':None,'mean_cosine':None,'min_cosine':None,'per_dimension_correlations':None,'distribution_differences':None,'status':'NOT_RUN_MISSING_CONFIRMED_ENCODER','passed':None}
   layerrows.append(row);csvrows.append(row)
  metrics[ds]={'status':'BLOCKED_SOURCE_RECOVERY','actual_cmerc_encoder_found':False,'successful_extraction':False,'layers':layerrows,'all_four_clean_parity_passed':None,'probability_gate_eligible':False,'numeric_feature_failure_observed':False,'validation_values_opened':False,'probability_status':'NOT_RUN','probability_metrics':{'max_absolute_error':None,'mean_absolute_error':None,'class_agreement':None,'maximum_confidence_difference':None},'can_begin_raw_text_perturbation_with_original_XP04':False}
 write('provenance/text_parity_status.json',json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'layer_comparison.csv').open('x',newline='') as f:
  fields=list(csvrows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in csvrows:
   writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else ('NA' if v is None else v) for k,v in row.items()})
 write('ROBERTA_SOURCE_RECOVERY.md','''# XP04 RoBERTa 文本提取来源恢复

**结果：未找到可确认实际生成 CMERC/XP04 四层文本特征的两套 encoder 权重，原始文本重提取阻断。** 这不是数值Parity失败，也不是模型不具备文本敏感性的结论。本轮没有使用替代权重、训练、提取或CMERC前向。

## 1. 论文与参考实现的证据层级

CMERC正式版《Multimodal Emotion Recognition Calibration in Conversations》，DOI [10.1145/3664647.3681515](https://doi.org/10.1145/3664647.3681515)，本地PDF SHA-256 `8b4133721eb6020eadc470ad15b6337ee2c06772404dd34f29dc4364351905e0`。PDF第3页§3.2引用Ghosal等[9]，描述最后四层、layer normalization与均值操作；第9页参考[9]是COSMIC。论文**没有明确说明直接以未微调的公开初始化权重提取，也没有给出encoder checkpoint身份**。不能把未写微调解释成没有微调。

COSMIC [正式论文PDF第3页§3.2](https://aclanthology.org/2020.findings-emnlp.224.pdf) 明确先把RoBERTa Large微调用于话语情绪分类，再取首特殊Token的最后四层。其公开脚本与此吻合。但引用关系、实现结构和特征包关联不能证明CMERC使用了哪一个微调run、Fairseq版本或当时的完整预处理。

固定代码版本：CMERC `815c7c060914f5b55db816c0ba3662b11d08552f`；COSMIC/conv-emotion `6128ca20e9c736605cce7e99d5d95db0356c35f5`。论文、源码定位、请求失败与范围见 [paper_source_evidence.json](provenance/paper_source_evidence.json)。

## 2. 权重与下载资源分类

| 对象 | 实际类型/核查结果 | 可否作为XP04原文本encoder |
|---|---|---|
| CMERC Dropbox已缓存514,504,220字节包 | 目录项加四个预提特征pkl；当前包SHA与历史下载一致 | 否，是特征文件 |
| CMERC官方Drive的IEMOCAP/MELD_checkpoint.pkl | README与模型实现指向下游CMERC/M3Net；本轮匿名文件夹列表SSL超时 | 不能当作RoBERTa encoder |
| Fairseq roberta.large.tar.gz | 官方README的微调初始化包；本轮HEAD200、655,283,069字节，无大下载 | 不能替代未知任务微调权重 |
| 本地HF roberta-large snapshot | 已有完整预训练MaskedLM，24层、1024维 | 无实际生产身份，未采用 |
| 本地IEMOCAP/MELD微调权重 | 自有2026年HF实验；包含speaker token、pooling、seed等与COSMIC不同的配置 | 无CMERC来源证明，未采用 |
| COSMIC官方Drive erc-training.zip | 实验特征资源；本轮通过HTTP Range完整列中央目录 | 未发现权重命名条目，不能据此恢复encoder |

本地重点权重哈希：

- HF预训练safetensors：`047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`。
- 自有IEMOCAP speaker-token encoder：`874ee62a61f6d4cb284957e4abfcd502a8eda26177332d4f8734a5a64655cf99`。
- 自有IEMOCAP seed2025 encoder：`6e50cf9565e855b9ea533faee4c75afaba513a3103436c327ab33044c28ba50a`。
- 自有MELD seed42 encoder：`a12a4505d09e75db1abe0e0e0def5e85fdf711f016b5956f62eee0ed6d480285`。

扫描范围、文件数、对应生成代码/配置、哈希与排除理由见 [local_weight_asset_audit.json](provenance/local_weight_asset_audit.json)。名称盘点未找到checkpoint_best.pt；9个现有Conda环境未发现Fairseq包目录。搜索不是全世界资产不存在的证明，也不能由当前安装状态倒推历史Fairseq版本。

## 3. COSMIC特征包的新增目录实查

官方ZIP共5,488,532,347字节；只用Range读取末尾131,072字节，由ZIP64记录定位4,475字节中央目录，列出36项。普通内容按命名是四个数据集的RoBERTa/COMET特征pkl及macOS元数据，没有.pt/.pth/.bin/.safetensors/.hdf5或checkpoint命名条目。未下载整个5.49GB包、未解压pkl、未读取任何原始语料或Test标签，也未加载模型。目录只证明可见条目，不声称检查了任意pickle内部的对象语义。

IEMOCAP/MELD的RoBERTa条目与CMERC已缓存官方ZIP的大小、CRC32都吻合：124,438,886字节/`c5e7cb82`；229,654,413字节/`6e6c53de`。这是较强的包元数据一致性证据，**不是本轮测得的COSMIC payload SHA-256相等，更不是实际encoder权重身份**。

[完整目录](provenance/cosmic_archive_directory.json)、[两包元数据比较](provenance/cosmic_cmerc_zip_metadata_comparison.json)、[限量网络探测](provenance/public_resource_probe.json)。

## 4. 四层表示、Tokenizer与处理规则

已核查的COSMIC参考代码加载 `checkpoints/iemocap/checkpoint_best.pt` 或 `checkpoints/meld/checkpoint_best.pt`，然后eval。预训练模型是训练初始化，不是这两条提取路径的实际权重。

| 项目 | 参考代码可确认内容 | 尚不能认定的实际CMERC信息 |
|---|---|---|
| 架构 | Fairseq roberta_large；各1024维 | 原权重SHA、run、环境 |
| 四层顺序 | 保存T1=hidden[-1]，T2=[-2]，T3=[-3]，T4=[-4] | 不能重排后再找Parity |
| Pooling | 每个话语独立encode，在每层取token index0的BOS表示 | 不用mean pooling替代 |
| batch/context | 同一Dialogue的话语作为batch，pad_idx=1；不拼接对话文本 | 真正encoder上下文配置仍待生产记录 |
| Tokenizer/BPE | GPT-2 byte BPE和Fairseq dictionary；训练init-token0、separator-token2 | 原checkpoint BPE参数、Fairseq版本及实际tokens |
| 文本 | IEMOCAP直接传存储字符串；MELD参考代码ASCII忽略非ASCII字符 | 不能未经确认就清洗或改写XP04原文 |
| 长度 | 训练max_positions512、shorten_method truncate | extraction没有显式截断/分块；不能擅自补造 |
| 保存归一化 | 四路BOS数组直接写pkl，没有可见额外zscore/L2/仿射 | 原生产可能的外部处理仍缺记录 |

三个参考tokenizer资产SHA：encoder.json `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`；vocab.bpe `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`；dict.txt `2c31c5b236661067fe8626775eb995ef5fe808ac87d2ad4a6ddbef68550f5cc6`。HF merges文件相同不等于权重或完整tokenization行为相同。

论文的LN/均值描述不能覆盖原checkpoint配置。本轮两份XP04 effective config实际都是 `norm=BN`，冻结model.py逐层归一化后求四层均值。该模型内处理与pkl生产是两个阶段；Clean Parity应逐层比较存储特征，不能先套LN/平均使差异消失，也不修改checkpoint的BN策略。

## 5. 处理失败与复用范围

新源码下载中6个raw请求出现代理502；其余成功获取字节已留档。历史来源清单的source_url/url字段差异、ZIP目录项计数以及两个README同名写出冲突也都留存失败记录/代码快照。修正后以固定commit、既有SHA核验的缓存源码补齐参考，不把缓存复用写成全部新网络请求成功。没有执行会遍历train/valid/test的上游init/extraction脚本。

ACM403、OpenReview浏览器验证、Drive SSL超时与Dropbox502保留。正式CMERC PDF复用本地归档，COSMIC论文从官方ACL下载；此类检索障碍没有用非正式摘要取代论文结论。

## 6. 阻断决定

本轮论文证据没有满足“明确允许直接用公开未微调权重”的条件，也没有找回可确认的实际任务encoder。因此不从HF缓存、其他本地微调模型或新微调模型启动Validation提取。不拟合任何转换器。需要补齐：两数据集实际生产encoder checkpoint及来源/哈希、训练与原始split记录、字典与BPE关联、Fairseq版本和抽特征文本/长度处理记录。

当前结果为BLOCKED/NOT_RUN，不能写成Clean Parity数值失败或通过。权重缺失不影响既有XP04模型使用原预提特征的历史结果，但尚不支持生成兼容的新文本特征。
''')
 for ds,n,seed,start in [('IEMOCAP',527,1475,2),('MELD',1109,67137,3)]:
  rec=index['datasets'][ds]
  table='| 保存层 | 参考hidden索引 | pkl槽位 | 预期参考形状 | 重提取形状 | 数值指标 |\n|---|---|---:|---|---|---|\n'
  for k in range(4):table+=f'| T{k+1} | [-{k+1}] | {start+k} | ({n},1024) | NA | NOT_RUN |\n'
  extra='''IEMOCAP必须沿用Seed1475 effective split，不能改用RoBERTa pkl自带valid_ids。历史已确认当前12个Validation对话中，11个出现在文本包自带train_ids，1个在其valid_ids。若这些IDs对应encoder有监督微调训练集合，则存在表示学习阶段与下游Validation重叠的条件风险；没有实际encoder/训练日志时，不能据此宣告已发生标签泄漏。这里只复用历史split审计及现有配置哈希，不使用包内列表替换当前split。
''' if ds=='IEMOCAP' else '''MELD采用Seed67137的114个真正Validation对话，共1109条；不拼接或修改说话人、顺序与Dialogue。COSMIC参考ASCII处理是否属于CMERC原生产仍未确认，不能默认删除原文非ASCII内容，也不以重清洗后更接近目标为由选择方案。
'''
  write(f'{ds}_TEXT_CLEAN_PARITY.md',f'''# {ds} 文本 Clean Feature Parity

**状态：BLOCKED / NOT_RUN。** 没有找到已确认实际生产权重，论文也没有明确支持以未微调预训练模型直接替代。本轮成功重提取话语数为0；没有测量特征误差，不是观测到Parity失败。

## 固定目标与索引

使用Seed{seed}真正Validation，共{n}条/{rec['validation_dialogues']}个Dialogue。现有ID清单已逐行核对effective config的对话顺序、row index及唯一性，保留本地。没有打开本轮参考特征值或原始话语文本，也没有编码任何tokens。表中形状是基于已锚定官方pkl与历史结构审计的预期目标，不能写成新提取或新测量的形状。

目标文件 `{Path(rec['reference_features']).name}` 当前SHA-256：`{rec['reference_sha256']}`，与此前官方Dropbox文件的完整SHA一致。Checkpoint SHA-256：`{rec['checkpoint_sha256']}`，仅按字节核验，没有反序列化。配置SHA-256：`{rec['config_sha256']}`。

{table}

{extra}

## 预先固定的兼容标准

[协议](configs/TEXT_RECOVERY_PROTOCOL.json)在任何本轮Validation目标值访问前冻结。四层必须分别满足完整覆盖和有限性、每元素allclose(atol=rtol=1e-4)、MAE≤1e-5、RMSE≤1e-4、非零话语余弦平均≥0.99999/最小≥0.9999、所有目标非常数维相关有定义且p05≥0.999；常数/零向量单独计数。逐维分布记录均值/总体标准差差异与一阶Wasserstein。这里的相关性是坐标跨话语的描述统计，不是把同对话话语当独立样本做显著性检验。

当前缺权重导致这些检查均未执行。最大绝对误差、MAE、RMSE、平均/最小余弦、逐维相关性和分布差异全部为null/NA，不填0、1或Pass。参见[逐层状态CSV](layer_comparison.csv)和[机器可读状态](provenance/text_parity_status.json)。没有逐维数值CSV，因为不存在重新提取的数值。

## 执行边界与结论

无文本改写/追加，无训练、回归转换器或仿射匹配，无Validation反馈调参；未处理音频/视觉，未索引Test特征或标签。当前四层未通过（原因是尚未执行，而非已测数值不匹配）。因此不执行Seed{seed}原XP04的概率比较，也不能直接开始原始文本干扰。原始特征与checkpoint保持不变。

来源与缺失环节见[RoBERTa恢复报告](ROBERTA_SOURCE_RECOVERY.md)。历史已经存在Validation结构审计，“独立”仅指未来锁定提取方案后一次性验证、不把本轮结果用于调参，不声称研究者此前未见过该Validation。
''')
 write('XP04_TEXT_PROBABILITY_PARITY.md','''# XP04 文本概率一致性验证

**IEMOCAP Seed1475、MELD Seed67137：均为NOT_RUN。** 四层文本特征没有完成Clean Feature Parity，因而按用户要求禁止继续概率验证；不是概率比较后发现失败。

| 数据集 | softmax最大误差 | softmax平均误差 | 类别一致率 | 最大置信度差异 | 原因 |
|---|---|---|---|---|---|
| IEMOCAP | NA | NA | NA | NA | 缺少已确认文本encoder，四层未验证 |
| MELD | NA | NA | NA | NA | 缺少已确认文本encoder，四层未验证 |

两checkpoint仅进行文件哈希核验，未反序列化、未执行CMERC模型前向。没有Softmax数组或预测CSV；不能把未测量写成零误差或100%一致，也不能宣称类别不一致。概率门未打开，因此本轮没有替换任何输入模态，更没有修改其他模态/话语/对话结构。

未来只有每数据集四层全部通过预定特征门，才能使用原模型冻结推理，保留音频/视觉、Speaker与Dialogue结构，仅替换四路文本；使用原norm=BN配置，不按论文LN描述改模型。原/重建两套softmax、类别和最大置信度须同时保存，概率最大误差≤1e-4、MAE≤1e-5、类别一致100%、最大置信度误差≤1e-4。

权重缺失与归一化阶段的证据见[来源恢复](ROBERTA_SOURCE_RECOVERY.md)；本轮状态为[JSON](provenance/text_parity_status.json)。
''')
 write('FINAL_TEXT_RECOVERY_DECISION.md','''# 文本特征恢复：最终研究决定

**本轮完成来源核查，但两数据集的重新提取和Clean Feature Parity均因实际encoder权重未确认而阻断。** 不使用预训练或其他本地微调模型替代，不新增微调，也不通过仿射/回归制造Parity。

| 最终问题 | IEMOCAP | MELD |
|---|---|---|
| 找到CMERC实际使用的RoBERTa权重？ | 未找到可确认对象 | 未找到可确认对象 |
| 成功重提取目标Validation文本？ | 否；0/527，NOT_RUN | 否；0/1109，NOT_RUN |
| 四层是否通过Clean Feature Parity？ | 未验证，不能判通过 | 未验证，不能判通过 |
| 可否直接用原XP04开展文本干扰？ | 当前不具备兼容特征生成条件 | 当前不具备兼容特征生成条件 |

## 已增加的实证

1. 核对CMERC正式论文§3.2及其COSMIC引用，确认论文没有明确支持用未微调公开模型直接替代；COSMIC原文和代码明确要求数据集微调后的encoder。
2. 扩展本地权重盘点，找到了预训练大模型和自有微调权重，依据配置与生成代码逐项排除原生产身份；不能以“目录里有RoBERTa”认定恢复完成。
3. 仅读取COSMIC官方5.49GB包的128KiB尾部，完整列出36项中央目录，未见checkpoint命名条目；IEMOCAP/MELD特征与CMERC官方包大小和CRC32一致。没有下载/解包数据，也没有获得实际encoder。此元数据关系强于仅看README，仍不能充当payload SHA或权重身份。
4. 对现有官方目标特征、原始Checkpoint、配置及历史资料留档哈希；真正Validation IDs为Seed1475的527条和Seed67137的1109条，未使用文本包自带split替代。
5. 保存了最后四层顺序、BOS/pad/BPE、长度和归一化已知/未知项，明确模型内BN/四层平均与pkl生产的区别。

## 没有执行的部分

原始文本提取、分词、token导出、四层数值对比和CMERC概率验证均未执行。逐层CSV的全部数值指标为NA，机器状态为null；不是负的性能结果，更不是以缺数据证明模型无效。没有修改原始文本、音频、视觉、预提特征或checkpoint，没有使用Test标签，没有原始数据干扰、CMERC训练、转换器拟合或Git Commit/Push。

## 精确缺失项与下一步条件

需要每数据集实际生产encoder checkpoint的权重与可追溯来源/哈希、该run采用的训练split、Fairseq版本、BPE/dictionary参数，以及真实文本清洗、超长序列和四层输出处理记录。任务微调模型的名称相同或维度相同不足以证明身份。

得到这些资料后，才在新协议下固定方案并一次性验证真正Validation。若研究者另行授权“公开预训练候选探索”，必须单列为候选研究，不能改称本轮已恢复的CMERC原流程，也不能以这次Validation反复选参数后再宣称独立通过。

本轮停止，等待研究者审核。报告：[来源恢复](ROBERTA_SOURCE_RECOVERY.md)、[IEMOCAP](IEMOCAP_TEXT_CLEAN_PARITY.md)、[MELD](MELD_TEXT_CLEAN_PARITY.md)、[概率门控](XP04_TEXT_PROBABILITY_PARITY.md)、[验证证据](verification.json)。完整媒体与权重仍只保留本地，没有发布动作。
''')
 write('README.md','''# XP04 文本特征恢复审计

结果：两数据集均因实际RoBERTa encoder未确认而BLOCKED；没有提取、数值Parity或CMERC概率推理。

入口：[最终决定](FINAL_TEXT_RECOVERY_DECISION.md)。逐层状态：[CSV](layer_comparison.csv)。来源、网络失败、有限包目录核查、本地权重身份与哈希在provenance/；参考代码及论文在sources/，不会作为新提取器自动执行。configs/保存访问Validation前冻结的协议。

scripts/仅包含来源/索引审计、限量公开资源探测、ZIP中央目录检查及报告/交付验证，不包含替代权重提取流程。原始特征pkl和模型只按字节哈希检查，未反序列化；本地ID清单位于local_only/，不含原始文本。COSMIC ZIP尾部是原始二进制元数据，仅本地保留，不是完整包。

本轮未Git Commit或Push。完整文本、特征、权重和私有路径不自动公开。
''')
 print(json.dumps({'reports_written':6,'layer_rows':8,'numeric_metrics':'NOT_RUN/null','text_extraction_runs':0}))
if __name__=='__main__':main()
