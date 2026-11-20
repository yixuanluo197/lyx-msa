#!/usr/bin/env python3
"""Render frozen numerical results without fitting, selecting or rerunning models."""
import csv
import json
from collections import defaultdict
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
A=RUN/'results/matched_analysis'
LABEL={'neutral_append':'中性无关信息追加','semantic_paraphrase':'情绪保持语义改写'}
METRIC={'mu':'μ','delta':'δ','js_mean':'JS（自然对数）','flip_rate':'翻转率'}
CAL={'all':'全部冻结配对','gap_le_0.02':'置信度差≤0.02','gap_le_0.01':'置信度差≤0.01'}


def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def csvout(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def number(x):return f'{float(x):.6f}' if x not in ('',None) else 'NA'


def ci(row):return f"[{number(row['ci95_low'])}, {number(row['ci95_high'])}]"


def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in data])+'\n'


def main():
    supports=read(A/'condition_support.csv');contrasts=read(A/'paired_metric_contrasts.csv')
    ranks=read(A/'conditional_pair_ranking.csv');pairs=read(A/'pair_behavior_and_eligibility.csv')
    controls=read(RUN/'results/clean_controls.csv');predictions=read(RUN/'results/perturbed_predictions.csv')
    verification=json.loads((RUN/'verification.json').read_text())
    eligible=lambda r:r['analysis_scope']=='AI_REVIEWED_EXPLORATORY' and r['residual_confidence_group']=='all'
    s=[r for r in supports if eligible(r)]
    c=[r for r in contrasts if eligible(r) and r['estimand']=='equal_dialogue']
    r=[row for row in ranks if eligible(row)]
    # Descriptive group means on the EXACT same eligible pairs, equal Dialogue weighting.
    summaries=[]
    for row in c:
        subset=[x for x in pairs if x['analysis_scope']=='AI_REVIEWED_EXPLORATORY'
                and x['perturbation_type']==row['perturbation_type'] and x['eligible']=='True']
        if row['confidence_caliper']!='all':
            threshold=.02 if row['confidence_caliper']=='gap_le_0.02' else .01
            subset=[x for x in subset if float(x['confidence_abs_gap'])<=threshold]
        values={}
        for side in ('correct','error'):
            clusters=defaultdict(list)
            for x in subset:clusters[x['dialogue_id']].append(float(x[f"{row['metric']}_{side}"]))
            values[side]=sum(sum(v)/len(v) for v in clusters.values())/len(clusters)
        summaries.append(dict(perturbation_type=row['perturbation_type'],analysis_scope=row['analysis_scope'],
            confidence_caliper=row['confidence_caliper'],metric=row['metric'],estimand='equal_dialogue',
            correct_mean=values['correct'],error_mean=values['error'],error_minus_correct=row['error_minus_correct'],
            ci95_low=row['ci95_low'],ci95_high=row['ci95_high'],eligible_pairs=row['eligible_pairs'],
            eligible_dialogues=row['eligible_dialogues'],bootstrap_repetitions=2000,exploratory_only=True))
    csvout(RUN/'results/correct_wrong_summary.csv',summaries)
    flips=[]
    for x in predictions:
        row={k:x[k] for k in ['sample_id','dialogue_id','utterance_index','pair_id','perturbation_type',
            'variant_id','status','original_correct','original_confidence','label_review_status','automatic_qc_pass']}
        success=x['status']=='SUCCESS'
        changed=success and int(x['perturbed_prediction'])!=int(x['original_prediction'])
        row.update(prediction_flip=changed if success else '',
            correct_to_wrong=(x['original_correct']=='1' and changed) if success else '',
            wrong_to_correct=(x['original_correct']=='0' and int(x['perturbed_prediction'])==int(x['true_label'])) if success else '',
            max_confidence_change=(float(x['perturbed_max_confidence'])-float(x['original_confidence'])) if success else '',
            original_class_probability_change=(float(x['perturbed_original_class_probability'])-float(x['original_confidence'])) if success else '')
        flips.append(row)
    csvout(RUN/'results/prediction_flips_and_confidence_changes.csv',flips)
    support_table=table(['干扰','置信度差限制','有效配对 / Dialogue','候选配对支持率','占完整Validation话语','描述性支持门'],[
        [LABEL[x['perturbation_type']],CAL[x['confidence_caliper']],f"{x['eligible_pairs']} / {x['eligible_dialogues']}",
         f"{100*float(x['candidate_pair_coverage']):.2f}%",f"{100*float(x['validation_target_coverage']):.2f}%",
         '通过（非功效证明）' if x['descriptive_support_floor_pass']=='True' else '不足'] for x in s])
    main_table=table(['干扰','指标','正确样本均值','错误样本均值','错误−正确','95% Dialogue CI'],[
        [LABEL[x['perturbation_type']],METRIC[x['metric']],number(x['correct_mean']),number(x['error_mean']),
         number(x['error_minus_correct']),f"[{number(x['ci95_low'])}, {number(x['ci95_high'])}]"]
        for x in summaries if x['confidence_caliper']=='all'])
    caliper_table=table(['干扰','限制','指标','错误−正确','95% Dialogue CI','有效配对 / Dialogue'],[
        [LABEL[x['perturbation_type']],CAL[x['confidence_caliper']],METRIC[x['metric']],number(x['error_minus_correct']),ci(x),
         f"{x['eligible_pairs']} / {x['eligible_dialogues']}"] for x in c if x['confidence_caliper']!='all'])
    priority_table=table(['干扰','JS 错误−正确','95% CI','两家庭校正97.5% CI'],[
        [LABEL[x['perturbation_type']],number(x['error_minus_correct']),ci(x),
         f"[{number(x['bonferroni_ci97_5_low'])}, {number(x['bonferroni_ci97_5_high'])}]"]
        for x in c if x['priority_two_family_js']=='True'])
    rankrows=[]
    for family in LABEL:
        for metric in METRIC:
            selected={x['ranking']:x for x in r if x['perturbation_type']==family and x['metric']==metric and x['confidence_caliper']=='all'}
            delta=selected['behavior_minus_confidence']
            rankrows.append([LABEL[family],METRIC[metric],number(selected['behavior']['estimate']),
                number(selected['confidence_only']['estimate']),number(delta['estimate']),ci(delta)])
    rank_table=table(['干扰','指标','行为配对排序','同配对1−置信度排序','排序差','排序差95% CI'],rankrows)
    diagnostic_table=table(['干扰','指标','全部输入QC通过：错误−正确','95% CI'],[
        [LABEL[x['perturbation_type']],METRIC[x['metric']],number(x['error_minus_correct']),ci(x)]
        for x in contrasts if x['analysis_scope']=='AUTOMATIC_INPUT_QC_DIAGNOSTIC' and x['confidence_caliper']=='all'
        and x['residual_confidence_group']=='all' and x['estimand']=='equal_dialogue'])
    feature_error=max(float(x['feature_max_abs_error']) for x in controls)
    probability_error=max(float(x['probability_max_abs_error']) for x in controls)
    report=f'''# MELD 文本可靠性增量探索结果

本轮完成冻结 MELD Seed 67137 的 203 对匹配样本分析。**语义改写存在局部条件关联信号，但尚未证明行为指标具有原始 Softmax 置信度之外的稳定增量价值；暂不支持直接训练 Reliability Score。** 中性追加没有出现清晰的正确/错误区分。两个预先指定的主 JS 对比均未排除零差异。

这是重复使用 Validation 的单 Seed 探索，含 AI 审核而非人工确认；不构成独立确认性实验，也不能用匹配样本的 50% 错误比例估计总体分类性能。

## 冻结模型、输入与实际执行

- 使用首轮 MELD Seed 67137 CMERC 与最佳 RoBERTa，不更新参数，不调用音频编码器，不增加 Seed，不使用 Test。
- 输入锁定于新增预测前；406 条目标各两类×两版，共 1,624 个固定条件。每次只替换一个目标的文本特征，再运行完整 Dialogue；音频、视觉、Speaker、标签和其他话语的特征逐位不变。
- 406 条真实 clean 负对照全部通过。特征比较使用继承的 `atol=rtol=1e-4` 组合判据，最大绝对差 {feature_error:.9g}；**这不是声称最大绝对差小于1e-4**。文本重提取输入后的最大概率差 {probability_error:.9g}，类别全部保持。模型、BN 状态及全 Validation 上下文指纹前后一致。
- 固定条件成功 1,544，输入QC失败 80：neutral 758成功/54失败，paraphrase 786成功/26失败。失败行仍保留，不修复、不删除、不降低K；812个目标×家庭中70个无法形成完整K=2。
- 匹配条件 AI PASS 1,037、UNCERTAIN 489、FAIL 98。四组新增版本单次生成后独立交叉AI复核，48个原PASS被降级；172个历史重叠版本及原审核记录逐项不变。未根据模型响应修改文本。
- 中性共54条输入失败，均为实际新增Token超预算：52条为5>4，另2条分别为6>5、6>4；其中53条集中在同一生成组，另一组1条。此前生成者将本组53条笼统归为5>4，最终以实际Tokenizer QC逐条核对结果为准，原日志保留。改写26条未改变实际输入，其中18条同时出现同家庭Token重复。没有原文截断。具体条件见 `failed_uncertain_review_records.csv` 与私有AI记录。生成者分组没有随机化，文本风格及质量可能形成混杂。

## 审核范围与匹配支持率

原始203对覆盖406/1109=36.61%的Validation话语、80/114个Dialogue。两类干扰采用同一初始目标集；QC和完整K门导致实际分析支持不同，不能重新配对补齐。单纯输入QC通过的诊断范围：中性153对/76Dialogue，改写186对/80Dialogue。AI审核探索性范围：

{support_table}

全部初始配对中，中性审核范围保留68/203=33.50%，改写86/203=42.36%。表中“候选配对支持率”在各置信度子集内计算，所有匹配/排除原因均留存。≤0.01子集中，中性仅21对/19Dialogue，改写仅29对/23Dialogue，均未达到预先固定的30对且20Dialogue描述性支持门；该门也不是正式功效或样本量充足证明。

`TEXT_REVIEW_PACKAGE.csv` 含历史全部480版及新增1452版，共1932行；完整Dialogue、原句和变换句均仅留本地。两套人工字段全部空白，PENDING=1932；**人工确认范围为0对，结果NA**，不能将AI PASS解释成人工通过。

## 指标及统计口径

对每个目标、每个家庭固定K=2：μ=mean(flip×干扰后最大概率)，δ=abs(原始最大概率−mean(干扰后最大概率))，JS为两次自然对数Jensen–Shannon散度的均值，Flip Rate=mean(flip)。μ不是纯翻转率，δ不是平均绝对置信度变化；后者及原始类别概率变化另保留逐话语记录。

每对先计算错误减正确，再在Dialogue内平均，最后对有效Dialogue等权。固定80个Dialogue为共同Bootstrap总体，种子20261120，2000次有放回抽样；两类干扰、三个审核范围、指标和置信度子集共享相同抽样。重复抽中的Dialogue按重数计权，始终保留一对双方；不将话语作为独立样本。所有非空主范围与固定caliper对比均有2000次有效抽样；极小的额外残余方向诊断如有不足，原CSV保留warning。

以下均为等Dialogue主估计量。等配对次估计量、残余置信度方向诊断、完整共享抽样索引与720组重抽样数组保存在CSV/NPZ中。只有两家庭“全部配对JS”是预先指定优先对比，另报97.5% Bonferroni区间；其他95%区间是多重、探索性描述，不据单个区间挑选“成功”条件。

## 正确与错误样本的行为差异

{main_table}

语义改写的μ和翻转率出现正差异；δ和JS主差异尚不明确。中性追加四个指标均没有排除零差异。

优先JS对比：

{priority_table}

两类主JS结果的95%和97.5%区间都包含零。不能因某个子集的JS区间不含零而改写这个主结论。

## 收紧置信度差后的固定敏感性分析

{caliper_table}

语义改写在预先固定的≤0.02子集，μ、δ、JS和翻转率的95%区间均为正，可作为后续复核候选。收紧到≤0.01后，μ、JS和翻转率区间包含零；δ仍有正差异，但仅29对/23Dialogue且属于次指标。故存在局部条件关联，尚不能称为跨阈值稳定的可靠性增量。中性追加没有稳定的差异信号。

同Dialogue控制是精确的，连续置信度控制是近似的。原始AI范围配对绝对置信度差均值：中性0.024478、改写0.022751；≤0.02后分别为0.009143和0.008435。即使≤0.01也不等于完全相同置信度，且文本长度、类别、语境和可改写性仍可混杂。

## 是否优于原始置信度

以下是同一配对内“行为分数更高的一方为错误”的排序一致率，平分=0.5，先Dialogue内再Dialogue间平均。比较基线为同一组配对上的1−原始置信度；**不是总体ROC-AUC，也不是置信度+行为联合评分的独立泛化增益**。

{rank_table}

全部配对范围的8个行为减置信度排序差95%区间均包含零。≤0.02子集也没有明确的行为相对置信度排序优势；≤0.01支持进一步减小，不能据个别边界区间宣称收益。由此本轮没有建立稳定额外预测信息，更没有建立校准收益。

## 完整失败/不确定范围诊断

{diagnostic_table}

上述诊断允许语义FAIL/UNCERTAIN，只要求输入QC及完整K。其改写JS差异为正，而审核探索性主范围JS区间包含零；两者支持集合不同。不能把未确认标签保持的诊断结果当作标签保持主证据，不能只报告更有利的范围。所有失败和不确定版本仍有原始AI记录及可追踪处理状态。`prediction_flips_and_confidence_changes.csv` 对所有成功条件保留沿用原标签的参考转移；其中FAIL/UNCERTAIN版本的Correct-to-Wrong、Wrong-to-Correct不能解释为扰动后的真实正确性或校准证据，应结合审核状态筛选。主分析仍要求完整AI PASS配对，不删除这些诊断行。

## 对研究问题的回答

1. **错误样本是否μ、δ、JS更高？** 改写的μ及翻转率有探索性信号；主δ和JS未明确，中性追加未明确。不能概括为三个指标都更高。
2. **控制Dialogue和置信度后是否仍有差异？** 改写≤0.02有局部差异；更严格≤0.01支持不足且μ/JS/翻转率区间跨零，尚不稳定。
3. **是否有Softmax置信度之外的信息？** 观察到条件关联候选，但配对排序相对1−置信度的增益区间跨零；未训练联合评分或做独立/OOF评估，额外预测价值未证明。
4. **样本量是否足以训练Reliability Score？** 不足以据此支持正式训练：有效主范围仅68/86对、严格子集21/29对，全部为反复探索的Validation，人工确认尚缺，并非可靠性训练/独立评估数据。未做正式功效证明。

建议优先完成人工复核后验证语义改写的信号，并在预先固定规则下补少量模型Seed核查随机性；中性追加保持低优先对照。新Seed不会使复用的Validation成为独立数据。本轮不训练模块、不自动加Seed或扩展数据。

## 复现与核验

协议和配对在新增预测前冻结。172个旧固定条件重新推理预测类别100%一致，最大概率差1.6391e-6，是工程重复性，不是独立科学验证。`verification.json` 独立核验 {verification['checks']:,} 项PASS，包括真实概率手算μ/δ/JS、冻结pair端点/方向/资格、全部720组Bootstrap重抽样、CI和来源/产物哈希。未修改历史文件、权重、原始特征或数据。

代码、锁、运行命令、PID、GPU UUID和原始日志留在本目录。一次统计CLI遗漏必填`--run-dir`，在参数解析阶段退出、未生成结果；补齐参数后仅执行一次完整正式分析，未改变任何统计规则或根据结果调参。
'''
    (RUN/'MELD_TEXT_RELIABILITY_REPORT.md').write_text(report,encoding='utf-8')
    decision='''# 下一阶段研究判断

结论：**保留语义改写作为可靠性研究候选，暂不进入正式 Reliability Score 模块开发或训练；先人工复核，再考虑少量 Seed 的预注册验证。** 本轮不是全盘无效，也没有证明稳定的置信度增量。

## 已完成

- Assertion、Probe、Sample：早期资料曾列名，但未找到可执行原始协议、专用实现或可归属结果，均待确认；未自行构造，也未将neutral_append/paraphrase改名。
- 冻结MELD Seed67137上，从完整1109Validation按同Dialogue×0.1置信度区间冻结203对、406目标、80Dialogue。完整保留703条未匹配话语及原因。
- 两类×两版共1624固定条件，1544成功、80输入QC失败；不重配、不修复失败版本。406clean负对照全部通过，原模型和非目标上下文不变。
- 盲审包1932行，历史480版和新增1452版完整覆盖；原AI审核与交叉AI复核留档。没有实际人工审核，所有人工栏PENDING，人工确认范围NA。
- 2000次共同Dialogue Bootstrap、95%区间、两个主JS的97.5%区间、置信度caliper和同配对置信度排序对照均完成。独立61791项产物/数值核验PASS。

## 哪些结果值得保留

语义改写AI范围86对/57Dialogue，错误减正确μ=0.049816，95%CI[0.014069,0.090665]；翻转率差=0.130117，CI[0.044023,0.223411]。在预定≤0.02子集45对/35Dialogue，JS差=0.015397，CI[0.005719,0.027495]。这些是局部探索性条件关联，值得优先复核语义保持真实性。

两个优先主JS对比均不明确：中性0.001595，97.5%CI[-0.010343,0.012425]；改写0.005147，97.5%CI[-0.001677,0.012157]。不能改以更有利的子集作为新的“主成功条件”。≤0.01后改写仅29对/23Dialogue，μ、JS、翻转率区间跨零；中性追加未显示明确区分。

## 是否进入Reliability Score

**目前不建议正式开发或训练。** 可以保留现有特征接口和候选机制，但本轮行为排序相对同配对1−置信度的差异区间均未在主范围排除零，没有验证置信度+行为联合评分的OOF/独立增益，更没有校准收益。有效样本和独立性不足，不能把本轮匹配Validation转成训练集。

后续若研究者批准，应先取得真正人工盲审，确认完整K及配对支持；冻结新的验证方案，再在Train内部通过Dialogue-level OOF构建可靠性训练信息，并保留独立评估边界。任何模块训练都属于后续另行授权任务，本轮未执行。

## 是否需要补充Seed

**形成一般性结论前需要。** 当前仅MELD67137，单Seed无法区分模型随机性。建议人工复核后，先对语义改写补少量Seed的固定协议复核，不立即启动五Seed；若信号稳定且相对置信度有额外价值，再决定是否扩展。补Seed只能验证模型随机性，不能使反复使用的Validation变为独立样本，也不能补救语义标签保持的不确定性。中性追加保持低优先对照，不为得到正结果继续改强度或调分箱。

## 保留事项与停止条件

完整文本、盲审包、编码特征及Dialogue概率仅保留本地，未上传公开GitHub。已有历史数据、模型和实验结果未覆盖；没有训练Baseline、Reliability Score或CaliDist，没有Test、没有新Seed。所有FAIL/UNCERTAIN及80个输入失败均保留。

本轮工作已完成并停止，等待研究者审核；Assertion/Probe/Sample的原始操作协议及人工盲审仍为后续待确认事项。
'''
    (RUN/'FINAL_NEXT_STAGE_DECISION.md').write_text(decision,encoding='utf-8')
    print(json.dumps(dict(reports_written=True,correct_wrong_summary_rows=len(summaries),per_condition_flip_confidence_rows=len(flips)),indent=2))


if __name__=='__main__':main()
