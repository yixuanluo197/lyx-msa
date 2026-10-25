"""Public aggregate-only CPU analysis. No model/checkpoint/private/test files are read."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS=('XP04_best','XP04_latest','XP12_best','XP12_latest')
GROUPS=('all','shared_backbone','text_gru','audio_projection','visual_projection','graph_shared','classification_head','other')
LOSSES=('classification','cc','mscl','cscl')
REFERENCES={'classification':1.,'cc':.002,'mscl':.15,'cscl':.15}
PUBLIC_FILES=('gradient_norms','gradient_pairs','pseudo_counts','pair_counts','class_relations','branch_counts','confidence_summary','contribution_summary','case_summary','sample_coverage','state_audit')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def csv_read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def csv_write(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def number(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (TypeError,ValueError):return None


def ratio(numerator,denominator):
    a,b=number(numerator),number(denominator)
    return a/b if a is not None and b is not None and b>0 else None


def quantile(values,p):
    assert values
    values=sorted(values);index=(len(values)-1)*p;lower=math.floor(index);upper=math.ceil(index)
    return values[lower]+(values[upper]-values[lower])*(index-lower)


def distribution(values):
    values=[number(v) for v in values];values=[v for v in values if v is not None]
    return {'n':len(values),'mean':statistics.mean(values) if values else None,'median':statistics.median(values) if values else None,
            'q25':quantile(values,.25) if values else None,'q75':quantile(values,.75) if values else None,
            'min':min(values) if values else None,'max':max(values) if values else None}


def assert_public(rows):
    for row in rows:
        for key,value in row.items():
            normalized=key.lower()
            assert not any(term in normalized for term in ('dialogue_id','utterance_id','utterance_index','raw_gradient','raw_gradients','checkpoint_path','private_path','main_pred','removed_pred'))
            if isinstance(value,str):assert not value.startswith('/') and 'private/' not in value and 'private\\' not in value


def valid_pair(row):
    if row.get('status')!='valid':return False
    cosine=number(row.get('cosine'));a=number(row.get('norm_a_common'));b=number(row.get('norm_b_common'))
    common=number(row.get('common_tensors'))
    return cosine is not None and -1.000001<=cosine<=1.000001 and a is not None and a>0 and b is not None and b>0 and common is not None and common>0


def summarize_pairs(rows):
    values=[number(r['cosine']) for r in rows if valid_pair(r)];stats=distribution(values)
    return {'total_n':len(rows),'valid_n':len(values),'conflict_n':sum(v<0 for v in values),
            'conflict_frequency':sum(v<0 for v in values)/len(values) if values else None,
            **{'cosine_'+key:value for key,value in stats.items() if key!='n'},'status_counts':dict(Counter(r.get('status','missing') for r in rows))}


def paired_deltas(rows,first,second):
    lookup={}
    for row in rows:
        if row['checkpoint_id'] not in (first,second):continue
        key=(row['batch_id'],row['loss_a'],row['loss_b'],row['group'])
        assert (row['checkpoint_id'],key) not in lookup
        lookup[(row['checkpoint_id'],key)]=row
    result=[]
    for cp,key in lookup:
        if cp!=first:continue
        a=lookup[(first,key)];b=lookup.get((second,key))
        if b is not None and valid_pair(a) and valid_pair(b):
            result.append({'checkpoint_from':first,'checkpoint_to':second,'batch_id':key[0],'loss_a':key[1],'loss_b':key[2],'group':key[3],
                           'cosine_from':number(a['cosine']),'cosine_to':number(b['cosine']),'delta_cosine':number(b['cosine'])-number(a['cosine'])})
    return result


def require_complete(state,verification,expected):
    assert state['phase']=='completed' and state['completed_cases']==expected
    assert state['optimizer_steps']==0 and state['test_forwards']==0
    assert verification['status']=='passed' and verification['cases']==expected
    assert verification['optimizer_steps']==0 and verification['test_forwards']==0


def case_coverage(rows,checkpoints,n_batches):
    assert len(rows)==len(checkpoints)*n_batches
    keys=[(r['checkpoint_id'],r['batch_id']) for r in rows];assert len(set(keys))==len(keys)
    pools=[{r['batch_id'] for r in rows if r['checkpoint_id']==cp} for cp in checkpoints]
    assert all(len(pool)==n_batches for pool in pools) and all(pool==pools[0] for pool in pools)
    return sorted(pools[0])


def reference_coefficient(loss):return REFERENCES[loss]
def actual_coefficient(checkpoint,loss):return 0. if checkpoint.startswith('XP12') and loss in ('mscl','cscl') else REFERENCES[loss]


def fmt(value):return 'N/A' if value is None else f'{value:.4f}'


def group_rows(rows,keys):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def aggregate_counts(rows,keys,count_fields):
    result=[]
    for values,part in sorted(group_rows(rows,keys).items()):
        output=dict(zip(keys,values))
        for key in count_fields:output[key]=sum(number(row.get(key)) or 0 for row in part)
        output['batch_records']=len(part);result.append(output)
    return result


def priority_screen(gradient_summary):
    winners=[];evidence=[]
    for cp in ('XP04_best','XP04_latest'):
        rows=[r for r in gradient_summary if r['checkpoint_id']==cp and r['group']=='shared_backbone' and r['loss_b'] in ('cc','mscl','cscl')]
        if len(rows)!=3 or any(r['valid_n']!=32 or r['norm_ratio_reference_weighted_median'] is None for r in rows):
            return {'status':'insufficient','mechanism':None,'reason':'XP04两快照共享主干的三项有效共同支持不足32个Batch，不能统一筛选。','evidence':evidence}
        for row in rows:evidence.append({key:row[key] for key in ('checkpoint_id','loss_b','valid_n','conflict_frequency','norm_ratio_reference_weighted_median')})
        conflict=max(r['conflict_frequency'] for r in rows);scale=max(r['norm_ratio_reference_weighted_median'] for r in rows)
        tied=[r['loss_b'] for r in rows if r['conflict_frequency']==conflict and r['norm_ratio_reference_weighted_median']==scale]
        winners.append(tied[0] if len(tied)==1 else None)
    if winners[0] is not None and winners[0]==winners[1]:
        return {'status':'descriptive_priority_only','mechanism':winners[0],
                'reason':'该项在XP04 best/latest均同时具有最高共同主干冲突率和最高reference加权norm比；仅优先核验其尺度/方向假设，不是有害或降系数可获益的证明。','evidence':evidence}
    return {'status':'insufficient','mechanism':None,'reason':'方向冲突与加权尺度排序未在两个XP04快照集中到同一项；证据不足以唯一选CC/MSCL/CSCL或系数。','evidence':evidence}


def summarize_wrappers(rows):
    seen=set();fallback=defaultdict(dict)
    for row in rows:
        key=(row['checkpoint_id'],row['batch_id'],row['loss']);assert key not in seen;seen.add(key)
        flag=str(row['fallback_observed']).lower() in ('true','1');nonfinite=str(row['nonfinite_before_fallback']).lower() in ('true','1')
        assert flag==nonfinite
        if flag:assert number(row['returned_loss'])==0.
        else:
            assert number(row['pre_fallback_loss']) is not None and number(row['returned_loss']) is not None
            assert math.isclose(float(row['pre_fallback_loss']),float(row['returned_loss']),rel_tol=1e-6,abs_tol=1e-7)
        fallback[(row['checkpoint_id'],row['loss'])][row['batch_id']]=flag
    return [{'checkpoint_id':key[0],'loss':key[1],'batches':len(value),'fallback_batches':sum(value.values())} for key,value in sorted(fallback.items())]


def normalize_pairs(rows):
    result=[]
    for row in rows:
        copy=dict(row)
        for key in ('cosine','norm_a_common','norm_b_common','common_tensors','norm_ratio_reference_weighted','norm_ratio_actual_weighted'):
            copy[key]=number(copy.get(key))
        # Retain the core status while explicitly certifying retained common tensor support.
        copy['source_status']=copy.get('status','missing')
        if copy['source_status'] in ('valid','ok','partial_support','partial_nonfinite'):
            copy['status']='valid'
            assert valid_pair(copy), 'core pair has invalid retained common support'
        assert copy['group'] in GROUPS
        result.append(copy)
    return result


def report(summary,registry):
    lines=['# CMERC辅助Loss快照诊断报告','',f"生成时间：{summary['generated_at_utc']}。固定四个Checkpoint×同一组32个训练Batch，共128个case；每case六次模型前向、最多四次独立autograd，eval+grad固定权重，无训练、无optimizer.step、无新增测试前向。",'',
           '## 研究设计与证据边界','',
           '主要比较XP04 best第8轮与latest第40轮；XP12 w/o HSCL best第5轮与latest第40轮仅辅助描述。原训练teacher在epoch入口snapshot后固定，学生在该epoch多步更新；本次每Batch teacher=当前Checkpoint学生，无法还原历史stale teacher。eval关闭Dropout且固定BN。这里测量eval快照梯度，不能称为历史训练梯度；四个离散Checkpoint不是完整优化轨迹。XP12原训练HSCL实际权重为0，其raw HSCL离线梯度和reference 0.15/0.15只是反事实尺度对照，不代表XP12优化了HSCL。','',
           '采样规则预先固定：训练池按对话长度四分层、各8个Batch16，固定seed67137，在可用类别内贪心覆盖七情绪并随机补齐，不看性能/梯度/测试挑Batch。32个Batch不是32个独立Seed，不做p值或统计显著检验。长度分层重新组织Batch会改变HSCL正负集合、padding及原均值移除的输入；冲突频率仅对应这32个固定Batch，不是整个训练池/历史训练冲突频率的估计。采样代表性、单Seed、历史时点和eval模式均限制外推。公共文件只保留per-Batch或聚合，无原始对话ID、逐话语预测/置信度或raw梯度。','',
           f"完成门禁：state completed，独立验收passed，case数{summary['case_count']}；实际forward={summary['execution_counts']['model_forward_calls']}、autograd={summary['execution_counts']['autograd_calls']}，training_runs=0、optimizer_steps=0、test_forwards=0。设备为{registry['host']} GPU{registry['gpu_index']}（`{registry['gpu_uuid']}`）；不将共享资源/离散采样解释为连续独占。",'',
           '## 梯度方向、尺度和有效分母','',
           '每个pair只在共同参数Tensor上计算：双方Tensor均非None、有限且整Tensor范数非零，保留该Tensor全部坐标；不使用逐坐标交集。无共同支持、零、缺失和非有限分别记录，不填0cos。原core的ok/partial_support/partial_nonfinite在保留共同norm有限非零的检验后，由分析副本显式认证status=valid；source_status原样保留，原CSV不改。一个valid pair也可能排除了部分缺失/零/非有限Tensor，见原状态和排除计数。冲突定义cos<0，分母是该Checkpoint/项/组中有效Batch数，不是固定32或所有参数组之和。分类头没有HSCL梯度时明确N/A。','',
           '| Checkpoint | 辅助项 | 组 | valid/总Batch | 负cos数 | 冲突频率 | cos中位数 | reference加权norm/分类norm中位数 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary['gradient_summary']:
        if row['group'] not in ('all','shared_backbone','classification_head') or row['loss_b'] not in ('cc','mscl','cscl'):continue
        lines.append(f"| {row['checkpoint_id']} | {row['loss_b']} | {row['group']} | {row['valid_n']}/{row['total_n']} | {row['conflict_n']} | {fmt(row['conflict_frequency'])} | {fmt(row['cosine_median'])} | {fmt(row['norm_ratio_reference_weighted_median'])} |")
    lines+=['','上述norm比使用相同pair的共同参数支持；完整各Loss各组raw/reference/actual梯度范数与缺失/零/非有限Tensor数另见原始gradient_norms.csv和聚合norm_summary.csv。负cos仅说明该快照局部一阶方向相反，不证明持续损害；norm大不等于有害，正cos也不证明泛化收益。','',
            '同Batch Checkpoint差值（仅两侧都valid的pair；latest−best）：','','| 配对 | 辅助项 | 组 | 有效配对n | cos变化中位数 | cos变化均值 |','|---|---|---|---:|---:|---:|']
    for row in summary['paired_summary']:
        if row['group']=='shared_backbone' and row['loss_b'] in ('cc','mscl','cscl'):
            lines.append(f"| {row['checkpoint_from']} → {row['checkpoint_to']} | {row['loss_b']} | {row['group']} | {row['n']} | {fmt(row['median'])} | {fmt(row['mean'])} |")
    lines+=['','同Batch匹配去掉缺失一侧，不以不匹配Batch替代。XP12轨迹变化不能归因为单独关闭HSCL：其权重轨迹不同，且诊断teacher与历史训练teacher不同。','',
            '## 五路二值关系伪标签与条件分支','',
            't/a/v/intra/inter均是移除前后模型关系标签，不是七类情绪标签。初始化0；主错/移除对为0，主对/移除错为1；二者都对时依真实类别置信度变化判0/1；二者都错保留默认0。因此0不一定表示正确反事实证据，必须连同both_wrong_default0与置信度正/负/相等的条件覆盖解释。','',
            '| Checkpoint | 关系 | label0数 | label1数 | label1比例 | both-wrong默认0数 | 默认0占全部 |','|---|---|---:|---:|---:|---:|---:|']
    for row in summary['pseudo_summary']:
        lines.append(f"| {row['checkpoint_id']} | {row['relation']} | {int(row['label0_count'])} | {int(row['label1_count'])} | {fmt(row['label1_fraction'])} | {int(row['both_wrong_default0_count'])} | {fmt(row['both_wrong_default0_fraction'])} |")
    lines+=['','情绪×关系热图按emotion_class独立分层，展示关系label1比例而非情绪准确率。confidence_aggregate.csv保留subset条件下的主/移除真实类概率均值、差值和正/负/相等计数；其均值按subset话语数加权，不能与无条件均值混用。','',
            '## CKSCL正负对、权重和数值路径','',
            '原clloss.py的CKSCL_base先归一化features、temperature=0.07；相等关系标签mask加1e−7，再去对角。故同类positive权重1+epsilon、同类negative权重−epsilon；异类positive权重epsilon、negative权重1−epsilon。这些是数学写法，FP32舍入后的实际mask值由审计CSV保留。epsilon加在关系mask，不是最终分母的通用稳定项。正负对数量按真实关系相等/不等的有序非对角pair报告，与带epsilon的有效权重质量分开；zero-positive/negative-anchor比例不能从总正负pair数猜测。','',
            'create_class_weight_SCL采用当前Batch二值关系类别逆频率n/n_c，而非情绪类别权重。CKSCL_tav/context把各关系logit乘对应类别权重，求和后按N话语取负均值；MSCL包含三关系，CSCL包含两关系，二者原始尺度不能只按单一关系直接比较。原wrapper遇非有限Loss返回零的fallback与真实zero梯度分开记录。不得把fallback或无梯度当作“损失未执行”。','',
            '数值与fallback计数（按Checkpoint/原wrapper/Batch去重）：','','| Checkpoint | 项 | wrapper Batch数 | fallback数 |','|---|---|---:|---:|']
    for row in summary['fallback_summary']:lines.append(f"| {row['checkpoint_id']} | {row['loss']} | {row['batches']} | {row['fallback_batches']} |")
    lines+=['','原CKSCL_base实际局部数值审计（按同组32个Batch累积计数；极值保留有限记录，非有限单列）：','','| Checkpoint | 关系 | anchor总数 | 非有限logit | 非有限相似度cell | 负mask cell | neg<0 anchor | pos+neg<0 anchor | 零/非有限分母 | pos范围 | neg范围 |','|---|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for row in summary['ckscl_numeric_summary']:
        lines.append(f"| {row['checkpoint_id']} | {row['relation']} | {int(row['n_anchors'])} | {int(row['nonfinite_logits'])} | {int(row['nonfinite_similarity_cells'])} | {int(row['negative_mask_cells'])} | {int(row['negative_neg_sum_anchors'])} | {int(row['negative_denominator_anchors'])} | {int(row['zero_denominator_anchors'])}/{int(row['nonfinite_denominator_anchors'])} | {fmt(row['pos_min'])}–{fmt(row['pos_max'])} | {fmt(row['neg_min'])}–{fmt(row['neg_max'])} |")
    lines+=['','负neg和（负对加权质量）不等于log比值的负分母：分母严格为pos+neg，负/零/非有限分母与neg<0独立计数，不混称。逆频率权重按每Batch中该关系类别存在时统计；缺失类别Batch单列，不当零权重观测混入。权重分布和条件confidence/贡献可用于提出标签覆盖假设，但不直接等于伪标签正确率。见[class_weight_summary.csv](../results/class_weight_summary.csv)。','','各关系positive/negative数量、zero-anchor、epsilon mask质量、n/nc权重、条件confidence与分层contribution见底层CSV；contribution_sum已经含原1/N，不二次除N。contribution_aggregate.csv只取stratum=all；有非有限Batch则完整目标均值保留N/A，有限子集均值单独标记。分层可能重叠，不能跨all/类别/branch直接求和。CKSCL实际pos/neg/logit数值聚合和wrapper审计单独保存，分析不替代原计算。','',
            '## 事实、推断与下一研究优先级','',
            '已证实事实：固定case覆盖、数值有效/缺失分母、上述离散快照方向/尺度、二值关系覆盖及原数学路径审计。尚未证实：某辅助Loss持续损害分类、伪标签等价于情绪质量、降低系数可获益、对其他Seed或测试集成立、论文差距的唯一根因。','',
            '**描述性筛选：** '+summary['priority']['reason'],'',
            ('**下一最值得核验：** '+summary['priority']['mechanism']+' 的尺度/方向假设；先检查其关系分支/数值路径，再另行登记规范验证选模的单机制或单系数对照。' if summary['priority']['mechanism'] else '**下一步：** 当前证据不足以唯一优先CC/MSCL/CSCL或给出新系数；先结合有效分母、默认0条件覆盖和局部数值审计提出可证伪假设，再单独登记。'),'',
            '筛选规则预先写入脚本：仅当同一项在XP04 best/latest均具有最高共同主干冲突率和最高reference加权norm比，且三项均有32个有效Batch时，才列为描述性优先检查。规则不等于优化决策或统计强证据。本批不训练、不step、不测试，不自动修改Loss公式/系数、追加Seed或开展五Seed。','',
            '本阶段零新增测试前向，未计算ECE/CEL；历史字段偶然访问已在先前阶段披露，不声称整个项目从未接触历史测试字段。本分析只读取本批公共聚合CSV，不读取private、混合旧summary、final_test或预测数组。作者课程生成史未知等前序来源限制仍保留。','',
            '## 交付与可追溯性','',
            '[预登记](../reports/PREREGISTRATION.md)、[配置](../configs/registry.json)、[Checkpoint清单](../preflight/checkpoint_inventory.json)、[图契约](../provenance/figure_contract.md)、[独立验收](../provenance/final_verification.json)、[汇总](../results/metrics_summary.json)、[9图18文件及SHA](../results/figures/manifest.json)。','',
            '[梯度方向聚合](../results/gradient_summary.csv)、[梯度范数聚合](../results/norm_summary.csv)、[同Batch配对](../results/paired_checkpoint_deltas.csv)、[伪标签聚合](../results/pseudo_summary.csv)、[分支聚合](../results/branch_summary.csv)、[条件置信度](../results/confidence_aggregate.csv)、[采样覆盖](../results/sample_coverage.csv)、[模型/RNG状态哈希审计](../results/state_audit.csv)、[原CKSCL局部数值](../results/ckscl_numeric.csv)、[原wrapper回退](../results/ckscl_wrapper_audit.csv)。','']
    return '\n'.join(lines)


def main():
    registry=read(ROOT/'configs/registry.json');state=read(ROOT/'state/status.json');verification=read(ROOT/'provenance/final_verification.json')
    require_complete(state,verification,registry['expected_cases'])
    assert registry['expected_cases']==128 and registry['n_batches']==32 and registry['batch_size']==16
    assert state['training_runs']==verification['training_runs']==0
    assert state['model_forward_calls']==verification['model_forward_calls']==768 and state['autograd_calls']==verification['autograd_calls']<=512
    assert tuple(r['checkpoint_id'] for r in registry['checkpoints'])==CHECKPOINTS
    files={name:ROOT/'results'/(name+'.csv') for name in PUBLIC_FILES}
    files.update({name:ROOT/'results'/(name+'.csv') for name in ('ckscl_numeric','ckscl_wrapper_audit')})
    data={name:csv_read(path) for name,path in files.items()}
    for values in data.values():assert_public(values)
    batch_ids=case_coverage(data['case_summary'],CHECKPOINTS,32)
    case_keys={(cp,batch) for cp in CHECKPOINTS for batch in batch_ids}
    pair_keys=[(r['checkpoint_id'],r['batch_id'],r['group'],r['loss_b']) for r in data['gradient_pairs'] if r['loss_a']=='classification']
    assert len(pair_keys)==len(set(pair_keys))
    assert {(cp,batch,group,loss) for cp,batch in case_keys for group in GROUPS for loss in ('cc','mscl','cscl')}.issubset(set(pair_keys))
    norm_keys=[(r['checkpoint_id'],r['batch_id'],r['group'],r['loss']) for r in data['gradient_norms']]
    assert len(norm_keys)==len(set(norm_keys))
    assert {(cp,batch,group,loss) for cp,batch in case_keys for group in GROUPS for loss in LOSSES}.issubset(set(norm_keys))
    for row in data['case_summary']:
        for key in ('parameters_buffers_unchanged','RNG_unchanged','grad_buffers_unchanged'):assert str(row[key]).lower() in ('true','1')
    pairs=normalize_pairs(data['gradient_pairs']);grouped=group_rows(pairs,('checkpoint_id','loss_a','loss_b','group'));gradients=[]
    for keys,part in sorted(grouped.items()):
        out=dict(zip(('checkpoint_id','loss_a','loss_b','group'),keys));stats=summarize_pairs(part)
        for key in ('norm_ratio_reference_weighted','norm_ratio_actual_weighted'):
            stats.update({key+'_'+name:value for name,value in distribution([r[key] for r in part if valid_pair(r)]).items()})
        stats['status_counts']=json.dumps(stats['status_counts'],sort_keys=True)
        stats['source_status_counts']=json.dumps(dict(Counter(r['source_status'] for r in part)),sort_keys=True)
        for key in ('excluded_missing_tensors','excluded_zero_tensors','excluded_nonfinite_tensors'):
            stats[key]=sum(number(r.get(key)) or 0 for r in part)
        gradients.append({**out,**stats})
    norms=[]
    for keys,part in sorted(group_rows(data['gradient_norms'],('checkpoint_id','loss','group')).items()):
        out=dict(zip(('checkpoint_id','loss','group'),keys));out['status_counts']=json.dumps(dict(Counter(r['status'] for r in part)),sort_keys=True)
        for key in ('raw_l2','reference_weighted_l2','actual_weighted_l2'):
            out.update({key+'_'+name:value for name,value in distribution([r.get(key) for r in part if r.get('status') not in ('missing','no_gradient','nonfinite')]).items()})
        for key in ('missing_tensors','zero_tensors','nonfinite_tensors','nonfinite_coordinates','present_tensors','total_tensors'):out[key]=sum(number(r.get(key)) or 0 for r in part)
        for row in part:
            if row['loss'] in LOSSES:
                assert math.isclose(float(row['reference_coefficient']),reference_coefficient(row['loss']))
                assert math.isclose(float(row['actual_coefficient']),actual_coefficient(row['checkpoint_id'],row['loss']))
        norms.append(out)
    paired=paired_deltas(pairs,'XP04_best','XP04_latest')+paired_deltas(pairs,'XP12_best','XP12_latest');paired_summary=[]
    for keys,part in sorted(group_rows(paired,('checkpoint_from','checkpoint_to','loss_a','loss_b','group')).items()):
        paired_summary.append({**dict(zip(('checkpoint_from','checkpoint_to','loss_a','loss_b','group'),keys)),**distribution([r['delta_cosine'] for r in part])})
    branches=aggregate_counts(data['branch_counts'],('checkpoint_id','relation','branch','pseudo_expected'),('n_utterances','n_delta_positive','n_delta_zero','n_delta_negative'))
    pseudo=[]
    for (cp,relation),part in sorted(group_rows(data['pseudo_counts'],('checkpoint_id','relation')).items()):
        counts={label:sum(number(r['n_pseudo']) or 0 for r in part if int(r['pseudo_label'])==label) for label in (0,1)};n=sum(counts.values())
        default=sum(r['n_utterances'] for r in branches if r['checkpoint_id']==cp and r['relation']==relation and r['branch']=='both_wrong_default0')
        pseudo.append({'checkpoint_id':cp,'relation':relation,'label0_count':counts[0],'label1_count':counts[1],'n_utterances':n,'label1_fraction':ratio(counts[1],n),'both_wrong_default0_count':default,'both_wrong_default0_fraction':ratio(default,n)})
    classes=aggregate_counts(data['class_relations'],('checkpoint_id','relation','emotion_class','pseudo_label'),('n_utterances','n_main_correct','n_removed_correct','n_default_zero'))
    confidence=[]
    for keys,part in sorted(group_rows(data['confidence_summary'],('checkpoint_id','relation','subset')).items()):
        out=dict(zip(('checkpoint_id','relation','subset'),keys));out['n_utterances']=sum(number(r['n_utterances']) or 0 for r in part)
        for key in ('mean_main_gt_probability','mean_removed_gt_probability','mean_gt_probability_delta'):
            usable=[r for r in part if number(r.get(key)) is not None and (number(r['n_utterances']) or 0)>0]
            out[key]=ratio(sum(float(r[key])*float(r['n_utterances']) for r in usable),sum(float(r['n_utterances']) for r in usable))
        for key in ('n_delta_positive','n_delta_zero','n_delta_negative'):out[key]=sum(number(r.get(key)) or 0 for r in part)
        confidence.append(out)
    # These flags come from the original wrapper return trace, never a recomputed loss.
    wrapper_keys=[(r['checkpoint_id'],r['batch_id'],r['loss']) for r in data['ckscl_wrapper_audit']]
    assert len(wrapper_keys)==len(set(wrapper_keys))==256
    assert set(wrapper_keys)=={(cp,batch,loss) for cp,batch in case_keys for loss in ('mscl','cscl')}
    fallback_summary=summarize_wrappers(data['ckscl_wrapper_audit'])
    numeric_keys=[(r['checkpoint_id'],r['batch_id'],r['relation']) for r in data['ckscl_numeric']]
    assert len(numeric_keys)==len(set(numeric_keys))==640
    assert set(numeric_keys)=={(cp,batch,rel) for cp,batch in case_keys for rel in ('t','a','v','intra','inter')}
    numeric=[]
    for keys,part in sorted(group_rows(data['ckscl_numeric'],('checkpoint_id','relation')).items()):
        output=dict(zip(('checkpoint_id','relation'),keys));output['batch_records']=len(part)
        for key in ('n_anchors','nonfinite_logits','nonfinite_similarity_cells','negative_mask_cells','negative_neg_sum_anchors','negative_denominator_anchors','zero_denominator_anchors','nonfinite_denominator_anchors'):output[key]=sum(number(r.get(key)) or 0 for r in part)
        for key in ('pos_min','neg_min'):
            values=[number(r.get(key)) for r in part if number(r.get(key)) is not None];output[key]=min(values) if values else None
        for key in ('pos_max','neg_max'):
            values=[number(r.get(key)) for r in part if number(r.get(key)) is not None];output[key]=max(values) if values else None
        numeric.append(output)
    weights=[]
    for keys,part in sorted(group_rows(data['pseudo_counts'],('checkpoint_id','relation','pseudo_label')).items()):
        present=[r for r in part if (number(r['n_pseudo']) or 0)>0]
        weights.append({**dict(zip(('checkpoint_id','relation','pseudo_label'),keys)),'present_batches':len(present),'absent_batches':len(part)-len(present),
                        **{'class_weight_'+name:value for name,value in distribution([r['class_weight'] for r in present]).items()}})
    contributions=[]
    # Relation contribution_sum already includes the original 1/N. Only stratum=all is additive across relations.
    all_contributions=[r for r in data['contribution_summary'] if r['stratum']=='all']
    for keys,part in sorted(group_rows(all_contributions,('checkpoint_id','relation')).items()):
        values=[number(r['contribution_sum']) for r in part]
        finite=[v for v in values if v is not None]
        contributions.append({**dict(zip(('checkpoint_id','relation'),keys)),'batch_records':len(part),'finite_batch_records':len(finite),
                              'all_batches_finite':len(finite)==len(part),'batch_mean_relation_contribution':statistics.mean(finite) if len(finite)==len(part) and finite else None,
                              'finite_batch_subset_mean_not_full_objective':statistics.mean(finite) if finite else None,
                              'status_counts':json.dumps(dict(Counter(r['status'] for r in part)),sort_keys=True)})
    priority=priority_screen(gradients)
    inputs=list(files.values())+[ROOT/name for name in ('configs/registry.json','state/status.json','provenance/final_verification.json','preflight/checkpoint_inventory.json','provenance/figure_contract.md','scripts/analyze_diagnostic.py','scripts/plot_diagnostic.py','scripts/test_diagnostic_analysis.py')]
    inputs += [p for p in [ROOT/'preflight/analysis_tests.json',ROOT/'provenance/frozen_execution.json',ROOT/'reports/PREREGISTRATION.md'] if p.exists()]
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'completed':True,'verification_status':'passed','case_count':128,'checkpoint_ids':list(CHECKPOINTS),
             'execution_counts':{key:state[key] for key in ('training_runs','optimizer_steps','test_forwards','model_forward_calls','autograd_calls')},
             'gradient_summary':gradients,'norm_summary':norms,'paired_summary':paired_summary,'pseudo_summary':pseudo,'fallback_summary':fallback_summary,'ckscl_numeric_summary':numeric,'class_weight_summary':weights,'contribution_aggregate':contributions,'priority':priority,
             'case_status_counts':dict(Counter(r.get('status','not_recorded') for r in data['case_summary'])),
             'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'statistics':'descriptive only; batches are not independent seeds; no p values',
             'scope':'public per-batch/aggregate train-only eval snapshot diagnostic; no private/checkpoint/test files read'}
    prose=report(summary,registry)
    output=ROOT/'results'
    for name,values in [('gradient_summary',gradients),('norm_summary',norms),('analysis_gradient_pairs',pairs),('paired_checkpoint_deltas',paired),('paired_summary',paired_summary),('pseudo_summary',pseudo),('branch_summary',branches),('class_relation_summary',classes),('confidence_aggregate',confidence),('fallback_summary',fallback_summary),('ckscl_numeric_summary',numeric),('class_weight_summary',weights),('contribution_aggregate',contributions)]:
        assert_public(values);csv_write(output/(name+'.csv'),values)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (ROOT/'reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md').write_text(prose)
    print(json.dumps({'cases':128,'priority':priority,'new_training_runs':0,'optimizer_steps':0,'test_forwards':0},ensure_ascii=False))


if __name__=='__main__':main()
