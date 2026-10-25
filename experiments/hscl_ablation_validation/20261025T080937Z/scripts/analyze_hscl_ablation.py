"""Strict CPU analysis of one preregistered HSCL ablation, with sanitized historical controls.

No model execution, scheduling, mixed supplements, test metric files or predictions
are read. Output is generated only after the single new run passes completion gates.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
LOSS=('classification_loss','cc_loss','mscl_loss','cscl_loss','total_loss')
CURVE=('train_weighted_f1','validation_weighted_f1','train_accuracy','validation_accuracy',
       'train_loss','validation_loss','sample_ratio','train_dialogues','max_gpu_memory_mb')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def csv_write(path,data):
    fields=list(dict.fromkeys(key for row in data for key in row))
    with Path(path).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(data)


def first_max(epochs):
    assert epochs and all(row['validation_weighted_f1']==round(row['validation_weighted_f1'],2) for row in epochs)
    return max(epochs,key=lambda row:row['validation_weighted_f1'])


def last10_mean(epochs,key='validation_weighted_f1'):
    assert len(epochs)>=10
    return statistics.mean(row[key] for row in epochs[-10:])


def weighted(row,args):
    return {'weighted_cc_loss':row['cc_loss']*args['rank_coff'] if args['calibrate'] else 0.,
            'weighted_mscl_loss':row['mscl_loss']*args['mscl_coff'] if args['contrastlearning'] else 0.,
            'weighted_cscl_loss':row['cscl_loss']*args['cscl_coff'] if args['contrastlearning'] else 0.}


def verify_epochs(epochs,args,split_hash,new=False):
    assert len(epochs)==40 and [r['epoch'] for r in epochs]==list(range(1,41))
    for row in epochs:
        assert row['selection_protocol']=='validation' and row['selection_split']=='validation'
        assert row['rng_policy']=='continuous' and row['batch_policy']=='capacity_safe' and row['split_hash']==split_hash
        assert row['selection_weighted_f1']==row['validation_weighted_f1']==round(row['validation_weighted_f1'],2)
        assert all(math.isfinite(row[key]) for key in LOSS+CURVE[:6])
        assert math.isclose(row['classification_loss']+sum(weighted(row,args).values()),row['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
        if new:assert row['test_weighted_f1'] is None and row['test_accuracy'] is None
    return first_max(epochs)


def stats(variant,source,epochs,watch,args):
    seed=args['seed'];dropout=args['dropout']
    best=first_max(epochs);tail=epochs[-10:]
    result={'variant':variant,'seed':seed,'lr':args['lr'],'batch_size':args['batch_size'],'validation_batch_size':args['batch_size'],
            'dropout':dropout,'rec_dropout_argument':args['rec_dropout'],'source':source,'epochs':40,
            'best_epoch':best['epoch'],'best_validation_wf1':best['validation_weighted_f1'],
            'last10_validation_wf1':last10_mean(epochs),'last10_validation_wf1_temporal_sd':statistics.stdev(row['validation_weighted_f1'] for row in tail),
            'best_validation_accuracy':best['validation_accuracy'],'last10_validation_accuracy':last10_mean(epochs,'validation_accuracy'),
            'best_train_weighted_f1':best['train_weighted_f1'],'last10_train_weighted_f1':last10_mean(epochs,'train_weighted_f1'),
            'best_train_accuracy':best['train_accuracy'],'last10_train_accuracy':last10_mean(epochs,'train_accuracy'),
            'best_train_classification_loss':best['classification_loss'],'best_validation_classification_loss':best['validation_loss'],
            'last10_train_classification_loss':last10_mean(epochs,'classification_loss'),'last10_validation_classification_loss':last10_mean(epochs,'validation_loss'),
            'max_epoch_allocated_mib':max(row['max_gpu_memory_mb'] for row in epochs)}
    for key in LOSS:
        result['best_train_'+key]=best[key];result['last10_train_'+key]=last10_mean(epochs,key)
        result['best_validation_'+key]=None if key!='classification_loss' else best['validation_loss']
        result['last10_validation_'+key]=None if key!='classification_loss' else last10_mean(epochs,'validation_loss')
    for prefix,value in [('best',best),('last10',{key:last10_mean(epochs,key) for key in LOSS})]:
        for key,number in weighted(value,args).items():result[prefix+'_train_'+key]=number
    if watch:
        train=[r for r in watch if r['split']=='train'];val=[r for r in watch if r['split']=='validation']
        assert len(watch)==80 and len(train)==len(val)==40
        for part in (train,val):assert [r['epoch'] for r in part]==list(range(1,41))
        assert all(r['gradients_finite'] is True and r['first_step_parameters_changed'] is True for r in train)
        assert all(round(v['weighted_f1'],2)==e['validation_weighted_f1'] and round(v['accuracy'],2)==e['validation_accuracy'] for e,v in zip(epochs,val))
        result.update(gradient_epoch_checks=40,gradient_abnormalities=0,gradient_l2_mean=statistics.mean(r['gradient_l2_mean'] for r in train),gradient_l2_max=max(r['gradient_l2_max'] for r in train))
        assert math.isfinite(result['gradient_l2_mean']) and math.isfinite(result['gradient_l2_max'])
        for key in LOSS:
            result['best_validation_'+key]=val[best['epoch']-1]['loss_components'][key]
            result['last10_validation_'+key]=statistics.mean(r['loss_components'][key] for r in val[-10:])
        result['classification_loss_precision']='watcher_full_batch_mean'
    else:result.update(gradient_epoch_checks=0,gradient_abnormalities=None,gradient_l2_mean=None,gradient_l2_max=None)
    if not watch:result['classification_loss_precision']='historical_validation_loss_4dp_train_component_full_precision'
    result['tail_window_sd_ddof']=1
    result['calibrate']=args['calibrate'];result['contrastlearning']=args['contrastlearning'];result['courselearning']=args['courselearning']
    for split in ('train','validation'):
        part=[r for r in watch if r['split']==split]
        result['best_'+split+'_macro_f1']=part[best['epoch']-1]['macro_f1'] if part else None
        result['last10_'+split+'_macro_f1']=statistics.mean(r['macro_f1'] for r in part[-10:]) if part else None
    return result



def fmt(value,signed=False):return "未记录" if value is None else format(value,"+.4f" if signed else ".4f")


def verify_watch(watch,epochs,args=None):
    assert len(watch)==80 and len(epochs)==40
    for split in ('train','validation'):
        part=[r for r in watch if r['split']==split]
        assert [r['epoch'] for r in part]==list(range(1,41))
        for observed,epoch in zip(part,epochs):
            assert observed['weighted_f1']==epoch[split+'_weighted_f1']
            assert observed['accuracy']==epoch[split+'_accuracy']
            assert observed['batches']>0
            component=observed['loss_components']
            assert all(math.isfinite(component[key]) for key in LOSS)
            if args:
                assert math.isclose(component['classification_loss']+sum(weighted(component,args).values()),component['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
            if split=='train':
                assert observed['gradients_finite'] is True and observed['first_step_parameters_changed'] is True
                assert math.isfinite(observed['gradient_l2_mean']) and math.isfinite(observed['gradient_l2_max'])




def qualification(best_delta,tail_delta):
    if best_delta<=0:return 'no_best_improvement'
    return 'positive' if tail_delta>0 else 'mixed_incomplete'


def validate_pair(actual,old):
    allowed={'contrastlearning','skip_final_test','output_dir','provenance'}
    assert {k:v for k,v in actual.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
    assert actual['contrastlearning'] is False and old['contrastlearning'] is True and actual['skip_final_test'] is True
    assert actual['calibrate'] is old['calibrate'] is True and actual['courselearning'] is old['courselearning'] is True
    assert (actual['Dataset'],actual['seed'],actual['lr'],actual['batch_size'],actual['dropout'],actual['rec_dropout'])==('MELD',67137,.0001,16,.4,.1)
    assert (actual['rank_coff'],actual['mscl_coff'],actual['cscl_coff'])==(.002,.15,.15)


def require_complete(status):
    assert status['phase']=='completed' and status['counts']=={'train':40,'validation':40,'test':0}
    assert status['test_forward_passes']==0 and status['no_final_test_artifacts']


def validation_component(epoch,observed):
    return observed['loss_components'] if observed else {key:epoch['validation_loss'] if key=='classification_loss' else None for key in LOSS}


def verify_final(value,run_id,selected_epoch,best):
    assert value['status']=='passed' and value['formal_runs']==1 and value['test_forwards']==0
    assert value['frozen_sources_and_inputs_unchanged'] is True
    run=value['runs'][run_id]
    assert run['status']=='passed' and run['epochs']==40 and run['test_forwards']==0
    assert run['selected_epoch']==selected_epoch and run['validation_best_wf1']==best and run['loss_and_gradient_finite'] is True
    assert run['contrastlearning'] is False and run['calibrate'] is True and run['courselearning'] is True and run['dropout']==.4
    assert set(run['checkpoints'])=={'best.pt','latest.pt'}
    for name,epoch in [('best.pt',selected_epoch),('latest.pt',40)]:
        c=run['checkpoints'][name];assert c['epoch']==epoch and c['model_and_adam_finite'] and c['full_rng_present']
    return run


def compare(control,new,role):
    bd=new['best_validation_wf1']-control['best_validation_wf1'];td=new['last10_validation_wf1']-control['last10_validation_wf1']
    return {'control':'XP04_full','comparison':new['variant'],'role':role,'seed':67137,
            'control_best_validation_wf1':control['best_validation_wf1'],'comparison_best_validation_wf1':new['best_validation_wf1'],
            'control_best_epoch':control['best_epoch'],'comparison_best_epoch':new['best_epoch'],
            'control_last10_validation_wf1':control['last10_validation_wf1'],'comparison_last10_validation_wf1':new['last10_validation_wf1'],
            'delta_best_validation_wf1':bd,'delta_last10_validation_wf1':td,
            'candidate_category':qualification(bd,td) if role=='primary' else None}


def advice(category):
    if category=='positive':return '最佳与末段均严格提高，提供该Seed关闭HSCL的正向线索，值得在独立目录预登记第二Seed确认；本次不自动追加。'
    if category=='mixed_incomplete':return '最佳提高但末段下降或不变，证据混合/不足；暂缓第二Seed及五Seed扩展，先解释已有轨迹，不更换Seed或按测试择优。'
    return '最佳验证未提高，不支持该Seed的主分类指标改善；不扩大五Seed或自动追加其他Seed/超参数，末段变化仅作辅助线索。'


def environment_versions(environment):
    return (environment['python'].split(' | ')[0],environment['torch'],environment['torch_cuda'],environment['pyg'])


def report(summary,registry,origin,environment,verification):
    protocols=summary['per_run'];primary=summary['comparison'][0];run=registry['runs'][0];verified=verification['runs'][run['run_id']]
    resource=verification.get('resource_evidence',{});policy=registry['resource_policy']
    python_version,torch_version,cuda_version,pyg_version=environment_versions(environment)
    lines=['# CMERC HSCL消融诊断报告','',f"生成时间：{summary['generated_at_utc']}。本批仅新增MELD seed67137 w/o HSCL完整40轮1次训练。XP04完整CMERC为主要历史控制；XP06 w/o CC仅作为已有描述性参照，不是本批新增训练，也不将XP12与XP06差异解释为单变量效应。",'',
           '**主要判定：** `'+primary['candidate_category']+'`。'+advice(primary['candidate_category']),'',
           '## 科学设置、源码开关与选模','',
           '唯一登记科学改动：contrastlearning=True→False；保留calibrate=True、courselearning=True。dropout=0.4、rec_dropout CLI元数据0.1、MELD seed67137、split_seed1475、lr=0.0001、L2=0.00003、Adam、Batch16/16、40 epochs、GRU/ATen、hyper/direct/concat_DHT、BN、L3/K3、CC系数0.002、MSCL/CSCL登记系数0.15/0.15、CL epoch_ratio0.4/scheduler_steps1、continuous RNG与capacity_safe保持XP04。工程skip_final_test=True取消最终测试，不改变训练数学。','',
           'train_cm.py:656使用contrastlearning or calibrate控制共享分支。本批CC仍开，故:667–674的五个移除辅助前向、:689–707的CC、:708–771的伪标签及原始MSCL/CSCL仍执行；原始HSCL不应伪造为零。:236–259仅按实际开关组合目标，关闭HSCL后effective MSCL=CSCL=0，总Loss=classification+0.002×rawCC。:1270课程学习独立控制，未因关闭HSCL而关闭CL。辅助teacher保持epoch入口快照/eval，网络和Loss公式未修改。','',
           'HSCL主要是确定性张量计算，采样使用固定private RNG，但这不等于已经证明所有随机调用与全程RNG轨迹匹配。只依据实际预检/首Batch审计报告初始化、RNG和梯度证据；关闭目标项会改变后续参数与预测，不要求后续数值轨迹相同，也不人为改源码强制RNG。','',
           f"train_cm.py SHA256 `{origin['training_source_sha256']}`；来源`{origin['copied_from']}`，与11组12个源码字节相同标记`{origin['training_sources_identical_to11']}`；Git HEAD `{origin['head_commit']}`。",'',
           f"Python {python_version}、PyTorch {torch_version}、CUDA build {cuda_version}、PyG {pyg_version}，复用环境、无新增依赖。",'',
           '选模为完整固定validation池话语级汇总W-F1，百分比round2后严格提高才保存，最高并列取首次，不是Batch F1平均。末段固定31–40轮均值，时间窗SD使用ddof=1，不是跨Seed波动或统计显著证据。positive要求最佳和末段均严格提高；最佳升而末段降或不变为mixed_incomplete；最佳未升为no_best_improvement。','',
           '## 三臂实际训练/验证结果','',
           '| 臂 | 最佳val W-F1 % | Epoch | 末10轮val W-F1 % | 末段SD | 最佳轮val ACC % | 末段val ACC % | 最佳轮val Macro-F1 % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in protocols:lines.append(f"| {r['variant']} | {r['best_validation_wf1']:.2f} | {r['best_epoch']} | {r['last10_validation_wf1']:.3f} | {r['last10_validation_wf1_temporal_sd']:.4f} | {r['best_validation_accuracy']:.2f} | {r['last10_validation_accuracy']:.3f} | {fmt(r['best_validation_macro_f1'])} |")
    lines+=['','| 相对XP04完整模型 | 用途 | 最佳变化 pp | 末段变化 pp | 判据 |','|---|---|---:|---:|---|']
    for r in summary['comparison']:lines.append(f"| {r['comparison']} | {r['role']} | {r['delta_best_validation_wf1']:+.2f} | {r['delta_last10_validation_wf1']:+.3f} | {r['candidate_category'] or '仅描述，不做候选判定'} |")
    lines+=['','单Seed不作显著性检验；验证峰值与末段是同一相关轨迹的两个摘要，不当独立重复证据。XP06历史运行设备/时点与本批不同，且干预因素不同，描述性列不提供“CC优于/劣于HSCL”的严格归因。','',
            '## Loss、梯度与训练/验证轨迹','',
            '| 臂 | train分类Loss 最佳→末段 | val分类Loss 最佳→末段 | 实际optimizer.step | 梯度异常Epoch |','|---|---|---|---|---|']
    for r in protocols:
        lines.append(f"| {r['variant']} | {fmt(r['best_train_classification_loss'])} → {fmt(r['last10_train_classification_loss'])} | {fmt(r['best_validation_classification_loss'])} → {fmt(r['last10_validation_classification_loss'])} | {r['optimizer_steps_observed'] if r['optimizer_steps_observed'] is not None else '未记录；CPU重建'+str(r['optimizer_steps_reconstructed'])} | {r['gradient_abnormalities'] if r['gradient_epoch_checks'] else '历史未记录'} |")
    for split,title in [('train','训练'),('validation','验证')]:
        lines+=['',title+'原始Loss（最佳验证轮→末10轮均值）：','','| 臂 | 分类 | CC | MSCL | CSCL | 总Loss |','|---|---|---|---|---|---|']
        for r in protocols:lines.append('| '+r['variant']+' | '+' | '.join(fmt(r['best_'+split+'_'+key])+' → '+fmt(r['last10_'+split+'_'+key]) for key in LOSS)+' |')
    lines+=['','本批raw MSCL/CSCL实测非零批次数为'+str(summary['new_raw_hscl_nonzero_batches'])+'，有效加权HSCL全部0；每个实际Batch核对目标class+0.002CC。旧档案validation辅助/总Loss未记录，保持None而不是填零。新臂分类Loss取watcher高精度均值，历史validation分类Loss仅4位小数日志。各项effective loss保存在loss_curves.csv；完整训练/验证W-F1、ACC、分类与总Loss以及raw/effective辅助Loss在图和CSV中。','']
    for r in protocols:
        lines.append(f"- {r['variant']}：最佳验证轮→末段，train分类Loss {r['last10_train_classification_loss']-r['best_train_classification_loss']:+.4f}，val分类Loss {r['last10_validation_classification_loss']-r['best_validation_classification_loss']:+.4f}，val W-F1 {r['last10_validation_wf1']-r['best_validation_wf1']:+.4f} pp。")
    lines+=['','train为在线训练态、val为固定epoch权重eval，不把二者差当严格泛化gap。train Loss下降同时val Loss上升/验证末段回落仅为过拟合或目标竞争线索；W-F1与Loss方向不一致时不宣称已解决过拟合。有限梯度和更新只证明数值执行，不排除伪目标质量或泛化问题。首Batch梯度审计仅预检额外autograd.grad，不是正式训练的新算法，也不能代替多Seed机制证据。','',
            '## 预检、资源和完整验收','',
            f"本批实际设备 {registry['host']} GPU{registry['gpu_index']}（`{registry['gpu_uuid']}`）；XP04同Seed控制为GPU0 UUID GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248，{'同物理卡' if registry['gpu_uuid']=='GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248' else '不同物理卡'}。共享资源、历史时点及CUDA scatter非确定性限制严格因果和速度解释，不宣称连续独占。准入free>={policy['admission_min_free_mib']}MiB/util<={policy['admission_max_utilization_percent']}%，{policy['samples_required']}次间隔{policy['sample_seconds']}秒，外部PID只披露、不终止。",'',
            f"独立验收`{verification['status']}`，本批唯一40train/40validation/test0，实际训练Batch {verified['train_batches']}；best/latest为第{verified['checkpoints']['best.pt']['epoch']}/{verified['checkpoints']['latest.pt']['epoch']}轮，模型/Adam有限且Python/NumPy/Torch/CUDA RNG完整，核对实际args/provenance、源/数据与采样顺序，历史保全{sum(verification.get('historical_files_unchanged',{}).values())}文件。",'',
            f"资源证据`{resource.get('status','unknown')}`：{resource.get('resource_samples','未记录')}次采样，既有共驻PID样本{resource.get('allowed_preexisting_pid_samples','未记录')}次、新PID样本{resource.get('new_foreign_pid_samples','未记录')}次、查询错误{resource.get('resource_query_errors','未记录')}次，正式最低空闲{resource.get('minimum_free_mib_during_formal','未记录')}MiB/最大利用率{resource.get('maximum_utilization_percent','未记录')}%；离散采样不能证明全程资源隔离。",'',
            '首Batch预检标量审计摘要（只有实际保存的字段，完整梯度/RNG证据见链接；只比较本批同一wo-HSCL probe与正式首步，不含额外full-HSCL probe，未与XP04历史初始化/RNG逐位配对）：','', '```json',json.dumps(summary['preflight_scalar_evidence'],ensure_ascii=False,indent=2),'```','',
            '## 测试与来源透明边界','',
            '本次零新增test loader遍历/零新增测试前向，未计算ECE/CEL，不根据测试选择配置或Epoch。原综合pickle与test loader元数据仍载入，零前向不等于完全未接触含测试数据的文件。历史XP06 README初读曾可见旧测试列，访问说明只记录事件、不引用数值，见[历史指标访问说明](../provenance/historical_metric_access_note.json)；不宣称“全程未读旧测试”。本分析仅读取净化训练/验证refs及本次记录，绝不读取历史mixed summary、final_test或预测数组。','',
            '作者fixed_author课程包含validation键且生成史未知；同输入配对不等于全链条validation-free。代码、历史Checkpoint、日志、结果均不覆盖。','',
            '## 已证实事实、推断和下一步','',
            '已证实：实际开关/数据/选模、40轮轨迹、原始Loss保留而有效HSCL置零、完整验收与数值执行。尚未证实：HSCL就是论文F1差距唯一根因、其伪标签质量、跨Seed普遍改善或测试泛化。若正向，仅支持该Seed消融线索；若无主指标提高，也不能断言HSCL在所有配置均无作用。','',
            '**直接建议：** '+advice(primary['candidate_category'])+' 本批不自动启动w/o CL、第二Seed、五Seed、其他超参数或测试。','',
            '[预登记](../reports/PREREGISTRATION.md)、[源码协议审计](../reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[配置](../configs/registry.json)、[源码来源](../provenance/source_origin.json)、[环境](../provenance/environment.json)、[冻结](../provenance/frozen_execution.json)、[XP04净化refs](../provenance/xp04_validation_references.json)、[XP06净化refs](../provenance/xp06_validation_reference.json)、[容量预检](../preflight/capacity_cpu.json)、[预检门禁](../preflight/gate.json)、[梯度审计](../preflight/'+run['run_id']+'/observer/gradient_audit.json)、[完整验收](../provenance/final_verification.json)。','',
            '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)、[9图18文件及SHA](../results/figures/manifest.json)。','',
            f'实验根目录：`{ROOT}`。仓库根副本由父级在验收后复制并调整链接。','']
    return '\n'.join(lines)


def main():
    registry=read(ROOT/'configs/registry.json');assert registry['formal_run_count']==len(registry['runs'])==1;run=registry['runs'][0]
    assert (run['run_id'],run['dataset'],run['seed'],run['lr'],run['epochs'],run['batch_size'],run['dropout'])==('MELD_wo_hscl_seed67137','MELD',67137,.0001,40,16,.4)
    result=ROOT/'results'/run['run_id'];status=read(result/'status.json');require_complete(status)
    reference=read(ROOT/'provenance/xp04_validation_references.json')['MELD_67137'];description=read(ROOT/'provenance/xp06_validation_reference.json')
    directory=Path(run['run_dir']);manifest=read(directory/'run_manifest.json');args=manifest['args'];validate_pair(args,reference['args'])
    assert manifest['split']==reference['split']==description['split'] and manifest['data']==reference['data']==description['data']
    capsule=read(ROOT/'provenance/frozen_execution.json')
    for key,value in manifest['source']['files'].items():assert sha(ROOT/'code_xp04'/key)==capsule['files']['code_xp04/'+key]==value
    for key,value in manifest['data'].items():assert sha(ROOT/'code_xp04'/key)==value
    assert not (directory/'final_test.json').exists() and not list(result.glob('*.npz'))
    assert all((directory/name).is_file() for name in ('best.pt','latest.pt'))
    epochs=rows(directory/'metrics.jsonl');watch=rows(result/'watch_epochs.jsonl');batches=rows(result/'watch_batches.jsonl')
    verify_epochs(epochs,args,run['split_hash'],True);verify_watch(watch,epochs,args)
    for ref in (reference,description):verify_epochs(ref['epochs'],ref['args'],run['split_hash'])
    old=stats('XP04_full','historical_primary',reference['epochs'],[],reference['args'])
    descriptive=stats('XP06_wo_cc','historical_descriptive_only',description['epochs'],[],description['args'])
    new=stats('XP12_wo_hscl','new_single_formal_run',epochs,watch,args)
    assert (old['best_epoch'],old['best_validation_wf1'])==(reference['best_epoch'],reference['best_validation_wf1'])
    assert math.isclose(old['last10_validation_wf1'],reference['last10_validation_wf1'],abs_tol=1e-12)
    assert status['selected_epoch']==new['best_epoch'] and status['selection_weighted_f1']==new['best_validation_wf1']
    final=read(ROOT/'provenance/final_verification.json');verified=verify_final(final,run['run_id'],new['best_epoch'],new['best_validation_wf1'])
    capacity=read(ROOT/'preflight/capacity_cpu.json');assert capacity['status']=='passed'
    caps={r['epoch']:r for r in capacity['records'] if r['run_id']==run['run_id']};assert set(caps)==set(range(1,41))
    nonzero={'mscl_loss':0,'cscl_loss':0}
    for epoch in range(1,41):
        part=[r for r in batches if r['epoch']==epoch];assert [r['batch'] for r in part]==list(range(1,len(part)+1))
        assert len(part)==caps[epoch]['train_batches']==next(w for w in watch if w['epoch']==epoch and w['split']=='train')['batches']
        assert next(w for w in watch if w['epoch']==epoch and w['split']=='validation')['batches']==caps[epoch]['validation_batches']
        for b in part:
            assert b['gradients_finite'] is True and math.isfinite(b['gradient_l2']) and all(math.isfinite(b[k]) for k in LOSS)
            effective=weighted(b,args);assert effective['weighted_mscl_loss']==effective['weighted_cscl_loss']==0
            assert math.isclose(b['classification_loss']+effective['weighted_cc_loss'],b['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
            for key in nonzero:nonzero[key]+=int(b[key]!=0)
            for key,value in effective.items():
                if key in b:assert math.isclose(b[key],value,rel_tol=1e-6,abs_tol=1e-6)
    assert len(batches)==sum(c['train_batches'] for c in caps.values())==verified['train_batches']
    assert all(nonzero[key]>0 for key in nonzero)
    protocols=[old,descriptive,new];datasets=[(old,reference['epochs'],[],reference['args']),(descriptive,description['epochs'],[],description['args']),(new,epochs,watch,args)]
    curves=[];losses=[];steps=[]
    for item,data,observations,config in datasets:
        lookup={(r['split'],r['epoch']):r for r in observations};total=0
        for epoch in data:
            base={'variant':item['variant'],'source':item['source'],'seed':67137,'epoch':epoch['epoch']};fields={k:epoch[k] for k in CURVE}
            for split in ('train','validation'):
                observed=lookup.get((split,epoch['epoch']))
                component=(observed['loss_components'] if observed else {key:epoch[key] for key in LOSS}) if split=='train' else validation_component(epoch,observed)
                effective=weighted(component,config) if component['cc_loss'] is not None else {key:None for key in ('weighted_cc_loss','weighted_mscl_loss','weighted_cscl_loss')}
                losses.append({**base,'split':split,**component,**effective})
                if observed:fields[split+'_loss']=component['classification_loss']
            curves.append({**base,**fields})
            n=lookup[('train',epoch['epoch'])]['batches'] if observations else math.ceil(epoch['train_dialogues']/16);total+=n
            steps.append({**base,'optimizer_steps':n,'basis':'observed' if observations else 'CPU_reconstructed_not_historical_step_log'})
        item.update(optimizer_steps_observed=total if observations else None,optimizer_steps_reconstructed=total)
    preflight=ROOT/'preflight'/run['run_id']/'observer';evidence={}
    for name in ('first.json','gradient_audit.json'):
        path=preflight/name
        if path.exists():evidence[name]={k:v for k,v in read(path).items() if isinstance(v,(str,bool,int,float)) or v is None}
    audit=read(preflight/'gradient_audit.json')
    assert audit['status']=='passed' and audit['preflight_only'] and audit['additional_model_forwards']==0
    assert audit['CC_backbone_gradient_nonzero'] and audit['total_equals_classification_plus_0p002_CC_gradient']
    assert audit['MSCL_gradient_wrt_total'] is None and audit['CSCL_gradient_wrt_total'] is None
    assert audit['parameter_grad_buffers_unchanged'] and audit['rng_before']==audit['rng_after']
    probe_first=read(preflight/'first.json');formal_first=read(result/'first.json')
    identity={key:probe_first[key]==formal_first[key] for key in ('initial_model_sha256','rng_before_train','rng_before_step','rng_after_step','dialogue_ids','lengths')}
    assert all(identity.values())
    evidence['probe_vs_formal_first_batch_exact_identity']=identity
    evidence['gradient_l2']=audit['gradient_l2']
    origin=read(ROOT/'provenance/source_origin.json');environment=read(ROOT/'provenance/environment.json')
    assert origin['training_sources_identical_to11'] and origin['training_source_sha256']==sha(ROOT/'code_xp04/train_cm.py')
    names=['configs/registry.json','provenance/xp04_validation_references.json','provenance/xp06_validation_reference.json','provenance/source_origin.json','provenance/environment.json','provenance/frozen_execution.json','provenance/data_inputs.json','provenance/final_verification.json','provenance/historical_metric_access_note.json','preflight/capacity_cpu.json','scripts/analyze_hscl_ablation.py','scripts/plot_hscl_ablation.py','scripts/test_hscl_ablation_analysis.py','reports/PREREGISTRATION.md','reports/SOURCE_AND_PROTOCOL_AUDIT.md']
    inputs=[ROOT/name for name in names]+[directory/'run_manifest.json',directory/'metrics.jsonl',result/'status.json',result/'watch_epochs.jsonl',result/'watch_batches.jsonl',result/'first.json',preflight/'gradient_audit.json']
    inputs += [p for p in [preflight/'first.json',ROOT/'preflight/gate.json',ROOT/'preflight/analysis_tests.json',ROOT/'provenance/test_metric_access_note.json',ROOT/'provenance/test_metric_access_incident.json',ROOT/'provenance/test_metric_access_notes.json'] if p.exists()]
    comparisons=[compare(old,new,'primary'),compare(old,descriptive,'descriptive_only')]
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,'new_runs_completed':1,'reused_historical_controls':2,
             'per_run':protocols,'comparison':comparisons,'optimizer_step_records':steps,'new_raw_hscl_nonzero_batches':nonzero,'preflight_scalar_evidence':evidence,
             'test_forward_passes':0,'ece':None,'cel':None,'automatic_additional_runs':False,
             'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'scope':'sanitized training/validation only; prior historic test-column access disclosed, no new test forwards'}
    prose=report(summary,registry,origin,environment,final)
    # All gates above precede any output mutation. Root report is copied by the parent.
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_run.csv',protocols),('comparison.csv',comparisons),('curves.csv',curves),('loss_curves.csv',losses)]:csv_write(output/name,data)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (ROOT/'reports/CMERC_HSCL_ABLATION_DIAGNOSIS.md').write_text(prose)
    print(json.dumps({'completed_new_runs':1,'primary':comparisons[0]},ensure_ascii=False))


if __name__=='__main__':main()
