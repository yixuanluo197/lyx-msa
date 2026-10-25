"""Strict CPU analysis of one new dropout confirmation plus one reused XP10 run.

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


def qualification(best_delta,tail_delta):
    if best_delta<=0:return 'no_best_improvement'
    return 'qualified' if tail_delta>=0 else 'mixed_evidence'


def confirmation(comparisons):
    assert len(comparisons)==2 and {r['seed'] for r in comparisons}=={2024,67137}
    second=next(r for r in comparisons if r['seed']==67137)
    if all(r['candidate_category']=='qualified' for r in comparisons):
        return {'decision':'preliminary_consistency_support',
                'interpretation':'两个Seed的最佳验证均提高且末段均未降低，提供初步跨Seed一致性支持；不是统计显著或测试泛化证明。',
                'recommendation':'值得另行预登记正式配置确认及配对预算审查；本次不自动开展五Seed、Baseline或任何其他训练/测试。'}
    if second['candidate_category']=='no_best_improvement':
        return {'decision':'not_confirmed','interpretation':'第二Seed最佳验证未提高，未获得预登记主指标的一致确认；保留末段变化作为辅助线索。',
                'recommendation':'不扩大五Seed；诊断已有训练/验证轨迹，不更换Seed或追加其他Dropout/lr/Batch来择优。'}
    return {'decision':'mixed_evidence','interpretation':'第二Seed最佳验证升、末段降，证据混合，未获得同时满足原登记条件的一致确认。',
            'recommendation':'暂缓扩大五Seed和正式配置确认；保持主指标，先解释已有曲线，不自动追加训练或测试。'}


def validate_pair(actual,old):
    allowed={'dropout','skip_final_test','output_dir','provenance'}
    assert {k:v for k,v in actual.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
    assert actual['dropout']==.5 and old['dropout']==.4 and actual['skip_final_test'] is True
    assert actual['lr']==old['lr']==1e-4 and actual['batch_size']==old['batch_size']==16
    assert actual['rec_dropout']==old['rec_dropout']==.1 and actual['seed']==old['seed']
    assert actual['seed'] in (2024,67137)


def require_complete(status):
    assert status['phase']=='completed' and status['counts']=={'train':40,'validation':40,'test':0}
    assert status['test_forward_passes']==0 and status['no_final_test_artifacts']

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


def stats(seed,dropout,source,epochs,watch,args):
    best=first_max(epochs);tail=epochs[-10:]
    result={'seed':seed,'lr':args['lr'],'batch_size':args['batch_size'],'validation_batch_size':args['batch_size'],
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


def verify_final(value,run_id,selected_epoch,best):
    assert value['status']=='passed' and value['formal_runs']==1 and value['test_forwards']==0
    assert value['cuda_initialized'] is False and value['frozen_sources_and_inputs_unchanged'] is True
    assert value['historical_files_unchanged'] and sum(value['historical_files_unchanged'].values())>0
    run=value['runs'][run_id]
    assert run['status']=='passed' and run['epochs']==40 and run['test_forwards']==0
    assert run['selected_epoch']==selected_epoch and run['validation_best_wf1']==best
    assert run['loss_and_gradient_finite'] is True
    assert run['every_epoch_order_matches_historical_and_registered_batch16_replay'] is True
    assert run['unchanged_sampler_AST_and_historical_batch16_replay_passed'] is True
    assert run['train_and_validation_batch_size']==16 and run['lr_unchanged_from_xp04']==1e-4
    assert run['dropout']==.5 and run['rec_dropout_metadata']==.1
    assert set(run['checkpoints'])=={'best.pt','latest.pt'}
    for name,epoch in [('best.pt',selected_epoch),('latest.pt',40)]:
        checkpoint=run['checkpoints'][name]
        assert checkpoint['epoch']==epoch and checkpoint['model_and_adam_finite'] and checkpoint['full_rng_present']
    return run


def compare(old,new):
    bd=new['best_validation_wf1']-old['best_validation_wf1']
    td=new['last10_validation_wf1']-old['last10_validation_wf1']
    return {'seed':old['seed'],'lr':1e-4,'batch_size':16,'rec_dropout_argument':.1,
            'control_dropout':.4,'new_dropout':.5,'control_best_validation_wf1':old['best_validation_wf1'],
            'new_best_validation_wf1':new['best_validation_wf1'],'control_best_epoch':old['best_epoch'],
            'new_best_epoch':new['best_epoch'],'control_last10_validation_wf1':old['last10_validation_wf1'],
            'new_last10_validation_wf1':new['last10_validation_wf1'],'delta_best_validation_wf1':bd,
            'delta_last10_validation_wf1':td,'candidate_category':qualification(bd,td)}


def report(summary,registry,origin,environment,verification):
    protocols=summary['per_run'];run=registry['runs'][0];decision=summary['confirmation']
    policy=registry['resource_policy'];verified=verification['runs'][run['run_id']]
    same_gpu=registry['gpu_uuid']=='GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248'
    lines=['# CMERC Dropout第二Seed确认报告','',
           f"生成时间：{summary['generated_at_utc']}。第11组仅新增1次MELD seed67137完整CMERC训练；复用第10组已完成的seed2024 dropout0.5训练1次，以及XP04两个对应Seed的dropout0.4历史控制。四臂均为40轮，不能将复用结果称为本批新训练。",'',
           '**当前判定：** `'+decision['decision']+'`。'+decision['interpretation'],'',
           '**直接建议：** '+decision['recommendation'],'',
           '## 预登记、配对及实际执行','',
           '唯一科学因素为通用Dropout参数0.4→0.5；两个Seed各自与同Seed XP04完整CMERC配对。lr=0.0001、L2=0.00003、Adam、train/validation Batch=16/16、CLI rec_dropout=0.1、40 epochs、GRU、hyper/direct/concat_DHT、BN、L3/K3、CC=0.002、MSCL/CSCL=0.15/0.15和CL=0.4/1全部保持。数据split_seed=1475、continuous RNG、ATen GRU和capacity_safe采样保持。逐项args、数据文件SHA与split精确核对；工程开关skip_final_test仅取消最后测试阶段。','',
           '使用完整固定验证池拼接话语后计算Weighted F1，保留原训练器按百分比四舍五入两位后严格提高才保存的规则；并列最高选首次。末段固定为31–40轮均值；末段时间窗SD使用ddof=1，不是跨Seed标准差，峰值和末段也不是独立证据。','',
           f"train_cm.py SHA256：`{origin['training_source_sha256']}`；源码来源`{origin['copied_from']}`，12个训练源码与第10组字节相同标记`{origin['training_sources_identical_to10']}`；记录Git HEAD `{origin['head_commit']}`。科学与工程来源见[源码和协议审计](../reports/SOURCE_AND_PROTOCOL_AUDIT.md)。",'',
           f"实际环境：Python {environment['python'].split(' | ')[0]}、PyTorch {environment['packages']['torch']}、CUDA build {environment['torch_cuda_build']}、PyG {environment['packages']['torch-geometric']}；复用已验证环境，无新增依赖。",'',
           'Dropout实际作用路径：train_cm.py:1040–1073传入args.dropout；model_cm.py:658的两层双向文本GRU层间Dropout和:713的hidden-feature Dropout（hyper路径约:892–896）均从0.4改到0.5。CLI rec_dropout=0.1未传给当前活跃图模型，仅保留元数据，不能解释为“实际GRU Dropout仍为0.1”。四模型实例属性已记录；辅助模型在train_cm.py:608–613设为eval，teacher前向Dropout关闭；不将其关闭状态等同于主模型训练Dropout关闭。','',
           '## 两Seed四臂结果（仅训练和验证）','',
           '| Seed | Dropout | 来源 | 最佳验证W-F1 % | 轮 | 末10轮W-F1 % | 末10轮SD | 最佳轮val ACC % | 末10轮val ACC % |',
           '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in protocols:
        lines.append(f"| {r['seed']} | {r['dropout']} | {r['source']} | {r['best_validation_wf1']:.2f} | {r['best_epoch']} | {r['last10_validation_wf1']:.3f} | {r['last10_validation_wf1_temporal_sd']:.4f} | {r['best_validation_accuracy']:.2f} | {r['last10_validation_accuracy']:.3f} |")
    lines+=['','| Seed | 最佳W-F1变化 pp | 末10轮均值变化 pp | 预登记分类 |','|---:|---:|---:|---|']
    for r in summary['comparison']:
        lines.append(f"| {r['seed']} | {r['delta_best_validation_wf1']:+.2f} | {r['delta_last10_validation_wf1']:+.3f} | `{r['candidate_category']}` |")
    lines+=['','分类门槛：最佳提高且末段不降为qualified；最佳提高但末段下降为mixed_evidence；最佳未提高为no_best_improvement，即使末段改善也不能替换主指标。两个Seed均qualified才是初步一致性支持。门槛满足不等于统计强证据，不给两Seed p值或统计显著结论，不推断测试泛化。','',
            '## Loss与训练稳定性','',
            '| Seed | Dropout | train分类Loss 最佳轮→末段 | val分类Loss 最佳轮→末段 | 梯度异常Epoch | optimizer.step |','|---:|---:|---|---|---|---|']
    for r in protocols:
        steps=r['optimizer_steps_observed'] if r['optimizer_steps_observed'] is not None else 'CPU重建 '+str(r['optimizer_steps_reconstructed'])
        lines.append(f"| {r['seed']} | {r['dropout']} | {fmt(r['best_train_classification_loss'])} → {fmt(r['last10_train_classification_loss'])} | {fmt(r['best_validation_classification_loss'])} → {fmt(r['last10_validation_classification_loss'])} | {r['gradient_abnormalities'] if r['gradient_epoch_checks'] else '历史未记录'} | {steps} |")
    for split,title in [('train','训练'),('validation','验证')]:
        lines+=['',title+'原始Loss（最佳验证轮→31–40轮均值）：','','| Seed | Dropout | 分类 | CC | MSCL | CSCL | 总Loss |','|---:|---:|---|---|---|---|---|']
        for r in protocols:
            values=[fmt(r['best_'+split+'_'+key])+' → '+fmt(r['last10_'+split+'_'+key]) for key in LOSS]
            lines.append(f"| {r['seed']} | {r['dropout']} | "+' | '.join(values)+' |')
    lines+=['','固定系数的加权CC/MSCL/CSCL、完整train/val曲线与每轮更新数见CSV和图。历史验证辅助Loss/总Loss没有正式记录，保留缺失而非填零；新两臂的watcher均有80条train/validation记录。新臂分类Loss使用watcher完整Batch均值精度；旧臂validation分类Loss来自4位小数原日志，不能把微小差值归为高精度优化效应。Loss仍按原训练器Batch均值汇总，不将原始Loss数值当作独立因果证据。','']
    for r in protocols:
        tc=r['last10_train_classification_loss']-r['best_train_classification_loss'];vc=r['last10_validation_classification_loss']-r['best_validation_classification_loss'];wc=r['last10_validation_wf1']-r['best_validation_wf1']
        lines.append(f"- Seed{r['seed']} dropout={r['dropout']}：最佳轮→末段，train分类Loss {tc:+.4f}、val分类Loss {vc:+.4f}、val W-F1 {wc:+.4f} pp。")
    lines+=['','**推断边界：** train分类Loss继续下降而val分类Loss上升、验证峰值后衰退，只是过拟合/目标竞争的轨迹线索；若W-F1和分类Loss方向不一致，也不能宣称过拟合已解决。训练拟合下降既可能是正则化作用也可能是欠拟合，不能只凭训练分数下降判定欠拟合。train为在线训练态不断变化的参数，val为固定epoch权重eval，二者间距不是严格泛化gap。有限梯度和真实更新验证数值执行，不能排除伪目标噪声、梯度竞争或泛化问题。','',
            '## 来源、资源、测试访问与完整验收','',
            f"本批实际设备：{registry['host']} GPU{registry['gpu_index']}，UUID `{registry['gpu_uuid']}`；XP04 seed67137控制为GPU0（GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248），{'同物理卡' if same_gpu else '不同物理卡'}。seed2024的XP04控制和复用XP10均用GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456）。共享资源、历史时点和CUDA scatter非确定性仍限制严格因果及速度解释。",'',
            f"登记准入为free>={policy['admission_min_free_mib']}MiB、util<={policy['admission_max_utilization_percent']}%、连续{policy['samples_required']}次/间隔{policy['sample_seconds']}秒；其他进程只披露，不终止，也不称连续独占。实际采样限制见下方验收。",'',
            '本批与复用第10组均零新增测试loader遍历/零新增测试前向，不根据测试选择配置或轮次，未计算ECE/CEL。综合pickle和test loader元数据仍由原流程加载，零前向不等于完全未加载含测试数据的文件。另一次只读GPU来源审计误将混合run_supplement整份输出，偶然访问了历史测试汇总字段；本分析脚本只读取训练/验证净化档案，事件说明不含测试分数。该访问不能隐瞒，故不宣称“全程未读取新旧测试指标”。科学配置由用户事先固定，未据该历史字段改变训练/选模/后续判据，见[测试字段偶然访问事件](../provenance/test_metric_access_incident.json)。','',
            '作者固定课程难度包含validation键且生成史未知；同输入保持配对，但不能宣称完整预处理链validation-free。四臂数据、源文件和实际args可追溯，不修改XP04、第10组或其他历史产物。','',
            f"独立完整验收状态`{verification['status']}`：本批1次正式训练，40train/40validation/test0，真实训练Batch {verified['train_batches']}；排列与历史和登记Batch16重放一致`{verified['every_epoch_order_matches_historical_and_registered_batch16_replay']}`，采样AST与历史重放通过`{verified['unchanged_sampler_AST_and_historical_batch16_replay_passed']}`。",
            f"best/latest为第{verified['checkpoints']['best.pt']['epoch']}/{verified['checkpoints']['latest.pt']['epoch']}轮，模型/Adam有限、完整Python/NumPy/Torch/CUDA RNG均由CPU验收确认；核对完整args/provenance和Adam lr，冻结源/输入未变`{verification['frozen_sources_and_inputs_unchanged']}`，历史文件保全{sum(verification['historical_files_unchanged'].values())}项。"]
    resource=verification.get('resource_evidence',{})
    lines+=['',f"资源证据`{resource.get('status','unknown')}`：采样{resource.get('resource_samples','未记录')}次，既有共驻PID样本{resource.get('allowed_preexisting_pid_samples','未记录')}次、新外部PID样本{resource.get('new_foreign_pid_samples','未记录')}次，外部PID `{resource.get('observed_foreign_pids','未记录')}`，查询错误{resource.get('resource_query_errors','未记录')}次，正式最小空闲{resource.get('minimum_free_mib_during_formal','未记录')}MiB、最大利用率{resource.get('maximum_utilization_percent','未记录')}%。离散采样不能证明连续资源隔离。",'',
            '四实例实际Dropout属性（本批无额外模型前向）：','','| 实例 | 文本GRU Dropout | 层数/双向 | hidden-feature Dropout |','|---|---:|---|---:|']
    for name,instance in summary['module_audit']['models'].items():
        gru=instance['language_gru'];lines.append(f"| {name} | {gru['dropout']} | {gru['num_layers']}/{gru['bidirectional']} | {instance['hidden_dropout_p']} |")
    lines+=['','## 已证实事实、未证实推断与下一步','',
            '已证实事实：本批单次40轮的执行、源/参数/划分一致性、验证选模、四臂真实训练/验证轨迹、数值执行及完整验收；2024的dropout0.5是第10组复用，不是重训。尚未证实：Dropout是论文F1差距唯一根因、对任意Seed稳定改善、统计显著、校准性能或测试泛化改善。','',
            '**下一步直接建议：** '+decision['recommendation']+' 本阶段暂缓Batch8，没有对Batch因素的新隔离证据；不自动追加五Seed、其他Dropout/lr/Batch、Baseline或测试。','',
            '## 交付和可追溯输入','',
            '[预登记](../reports/PREREGISTRATION.md)、[源码/协议审计](../reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[配置](../configs/registry.json)、[环境](../provenance/environment.json)、[源码来源](../provenance/source_origin.json)、[冻结](../provenance/frozen_execution.json)、[数据输入](../provenance/data_inputs.json)、[XP04净化控制](../provenance/xp04_validation_references.json)、[复用XP10档案](../provenance/xp10_validation_reference.json)、[容量预检](../preflight/capacity_cpu.json)、[预检门禁](../preflight/gate.json)、[完整验收](../provenance/final_verification.json)。','',
            '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)、[22图44文件及SHA清单](../results/figures/manifest.json)。','']
    for seed in (2024,67137):
        lines.append(f"Seed{seed}：[W-F1](../results/figures/MELD_seed{seed}_dropout_wf1.png)、[ACC](../results/figures/MELD_seed{seed}_dropout_accuracy.png)、[分类Loss](../results/figures/MELD_seed{seed}_dropout_classification_loss.png)、[总Loss](../results/figures/MELD_seed{seed}_dropout_total_loss.png)、[更新次数](../results/figures/MELD_seed{seed}_dropout_optimizer_steps.png)。")
    lines+=['',f'实验根目录：`{ROOT}`。仓库根报告正文相同，相对链接按位置调整。','']
    return '\n'.join(lines)


def main():
    registry=read(ROOT/'configs/registry.json');assert registry['formal_run_count']==len(registry['runs'])==1
    run=registry['runs'][0]
    assert (run['dataset'],run['seed'],run['lr'],run['batch_size'],run['validation_batch_size'],run['dropout'],run['rec_dropout'],run['epochs'])==('MELD',67137,1e-4,16,16,.5,.1,40)
    result=ROOT/'results'/run['run_id'];status=read(result/'status.json');require_complete(status)
    # No files are emitted until all real-run, reuse, provenance and final-verifier gates pass.
    references=read(ROOT/'provenance/xp04_validation_references.json');reuse=read(ROOT/'provenance/xp10_validation_reference.json')
    assert reuse['seed']==2024 and reuse['verification_status']=='passed' and reuse['summary']['formal_training_complete']
    assert reuse['summary']['test_forward_passes']==0 and reuse['summary']['new_runs_completed']==1
    directory=Path(run['run_dir']);manifest=read(directory/'run_manifest.json');args=manifest['args']
    validate_pair(args,references['MELD_67137']['args']);validate_pair(reuse['args'],references['MELD_2024']['args'])
    assert manifest['split']==references['MELD_67137']['split'] and manifest['data']==references['MELD_67137']['data']
    assert references['MELD_2024']['split']==manifest['split'] and references['MELD_2024']['data']==manifest['data']
    assert reuse['args']['provenance']['split']==manifest['split'] and reuse['args']['provenance']['data']==manifest['data']
    capsule=read(ROOT/'provenance/frozen_execution.json')
    for key,value in manifest['source']['files'].items():
        assert sha(ROOT/'code_xp04'/key)==capsule['files']['code_xp04/'+key]==value
        assert reuse['args']['provenance']['source']['files'][key]==value
    for key,value in manifest['data'].items():assert sha(ROOT/'code_xp04'/key)==value
    assert not (directory/'final_test.json').exists() and not list(result.glob('*.npz'))
    assert all((directory/name).is_file() for name in ('best.pt','latest.pt'))
    epochs=rows(directory/'metrics.jsonl');watch=rows(result/'watch_epochs.jsonl');batches=rows(result/'watch_batches.jsonl')
    verify_epochs(epochs,args,run['split_hash'],True);verify_watch(watch,epochs,args)
    verify_epochs(reuse['epochs'],reuse['args'],run['split_hash']);verify_watch(reuse['watch_epochs'],reuse['epochs'],reuse['args'])
    module_audit=read(result/'module_audit.json')
    assert module_audit['cli_dropout']==.5 and module_audit['cli_rec_dropout_metadata']==.1
    assert set(module_audit['models'])=={'model','model_mm','model_intra','model_inter'}
    for instance in module_audit['models'].values():
        assert instance['language_gru']=={'dropout':.5,'num_layers':2,'bidirectional':True}
        assert instance['hidden_dropout_p']==.5
    protocols=[];datasets=[]
    for seed in (2024,67137):
        ref=references['MELD_'+str(seed)];verify_epochs(ref['epochs'],ref['args'],run['split_hash'])
        old=stats(seed,.4,'XP04_control',ref['epochs'],[],ref['args'])
        assert (old['best_epoch'],old['best_validation_wf1'])==(ref['best_epoch'],ref['best_validation_wf1'])
        assert math.isclose(old['last10_validation_wf1'],ref['last10_validation_wf1'],abs_tol=1e-12)
        data,observations,config=(reuse['epochs'],reuse['watch_epochs'],reuse['args']) if seed==2024 else (epochs,watch,args)
        new=stats(seed,.5,'XP10_reused' if seed==2024 else 'XP11_new',data,observations,config)
        if seed==2024:
            saved=next(r for r in reuse['summary']['per_run'] if r['dropout']==.5)
            assert (new['best_epoch'],new['best_validation_wf1'])==(saved['best_epoch'],saved['best_validation_wf1'])
            assert math.isclose(new['last10_validation_wf1'],saved['last10_validation_wf1'],abs_tol=1e-12)
        protocols.extend((old,new));datasets.extend([(old,ref['epochs'],[],ref['args']),(new,data,observations,config)])
    comparisons=[compare(*[r for r in protocols if r['seed']==seed]) for seed in (2024,67137)]
    assert comparisons[0]['candidate_category']=='qualified'  # Frozen XP10 eligibility is confirmed, never changed by second Seed.
    new=protocols[-1];assert status['selected_epoch']==new['best_epoch'] and status['selection_weighted_f1']==new['best_validation_wf1']
    final=read(ROOT/'provenance/final_verification.json');verified=verify_final(final,run['run_id'],new['best_epoch'],new['best_validation_wf1'])
    capacity=read(ROOT/'preflight/capacity_cpu.json');assert capacity['status']=='passed' and capacity['historical_batch16_replay_passed']
    caps={r['epoch']:r for r in capacity['records'] if r['run_id']==run['run_id']};assert set(caps)==set(range(1,41))
    steps=[]
    for epoch in range(1,41):
        actual=[b for b in batches if b['epoch']==epoch];cap=caps[epoch]
        assert [b['batch'] for b in actual]==list(range(1,len(actual)+1))
        assert len(actual)==cap['train_batches']==next(w for w in watch if w['split']=='train' and w['epoch']==epoch)['batches']
        assert next(w for w in watch if w['split']=='validation' and w['epoch']==epoch)['batches']==cap['validation_batches']
        for batch in actual:
            assert batch['gradients_finite'] is True and math.isfinite(batch['gradient_l2']) and all(math.isfinite(batch[key]) for key in LOSS)
            assert math.isclose(batch['classification_loss']+sum(weighted(batch,args).values()),batch['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
    assert len(batches)==sum(c['train_batches'] for c in caps.values())==verified['train_batches']
    for item,data,observations,config in datasets:
        observed={r['epoch']:r for r in observations if r['split']=='train'}
        total=0
        for epoch in data:
            step=observed[epoch['epoch']]['batches'] if observed else math.ceil(epoch['train_dialogues']/16)
            total+=step
            steps.append({'seed':item['seed'],'dropout':item['dropout'],'epoch':epoch['epoch'],'optimizer_steps':step,
                          'basis':'observed_real_optimizer_steps' if observed else 'CPU_reconstructed_historical'})
        item.update(optimizer_steps_observed=total if observed else None,optimizer_steps_reconstructed=total)
    curves=[];losses=[]
    for item,data,observations,config in datasets:
        valwatch={r['epoch']:r for r in observations if r['split']=='validation'}
        trainwatch={r['epoch']:r for r in observations if r['split']=='train'}
        for epoch in data:
            fields={key:epoch[key] for key in CURVE}
            if observations:
                fields['train_loss']=trainwatch[epoch['epoch']]['loss_components']['classification_loss']
                fields['validation_loss']=valwatch[epoch['epoch']]['loss_components']['classification_loss']
            base={'seed':item['seed'],'dropout':item['dropout'],'lr':1e-4,'batch_size':16,'source':item['source'],'epoch':epoch['epoch']}
            curves.append({**base,**fields})
            for split in ('train','validation'):
                observed=(trainwatch if split=='train' else valwatch).get(epoch['epoch'])
                component=observed['loss_components'] if observed else ({key:epoch[key] for key in LOSS} if split=='train' else {key:epoch['validation_loss'] if key=='classification_loss' else None for key in LOSS})
                weighted_component=weighted(component,config) if component['cc_loss'] is not None else {key:None for key in ('weighted_cc_loss','weighted_mscl_loss','weighted_cscl_loss')}
                losses.append({**base,'split':split,**component,**weighted_component})
    origin=read(ROOT/'provenance/source_origin.json');environment=read(ROOT/'provenance/environment.json')
    assert origin['training_sources_identical_to10'] and origin['training_source_sha256']==sha(ROOT/'code_xp04/train_cm.py')
    # The parent supplies an incident description without old test-score values. Do not open mixed supplements.
    assert (ROOT/'provenance/test_metric_access_incident.json').is_file()
    names=['configs/registry.json','provenance/xp04_validation_references.json','provenance/xp10_validation_reference.json',
           'provenance/source_origin.json','provenance/environment.json','provenance/frozen_execution.json','provenance/data_inputs.json',
           'provenance/test_metric_access_incident.json','provenance/final_verification.json','preflight/capacity_cpu.json',
           'scripts/analyze_dropout_confirmation.py','scripts/plot_dropout_confirmation.py','scripts/test_dropout_confirmation_analysis.py',
           'reports/PREREGISTRATION.md','reports/SOURCE_AND_PROTOCOL_AUDIT.md']
    inputs=[ROOT/name for name in names]+[directory/'run_manifest.json',directory/'metrics.jsonl',result/'watch_epochs.jsonl',result/'watch_batches.jsonl',result/'status.json',result/'module_audit.json']
    inputs += [p for p in [ROOT/'preflight/gate.json',ROOT/'preflight/analysis_tests.json',ROOT/'provenance/gpu_selection.json'] if p.exists()]
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,
             'new_runs_completed':1,'reused_xp10_completed_runs':1,'dropout05_completed_runs_across_two_batches':2,
             'per_run':protocols,'comparison':comparisons,'confirmation':confirmation(comparisons),'optimizer_step_records':steps,
             'module_audit':module_audit,'test_forward_passes':0,'automatic_additional_runs':False,
             'historical_test_field_access_disclosed':True,'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
             'scope':'Current CPU analysis uses sanitized train/validation records only; historical mixed-field access by separate audit is disclosed; no new test forwards.'}
    prose=report(summary,registry,origin,environment,final)
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_run.csv',protocols),('comparison.csv',comparisons),('curves.csv',curves),('loss_curves.csv',losses)]:csv_write(output/name,data)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    path=ROOT/'reports/CMERC_DROPOUT_SECOND_SEED_CONFIRMATION.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    prefix=ROOT.relative_to(REPO).as_posix()
    for section in ('reports','configs','provenance','preflight','results'):prose=prose.replace('../'+section+'/', '../'+prefix+'/'+section+'/')
    path=REPO/'reports/CMERC_DROPOUT_SECOND_SEED_CONFIRMATION.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    print(json.dumps({'new_runs_completed':1,'reused_completed':1,'decision':summary['confirmation']},ensure_ascii=False))


if __name__=='__main__':main()
