"""Strict CPU train/validation-only analysis of the registered dropout diagnosis."""
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
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):value.update(chunk)
    return value.hexdigest()


def csv_write(path,data):
    keys=list(dict.fromkeys(key for row in data for key in row))
    with Path(path).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(data)


def first_max(epochs):
    assert epochs
    assert all(r['validation_weighted_f1']==round(r['validation_weighted_f1'],2) for r in epochs)
    return max(epochs,key=lambda row:row['validation_weighted_f1'])


def last10_mean(epochs,key='validation_weighted_f1'):
    assert len(epochs)>=10
    return statistics.mean(row[key] for row in epochs[-10:])


def qualification(best_delta,tail_delta):
    if best_delta<=0:return 'no_best_improvement'
    return 'second_seed_candidate' if tail_delta>=0 else 'mixed_evidence'


def recommendation(label):
    if label=='second_seed_candidate':
        return '本Seed满足预登记资格，值得另行登记第二Seed确认；该资格不是统计强证据。本批不自动追加Seed、学习率、Dropout、Batch Size、五Seed训练或测试。'
    if label=='mixed_evidence':
        return '最佳验证提高但末段下降，证据混合，暂不推荐扩大第二Seed或五Seed；保留曲线线索，不更换主指标，不追加配置择优。'
    assert label=='no_best_improvement'
    return '主指标未改善，本Seed不支持扩大第二Seed或五Seed；继续分析现有训练/验证证据，不改试其他Dropout、lr或Seed来凑结果。'


def validate_pair(actual,historical):
    allowed={'dropout','skip_final_test','output_dir','provenance'}
    assert {k:v for k,v in actual.items() if k not in allowed}=={k:v for k,v in historical.items() if k not in allowed}
    assert actual['dropout']==.5 and historical['dropout']==.4
    assert actual['lr']==historical['lr']==1e-4
    assert actual['batch_size']==historical['batch_size']==16
    assert actual['rec_dropout']==historical['rec_dropout']==.1
    assert actual['seed']==historical['seed']==2024
    assert actual['skip_final_test'] is True


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


def stats(dropout,source,epochs,watch,args):
    best=first_max(epochs);tail=epochs[-10:]
    result={'seed':2024,'lr':args['lr'],'batch_size':args['batch_size'],'validation_batch_size':args['batch_size'],
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
    else:result.update(gradient_epoch_checks=0,gradient_abnormalities=None,gradient_l2_mean=None,gradient_l2_max=None)
    return result


def fmt(value,signed=False):return '未记录' if value is None else format(value,'+.4f' if signed else '.4f')


def report(summary,registry):
    comparison=summary['comparison'][0];protocols=summary['per_run'];run=registry['runs'][0]
    environment=read(ROOT/'provenance/environment.json');origin=read(ROOT/'provenance/source_origin.json');policy=registry['resource_policy']
    pid_list=policy.get('preexisting_gpu_pids',policy.get('preexisting_gpu3_pids',[]))
    same_gpu=registry['gpu_index']==3 and registry['gpu_uuid']=='GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'
    label={'second_seed_candidate':'满足第二Seed候选资格（非统计强证据）','mixed_evidence':'最佳升、末段降：混合线索','no_best_improvement':'主指标未改善'}
    lines=['# CMERC Dropout 诊断', '',f"生成时间：{summary['generated_at_utc']}。仅1/1正式运行完成，MELD seed2024、dropout0.5、lr1e-4、batch16、40轮。", '',
           '## 实际科学协议与Dropout路径', '',
           '直接对照XP04完整CMERC同Seed、同lr、同Batch16，唯一登记变化为args.dropout 0.4→0.5。rec_dropout参数固定0.1、Adam/L2=3e-5、dropout以外全部模型参数不变；GRU、ATen、continuous RNG、capacity_safe、BN、hyper/direct、concat_DHT、L3/K3、FP32、worker0、split_seed1475、CC/MSCL/CSCL=.002/.15/.15、CL比例步长=.4/1。训练池1038、验证114对话，train/val均Batch16，划分SHA `'+run['split_hash']+'`。', '',
           '**rec_dropout的实际边界：** train_cm.py:1040–1073的活跃graph Model入口传递dropout=args.dropout，未传args.rec_dropout。model_cm.py:658的文本两层双向GRU使用dropout=dropout，:713定义nn.Dropout(self.dropout)；因此本次通用dropout变化也作用于活跃GRU层间dropout。rec_dropout=.1是保持不变的CLI元数据，不能声称真实GRU dropout固定.1或本次只改变非循环层Dropout。av_using_lstm=False时音/视GRU分支不启用。hyper输出的hidden-feature Dropout在model_cm.py:892–896参与分类；辅助模型按原train_cm.py:608–613始终eval，辅助前向不因配置中的概率提高而启用Dropout。', '',
           f"实际训练源SHA `{sha(ROOT/'code_xp04/train_cm.py')}`；来源`{origin['copied_from']}`，Git HEAD `{origin['head_commit']}`，12训练源保持09字节版本，仅保留既有skip-final-test工程门禁，不改网络或Loss数学。环境Python {environment['python'].split()[0]}、PyTorch {environment['packages']['torch']}、PyG {environment['packages']['torch-geometric']}、CUDA build {environment['torch_cuda_build']}。", '',
           '规范选模：验证Weighted F1按全验证池labels/predictions汇总，百分比四舍五入两位，严格提高才换Checkpoint，同分取首次最高轮。末10轮固定31–40；最佳值与末段来自同一验证曲线，是相关描述统计，不是独立证据。', '',
           '## 验证结果与登记判据', '', '| Dropout | 来源 | 最佳验证W-F1 % | Epoch | 末10轮W-F1 % | 最佳轮验证ACC % | 末10轮验证ACC % | 末段W-F1时间窗SD |', '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in protocols:lines.append(f"| {r['dropout']} | {r['source']} | {fmt(r['best_validation_wf1'])} | {r['best_epoch']} | {fmt(r['last10_validation_wf1'])} | {fmt(r['best_validation_accuracy'])} | {fmt(r['last10_validation_accuracy'])} | {fmt(r['last10_validation_wf1_temporal_sd'])} |")
    lines+=['',f"dropout0.5−0.4：Δ最佳验证 **{fmt(comparison['delta_best_validation_wf1'],True)} pp**，Δ末10轮 **{fmt(comparison['delta_last10_validation_wf1'],True)} pp**；分类：**{label[comparison['candidate_category']]}**。", '',
            recommendation(comparison['candidate_category']), '',
            '原登记资格不变：最佳提高且末段不降才是第二Seed候选；最佳升、末段降为mixed；最佳未升为no_best_improvement。辅助末段改善不能替换预登记主指标。资格无最小效应量或置信区间门槛，一个Seed不作p值、显著性、稳定复现或等价结论。', '',
            '## Loss与训练稳定性事实', '', '| Dropout | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 非有限梯度Epoch | optimizer.step实际记录数 |', '|---:|---:|---:|---:|---:|---|---|']
    for r in protocols:lines.append(f"| {r['dropout']} | {fmt(r['best_train_classification_loss'])} | {fmt(r['best_validation_classification_loss'])} | {fmt(r['last10_train_classification_loss'])} | {fmt(r['last10_validation_classification_loss'])} | {r['gradient_abnormalities'] if r['gradient_epoch_checks'] else '历史未记录'} | {r['optimizer_steps_observed'] if r['optimizer_steps_observed'] is not None else '历史未记录；CPU重建'+str(r['optimizer_steps_reconstructed'])} |")
    for split,heading in [('train','训练'),('validation','验证')]:
        lines+=['',heading+'原始分项（最佳验证轮→末10轮均值）：', '', '| Dropout | 分类Loss | CC | MSCL | CSCL | 总Loss |', '|---:|---|---|---|---|---|']
        for r in protocols:
            values=[fmt(r['best_'+split+'_'+key])+' → '+fmt(r['last10_'+split+'_'+key]) for key in LOSS]
            lines.append(f"| {r['dropout']} | "+' | '.join(values)+' |')
    lines+=['', 'XP04历史验证辅助Loss没有正式记录，表中缺失保持“未记录”，不补造零值；新watcher记录完整80次train/val分项。所有固定系数加权CC/MSCL/CSCL及总Loss保存在loss_curves.csv与图中。', '']
    for r in protocols:
        tc=r['last10_train_classification_loss']-r['best_train_classification_loss'];vc=r['last10_validation_classification_loss']-r['best_validation_classification_loss'];wc=r['last10_validation_wf1']-r['best_validation_wf1']
        lines.append(f"- dropout={r['dropout']}：最佳验证轮→末段，train分类Loss {tc:+.4f}、val分类Loss {vc:+.4f}、val W-F1 {wc:+.4f} pp。")
    lines+=['', '**推断边界：** train Loss下降且val Loss上升、验证末段回落，可作为过拟合/目标竞争的轨迹线索，不能证明唯一根因。Dropout提高后若训练拟合降低，既可能是正则化作用也可能是欠拟合，必须同时看验证轨迹；仅训练分数低不能判定欠拟合。train在线训练态与val固定模型eval不同，间距不是严格泛化gap。时间窗SD不是跨Seed标准差。finite梯度和首步真实更新只验证执行数值，不排除伪目标噪声、梯度竞争或泛化问题。', '',
            '## 资源、零测试与来源限制', '',
            f"本批实际设备为同host GPU{registry['gpu_index']}（{registry['gpu_uuid']}）；XP04控制为GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456），{'同物理卡' if same_gpu else '不同物理卡'}。历史时点、CUDA scatter非确定性和共享资源仍限制严格因果与速度解释，不宣称连续独占。登记共驻PID `{pid_list}`；实际采样见最终验收。资源准入以registry记录为准：free>={policy['admission_min_free_mib']}MiB、util<={policy['admission_max_utilization_percent']}%、{policy['samples_required']}次/{policy['sample_seconds']}秒。不终止外部项目，不因分数重跑。", '',
            '本次零测试loader遍历/零测试前向，不读取新旧测试指标或预测数组、不据测试选配置；本阶段未计算ECE/CEL。综合pickle和测试loader元数据仍由原流程加载，零前向不等于完全未加载含测试数据的文件。固定作者课程包含validation键且生成史未知；同一输入保持配对，不称完整预处理链validation-free。', '',
            '## 事实、推断和后续建议', '',
            '已证实事实：登记配置、固定池选模、实际完整40轮轨迹、Loss身份和数值执行记录；推断仍限于上述拟合/验证变化线索。Dropout跨Seed效应、校准改善、泛化到测试集及论文差距唯一根因均未证实。', '',
            '**当前直接建议：** '+recommendation(comparison['candidate_category']), '',
            '本批不自动开展任何新Seed、lr、Dropout、Batch Size、五Seed或最终测试。若将来需要确认，须在独立目录重新登记固定候选与预算，不能根据测试分数或临时扩大搜索。', '',
            '## 产物和验收', '',
            '[预登记](../reports/PREREGISTRATION.md)、[配置](../configs/registry.json)、[协议与Dropout路径审计](../reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[环境](../provenance/environment.json)、[源码来源](../provenance/source_origin.json)、[冻结清单](../provenance/frozen_execution.json)、[数据](../provenance/data_inputs.json)、[容量CPU预检](../preflight/capacity_cpu.json)、[预检门禁](../preflight/gate.json)、[完整验收](../provenance/final_verification.json)。', '',
            '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)。', '',
            '[W-F1](../results/figures/MELD_dropout_wf1.png)、[ACC](../results/figures/MELD_dropout_accuracy.png)、[分类Loss](../results/figures/MELD_dropout_classification_loss.png)、[总Loss](../results/figures/MELD_dropout_total_loss.png)、[更新次数](../results/figures/MELD_dropout_optimizer_steps.png)；完整原始和加权辅助曲线及输入/输出SHA见[图表清单](../results/figures/manifest.json)。']
    module_audit=summary['module_audit']
    lines+=['', '从实际四实例只读属性记录的Dropout（无额外模型前向）：', '',
            '| 实例 | 文本GRU层间Dropout | GRU层数/双向 | hidden-feature Dropout |', '|---|---:|---|---:|']
    for name,instance in module_audit['models'].items():
        gru=instance['language_gru']
        lines.append(f"| {name} | {gru['dropout']} | {gru['num_layers']}/{gru['bidirectional']} | {instance['hidden_dropout_p']} |")
    lines+=['', '[四实例module_audit.json](../results/'+run['run_id']+'/module_audit.json)仅验证真实配置属性，不证明性能改善或辅助损失作用方向。']
    final=ROOT/'provenance/final_verification.json'
    if final.exists():
        verification=read(final);lines+=['',f"最终验收实际状态：`{verification.get('status','unknown')}`。"]
        if verification.get('status')=='passed':
            verified=verification['runs'][run['run_id']];ck=verified['checkpoints'];history=sum(verification['historical_files_unchanged'].values())
            lines += [f"单次40train/40validation，真实训练Batch {verified['train_batches']}，test前向{verified['test_forwards']}；每轮排列与历史及登记Batch16重放匹配`{verified['every_epoch_order_matches_historical_and_registered_batch16_replay']}`，train/val Batch均`{verified['train_and_validation_batch_size']}`，dropout/rec_dropout元数据为`{verified['dropout']}`/`{verified['rec_dropout_metadata']}`。best/latest第{ck['best.pt']['epoch']}/{ck['latest.pt']['epoch']}轮，模型/Adam有限`{ck['best.pt']['model_and_adam_finite']}`/`{ck['latest.pt']['model_and_adam_finite']}`，完整RNG存在`{ck['best.pt']['full_rng_present']}`/`{ck['latest.pt']['full_rng_present']}`；验收脚本核对完整args/provenance和Adam lr。best SHA `{ck['best.pt']['sha256']}`，latest SHA `{ck['latest.pt']['sha256']}`。冻结源/输入未变`{verification['frozen_sources_and_inputs_unchanged']}`，历史文件保全{history}项。"]
            resource=verification.get('resource_evidence',{})
            if resource:lines+=['',f"资源证据`{resource.get('status','unknown')}`：{resource.get('resource_samples','未记录')}次采样，既有共驻PID样本{resource.get('allowed_preexisting_pid_samples','未记录')}次，新外部PID样本{resource.get('new_foreign_pid_samples','未记录')}次，查询错误{resource.get('resource_query_errors','未记录')}次，正式最小空闲{resource.get('minimum_free_mib_during_formal','未记录')}MiB，最大利用率{resource.get('maximum_utilization_percent','未记录')}%；离散采样不能证明连续资源隔离。"]
        else:lines+=['最终验收未通过，不宣称完整验收成功。']
    else:lines+=['','最终验收尚未生成；分析门禁不代替独立完整验收。']
    lines+=['',f'实验根目录：`{ROOT}`。仓库根报告正文相同，相对链接按位置调整。','']
    return '\n'.join(lines)


def main():
    registry=read(ROOT/'configs/registry.json');assert registry['formal_run_count']==len(registry['runs'])==1
    run=registry['runs'][0];assert (run['seed'],run['lr'],run['batch_size'],run['dropout'],run['rec_dropout'],run['epochs'])==(2024,1e-4,16,.5,.1,40)
    reference=read(ROOT/'provenance/xp04_validation_references.json')['MELD_2024'];result=ROOT/'results'/run['run_id'];status=read(result/'status.json')
    assert status['phase']=='completed' and status['counts']=={'train':40,'validation':40,'test':0} and status['test_forward_passes']==0 and status['no_final_test_artifacts']
    directory=Path(run['run_dir']);manifest=read(directory/'run_manifest.json');args=manifest['args'];validate_pair(args,reference['args'])
    assert manifest['split']==reference['split'] and manifest['data']==reference['data']
    capsule=read(ROOT/'provenance/frozen_execution.json')
    for key,value in manifest['source']['files'].items():assert sha(ROOT/'code_xp04'/key)==capsule['files']['code_xp04/'+key]==value
    for key,value in manifest['data'].items():assert sha(ROOT/'code_xp04'/key)==value
    assert not (directory/'final_test.json').exists() and not list(result.glob('*.npz'))
    assert (directory/'best.pt').is_file() and (directory/'latest.pt').is_file()
    epochs=rows(directory/'metrics.jsonl');watch=rows(result/'watch_epochs.jsonl');batches=rows(result/'watch_batches.jsonl')
    module_audit=read(result/'module_audit.json')
    assert module_audit['cli_dropout']==.5 and module_audit['cli_rec_dropout_metadata']==.1
    assert set(module_audit['models'])=={'model','model_mm','model_intra','model_inter'}
    for instance in module_audit['models'].values():
        assert instance['language_gru']['dropout']==.5 and instance['language_gru']['num_layers']==2 and instance['language_gru']['bidirectional']
        assert instance['hidden_dropout_p']==.5
    verify_epochs(epochs,args,run['split_hash'],True);verify_epochs(reference['epochs'],reference['args'],run['split_hash'])
    old=stats(.4,'XP04_control',reference['epochs'],[],reference['args']);new=stats(.5,'10_dropout0p5',epochs,watch,args)
    assert status['selected_epoch']==new['best_epoch'] and status['selection_weighted_f1']==new['best_validation_wf1']
    assert old['best_epoch']==reference['best_epoch'] and old['best_validation_wf1']==reference['best_validation_wf1']
    assert math.isclose(old['last10_validation_wf1'],reference['last10_validation_wf1'],abs_tol=1e-12)
    capacity=read(ROOT/'preflight/capacity_cpu.json');caps={r['epoch']:r for r in capacity['records'] if r['run_id']==run['run_id']};assert set(caps)==set(range(1,41))
    steps=[]
    for epoch in range(1,41):
        actual=[b for b in batches if b['epoch']==epoch];cap=caps[epoch]
        assert [b['batch'] for b in actual]==list(range(1,len(actual)+1))
        assert len(actual)==cap['train_batches']==next(w for w in watch if w['split']=='train' and w['epoch']==epoch)['batches']
        assert next(w for w in watch if w['split']=='validation' and w['epoch']==epoch)['batches']==cap['validation_batches']
        for batch in actual:
            assert batch['gradients_finite'] and math.isfinite(batch['gradient_l2'])
            assert all(math.isfinite(batch[key]) for key in LOSS)
            assert math.isclose(batch['classification_loss']+sum(weighted(batch,args).values()),batch['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
        for d,basis in [(.4,'CPU_reconstructed_historical'),(.5,'observed_real_optimizer_steps')]:
            steps.append({'dropout':d,'epoch':epoch,'optimizer_steps':cap['train_batches'],'validation_batches':cap['validation_batches'],'basis':basis})
    expected=sum(c['train_batches'] for c in caps.values());assert len(batches)==expected
    old.update(optimizer_steps_observed=None,optimizer_steps_reconstructed=expected);new.update(optimizer_steps_observed=len(batches),optimizer_steps_reconstructed=expected)
    bd=new['best_validation_wf1']-old['best_validation_wf1'];td=new['last10_validation_wf1']-old['last10_validation_wf1']
    comparison={'seed':2024,'lr':1e-4,'batch_size':16,'rec_dropout_argument':.1,'control_dropout':.4,'new_dropout':.5,
                'control_best_validation_wf1':old['best_validation_wf1'],'new_best_validation_wf1':new['best_validation_wf1'],'control_best_epoch':old['best_epoch'],'new_best_epoch':new['best_epoch'],
                'control_last10_validation_wf1':old['last10_validation_wf1'],'new_last10_validation_wf1':new['last10_validation_wf1'],
                'delta_best_validation_wf1':bd,'delta_last10_validation_wf1':td,'candidate_category':qualification(bd,td)}
    curves=[];losses=[]
    for item,data,observations,config in [(old,reference['epochs'],[],reference['args']),(new,epochs,watch,args)]:
        for epoch in data:
            curves.append({'dropout':item['dropout'],'seed':2024,'lr':1e-4,'batch_size':16,'source':item['source'],'epoch':epoch['epoch'],**{key:epoch[key] for key in CURVE}})
            losses.append({'dropout':item['dropout'],'source':item['source'],'split':'train','epoch':epoch['epoch'],**{key:epoch[key] for key in LOSS},**weighted(epoch,config)})
        for observed in observations:
            if observed['split']=='validation':
                component=observed['loss_components'];losses.append({'dropout':item['dropout'],'source':item['source'],'split':'validation','epoch':observed['epoch'],**component,**weighted(component,config)})
    inputs=[ROOT/name for name in ['configs/registry.json','provenance/xp04_validation_references.json','provenance/source_origin.json','provenance/environment.json','provenance/frozen_execution.json','preflight/capacity_cpu.json','scripts/analyze_dropout.py']]
    inputs += [directory/'run_manifest.json',directory/'metrics.jsonl',result/'watch_epochs.jsonl',result/'watch_batches.jsonl',result/'status.json',result/'module_audit.json']
    inputs += [p for p in [ROOT/'preflight/gate.json',ROOT/'provenance/final_verification.json',ROOT/'reports/SOURCE_AND_PROTOCOL_AUDIT.md'] if p.exists()]
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,'new_runs_completed':1,'per_run':[old,new],'comparison':[comparison],'optimizer_step_records':steps,'module_audit':module_audit,
             'test_forward_passes':0,'automatic_additional_runs':False,'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
             'scope':'CPU training/validation records only; no test scores or prediction arrays read'}
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_run.csv',[old,new]),('comparison.csv',[comparison]),('curves.csv',curves),('loss_curves.csv',losses)]:csv_write(output/name,data)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    prose=report(summary,registry);path=ROOT/'reports/CMERC_DROPOUT_DIAGNOSIS.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    prefix=ROOT.relative_to(REPO).as_posix()
    for section in ('reports','configs','provenance','preflight','results'):prose=prose.replace('../'+section+'/', '../'+prefix+'/'+section+'/')
    path=REPO/'reports/CMERC_DROPOUT_DIAGNOSIS.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    print(json.dumps({'completed':1,'comparison':comparison},ensure_ascii=False))


if __name__=='__main__':main()
