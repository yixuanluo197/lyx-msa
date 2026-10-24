"""CPU-only, zero-test-result analysis of the registered MELD LR confirmation."""
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
LOSSES=('classification_loss','cc_loss','mscl_loss','cscl_loss','total_loss')
CURVES=('train_weighted_f1','validation_weighted_f1','train_accuracy','validation_accuracy',
        'train_loss','validation_loss','sample_ratio','train_dialogues','max_gpu_memory_mb')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_csv(path,data):
    keys=list(dict.fromkeys(key for row in data for key in row))
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader();writer.writerows(data)


def category(best_delta,late_delta):
    if best_delta<=0:
        return 'no_best_improvement'
    return 'strong_candidate' if late_delta>=0 else 'weak_mixed_candidate'


def confirmation_decision(comparisons):
    assert len(comparisons)==2 and {row['seed'] for row in comparisons}=={67137,2024}
    if all(row['candidate_category']=='strong_candidate' for row in comparisons):
        return {'decision':'both_seeds_meet_registered_qualification',
                'interpretation':'两个Seed均满足原登记资格，提供进一步独立复核的线索；不是显著性或五Seed稳定性证明。',
                'next_step':'建议：值得进入五Seed完整CMERC与Baseline配对方案的独立预登记和预算审查；当前两Seed仅为进一步验证线索，本批不自动启动五Seed、Baseline、Batch8或其他学习率。'}
    second=next(row for row in comparisons if row['seed']==2024)
    return {'decision':'not_consistently_confirmed',
            'interpretation':('第二Seed为最佳升但末段降的混合证据，未满足原登记确认资格。' if second['candidate_category']=='weak_mixed_candidate'
                              else '第二Seed未提高最佳验证W-F1，未确认首Seed的主指标收益。'),
            'next_step':'暂缓扩大五Seed；分析已有训练/验证轨迹，不临时改试另一个学习率、换Seed或追加Batch8。'}


def weighted(row,args):
    return {'weighted_cc_loss':args['rank_coff']*row['cc_loss'] if args['calibrate'] else 0.,
            'weighted_mscl_loss':args['mscl_coff']*row['mscl_loss'] if args['contrastlearning'] else 0.,
            'weighted_cscl_loss':args['cscl_coff']*row['cscl_loss'] if args['contrastlearning'] else 0.}


def verify_epochs(epochs,split_hash,original=False):
    assert len(epochs)==40
    assert [row['epoch'] for row in epochs]==list(range(1,41))
    for row in epochs:
        assert row['selection_protocol']=='validation' and row['selection_split']=='validation'
        assert row['split_hash']==split_hash
        assert row['validation_weighted_f1']==round(row['validation_weighted_f1'],2)
        assert row['selection_weighted_f1']==row['validation_weighted_f1']
        assert row['rng_policy']=='continuous' and row['batch_policy']=='capacity_safe'
        assert all(math.isfinite(row[key]) for key in LOSSES+CURVES[:6])
        if original:
            assert row['test_weighted_f1'] is None and row['test_accuracy'] is None
    return max(epochs,key=lambda row:row['validation_weighted_f1'])


def stats(seed,lr,source,epochs,watch,args):
    best=max(epochs,key=lambda row:row['validation_weighted_f1']);tail=epochs[-10:]
    result={'seed':seed,'lr':lr,'source':source,'epochs':len(epochs),'best_epoch':best['epoch'],
            'best_validation_wf1':best['validation_weighted_f1'],
            'last10_validation_wf1':statistics.mean(row['validation_weighted_f1'] for row in tail),
            'last10_validation_wf1_temporal_sd':statistics.stdev(row['validation_weighted_f1'] for row in tail),
            'last_validation_wf1':epochs[-1]['validation_weighted_f1'],
            'best_train_classification_loss':best['classification_loss'],
            'best_validation_classification_loss':best['validation_loss'],
            'last10_train_classification_loss':statistics.mean(row['classification_loss'] for row in tail),
            'last10_validation_classification_loss':statistics.mean(row['validation_loss'] for row in tail),
            'last10_total_training_loss':statistics.mean(row['total_loss'] for row in tail),
            'max_epoch_allocated_mib':max(row['max_gpu_memory_mb'] for row in epochs)}
    for key in ('cc_loss','mscl_loss','cscl_loss','total_loss'):
        result['best_train_'+key]=best[key]
        result['last10_train_'+key]=statistics.mean(row[key] for row in tail)
    train=[row for row in watch if row['split']=='train']
    if watch:
        assert len(watch)==80 and len(train)==40
        val=[row for row in watch if row['split']=='validation']
        assert len(val)==40 and all(row['split'] in ('train','validation') for row in watch)
        for part in (train,val):
            assert [row['epoch'] for row in part]==list(range(1,41))
        assert all(row['gradients_finite'] is True for row in train)
        for raw,observed in zip(epochs,val):
            assert round(observed['weighted_f1'],2)==raw['validation_weighted_f1']
        result['gradient_abnormalities']=0
        result['gradient_epoch_checks']=40
        result['gradient_l2_mean']=statistics.mean(row['gradient_l2_mean'] for row in train)
        result['gradient_l2_max']=max(row['gradient_l2_max'] for row in train)
        assert math.isfinite(result['gradient_l2_mean']) and math.isfinite(result['gradient_l2_max'])
    else:
        result.update(gradient_abnormalities=None,gradient_epoch_checks=0,gradient_l2_mean=None,gradient_l2_max=None)
    for row in epochs:
        expected=row['classification_loss']+sum(weighted(row,args).values())
        assert math.isclose(expected,row['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
    return result


def fmt(value,signed=False):
    return '未记录' if value is None else format(value,'+.4f' if signed else '.4f')


def report(summary,registry):
    decision=summary['confirmation'];run=registry['runs'][0]
    origin=read(ROOT/'provenance/source_origin.json')
    environment=read(ROOT/'provenance/environment.json')
    protocol=read(ROOT/'provenance/protocol_audit.json')
    training_sha=sha(ROOT/'code_xp04/train_cm.py')
    lines=['# CMERC 学习率第二Seed确认', '', f"生成时间：{summary['generated_at_utc']}。本批仅1/1正式运行完成：MELD seed2024、lr2e-4、40轮。", '',
           '## 设计与零测试协议', '',
           '首Seed67137在第07组依据预登记的最佳验证值优先排序选定lr2e-4；seed2024已在该阶段预留。本批只运行已选学习率的第二Seed，与XP04 seed2024完整CMERC lr1e-4作历史控制，不追加另一学习率、Baseline、Batch8、其他Seed或重训。两Seed中首Seed承担配置选择，第二Seed承担独立确认；不能把两点均值当作预先固定方案的五Seed性能估计。', '',
           '训练科学配置沿用XP04完整CMERC：Adam、batch16、40轮、dropout0.4、GRU、ATen、continuous RNG、BN、capacity_safe、hyper/direct、concat_DHT、L3/K3、split_seed1475、L2=3e-5；CC/MSCL/CSCL系数0.002/0.15/0.15、CL比例步长0.4/1。唯一科学配置变量为lr1e-4→2e-4。规范验证集Weighted F1按百分比两位小数选首个最高轮，严格提高才更新Checkpoint。', '',
           f"实际train_cm.py SHA-256：`{training_sha}`。源码复制自07的`{origin['copied_from']}`，12源字节一致审计状态为`{protocol['same_07_twelve_sources']}`，只保留07 skip-final-test补丁；来源记录中的Git HEAD为`{origin['head_commit']}`。", '',
           f"实际冻结环境：Python {environment['python'].split()[0]}、PyTorch {environment['packages']['torch']}、PyG {environment['packages']['torch-geometric']}、CUDA build {environment['torch_cuda_build']}、cuDNN {environment['cudnn']}；本批未安装或升级依赖。", '',
           '本阶段零测试loader遍历、零测试前向，不解析新旧测试指标、不读取预测数组或据测试筛选。综合pickle及测试loader元数据仍由原流程加载，不能称完全未加载含测试数据的文件。skip-final-test仅跳过训练后的原自动测试块，训练/验证数学与保存规则不变。本阶段未计算ECE/CEL，不能以未测值声称校准改善。', '',
           '## 两Seed的两个学习率', '',
           'W-F1单位为百分比，差值为百分点。末10轮固定指31–40轮。', '',
           '| Seed | lr | 来源 | 最佳验证W-F1 | Epoch | 末10轮均值 |', '|---:|---:|---|---:|---:|---:|']
    for row in summary['per_run']:
        lines.append(f"| {row['seed']} | {row['lr']:g} | {row['source']} | {fmt(row['best_validation_wf1'])} | {row['best_epoch']} | {fmt(row['last10_validation_wf1'])} |")
    lines+=['', '| Seed | Δ最佳验证 | Δ末10轮均值 | 按07登记的分类 |', '|---:|---:|---:|---|']
    labels={'strong_candidate':'满足候选资格（非统计强证据）','weak_mixed_candidate':'最佳升、末段降：混合线索','no_best_improvement':'主指标未改善'}
    for row in summary['comparison']:
        lines.append(f"| {row['seed']} | {fmt(row['delta_best_validation_wf1'],True)} | {fmt(row['delta_last10_validation_wf1'],True)} | {labels[row['candidate_category']]} |")
    second=next(row for row in summary['comparison'] if row['seed']==2024)
    trend=(f"Seed2024的最佳验证值变化{second['delta_best_validation_wf1']:+.4f} pp，末10轮均值变化{second['delta_last10_validation_wf1']:+.4f} pp。末段改善作为训练轨迹的辅助线索保留；本轮仍以预先确定的最佳验证W-F1为主指标，不据末段结果改变选模或确认规则。"
           if second['delta_best_validation_wf1']<=0 and second['delta_last10_validation_wf1']>0 else '')
    lines+=['', '**确认结论：** '+decision['interpretation'], '', trend, '', decision['next_step'], '',
            '资格门槛沿用07：Δ最佳>0且Δ末10轮>=0才合格；最佳升但末段降仅为弱线索，无候选资格。该门槛没有预设最小效应量或置信区间，不等于统计强证据。最佳与末段来自同一验证曲线，是相关统计；两Seed不作p值、显著性、等价或稳定复现声明。', '',
            '## 训练Loss与数值稳定性', '',
            '| Seed/lr | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 末段W-F1时间窗SD | 非有限梯度Epoch |', '|---|---:|---:|---:|---:|---:|---|']
    for row in summary['per_run']:
        lines.append(f"| {row['seed']}/{row['lr']:g} | {fmt(row['best_train_classification_loss'])} | {fmt(row['best_validation_classification_loss'])} | {fmt(row['last10_train_classification_loss'])} | {fmt(row['last10_validation_classification_loss'])} | {fmt(row['last10_validation_wf1_temporal_sd'])} | {row['gradient_abnormalities'] if row['gradient_epoch_checks'] else '历史未记录'} |")
    lines+=['']
    for row in summary['per_run']:
        tc=row['last10_train_classification_loss']-row['best_train_classification_loss'];vc=row['last10_validation_classification_loss']-row['best_validation_classification_loss'];fc=row['last10_validation_wf1']-row['best_validation_wf1']
        lines.append(f"- Seed{row['seed']}/lr={row['lr']:g}：最佳验证轮→末10轮均值，train分类Loss变化{tc:+.4f}，val分类Loss变化{vc:+.4f}，val W-F1变化{fc:+.4f} pp。")
    lines+=['', '训练原始辅助Loss（最佳验证轮→第31–40轮均值；固定系数CC=.002、MSCL/CSCL=.15）：', '',
            '| Seed/lr | CC | MSCL | CSCL | 总训练Loss |', '|---|---|---|---|---|']
    for row in summary['per_run']:
        parts=[fmt(row['best_train_'+key])+' → '+fmt(row['last10_train_'+key]) for key in ('cc_loss','mscl_loss','cscl_loss','total_loss')]
        lines.append(f"| {row['seed']}/{row['lr']:g} | "+' | '.join(parts)+' |')
    lines+=['', '以上是拟合继续增强或末段表现回落的描述线索。train为在线训练态逐Batch预测，val为固定权重eval，模式和参数时点不同，不能把两条Loss曲线间距当严格泛化gap。时间窗SD是相关Epoch内起伏，不是跨Seed标准差。finite梯度只证明所检数值有限，不能据此排除优化竞争、伪目标噪声或泛化问题。', '',
            '保留全部分类、原始CC/MSCL/CSCL及固定系数加权曲线。在线训练辅助项使用epoch入口辅助快照，Loss数值不等于参数梯度方向或校准质量。原实现Adam(weight_decay=l2)为耦合L2；L2系数固定不意味着改变lr后有效正则化更新轨迹相同。', '',
            '## 配对与资源限制', '',
            'seed2024的XP04控制和本次均为GPU3、UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456，改善了同物理卡配对；67137历史控制为GPU0，而07新lr2e-4为GPU3。CUDA scatter非确定性、历史运行时点和当前其他进程共驻仍限制严格因果解释。共驻任务及资源准入见[registry](../configs/registry.json)和资源日志；不终止其他项目，不把共享资源下耗时称纯模型速度。', '',
            '固定作者课程包含validation键，生成史未知；两臂同文件保持协议配对，但不能声称完整预处理链validation-free。单次验证选模与两Seed结果不能证明泛化到测试集或五Seed平均表现，更不能宣称lr是论文差距唯一根因。', '',
            '## 后续实验边界', '',
            '对最终五Seed的直接建议：'+decision['next_step'], '',
            '本阶段暂缓Batch8，没有独立隔离证据支持改变Batch。它同时改变更新步数、BN统计、HSCL对和未归一化CC尺度，需要另外预登记；暂缓属于控制变量与解释性安排，不是证明Batch8无益。不得通过换Seed或追加lr凑结果。', '',
            '## 产物与验收', '',
            '[预登记](../reports/PREREGISTRATION.md)、[配置](../configs/registry.json)、[训练环境](../provenance/environment.json)、[数据输入](../provenance/data_inputs.json)、[冻结清单](../provenance/frozen_execution.json)、[仅跳过最终测试的补丁](../provenance/skip_final_test.patch)、[预检](../preflight/gate.json)与[完整验收](../provenance/final_verification.json)。检查路径存在性不等于这些门禁已经通过；验收状态以对应JSON实际内容为准。', '',
            '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)、[metrics_summary.json](../results/metrics_summary.json)。所有CSV仅含train/validation字段。', '',
            '| Seed | W-F1轨迹 | 分类Loss | 加权总Loss |', '|---:|---|---|---|']
    for seed in (67137,2024):
        lines.append(f"| {seed} | [PNG](../results/figures/MELD_seed{seed}_wf1.png) | [PNG](../results/figures/MELD_seed{seed}_classification_loss.png) | [PNG](../results/figures/MELD_seed{seed}_weighted_loss.png) |")
    gate_path=ROOT/'preflight/gate.json'
    if gate_path.exists():
        gate=read(gate_path)
        lines+=['',f"正式预检门禁状态：`{gate.get('status','unknown')}`；其首步探针状态`{gate.get('probe_evidence',{}).get('status','unknown')}`，参数实际更新记录`{gate.get('probe_evidence',{}).get('first_step_parameters_changed','未记录')}`，test前向计数`{gate.get('test_forward_passes','未记录')}`。"]
    verification=ROOT/'provenance/final_verification.json'
    if verification.exists():
        v=read(verification);lines+=['',f"最终验收状态：`{v.get('status','unknown')}`。"]
        if v.get('status')=='passed':
            verified=v['runs'][run['run_id']]
            checks=verified['checkpoints']
            history_count=sum(v['historical_files_unchanged'].values())
            lines+=[f"单次完整验收覆盖40次train/40次validation、{verified['train_batches']}个训练Batch；唯一正式运行数{v['formal_runs']}，test前向{verified['test_forwards']}。逐轮训练池/排列重放匹配状态`{verified['every_epoch_order_matches_xp04_sampler']}`，Loss和梯度有限检查`{verified['loss_and_gradient_finite']}`。", '',
                    f"best/latest共{len(checks)}个Checkpoint：best第{checks['best.pt']['epoch']}轮，latest第{checks['latest.pt']['epoch']}轮；模型及Adam状态有限分别为`{checks['best.pt']['model_and_adam_finite']}`/`{checks['latest.pt']['model_and_adam_finite']}`，完整RNG字段存在分别为`{checks['best.pt']['full_rng_present']}`/`{checks['latest.pt']['full_rng_present']}`。验收脚本同时比对Checkpoint完整args/provenance与实际manifest，检查Adam参数组lr和完整RNG结构；通过状态不等于跨运行逐位数值复现。", '',
                    f"best SHA-256：`{checks['best.pt']['sha256']}`；latest SHA-256：`{checks['latest.pt']['sha256']}`。冻结源/输入未变状态`{v['frozen_sources_and_inputs_unchanged']}`；历史字节哈希保全合计{history_count}文件，具体根目录计数见验收JSON。"]
            resource=v.get('resource_evidence',{})
            if resource:
                lines+=['',f"资源证据状态`{resource.get('status','unknown')}`：{resource.get('resource_samples','未记录')}次离散采样，许可共驻PID采样{resource.get('allowed_preexisting_pid_samples','未记录')}次、新外部PID采样{resource.get('new_foreign_pid_samples','未记录')}次、查询错误{resource.get('resource_query_errors','未记录')}次。正式阶段最小空闲显存{resource.get('minimum_free_mib_during_formal','未记录')} MiB，最大利用率{resource.get('maximum_utilization_percent','未记录')}%；三次样本准入政策通过`{resource.get('admissions_passed_registered_three_sample_policy','未记录')}`。", '',
                        '资源采样限制：许可的既有PID共驻不等于独占GPU；运行期间新外部PID或资源压力只记录、不终止其他任务；离散nvidia-smi采样不能证明连续资源隔离。']
        else:lines+=['最终验收未通过，不宣称完整产物或资源检查已通过。']
    else:lines+=['','最终独立验收尚未生成；本报告不代替完整验收。']
    lines+=['',f'实验根目录：`{ROOT}`。仓库根报告正文一致，相对链接按位置调整。','']
    return '\n'.join(lines)


def main():
    registry=read(ROOT/'configs/registry.json');assert len(registry['runs'])==registry['formal_run_count']==1
    run=registry['runs'][0];assert run['seed']==2024 and run['dataset']=='MELD' and run['lr']==2e-4 and run['epochs']==40
    refs=read(ROOT/'provenance/xp04_validation_references.json');prior=read(ROOT/'provenance/seed67137_lr2e4_reference.json')
    result=ROOT/'results'/run['run_id'];status=read(result/'status.json')
    assert status['phase']=='completed' and status['counts']=={'train':40,'validation':40,'test':0}
    assert status['test_forward_passes']==0 and status['no_final_test_artifacts']
    current=rows(Path(run['run_dir'])/'metrics.jsonl');watch=rows(result/'watch_epochs.jsonl')
    manifest=read(Path(run['run_dir'])/'run_manifest.json');args=manifest['args'];reference=refs['MELD_2024']
    skip={'lr','output_dir','provenance','skip_final_test'}
    assert {k:v for k,v in args.items() if k not in skip}=={k:v for k,v in reference['args'].items() if k not in skip}
    assert args['lr']==2e-4 and args['skip_final_test']
    assert manifest['split']==reference['split'] and manifest['data']==reference['data']
    verify_epochs(current,run['split_hash'],original=True)
    assert not (Path(run['run_dir'])/'final_test.json').exists() # Guard only, never opens it.
    assert (Path(run['run_dir'])/'best.pt').is_file() and (Path(run['run_dir'])/'latest.pt').is_file()
    protocols=[];curves=[];losses=[];by_seed={}
    for seed in (67137,2024):
        ref=refs[f'MELD_{seed}'];verify_epochs(ref['epochs'],ref['split']['hash'])
        old=stats(seed,1e-4,'XP04_control',ref['epochs'],[],ref['args']);assert old['best_epoch']==ref['best_epoch'] and old['best_validation_wf1']==ref['best_validation_wf1']
        assert math.isclose(old['last10_validation_wf1'],ref['last10_validation_wf1'],abs_tol=1e-12)
        epochs,new_watch=(prior['epochs'],prior['watch_epochs']) if seed==67137 else (current,watch)
        verify_epochs(epochs,ref['split']['hash'])
        new=stats(seed,2e-4,'07_selected_lr' if seed==67137 else '08_confirmation',epochs,new_watch,{**ref['args'],'lr':2e-4})
        if seed==2024:
            assert status['selected_epoch']==new['best_epoch'] and status['selection_weighted_f1']==new['best_validation_wf1']
        else:
            assert new['best_epoch']==prior['summary']['best_epoch'] and new['best_validation_wf1']==prior['summary']['best_validation_wf1']
            assert math.isclose(new['last10_validation_wf1'],prior['summary']['last10_validation_wf1'],abs_tol=1e-12)
        protocols.extend([old,new]);by_seed[seed]=(old,new)
        for config,items,observations in [(old,ref['epochs'],[]),(new,epochs,new_watch)]:
            for epoch in items:
                curves.append({'seed':seed,'lr':config['lr'],'source':config['source'],'epoch':epoch['epoch'],**{key:epoch[key] for key in CURVES}})
                losses.append({'seed':seed,'lr':config['lr'],'source':config['source'],'split':'train','epoch':epoch['epoch'],**{key:epoch[key] for key in LOSSES},**weighted(epoch,ref['args'])})
            for observed in observations:
                if observed['split']=='validation':
                    component=observed['loss_components'];losses.append({'seed':seed,'lr':config['lr'],'source':config['source'],'split':'validation','epoch':observed['epoch'],**component,**weighted(component,ref['args'])})
    comparisons=[]
    for seed,(old,new) in by_seed.items():
        bd=new['best_validation_wf1']-old['best_validation_wf1'];ld=new['last10_validation_wf1']-old['last10_validation_wf1']
        comparisons.append({'seed':seed,'control_lr':1e-4,'new_lr':2e-4,'control_best_validation_wf1':old['best_validation_wf1'],'new_best_validation_wf1':new['best_validation_wf1'],
                            'control_best_epoch':old['best_epoch'],'new_best_epoch':new['best_epoch'],'control_last10_validation_wf1':old['last10_validation_wf1'],'new_last10_validation_wf1':new['last10_validation_wf1'],
                            'delta_best_validation_wf1':bd,'delta_last10_validation_wf1':ld,'candidate_category':category(bd,ld)})
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,'new_runs_completed':1,'historical_control_seeds':[67137,2024],
             'per_run':protocols,'comparison':comparisons,'confirmation':confirmation_decision(comparisons),'automatic_additional_runs':False,
             'test_forward_passes':0,'scope':'CPU-only train/validation JSON analysis; no test scores or prediction arrays opened',
             'inputs_sha256':{name:sha(ROOT/name) for name in ['configs/registry.json','provenance/xp04_validation_references.json','provenance/seed67137_lr2e4_reference.json','scripts/analyze_second_seed.py']}}
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_run.csv',protocols),('comparison.csv',comparisons),('curves.csv',curves),('loss_curves.csv',losses)]:output_csv(output/name,data)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    prose=report(summary,registry);path=ROOT/'reports/CMERC_LR_SECOND_SEED_CONFIRMATION.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    prefix=ROOT.relative_to(REPO).as_posix()
    for directory in ('reports','configs','provenance','preflight','results'):prose=prose.replace('../'+directory+'/', '../'+prefix+'/'+directory+'/')
    path=REPO/'reports/CMERC_LR_SECOND_SEED_CONFIRMATION.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    print(json.dumps({'completed':1,'confirmation':summary['confirmation']},ensure_ascii=False))


if __name__=='__main__':
    main()
