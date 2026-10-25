"""Strict CPU-only analysis of the single registered batch-size experiment.

Reads training/validation logs only. No checkpoint inference, test result reads,
provisional output, scheduling or training is performed by this script.
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


def csv_write(path,records):
    keys=list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)


def qualification(best_delta,tail_delta):
    if best_delta<=0:return 'no_best_improvement'
    return 'second_seed_candidate' if tail_delta>=0 else 'mixed_evidence'


def recommendation(label):
    if label=='second_seed_candidate':
        return '本Seed满足原登记资格，值得另行登记第二Seed确认；这不是统计强证据。本批不自动启动第二Seed、最终五Seed、其他Batch Size或学习率。'
    if label=='mixed_evidence':
        return '最佳升而末段降，证据混合，暂不推荐扩大第二Seed或五Seed；分析现有训练/验证轨迹，不追加训练择优。'
    assert label=='no_best_improvement'
    return '预登记主指标未改善，本Seed不支持扩大第二Seed或五Seed；保留末段与Loss辅助线索，不改主指标，不自动换学习率、Batch Size或Seed。'


def weighted(row,args):
    return {'weighted_cc_loss':args['rank_coff']*row['cc_loss'] if args['calibrate'] else 0.,
            'weighted_mscl_loss':args['mscl_coff']*row['mscl_loss'] if args['contrastlearning'] else 0.,
            'weighted_cscl_loss':args['cscl_coff']*row['cscl_loss'] if args['contrastlearning'] else 0.}


def verify_trajectory(epochs,args,split_hash,new=False):
    assert len(epochs)==40 and [row['epoch'] for row in epochs]==list(range(1,41))
    for row in epochs:
        assert row['selection_protocol']=='validation' and row['selection_split']=='validation'
        assert row['split_hash']==split_hash
        assert row['rng_policy']=='continuous' and row['batch_policy']=='capacity_safe'
        assert row['selection_weighted_f1']==row['validation_weighted_f1']==round(row['validation_weighted_f1'],2)
        assert all(math.isfinite(row[key]) for key in LOSS+CURVE[:6])
        assert math.isclose(row['classification_loss']+sum(weighted(row,args).values()),row['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
        if new:
            assert row['test_weighted_f1'] is None and row['test_accuracy'] is None
    return max(epochs,key=lambda row:row['validation_weighted_f1'])


def stats(batch,source,epochs,watch,args):
    best=max(epochs,key=lambda row:row['validation_weighted_f1']);tail=epochs[-10:]
    result={'seed':2024,'lr':args['lr'],'batch_size':batch,'validation_batch_size':batch,'source':source,'epochs':40,
            'best_epoch':best['epoch'],'best_validation_wf1':best['validation_weighted_f1'],
            'last10_validation_wf1':statistics.mean(row['validation_weighted_f1'] for row in tail),
            'last10_validation_wf1_temporal_sd':statistics.stdev(row['validation_weighted_f1'] for row in tail),
            'last_validation_wf1':epochs[-1]['validation_weighted_f1'],
            'best_train_classification_loss':best['classification_loss'],'best_validation_classification_loss':best['validation_loss'],
            'last10_train_classification_loss':statistics.mean(row['classification_loss'] for row in tail),
            'last10_validation_classification_loss':statistics.mean(row['validation_loss'] for row in tail),
            'max_epoch_allocated_mib':max(row['max_gpu_memory_mb'] for row in epochs)}
    for key in LOSS:
        result['best_train_'+key]=best[key];result['last10_train_'+key]=statistics.mean(row[key] for row in tail)
    for prefix,value in [('best',best),('last10',{k:statistics.mean(row[k] for row in tail) for k in LOSS})]:
        for field,weighted_value in weighted(value,args).items():result[prefix+'_train_'+field]=weighted_value
    train=[row for row in watch if row['split']=='train'];val=[row for row in watch if row['split']=='validation']
    if watch:
        assert len(watch)==80 and len(train)==len(val)==40
        assert all(row['split'] in ('train','validation') for row in watch)
        for part in (train,val):assert [row['epoch'] for row in part]==list(range(1,41))
        assert all(row['gradients_finite'] is True and row['first_step_parameters_changed'] is True for row in train)
        assert all(round(v['weighted_f1'],2)==e['validation_weighted_f1'] for e,v in zip(epochs,val))
        result.update(gradient_epoch_checks=40,gradient_abnormalities=0,
                      gradient_l2_mean=statistics.mean(row['gradient_l2_mean'] for row in train),
                      gradient_l2_max=max(row['gradient_l2_max'] for row in train))
        assert math.isfinite(result['gradient_l2_mean']) and math.isfinite(result['gradient_l2_max'])
    else:result.update(gradient_epoch_checks=0,gradient_abnormalities=None,gradient_l2_mean=None,gradient_l2_max=None)
    return result


def fmt(value,signed=False):return '历史未记录' if value is None else format(value,'+.4f' if signed else '.4f')


def report(summary,registry):
    comp=summary['comparison'][0];protocols=summary['per_run'];run=registry['runs'][0]
    origin=read(ROOT/'provenance/source_origin.json');environment=read(ROOT/'provenance/environment.json')
    policy=registry['resource_policy']
    co_resident_pids=policy.get('preexisting_gpu_pids',policy.get('preregistered_co_resident_pids',policy.get('preexisting_gpu3_pids',[])))
    historical_gpu_uuid='GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'
    same_physical=(registry['gpu_index']==3 and registry['gpu_uuid']==historical_gpu_uuid)
    pairing_note=('本次与XP04控制同物理卡。' if same_physical else '本次与XP04控制为同host、同型号、不同物理卡；不能声称同物理卡配对。')
    current_sha=sha(ROOT/'code_xp04/train_cm.py')
    labels={'second_seed_candidate':'满足第二Seed候选资格（非统计强证据）','mixed_evidence':'最佳升、末段降：混合线索','no_best_improvement':'预登记主指标未改善'}
    lines=['# CMERC Batch Size 诊断', '', f"生成时间：{summary['generated_at_utc']}。本批仅1/1正式训练完成：MELD seed2024、lr1e-4、Batch8、40轮。", '',
           '## 设计、参数与来源', '',
           '本批直接对照XP04完整CMERC seed2024、lr1e-4、Batch16；不继承08的lr2e-4，不改变任何损失公式或系数。实际args只允许batch_size、skip_final_test、output_dir和provenance与历史不同；lr必须始终为0.0001。', '',
           '实际模型：Adam、L2=3e-5、dropout0.4、GRU、ATen、continuous RNG、BN、capacity_safe、hyper/direct、concat_DHT、L3/K3、FP32、worker0、split_seed1475。CC/MSCL/CSCL=0.002/0.15/0.15，CL epoch_ratio0.4、scheduler_steps1，40轮、train1038/validation114对话。验证集完整池和ID顺序固定，split SHA-256为`'+run['split_hash']+'`。', '',
           f"实际train_cm.py SHA-256：`{current_sha}`；12个训练源字节同08的来源记录`{origin['copied_from']}`，来源Git HEAD为`{origin['head_commit']}`。只继承既有skip-final-test工程门禁，训练/验证公式未改。", '',
           f"冻结环境：Python {environment['python'].split()[0]}、PyTorch {environment['packages']['torch']}、PyG {environment['packages']['torch-geometric']}、CUDA build {environment['torch_cuda_build']}；依赖未新增。", '',
           '## Batch变量的实际作用范围', '',
           '原始--batch-size同时控制训练和验证loader，因此本次train与validation分组均16→8。比较对象是这一公开参数的整体影响，不能称只改变训练Batch而保持验证分组不变。验证W-F1由完整固定验证池的全部labels/predictions汇总计算，不是逐Batch F1的平均；按百分比两位小数严格提高才换Checkpoint，同分取首个最高轮。', '',
           'Batch8同时改变每Epoch optimizer更新次数、BN统计、HSCL正负对组成及未按句数归一化CC的每Batch尺度。RNG策略和初始Seed相同不意味着global dropout RNG轨迹相同：Batch数量、张量形状与前向次数改变随机数消耗。L2与lr系数固定也不代表总更新次数或有效正则化轨迹相同。', '',
           '源码model_cm.py:658,758–762,824显示双向GRU直接处理未pack的padded序列，BN亦包含padding；因此验证分组变化可能改变固定权重预测，不能预设它只改变Loss统计标尺。具体源码行号及mask/配对路径见[协议审计](../reports/PROTOCOL_AUDIT.md)。', '',
           '原训练/验证Loss采用Batch均值汇总；不同大小及末Batch的权重随分组变化。原始CC/MSCL/CSCL均值跨Batch Size不能直接解释为同一个总样本归一化目标的增减，更不能仅凭数值减半判断校准作用增强或减弱。', '',
           '## 最佳值、末段与原登记判断', '', '| Batch Size（train/val） | 来源 | 最佳验证W-F1 % | Epoch | 末10轮均值 % | 末10轮时间窗SD |', '|---:|---|---:|---:|---:|---:|']
    for row in protocols:lines.append(f"| {row['batch_size']} | {row['source']} | {fmt(row['best_validation_wf1'])} | {row['best_epoch']} | {fmt(row['last10_validation_wf1'])} | {fmt(row['last10_validation_wf1_temporal_sd'])} |")
    lines+=['',f"Batch8−Batch16：最佳验证 **{fmt(comp['delta_best_validation_wf1'],True)} pp**，末10轮均值 **{fmt(comp['delta_last10_validation_wf1'],True)} pp**；分类：**{labels[comp['candidate_category']]}**。", '',
            recommendation(comp['candidate_category']), '',
            '规则保持训练前登记：最佳值升且末10轮不降才具备第二Seed候选资格；最佳升末段降为mixed；最佳不升为no_best_improvement，不能凭辅助末段结果更换主指标。末10轮固定31–40，与最佳值来自同一曲线，是相关统计而非独立证据。一个Seed不提供p值、显著性、等价或五Seed稳定复现声明。', '',
            '## 实际更新次数与Loss', '', '| Batch | 40轮训练更新数 | 计数依据 | 40轮验证Batch总数 |', '|---:|---:|---|---:|']
    for row in protocols:
        steps=row['optimizer_steps_observed'] if row['optimizer_steps_observed'] is not None else row['optimizer_steps_reconstructed']
        basis='每次真实optimizer.step后的watch日志计数，并与容量重放核对' if row['optimizer_steps_observed'] is not None else '历史Batch16 CPU采样重建；不是原运行step日志观测'
        lines.append(f"| {row['batch_size']} | {steps} | {basis} | {row['validation_batches_total']} |")
    lines+=['', '| Batch | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 梯度异常Epoch |', '|---:|---:|---:|---:|---:|---|']
    for row in protocols:lines.append(f"| {row['batch_size']} | {fmt(row['best_train_classification_loss'])} | {fmt(row['best_validation_classification_loss'])} | {fmt(row['last10_train_classification_loss'])} | {fmt(row['last10_validation_classification_loss'])} | {row['gradient_abnormalities'] if row['gradient_epoch_checks'] else '历史未记录'} |")
    lines+=['', '最佳验证轮→末10轮训练分项均值：', '', '| Batch | 原始CC | 原始MSCL | 原始CSCL | 加权CC | 加权MSCL | 加权CSCL | 总Loss |', '|---:|---|---|---|---|---|---|---|']
    for row in protocols:
        fields=('cc_loss','mscl_loss','cscl_loss','weighted_cc_loss','weighted_mscl_loss','weighted_cscl_loss','total_loss')
        lines.append(f"| {row['batch_size']} | "+' | '.join(fmt(row['best_train_'+key])+' → '+fmt(row['last10_train_'+key]) for key in fields)+' |')
    lines+=['']
    for row in protocols:
        tc=row['last10_train_classification_loss']-row['best_train_classification_loss'];vc=row['last10_validation_classification_loss']-row['best_validation_classification_loss'];wc=row['last10_validation_wf1']-row['best_validation_wf1']
        lines.append(f"- Batch{row['batch_size']}最佳轮→末段：train分类Loss {tc:+.4f}，val分类Loss {vc:+.4f}，val W-F1 {wc:+.4f} pp。")
    lines+=['', 'train为在线训练态、val为固定模型eval，不同模式与参数时点下的曲线间距不是严格泛化gap。末段时间窗SD不是跨Seed标准差；有限梯度与真实参数更新只能证明所检数值和执行路径正常，不能据此排除目标竞争、伪目标噪声或泛化根因。训练辅助模型保留epoch入口快照语义。', '',
            '## 零测试与配对边界', '',
            '本次零测试loader遍历、零测试前向；不解析测试指标或预测数组、不据测试筛选。本阶段未计算ECE/CEL。原综合pickle及测试loader元数据仍加载，零前向不等于完全未加载含测试数据的文件。训练后原自动测试由明确skip-final-test门禁跳过，不进行新测试评估。', '',
            f"XP04 seed2024控制位于本地user23 GPU3（{historical_gpu_uuid}）；本批实际设备为同host GPU{registry['gpu_index']}（{registry['gpu_uuid']}）。{pairing_note}CUDA scatter数值非确定性、硬件/历史时点和共享资源仍限制严格因果与速度解释；不得称独占资源或连续隔离，不终止其他项目。registry当前登记的共驻PID为`{co_resident_pids}`，实际观察到的PID另见最终资源证据。作者fixed_author课程包含validation键、生成史未知；使用同一文件保持配对，不称完整预处理链validation-free。", '',
            '设备选择在任何GPU探针和正式训练前由CPU selector监控同host健康TITAN Xp候选GPU0/1/3；首个连续三次新采样满足资源v2门槛者作为唯一设备，记录实际registry和冻结清单后不再换卡。完整候选、时间及实际选择见[XP设备选择工程修订v3](../reports/XP_DEVICE_SELECTION_AMENDMENT_V3.md)与[设备选择记录](../provenance/gpu_selection.json)。本修订不改变科学配置、候选资格或唯一训练预算。', '',
            f"资源准入v2按registry实际值执行：free>={policy['admission_min_free_mib']} MiB、util<={policy['admission_max_utilization_percent']}%、{policy['samples_required']}次新采样、间隔{policy['sample_seconds']}秒；本任务估计预算{policy['estimated_own_peak_mib']} MiB。新增外部PID是否暂停准入的登记值为`{policy['new_foreign_pids_pause_admission']}`，不再以PID更替本身作硬拒绝；身份、资源和查询有效性仍需核验。", '',
            'v2仅修订资源管理，原v1门禁与等待记录已原字节归档；科学配置、唯一正式预算、零测试与候选资格不变。时间和资源依据见[资源准入工程修订v2](../reports/RESOURCE_ADMISSION_AMENDMENT_V2.md)，不将等待结束或门禁放行解释为训练成功。', '',
            '## 建议及证据状态', '', '**当前直接建议：** '+recommendation(comp['candidate_category']), '',
            '已证实事实是本次固定配置、日志轨迹、数值执行和登记主指标的比较；优化机制、校准收益、跨Seed稳定性和论文差距根因仍未证实。一个Batch参数改变产生多项相互耦合的执行变化，不能将观测差值分摊为BN、HSCL或CC的单独效应。后续仅提出独立预登记建议，不自动开展五Seed、其他Batch Size、lr、Seed或测试。', '',
            '## 可追溯材料', '',
            '[预登记](../reports/PREREGISTRATION.md)、[配置](../configs/registry.json)、[环境](../provenance/environment.json)、[数据](../provenance/data_inputs.json)、[来源](../provenance/source_origin.json)、[冻结清单](../provenance/frozen_execution.json)、[工程补丁](../provenance/skip_final_test.patch)、[容量与两种Batch的重放](../preflight/capacity_cpu.json)、[正式预检门禁](../preflight/gate.json)、[最终验收](../provenance/final_verification.json)。', '',
            '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)。', '',
            '[W-F1曲线](../results/figures/MELD_batch_size_wf1.png)、[分类Loss](../results/figures/MELD_batch_size_classification_loss.png)、[加权总Loss](../results/figures/MELD_batch_size_weighted_loss.png)、[每轮更新计数](../results/figures/MELD_batch_size_optimizer_steps.png)。其余原始及加权辅助Loss图和全部输入/输出SHA见[图表清单](../results/figures/manifest.json)。']
    gate=ROOT/'preflight/gate.json'
    if gate.exists():
        g=read(gate);lines+=['',f"预检门禁实际状态：`{g.get('status','unknown')}`，test前向计数`{g.get('test_forward_passes','未记录')}`。"]
    final=ROOT/'provenance/final_verification.json'
    if final.exists():
        verification=read(final);lines+=['',f"最终验收实际状态：`{verification.get('status','unknown')}`。"]
        if verification.get('status')=='passed':
            v=verification['runs'][run['run_id']];ck=v['checkpoints'];history=sum(verification['historical_files_unchanged'].values())
            lines += [f"单run完成40次train/40次validation，记录{v['train_batches']}个真实训练Batch，test前向{v['test_forwards']}。登记Batch8排列重放匹配`{v['every_epoch_order_matches_registered_batch8_replay']}`，原sampler AST与历史Batch16重放通过`{v['unchanged_sampler_AST_and_historical_batch16_replay_passed']}`，实际train/val Batch均`{v['train_and_validation_batch_size']}`、lr保持`{v['lr_unchanged_from_xp04']}`。", '',
                      f"best/latest两个Checkpoint分别第{ck['best.pt']['epoch']}/{ck['latest.pt']['epoch']}轮，模型和Adam有限`{ck['best.pt']['model_and_adam_finite']}`/`{ck['latest.pt']['model_and_adam_finite']}`，完整RNG存在`{ck['best.pt']['full_rng_present']}`/`{ck['latest.pt']['full_rng_present']}`；验收脚本同时核对完整args/provenance和Adam lr。best SHA `{ck['best.pt']['sha256']}`，latest SHA `{ck['latest.pt']['sha256']}`。冻结源/输入未变`{verification['frozen_sources_and_inputs_unchanged']}`，历史文件哈希保全合计{history}项。"]
            resource=verification.get('resource_evidence',{})
            if resource:lines+=['',f"资源证据`{resource.get('status','unknown')}`：采样{resource.get('resource_samples','未记录')}次、预登记共驻PID采样{resource.get('allowed_preexisting_pid_samples','未记录')}次、新外部PID采样{resource.get('new_foreign_pid_samples','未记录')}次、查询错误{resource.get('resource_query_errors','未记录')}次；正式阶段最小空闲{resource.get('minimum_free_mib_during_formal','未记录')} MiB、最大利用率{resource.get('maximum_utilization_percent','未记录')}%。", '',
                                 f"最终验收记录的预登记共驻PID：`{resource.get('preregistered_co_resident_pids',co_resident_pids)}`；实际观察到的外部PID：`{resource.get('observed_foreign_pids','未记录')}`。验收资源门槛free>={resource.get('admission_min_free_mib',policy['admission_min_free_mib'])} MiB、util<={resource.get('admission_max_utilization_percent',policy['admission_max_utilization_percent'])}%，新增PID暂停准入值`{resource.get('new_foreign_pids_pause_admission',policy['new_foreign_pids_pause_admission'])}`。离散采样不能证明连续独占或资源隔离；运行期压力和新增PID仅记录，不终止他人任务。"]
        else:lines+=['验收尚未通过，不宣称完整验收成功。']
    else:lines+=['','最终验收文件尚未生成；统计门禁不代替独立产物验收。']
    lines+=['',f'本批根目录：`{ROOT}`。仓库根报告正文一致，链接按所在位置调整。','']
    return '\n'.join(lines)


def main():
    registry=read(ROOT/'configs/registry.json');assert registry['formal_run_count']==len(registry['runs'])==1
    run=registry['runs'][0];assert run['seed']==2024 and run['lr']==1e-4 and run['batch_size']==8 and run['epochs']==40
    reference=read(ROOT/'provenance/xp04_validation_references.json')['MELD_2024']
    result=ROOT/'results'/run['run_id'];status=read(result/'status.json')
    assert status['phase']=='completed' and status['counts']=={'train':40,'validation':40,'test':0}
    assert status['test_forward_passes']==0 and status['no_final_test_artifacts']
    directory=Path(run['run_dir']);manifest=read(directory/'run_manifest.json');args=manifest['args']
    allowed={'batch_size','skip_final_test','output_dir','provenance'}
    assert {k:v for k,v in args.items() if k not in allowed}=={k:v for k,v in reference['args'].items() if k not in allowed}
    assert args['lr']==reference['args']['lr']==1e-4 and args['batch_size']==8 and reference['args']['batch_size']==16 and args['skip_final_test']
    assert manifest['split']==reference['split'] and manifest['data']==reference['data']
    capsule=read(ROOT/'provenance/frozen_execution.json')
    for key,value in manifest['source']['files'].items():assert sha(ROOT/'code_xp04'/key)==capsule['files']['code_xp04/'+key]==value
    for key,value in manifest['data'].items():assert sha(ROOT/'code_xp04'/key)==value
    assert not (directory/'final_test.json').exists() and not list(result.glob('*.npz'))
    assert (directory/'best.pt').is_file() and (directory/'latest.pt').is_file()
    epochs=rows(directory/'metrics.jsonl');watch=rows(result/'watch_epochs.jsonl');batches=rows(result/'watch_batches.jsonl')
    old_epochs=reference['epochs'];verify_trajectory(epochs,args,run['split_hash'],new=True);verify_trajectory(old_epochs,reference['args'],run['split_hash'])
    new=stats(8,'09_batch8',epochs,watch,args);old=stats(16,'XP04_batch16',old_epochs,[],reference['args'])
    assert status['selected_epoch']==new['best_epoch'] and status['selection_weighted_f1']==new['best_validation_wf1']
    assert old['best_epoch']==reference['best_epoch'] and old['best_validation_wf1']==reference['best_validation_wf1']
    assert math.isclose(old['last10_validation_wf1'],reference['last10_validation_wf1'],abs_tol=1e-12)
    capacity=read(ROOT/'preflight/capacity_cpu.json');assert capacity['historical_batch16_replay_passed'] is True
    caps={r['epoch']:r for r in capacity['records'] if r['run_id']==run['run_id']};assert set(caps)==set(range(1,41))
    step_records=[]
    for epoch in range(1,41):
        steps=[b for b in batches if b['epoch']==epoch];cap=caps[epoch]
        assert [b['batch'] for b in steps]==list(range(1,len(steps)+1))
        assert len(steps)==cap['train_batches']==next(r for r in watch if r['split']=='train' and r['epoch']==epoch)['batches']
        assert next(r for r in watch if r['split']=='validation' and r['epoch']==epoch)['batches']==cap['validation_batches']
        assert all(b['gradients_finite'] and math.isfinite(b['gradient_l2']) for b in steps)
        for b in steps:assert math.isclose(b['classification_loss']+sum(weighted(b,args).values()),b['total_loss'],rel_tol=3e-6,abs_tol=3e-5)
        for batch,label,train_count,val_count in [(8,'observed_after_real_step',len(steps),cap['validation_batches']),(16,'CPU_historical_reconstruction',cap['historical16']['train_batches'],cap['historical16']['validation_batches'])]:
            step_records.append({'epoch':epoch,'batch_size':batch,'optimizer_steps':train_count,'validation_batches':val_count,'basis':label})
    new.update(optimizer_steps_observed=len(batches),optimizer_steps_reconstructed=sum(c['train_batches'] for c in caps.values()),validation_batches_total=sum(c['validation_batches'] for c in caps.values()))
    old.update(optimizer_steps_observed=None,optimizer_steps_reconstructed=sum(c['historical16']['train_batches'] for c in caps.values()),validation_batches_total=sum(c['historical16']['validation_batches'] for c in caps.values()))
    assert new['optimizer_steps_observed']==new['optimizer_steps_reconstructed']
    delta_best=new['best_validation_wf1']-old['best_validation_wf1'];delta_tail=new['last10_validation_wf1']-old['last10_validation_wf1']
    comparison={'seed':2024,'lr':1e-4,'historical_batch_size':16,'new_batch_size':8,'control_best_validation_wf1':old['best_validation_wf1'],'new_best_validation_wf1':new['best_validation_wf1'],
                'control_best_epoch':old['best_epoch'],'new_best_epoch':new['best_epoch'],'control_last10_validation_wf1':old['last10_validation_wf1'],'new_last10_validation_wf1':new['last10_validation_wf1'],
                'delta_best_validation_wf1':delta_best,'delta_last10_validation_wf1':delta_tail,'candidate_category':qualification(delta_best,delta_tail)}
    curve_records=[];loss_records=[]
    for item,data,observations,config in [(old,old_epochs,[],reference['args']),(new,epochs,watch,args)]:
        for epoch in data:
            curve_records.append({'batch_size':item['batch_size'],'seed':2024,'lr':1e-4,'source':item['source'],'epoch':epoch['epoch'],**{k:epoch[k] for k in CURVE}})
            loss_records.append({'batch_size':item['batch_size'],'source':item['source'],'split':'train','epoch':epoch['epoch'],**{k:epoch[k] for k in LOSS},**weighted(epoch,config)})
        for observed in observations:
            if observed['split']=='validation':
                comp=observed['loss_components'];loss_records.append({'batch_size':item['batch_size'],'source':item['source'],'split':'validation','epoch':observed['epoch'],**comp,**weighted(comp,config)})
    input_files=[ROOT/name for name in ['configs/registry.json','provenance/xp04_validation_references.json','preflight/capacity_cpu.json','scripts/analyze_batch_size.py','provenance/source_origin.json','provenance/environment.json','provenance/frozen_execution.json']]
    input_files += [directory/'metrics.jsonl',directory/'run_manifest.json',result/'status.json',result/'watch_epochs.jsonl',result/'watch_batches.jsonl']
    input_files += [path for path in [ROOT/'preflight/gate.json',ROOT/'provenance/final_verification.json',ROOT/'reports/RESOURCE_ADMISSION_AMENDMENT_V2.md',ROOT/'reports/XP_DEVICE_SELECTION_AMENDMENT_V3.md',ROOT/'provenance/gpu_selection.json'] if path.exists()]
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,'new_runs_completed':1,'per_run':[old,new],'comparison':[comparison],
             'optimizer_step_records':step_records,'test_forward_passes':0,'automatic_additional_runs':False,
             'inputs_sha256':{str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):sha(path) for path in input_files},
             'scope':'CPU train/validation analysis only; no test metrics or predictions read'}
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_run.csv',[old,new]),('comparison.csv',[comparison]),('curves.csv',curve_records),('loss_curves.csv',loss_records)]:csv_write(output/name,data)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    prose=report(summary,registry);path=ROOT/'reports/CMERC_BATCH_SIZE_DIAGNOSIS.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    prefix=ROOT.relative_to(REPO).as_posix()
    for section in ('reports','configs','provenance','preflight','results'):prose=prose.replace('../'+section+'/', '../'+prefix+'/'+section+'/')
    path=REPO/'reports/CMERC_BATCH_SIZE_DIAGNOSIS.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(prose)
    print(json.dumps({'completed':1,'comparison':comparison},ensure_ascii=False))


if __name__=='__main__':main()
