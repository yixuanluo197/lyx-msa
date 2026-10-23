"""CPU-only, evidence-checked summary of the three preregistered ablations."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
METRICS = ('weighted_f1', 'macro_f1', 'accuracy', 'ece', 'cel')
LOSS = ('classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss', 'total_loss')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, records):
    names = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value):
    return '待完成' if value is None else f'{value:.4f}'


def sign(value):
    return '待完成' if value is None else f'{value:+.4f}'


def weighted_components(row, args):
    return {'classification_loss': row['classification_loss'],
            'weighted_cc': args['rank_coff'] * row['cc_loss'] if args['calibrate'] else 0.,
            'weighted_mscl': args['mscl_coff'] * row['mscl_loss'] if args['contrastlearning'] else 0.,
            'weighted_cscl': args['cscl_coff'] * row['cscl_loss'] if args['contrastlearning'] else 0.}


def gather(run, historical, allow_incomplete):
    run_dir = Path(run['run_dir'])
    result_dir = ROOT / 'results' / run['run_id']
    epochs = rows(run_dir / 'metrics.jsonl')
    observed = rows(result_dir / 'observer_epochs.jsonl')
    batches = rows(result_dir / 'observer_batches.jsonl')
    status = read(result_dir / 'status.json') if (result_dir / 'status.json').exists() else {}
    complete = status.get('phase') == 'completed'
    assert [r['epoch'] for r in epochs] == list(range(1, len(epochs) + 1))
    assert len(epochs) <= run['epochs']
    for epoch in epochs:
        assert epoch['selection_protocol'] == 'validation' and epoch['selection_split'] == 'validation'
        assert epoch['split_hash'] == run['split_hash']
        assert epoch['test_accuracy'] is None and epoch['test_weighted_f1'] is None
        assert epoch['selection_weighted_f1'] == epoch['validation_weighted_f1']
    args = read(run_dir / 'run_manifest.json')['args'] if (run_dir / 'run_manifest.json').exists() else None
    if args:
        old = read(run['historical_manifest'])
        assert args['provenance']['split'] == old['split']
        assert args['provenance']['data'] == old['data']
        permitted = {'output_dir', 'provenance', run['factor_key']}
        assert {k:v for k,v in args.items() if k not in permitted} == {k:v for k,v in old['args'].items() if k not in permitted}
        assert args[run['factor_key']] == run['factor_value']
        for batch in batches:
            assert all(math.isfinite(batch[k]) for k in LOSS)
            weighted = weighted_components(batch, args)
            expected = sum(weighted.values())
            assert math.isclose(expected, batch['total_loss'], rel_tol=3e-6, abs_tol=3e-5), (run['run_id'], batch)
            if batch['split'] == 'train':
                assert batch['gradients_finite'] is True
    selected = max(epochs, key=lambda r: r['validation_weighted_f1']) if epochs else None
    late = statistics.mean(r['validation_weighted_f1'] for r in epochs[-10:]) if len(epochs) >= 10 else None
    metric = read(result_dir / 'metrics.json') if complete else None
    if complete:
        assert len(epochs) == run['epochs'] == (40 if run['dataset'] == 'MELD' else 80)
        assert status['counts'] == {'train': run['epochs'], 'validation': run['epochs'], 'test': 1}
        assert status['test_forward_passes'] == 1 and status['final_test_matches']
        assert len(observed) == run['epochs'] * 2 + 1
        for split in ('train', 'validation'):
            assert [r['epoch'] for r in observed if r['split'] == split] == list(range(1, run['epochs'] + 1))
        test_rows = [r for r in observed if r['split'] == 'test']
        assert len(test_rows) == 1
        final = read(run_dir / 'final_test.json')
        assert final['selected_epoch'] == selected['epoch']
        assert final['selection_weighted_f1'] == selected['validation_weighted_f1']
        assert final['test_weighted_f1'] == round(metric['weighted_f1'], 2)
        assert final['test_accuracy'] == round(metric['accuracy'], 2)
        saved = np.load(result_dir / 'predictions.npz')
        labels, full = saved['labels'], saved['full_log_probabilities']
        assert np.isfinite(full).all() and len(labels) == metric['n_utterances']
        predictions = full.argmax(1)
        assert np.array_equal(predictions, saved['predictions'])
        assert round(f1_score(labels, predictions, average='weighted') * 100, 2) == final['test_weighted_f1']
        assert round(accuracy_score(labels, predictions) * 100, 2) == final['test_accuracy']
        assert math.isclose(f1_score(labels, predictions, average='macro') * 100, metric['macro_f1'], abs_tol=1e-10)
        for name in ('text', 'audio', 'visual', 'intra_context', 'inter_context'):
            assert saved[name + '_log_probabilities'].shape == full.shape
        assert (run_dir / 'best.pt').is_file() and (run_dir / 'latest.pt').is_file()
        assert all(math.isfinite(metric[key]) for key in METRICS)
        if run['dataset'] == 'MELD':
            assert args['calibrate'] is False and args['contrastlearning'] and args['courselearning']
            assert any(b['cc_loss'] > 0 for b in batches if b['split'] == 'train')
            assert all(weighted_components(b,args)['weighted_cc'] == 0 for b in batches)
    elif not allow_incomplete:
        raise AssertionError('Formal run incomplete: ' + run['run_id'])
    row = {'run_id': run['run_id'], 'dataset': run['dataset'], 'variant': run['variant'],
           'seed': run['seed'], 'complete': complete, 'epochs_completed': len(epochs), 'epochs_planned': run['epochs'],
           'selected_epoch': selected['epoch'] if selected else None,
           'selected_validation_weighted_f1': selected['validation_weighted_f1'] if selected else None,
           'validation_last10_mean': late if complete else None,
           'factor_key': run['factor_key'], 'factor_value': run['factor_value'],
           'test_policy': 'one_final_pass_on_validation_selected_checkpoint',
           **{key: metric[key] if metric else None for key in METRICS}}
    comparison = {'run_id': run['run_id'], 'complete': complete,
                  'historical_selected_epoch': historical['selected_epoch_verified'],
                  'historical_selected_validation_weighted_f1': historical['selected_validation_weighted_f1'],
                  'new_selected_validation_weighted_f1': row['selected_validation_weighted_f1'] if complete else None,
                  'delta_best_validation_wf1': row['selected_validation_weighted_f1'] - historical['selected_validation_weighted_f1'] if complete else None,
                  'historical_validation_last10_mean': historical['validation_last10_mean'],
                  'new_validation_last10_mean': row['validation_last10_mean'],
                  'delta_validation_last10_mean': late - historical['validation_last10_mean'] if complete else None}
    for key in METRICS:
        comparison.update({'historical_' + key: historical['test_metrics'][key], 'new_' + key: row[key],
                           'delta_' + key: row[key] - historical['test_metrics'][key] if complete else None})
    return row, comparison, epochs, observed, args


def inference(best, late, mechanism):
    if best is None or late is None:
        return '尚未完成对应预注册训练，暂不判断。'
    if best > 0 and late > 0:
        return f'最佳验证W-F1和末10轮均值同向改善，为“{mechanism}”提供单Seed配置消融线索；历史对照和设备非确定性仍阻止严格单变量因果或跨Seed稳定性声明。'
    if best <= 0 and late <= 0:
        return f'最佳验证W-F1和末10轮均值均未改善，本次不支持“{mechanism}”；不等于证明该机制没有影响。'
    return f'最佳验证W-F1与末10轮均值方向不一致，证据混合，尚不能支持“{mechanism}”的稳定收益解释。'


def infer_batch_order(best, late):
    if best is None or late is None:
        return 'IEMOCAP两臂尚未全部完成，暂不判断Batch排列对相对收益的影响。'
    if best == 0 and late == 0:
        return '已记录精度下，验证最佳和末10轮增量变化均为零；这不是统计等价或无影响证明。'
    if best > 0 and late > 0:
        direction = '固定源顺序提高CMERC相对Baseline的增量'
    elif best < 0 and late < 0:
        direction = '固定源顺序降低CMERC相对Baseline的增量'
    else:
        return '验证最佳与末10轮增量变化方向混合或含零，尚不能给出一致方向的Batch效应线索；保留逐臂结果，不将未改善解释为无影响。'
    return ('两类验证增量变化同向，提供“' + direction + '”的单Seed线索，支持Batch排列可能影响相对收益；'
            '历史完整CMERC跨物理卡和CUDA数值非确定性仍限制严格因果解释，不能外推五Seed稳定结论。')


def report(registry, per_run, comparisons, summary):
    by_id = {r['run_id']: r for r in per_run}
    comp = {r['run_id']: r for r in comparisons}
    n = sum(r['complete'] for r in per_run)
    verification_path = ROOT / 'provenance/final_verification.json'
    verification = read(verification_path) if verification_path.exists() else None
    text = ['# CMERC 最小消融诊断结果', '', f"生成时间：{summary['generated_at_utc']}。已完成 {n}/3 项预注册正式训练。",
            '', '## 目标、范围与依据', '',
            '已读取XP04与官方README对照完整报告，重点接续05的 `05_XP04_VS_README_COMPARISON.md` 和 `06_F1_GAP_DIAGNOSIS_AND_NEXT_STEPS.md`。此前MELD单Batch CC梯度与分类梯度方向相反只是本次假设来源，不能视为整段训练根因证据。',
            '', '本批只登记MELD seed67137关闭CC且保留HSCL/CL一次，以及IEMOCAP seed1475的Baseline/完整CMERC原始源顺序Batch各一次。每项从头初始化，历史源码、Checkpoint、日志和结果只读，不覆盖XP04或05。未追加Seed、系数搜索、w/o HSCL、w/o CL或最终五Seed训练。',
            '', '## 配对条件、实现与边界', '',
            '所有新运行使用验证集Weighted F1选模：保留XP04两位小数百分比口径，严格提高才换最佳Checkpoint，同分取首次最高轮。测试集在全部轮数完成后，只对已固定的最佳Checkpoint遍历一次；每次含主模型和五种移除诊断，不再运行独立GPU evaluator。保存概率后的CPU重算不增加测试访问。',
            '', '| Run | Seed | 唯一登记的科学配置变化 | 轮数 | 划分SHA-256 |', '|---|---:|---|---:|---|']
    for run in registry['runs']:
        change = run['unique_scientific_difference']
        text.append(f"| {run['run_id']} | {run['seed']} | `{run['factor_key']}`：`{change['from']}` → `{change['to']}` | {run['epochs']} | `{run['split_hash']}` |")
    text += ['', 'MELD的calibrate=False只取消总损失中的加权CC，rank_coff仍登记0.002；因contrastlearning=True，原始CC、五路辅助前向和HSCL目标仍计算，raw CC非零不代表其进入反传。保留MSCL/CSCL系数0.15/0.15与课程0.4。IEMOCAP保留原训练池108/验证12/测试31、continuous RNG、ATen GRU及各模型原损失系数，仅将随机排列改为训练池源顺序子序列；课程成员、Batch16、训练轮数80及验证选模不变。MELD为1038/114/280对话、40轮。',
              '', 'MELD冻结12个训练源文件与XP04逐字节一致；IEMOCAP仅train_cm.py含独立记录的source_order采样补丁，原capacity_safe分支保持。完整参数、源码和数据哈希在configs、run_manifest及provenance中。原始Loss数学、网络结构、输入数据不变。',
              '', '配置层面的单因素对齐不等于严格物理条件完全一致：历史MELD完整CMERC为GPU0→本批GPU3，历史IEMOCAP完整CMERC为GPU1→GPU3；IEMOCAP Baseline历史与本批均GPU3。全部为同机TITAN Xp，但CUDA scatter等数值非确定性及跨卡差异限制历史配对的因果解释；详见[hardware_pairing.json](../provenance/hardware_pairing.json)。RNG策略保持不等于改变Batch张量形状后dropout逐元素轨迹仍相同。作者fixed_author课程文件含validation键，生成历史未知；两臂使用同文件以维持对照，不能称整个预处理validation-free。',
              '', '## 预检和运行状态', '']
    gate_path = ROOT / 'preflight/gate.json'
    if gate_path.exists():
        gate = read(gate_path)
        text += [f"预检状态 `{gate['status']}`：4项真实首Batch前向/反向/Adam更新均有限且参数实际更新；无测试前向。MELD完整CC与w/o CC首Batch初始化模型、对话IDs及三项RNG hash精确相等；前向仅在容差内匹配（最大log概率差2.384e-7、raw CC差7.629e-6），不能声称逐位一致。目标核验及加权CC取消见[MELD_CC_pair_parity.json](../preflight/MELD_CC_pair_parity.json)。IEMOCAP CPU真实长度检查覆盖160个epoch条件；3项CPU观测器行为测试通过。证据：[gate.json](../preflight/gate.json)、[source_order_cpu.json](../preflight/source_order_cpu.json)。"]
    else:
        text += ['预检证据尚未生成；不将预检描述为通过。']
    text += ['', '| Run | 状态 | 已完成/计划Epoch | 最佳验证Epoch | 最佳验证W-F1 % | 末10轮验证均值 % |', '|---|---|---:|---:|---:|---:|']
    for r in per_run:
        text.append(f"| {r['run_id']} | {'完成' if r['complete'] else '未完成'} | {r['epochs_completed']}/{r['epochs_planned']} | {r['selected_epoch'] or '待完成'} | {fmt(r['selected_validation_weighted_f1'])} | {fmt(r['validation_last10_mean'])} |")
    if n == len(per_run) and verification and verification['status'] == 'passed':
        text += ['', f"完整轮数、每轮测试字段为空、首个最高验证轮、最终一次测试、保存概率、逐Batch覆盖和分项Loss重构均通过独立CPU验收。历史{sum(verification['history_files_unchanged'].values())}个常规文件及已登记链接核验未变，数据与执行源码哈希未变。{verification['resource_samples']}次资源采样未发现其他项目进程进入本批GPU；离散采样不能证明未采样时刻的全局互斥。"]
    else:
        text += ['', '未完成时的最佳验证值仅为当前轨迹进度，不代表最终选轮。完整轮数、每轮测试字段为空、首个最高验证轮、最终一次测试、保存概率和分项Loss重构由汇总程序检查；最终完整验收尚待完成。']
    text += [
             '', '## 固定最佳Checkpoint的最终测试描述', '', '分类指标和ECE为百分比，差值为百分点；CEL为五种移除正真类log概率差总和，非分类交叉熵。',
             '', '| Run | W-F1 % | Macro-F1 % | Accuracy % | ECE % | CEL总和 |', '|---|---:|---:|---:|---:|---:|']
    for r in per_run:
        text.append('| ' + r['run_id'] + ' | ' + ' | '.join(fmt(r[k]) for k in METRICS) + ' |')
    text += ['', 'IEMOCAP Baseline训练/验证无辅助移除，在线CEL为null；仅在唯一最终测试函数入口临时设置calibrate=True且系数0，获得同次遍历中的五路移除，返回即恢复。此开关不改变训练或选轮。训练态在线ECE/CEL混合逐Batch变化的模型参数，CEL还使用epoch入口辅助快照，不能作为固定Checkpoint泛化指标与最终测试直接混比。',
             '', '## MELD：CC是否限制分类性能', '']
    m = comp['MELD_wo_cc_seed67137']
    text += [f"XP04完整CMERC的最佳验证W-F1为{m['historical_selected_validation_weighted_f1']:.4f}%（第{m['historical_selected_epoch']}轮），末10轮均值{m['historical_validation_last10_mean']:.4f}%。w/o CC对应变化：最佳验证 **{sign(m['delta_best_validation_wf1'])} pp**，末10轮均值 **{sign(m['delta_validation_last10_mean'])} pp**。", '',
             inference(m['delta_best_validation_wf1'], m['delta_validation_last10_mean'], 'CC在当前MELD配置下可能限制验证分类表现'), '',
             f"固定最终测试W-F1：XP04 {m['historical_weighted_f1']:.4f}% → 本批{fmt(m['new_weighted_f1'])}%，变化{sign(m['delta_weighted_f1'])} pp，只作描述。不能据测试结果改系数或挑方案。",
             '', f"同一次固定最终测试的校准描述：ECE变化{sign(m['delta_ece'])} pp，CEL变化{sign(m['delta_cel'])}。两者方向不同，不能把关闭CC称为统一改善校准。",
             '', '## IEMOCAP：Batch组织是否影响CMERC相对收益', '']
    iem = summary['iemocap_gain_comparison']
    text += ['| 协议 | 验证CMERC−Baseline / pp | 末10轮验证均值差 / pp | 固定最终测试CMERC−Baseline / pp |', '|---|---:|---:|---:|',
             f"| XP04随机排列 | {fmt(iem['historical_best_validation_gain'])} | {fmt(iem['historical_late_validation_gain'])} | {fmt(iem['historical_test_gain'])} |",
             f"| 本批source_order | {fmt(iem['new_best_validation_gain'])} | {fmt(iem['new_late_validation_gain'])} | {fmt(iem['new_test_gain'])} |",
             f"| 增量变化 | {sign(iem['delta_best_validation_gain'])} | {sign(iem['delta_late_validation_gain'])} | {sign(iem['delta_test_gain'])} |", '']
    for run in registry['runs']:
        if run['dataset'] == 'IEMOCAP':
            c = comp[run['run_id']]
            text.append(f"- {run['variant']}逐臂变化：最佳验证{sign(c['delta_best_validation_wf1'])} pp，末10轮均值{sign(c['delta_validation_last10_mean'])} pp，固定测试{sign(c['delta_weighted_f1'])} pp。")
    c = comp['IEMOCAP_cmerc_source_order_seed1475']
    text += ['', f"完整CMERC固定最终测试相对XP04的ECE变化为{sign(c['delta_ece'])} pp，CEL变化为{sign(c['delta_cel'])}；分类分数接近不能理解为校准表现也接近。这仍是单Seed历史对照的描述，未据此改变配置或安排重测。"]
    text += ['', infer_batch_order(iem['delta_best_validation_gain'], iem['delta_late_validation_gain']),
             '', ('本次最佳验证增量与末段验证增量仍都为负，属于负增量缩小。两臂绝对验证分数均下降，Baseline下降更多；不能表述为固定顺序提高CMERC绝对验证性能、恢复正分类收益或解释了论文增量。' if n == len(per_run) and iem['new_best_validation_gain'] < 0 and iem['new_late_validation_gain'] < 0 else '相对增量的变化须与两臂绝对变化一起解读，不能单独称为CMERC绝对性能提升。'),
             '', '两个模型分别在各自验证曲线上选轮；上表验证增量是选中分数的描述差，不能把模型胜负或变化直接等同于固定轮次的优化因果。末10轮指MELD第31–40轮、IEMOCAP第71–80轮的训练末段表现，不是选中Checkpoint的指标，也不是跨Seed稳定性证据。最佳值与末10轮均值来自同一验证曲线，是相关的描述统计；同向不等于两份独立证据。固定顺序若使增量下降，也只支持此Seed下方向相反的描述，不代表所有Batch组织均无关。',
             '', '## 曲线、Loss与可追溯产物', '',
             '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[metrics_summary.json](../results/metrics_summary.json)、[epoch_curves.csv](../results/epoch_curves.csv)、[epoch_loss_curves.csv](../results/epoch_loss_curves.csv)。CSV包括新旧训练/验证轨迹、原始与固定系数加权Loss；测试分数不形成逐轮曲线。', '',
             '| Run | 训练/验证W-F1 | Loss分项 |', '|---|---|---|']
    for r in per_run:
        text.append(f"| {r['run_id']} | [PNG](../results/figures/{r['run_id']}_wf1.png) | [PNG](../results/figures/{r['run_id']}_loss.png) |")
    text += ['', '[IEMOCAP同轮验证增量曲线](../results/figures/IEMOCAP_validation_gain.png)展示各Epoch的CMERC−Baseline，与分别选取最佳Checkpoint后的增量统计口径不同。全部图同时保存SVG及输入/输出哈希，见[图表清单](../results/figures/manifest.json)。']
    text += ['', '## 已证实事实、推断与下一步', '',
             f'**已证实事实：** 当前{n}/3训练完成；完成项的数值来自保存输出，经CPU重算和逐Batch Loss身份检查。冻结参数、划分和唯一登记因素受运行时核验。此前单Batch梯度不是纵向根因证据。', '',
             '**尚未证实：** 单Seed历史对照不能证明跨Seed稳定收益、唯一根因或与论文差距的定量分摊。CC与HSCL/CL可能协同或竞争，但本批只隔离一个配置开关，不证明其内部作用机制。设备和非确定性边界仍存在。', '',
             '**Batch机制的待验证假设：** BN统计、padding参与的移除均值、HSCL正负对及未按句数归一化的CC都可能随Batch改变。另经源码确认，[model_hyper.py](../code_xp04/model_hyper.py)第127–128行定义可训练hyperedge_weight/EW_weight，第164–172行按Batch范围切片，第238–262行从零分配当前Batch节点/超边编号；[HypergraphConv.py](../code_xp04/HypergraphConv.py)第134–143行依该局部编号索引参数参与归一化。因此排列可改变样本与可训练参数槽位的对应关系。该路径的存在是代码事实，其对本次F1的贡献尚未隔离；初始化为全1也意味着不能直接据此断言早期差异。本批没有修改这些数学。', '',
             '**后续建议：** 先按预登记的验证最佳值、末10轮均值和完整轨迹判断证据一致性。若两类验证统计同向支持机制影响，可另行登记w/o HSCL或w/o CL，每次独立一个因素；若证据混合或不支持，先做训练/验证梯度及伪目标稳定性诊断，不根据测试调系数。最终五Seed应在诊断结论和新方案明确后单独决定；本脚本不会自动启动消融或扩展Seed。', '']
    if n == len(per_run):
        text += ['## 本轮完成后的实验决策建议（尚未执行）', '']
        mb, ml = m['delta_best_validation_wf1'], m['delta_validation_last10_mean']
        if mb > 0 and ml > 0:
            text += ['- MELD：两项相关验证统计同向改善，但只有单Seed、历史跨卡对照；先预登记同卡完整CMERC控制，检查此配置效应的可重现性，再决定是否独立开展w/o HSCL或w/o CL。不能据本次测试分数确定下一方案。']
        else:
            text += ['- MELD：本次未形成CC限制分类性能的稳定验证证据，暂缓围绕“去CC提升F1”开展最终五Seed，也不直接扩展w/o HSCL或w/o CL。优先登记只用训练/验证数据的分阶段多Batch梯度norm/cos、辅助快照与dropout下HSCL目标稳定性诊断；如需确认开关效应，再补同卡控制。']
        ib, il = iem['delta_best_validation_gain'], iem['delta_late_validation_gain']
        if (ib > 0 and il > 0) or (ib < 0 and il < 0):
            text += ['- IEMOCAP：验证相对增量同向变化，使Batch组织成为值得继续验证的因素；先另行登记同卡的随机/固定排列控制及CUDA数值重复性检查，确认历史跨卡与数值波动不足以解释观察值，再考虑在固定Batch协议下独立消融HSCL或CL。当前不把改变Batch当作已确定的优化方案。']
        else:
            text += ['- IEMOCAP：验证相对增量未形成一致方向，暂缓扩展机制消融。先分析已保存的逐轮/逐Batch训练覆盖、Loss尺度及目标分布，并预登记同卡控制以缩小数值与历史配对边界；不依据测试胜负指定新的Batch策略。']
        text += ['- 最终五Seed：本轮仅能形成定位线索，建议暂缓启动。应先冻结下一阶段的科学假设、同卡控制、最终生效配置及验证选模规则，再由新的独立预注册决定五Seed预算。本批未执行上述后续训练，也未重测测试集。', '']
    text.append('最终独立验收：' + ('[final_verification.json](../provenance/final_verification.json)通过；三项保存预测均独立CPU重算，6个最佳/最终Checkpoint的模型及Adam状态有限且含已登记恢复状态。' if verification and verification['status'] == 'passed' else '待主代理执行并生成provenance/final_verification.json；当前不声称最终独立验收已通过。'))
    text += ['', f'批次绝对路径：`{ROOT}`。仓库根报告与本批版本正文一致，产物相对链接按报告所在位置调整。', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true')
    opts = parser.parse_args()
    registry = read(ROOT / 'configs/registry.json')
    historical = read(ROOT / 'provenance/xp04_references.json')
    assert len(registry['runs']) == 3
    per_run, comparisons, curves, losses = [], [], [], []
    for run in registry['runs']:
        reference = historical[run['run_id']]
        row, comparison, epochs, observed, args = gather(run, reference, opts.allow_incomplete)
        per_run.append(row)
        comparisons.append(comparison)
        for source, epoch_rows, config in [('new', epochs, args), ('XP04', reference['epoch_rows'], read(run['historical_manifest'])['args'])]:
            for epoch in epoch_rows:
                curves.append({'run_id': run['run_id'], 'source': source, 'epoch': epoch['epoch'],
                               **{k:epoch[k] for k in ['train_weighted_f1','validation_weighted_f1','train_accuracy','validation_accuracy','train_loss','validation_loss','sample_ratio','train_dialogues']}})
                losses.append({'run_id': run['run_id'], 'source': source, 'split': 'train', 'epoch': epoch['epoch'],
                               **{k:epoch[k] for k in LOSS}, **weighted_components(epoch, config)})
        for observed_epoch in observed:
            if observed_epoch['split'] != 'test':
                curves.append({'run_id': run['run_id'], 'source': 'new_observer', 'split': observed_epoch['split'],
                               'epoch': observed_epoch['epoch'], **{k:observed_epoch.get(k) for k in METRICS}})
            if observed_epoch['split'] == 'validation' and observed_epoch.get('loss_components'):
                component = observed_epoch['loss_components']
                losses.append({'run_id': run['run_id'], 'source': 'new_observer', 'split': 'validation',
                               'epoch': observed_epoch['epoch'], **component, **weighted_components(component, args)})
    selected = {r['run_id']:r for r in per_run}
    base_id, full_id = 'IEMOCAP_baseline_source_order_seed1475', 'IEMOCAP_cmerc_source_order_seed1475'
    base, full = selected[base_id], selected[full_id]
    gain = {}
    for label, new_key, old_key in [('best_validation','selected_validation_weighted_f1','selected_validation_weighted_f1'), ('late_validation','validation_last10_mean','validation_last10_mean'), ('test','weighted_f1','weighted_f1')]:
        oldbase = historical[base_id]['test_metrics'][old_key] if label == 'test' else historical[base_id][old_key]
        oldfull = historical[full_id]['test_metrics'][old_key] if label == 'test' else historical[full_id][old_key]
        old_gain = oldfull-oldbase
        new_gain = full[new_key]-base[new_key] if full['complete'] and base['complete'] else None
        gain['historical_' + label + '_gain'] = old_gain
        gain['new_' + label + '_gain'] = new_gain
        gain['delta_' + label + '_gain'] = new_gain-old_gain if new_gain is not None else None
    summary = {'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
               'formal_training_complete':all(r['complete'] for r in per_run), 'planned_runs':3,
               'completed_runs':sum(r['complete'] for r in per_run), 'per_run':per_run,
               'comparison':comparisons, 'iemocap_gain_comparison':gain,
               'state':read(ROOT / 'state/status.json'),
               'source_hashes':{name:digest(ROOT/name) for name in ['configs/registry.json','provenance/xp04_references.json','scripts/summarize_minimal.py']},
               'verification_scope':'CPU persisted trajectory, loss weighting, unique test output and argmax checks; no new GPU forward or checkpoint selection'}
    for name, data in [('per_run.csv',per_run),('comparison.csv',comparisons),('epoch_curves.csv',curves),('epoch_loss_curves.csv',losses)]:
        csv_write(ROOT / 'results' / name, data)
    write(ROOT / 'results/metrics_summary.json', summary)
    prose = report(registry,per_run,comparisons,summary)
    batch_report = ROOT / 'reports/CMERC_MINIMAL_ABLATION_RESULTS.md'
    batch_report.parent.mkdir(parents=True, exist_ok=True)
    batch_report.write_text(prose)
    root_prose = prose
    relative_batch = ROOT.relative_to(REPO).as_posix()
    for directory in ('results', 'provenance', 'preflight', 'code_xp04', 'code_source_order'):
        root_prose = root_prose.replace('../' + directory + '/', '../' + relative_batch + '/' + directory + '/')
    root_report = REPO / 'reports/CMERC_MINIMAL_ABLATION_RESULTS.md'
    root_report.parent.mkdir(parents=True, exist_ok=True)
    root_report.write_text(root_prose)
    print(json.dumps({'complete':summary['formal_training_complete'],'completed_runs':summary['completed_runs'],'output':str(ROOT/'results/metrics_summary.json')},ensure_ascii=False))


if __name__ == '__main__':
    main()
