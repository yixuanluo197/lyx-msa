"""Analyze only preregistered CMERC training/validation traces, entirely on CPU.

Never opens test artifacts or invokes a model. Baselines are copied validation
references, not refitted runs. --allow-incomplete is an explicitly provisional view.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
LOSS_KEYS = ('classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss', 'total_loss')
CURVE_KEYS = ('train_weighted_f1', 'validation_weighted_f1', 'train_accuracy', 'validation_accuracy',
              'train_loss', 'validation_loss', 'sample_ratio', 'train_dialogues', 'max_gpu_memory_mb')


def read(path):
    return json.loads(Path(path).read_text())


def json_rows(path):
    if not Path(path).exists():
        return []
    # Ignore an in-progress final partial write only in the incremental reader.
    content = Path(path).read_text()
    lines = content.splitlines()
    if content and not content.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, records):
    keys = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def textnum(value, signed=False):
    if value is None:
        return '待完成'
    return format(value, '+.4f' if signed else '.4f')


def basic_stats(epochs):
    if not epochs:
        return {}
    best = max(epochs, key=lambda row: row['validation_weighted_f1'])
    tail = epochs[-10:]
    return {'best_epoch': best['epoch'], 'best_validation_wf1': best['validation_weighted_f1'],
            'last10_validation_wf1': statistics.mean(row['validation_weighted_f1'] for row in tail) if len(epochs) >= 10 else None,
            'last_epoch_validation_wf1': epochs[-1]['validation_weighted_f1'],
            'best_epoch_train_classification_loss': best['classification_loss'],
            'best_epoch_validation_classification_loss': best['validation_loss'],
            'last10_train_classification_loss': statistics.mean(row['classification_loss'] for row in tail) if len(epochs) >= 10 else None,
            'last10_validation_classification_loss': statistics.mean(row['validation_loss'] for row in tail) if len(epochs) >= 10 else None,
            'last10_total_train_loss': statistics.mean(row['total_loss'] for row in tail) if len(epochs) >= 10 else None,
            'max_epoch_allocated_mib': max(row['max_gpu_memory_mb'] for row in epochs)}


def loss_weighted(row, args):
    return {'classification_loss': row['classification_loss'],
            'weighted_cc_loss': args['rank_coff'] * row['cc_loss'] if args['calibrate'] else 0.,
            'weighted_mscl_loss': args['mscl_coff'] * row['mscl_loss'] if args['contrastlearning'] else 0.,
            'weighted_cscl_loss': args['cscl_coff'] * row['cscl_loss'] if args['contrastlearning'] else 0.}


def validate_epoch_rows(epochs, run, args, historical=False):
    assert [row['epoch'] for row in epochs] == list(range(1, len(epochs) + 1))
    assert len(epochs) <= args['epochs']
    for row in epochs:
        assert row['selection_protocol'] == 'validation' and row['selection_split'] == 'validation'
        assert row['split_hash'] == run['split_hash']
        assert row['rng_policy'] == 'continuous' and row['batch_policy'] == 'capacity_safe'
        assert row['validation_weighted_f1'] == round(row['validation_weighted_f1'], 2)
        assert row['selection_weighted_f1'] == row['validation_weighted_f1']
        if not historical:
            # Original trainer fields must remain explicitly empty; no test scores enter exports.
            assert 'test_weighted_f1' in row and row['test_weighted_f1'] is None
            assert 'test_accuracy' in row and row['test_accuracy'] is None
        assert all(math.isfinite(row[key]) for key in LOSS_KEYS + CURVE_KEYS[:6])
        expected = sum(loss_weighted(row, args).values())
        assert math.isclose(expected, row['total_loss'], rel_tol=3e-6, abs_tol=3e-5)


def candidate_category(best_delta, late_delta):
    if best_delta > 0:
        return 'strong_candidate' if late_delta >= 0 else 'weak_mixed_candidate'
    return 'no_best_improvement'


def watcher_status(result):
    for name in ('status.json', 'watch_status.json'):
        path = result / name
        if path.exists():
            return read(path)
    return {}


def gather(run, reference, allow_incomplete):
    run_dir = Path(run['run_dir'])
    result = ROOT / 'results' / run['run_id']
    epochs = json_rows(run_dir / 'metrics.jsonl')
    watched = json_rows(result / 'watch_epochs.jsonl')
    status = watcher_status(result)
    complete = status.get('phase', status.get('status')) in ('completed', 'complete')
    args = {**reference['args'], 'lr': run['lr'], 'skip_final_test': True}
    manifest_path = run_dir / 'run_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        args = manifest['args']
        permitted = {'lr', 'output_dir', 'provenance', 'skip_final_test'}
        assert {key:val for key,val in args.items() if key not in permitted} == {key:val for key,val in reference['args'].items() if key not in permitted}
        assert args['lr'] == run['lr'] and args['skip_final_test']
        assert args['provenance']['split'] == reference['split']
        assert args['provenance']['data'] == reference['data']
    validate_epoch_rows(epochs, run, args)
    assert args['epochs'] == run['epochs'] == (80 if run['dataset'] == 'IEMOCAP' else 40)
    train_watch = [row for row in watched if row['split'] == 'train']
    val_watch = [row for row in watched if row['split'] == 'validation']
    assert all(row['split'] in ('train', 'validation') for row in watched), 'Test call in watcher'
    for subset in (train_watch, val_watch):
        assert [row['epoch'] for row in subset] == list(range(1, len(subset) + 1))
    finite_flags = [row.get('gradients_finite') for row in train_watch if 'gradients_finite' in row]
    finite_flags += [row.get('all_gradients_finite') for row in train_watch if 'all_gradients_finite' in row]
    assert all(flag is True for flag in finite_flags), 'Gradient abnormality reported by watcher'
    for row in watched:
        for key in ('weighted_f1', 'macro_f1', 'accuracy', 'ece', 'cel'):
            if key in row and row[key] is not None:
                assert math.isfinite(row[key]), ('Nonfinite watcher metric', key)
    stats = basic_stats(epochs)
    if complete:
        assert len(epochs) == run['epochs']
        assert len(train_watch) == len(val_watch) == run['epochs']
        assert len(watched) == 2 * run['epochs']
        if 'counts' in status:
            assert status['counts'].get('train') == status['counts'].get('validation') == run['epochs']
            assert status['counts'].get('test', 0) == 0
        for key in ('test_calls', 'test_forward_passes', 'test_batches'):
            if key in status:
                assert status[key] == 0
        # Presence guards do not open/read any test artifacts.
        assert not (run_dir / 'final_test.json').exists()
        assert (run_dir / 'best.pt').is_file() and (run_dir / 'latest.pt').is_file()
        for row in epochs:
            val = val_watch[row['epoch'] - 1]
            assert round(val['weighted_f1'], 2) == row['validation_weighted_f1']
        if 'best_epoch' in status:
            assert status['best_epoch'] == stats['best_epoch']
        if 'selected_epoch' in status:
            assert status['selected_epoch'] == stats['best_epoch']
        if 'selection_weighted_f1' in status:
            assert status['selection_weighted_f1'] == stats['best_validation_wf1']
        if 'no_final_test_artifacts' in status:
            assert status['no_final_test_artifacts'] is True
    elif not allow_incomplete:
        raise AssertionError('Incomplete formal run: ' + run['run_id'])
    summary = {'run_id': run['run_id'], 'dataset': run['dataset'], 'seed': run['seed'], 'lr': run['lr'],
               'source': 'new', 'complete': complete, 'watcher_phase': status.get('phase'), 'epochs_completed': len(epochs), 'epochs_planned': run['epochs'],
               **stats, 'nonfinite_loss_epochs': 0,
               'gradient_checks_observed': len(finite_flags),
               'gradient_abnormalities': 0 if finite_flags else None,
               'gradient_l2_mean': statistics.mean(row['gradient_l2_mean'] for row in train_watch if row.get('gradient_l2_mean') is not None) if any(row.get('gradient_l2_mean') is not None for row in train_watch) else None,
               'gradient_l2_max': max(row['gradient_l2_max'] for row in train_watch if row.get('gradient_l2_max') is not None) if any(row.get('gradient_l2_max') is not None for row in train_watch) else None,
               'test_policy': 'zero_test_iterations_zero_test_forwards',
               'historical_best_validation_wf1': reference['best_validation_wf1'],
               'historical_last10_validation_wf1': reference['last10_validation_wf1']}
    summary['delta_best_validation_wf1'] = stats['best_validation_wf1'] - reference['best_validation_wf1'] if complete else None
    summary['delta_last10_validation_wf1'] = stats['last10_validation_wf1'] - reference['last10_validation_wf1'] if complete else None
    summary['candidate_category'] = candidate_category(summary['delta_best_validation_wf1'], summary['delta_last10_validation_wf1']) if complete else 'pending'
    if complete and val_watch:
        selected = val_watch[stats['best_epoch'] - 1]
        for key in ('weighted_f1', 'macro_f1', 'accuracy', 'ece', 'cel'):
            summary['selected_validation_' + key] = selected.get(key)
    return summary, epochs, watched, args


def choose_candidate(rows):
    if not all(row['complete'] for row in rows):
        return {'decision': 'pending', 'recommended_lr': None, 'run_id': None}
    improving = [row for row in rows if row['delta_best_validation_wf1'] > 0]
    if not improving:
        return {'decision': 'stop_lr_search', 'recommended_lr': None, 'run_id': None,
                'reason': '两项lr均未提高预登记主指标；停止，不扩大搜索范围。'}
    qualified = [row for row in improving if row['delta_last10_validation_wf1'] >= 0]
    weak_ids = [row['run_id'] for row in improving if row['delta_last10_validation_wf1'] < 0]
    if not qualified:
        return {'decision': 'weak_mixed_candidate', 'recommended_lr': None, 'run_id': None,
                'weak_run_ids': weak_ids,
                'reason': '仅存在最佳值改善但末段下降的混合线索，没有符合原登记资格的第二Seed候选；只报告，不推荐扩展。'}
    best = max(qualified, key=lambda row: (row['best_validation_wf1'], row['last10_validation_wf1'], row['lr'] == 5e-5))
    return {'decision': 'strong_candidate', 'recommended_lr': best['lr'], 'run_id': best['run_id'],
            'weak_run_ids': weak_ids,
            'reason': '先按原登记筛选最佳值改善且末段不低于控制的合格强候选；仅在此集合内按最佳验证W-F1、末10轮均值排序，完全相同优先5e-5；不启动第二Seed。'}


def generate_report(summary, registry):
    per = summary['per_run']
    lines = ['# CMERC 学习率诊断', '', f"生成：{summary['generated_at_utc']}；预登记4项已完成{summary['completed_runs']}/4。",
             '', '## 范围与协议', '',
             '本批接续XP04完整CMERC训练/验证控制，仅比较lr=5e-5、1e-4、2e-4。1e-4为既有XP04同Seed历史轨迹，新增4次从头训练：IEMOCAP seed1475两项、MELD seed67137两项。未继承第06组source_order或w/o CC设置，CC、HSCL和CL均保持开启及原系数。', '',
             '所有新增任务只训练和验证；验证Weighted F1按百分比四舍五入两位、严格提高才更新Checkpoint，同分取首次最高轮。训练结束跳过测试，不进行测试前向，不解析或使用新旧测试指标筛选学习率。本报告与候选排序完全不含测试结果。原始综合pickle和测试loader元数据仍由既有代码加载，零测试前向不等于完全未加载含测试数据的文件；历史产物仅作字节哈希保全。skip_final_test为独立记录的工程开关，训练数学不变。', '',
             'IEMOCAP沿用108训练/12验证对话、80轮、dropout0.5、K4、rank/MSCL/CSCL系数0.005/0.05/0.05、课程0.15；MELD1038训练/114验证、40轮、dropout0.4、K3、系数0.002/0.15/0.15、课程0.4。两者均batch16、GRU、ATen、continuous RNG、capacity_safe、BN、lr以外全部模型超参数保持历史值，split_seed1475、L2系数3e-5。完整登记见[registry.json](../configs/registry.json)。', '',
             'XP04完整模型历史IEMOCAP seed1475在GPU1、MELD seed67137在GPU0，本批统一同机TITAN Xp GPU3。跨物理卡与CUDA scatter非确定性限制严格单变量物理因果解释。固定作者课程文件包含validation键、生成史未知；相同来源保持配对，但不能称整个预处理validation-free。', '',
             '## 三个学习率的训练/验证对照', '',
             'W-F1为百分比，差值为百分点；训练分类Loss为在线逐Batch均值，验证Loss为固定模型eval统计。', '',
             '| 数据集 | lr | 来源/状态 | 最佳验证W-F1 | Epoch | 末10轮验证均值 | Δ最佳 | Δ末10轮 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in summary['all_protocols']:
        state = '历史完整' if row['source'] == 'XP04' else ('新训练完成' if row['complete'] else f"进行中{row['epochs_completed']}/{row['epochs_planned']}")
        lines.append(f"| {row['dataset']} | {row['lr']:g} | {state} | {textnum(row.get('best_validation_wf1'))} | {row.get('best_epoch','待完成')} | {textnum(row.get('last10_validation_wf1'))} | {textnum(row.get('delta_best_validation_wf1'),True)} | {textnum(row.get('delta_last10_validation_wf1'),True)} |")
    completeness = '四项新增训练均完成固定Epoch预算。' if summary['formal_training_complete'] else '未完成轨迹的当前最佳值不是最终结果；不足完整训练预算不进行候选判定。'
    lines += ['', completeness + '末10轮为IEMOCAP71–80轮、MELD31–40轮，与最佳值属于同一验证曲线上的相关描述统计，不是两份独立证据。', '',
              '## Loss、稳定性与过拟合线索', '',
              '| 数据集/lr | 最佳轮train分类Loss | 最佳轮val分类Loss | 末10轮train分类Loss | 末10轮val分类Loss | 训练Epoch峰值显存MiB | 梯度异常 |', '|---|---:|---:|---:|---:|---:|---|']
    for row in summary['all_protocols']:
        grad = '历史正式轨迹未记录' if row['source'] == 'XP04' else (str(row['gradient_abnormalities']) if row.get('gradient_checks_observed') else '当前无观测值，不能宣称零异常')
        lines.append(f"| {row['dataset']}/{row['lr']:g} | {textnum(row.get('best_epoch_train_classification_loss'))} | {textnum(row.get('best_epoch_validation_classification_loss'))} | {textnum(row.get('last10_train_classification_loss'))} | {textnum(row.get('last10_validation_classification_loss'))} | {textnum(row.get('max_epoch_allocated_mib'))} | {grad} |")
    lines.append('')
    for row in summary['all_protocols']:
        train_tail = row.get('last10_train_classification_loss')
        val_tail = row.get('last10_validation_classification_loss')
        if train_tail is not None and val_tail is not None and row.get('complete'):
            train_change = train_tail - row['best_epoch_train_classification_loss']
            val_change = val_tail - row['best_epoch_validation_classification_loss']
            compatible = '呈现训练Loss继续下降而验证Loss上升的轨迹线索' if train_change < 0 and val_change > 0 else '未同时满足训练Loss下降且验证Loss上升这一描述条件'
            lines.append(f"- {row['dataset']}/lr={row['lr']:g}：末10轮均值相对最佳验证轮，train分类Loss变化{train_change:+.4f}、val分类Loss变化{val_change:+.4f}；{compatible}。")
    lines += ['', '本阶段watcher不保存完整概率，未计算验证ECE/CEL；不得把未采集值当零或声称校准表现已验证。', '', '非有限Loss与watcher显式报告的非有限梯度会使严格汇总失败；历史XP04正式轮级轨迹无梯度norm，预检梯度不能冒充纵向对照。新watcher若只验证finite，也不能由此宣称各辅助项整段梯度方向已解决。显存为原训练器逐Epochallocated峰值，不等于整卡占用或训练进程完整生命周期峰值。', '',
              'train Loss下降而val Loss上升、val最佳早于训练末段，可以提供过拟合/目标竞争线索；训练在线预测与eval固定模型的模式和参数时点不同，不能把两条曲线间距称为严格泛化gap，也不能仅凭早峰判定lr是唯一根因。原始CC/MSCL/CSCL与加权分项见loss_curves.csv；CC为未按句数归一化的原作者项，本轮不改公式。', '',
              'L2系数始终3e-5；原实现是Adam(weight_decay=l2)耦合L2。改变lr会同时改变包含L2项在内的更新轨迹及有效正则化过程，不能解释成仅改变分类梯度步长，也不能声称三条轨迹的总正则化作用相同。', '',
              '## 预登记候选规则与诊断结论', '',
              '强候选：最佳验证W-F1高于同Seed 1e-4控制且末10轮均值不低于控制。弱候选/混合：最佳提高但末段下降。仅多个合格强候选按最佳值、再末10轮均值排序，完全相同优先5e-5。弱线索不具备第二Seed候选资格，无强候选时不推荐任何lr；详见[分析资格澄清](../reports/ANALYSIS_RULE_CLARIFICATION.md)。两个新lr均无最佳改善则停止lr搜索，不扩大范围。']
    for dataset, decision in summary['dataset_decisions'].items():
        labels = {'pending':'待完成','strong_candidate':'满足预登记第二Seed候选资格（非统计强证据）','weak_mixed_candidate':'弱候选/混合证据','stop_lr_search':'停止lr搜索'}
        lines += ['', f"**{dataset}：{labels[decision['decision']]}。**" + (f" 预登记排序首选lr={decision['recommended_lr']:g}；仅供下一阶段评估。" if decision['recommended_lr'] else ''), '']
        for row in per:
            if row['dataset'] == dataset:
                lines.append(f"- lr={row['lr']:g}：Δ最佳{ textnum(row['delta_best_validation_wf1'],True)} pp，Δ末10轮{textnum(row['delta_last10_validation_wf1'],True)} pp；类别`{row['candidate_category']}`。")
        lines.append('')
        if 'reason' in decision:
            lines.append(decision['reason'])
        if decision['decision'] == 'strong_candidate':
            eligible = [r for r in per if r['dataset'] == dataset and r['candidate_category'] == 'strong_candidate']
            selected = next(r for r in eligible if r['run_id'] == decision['run_id'])
            lines.append('该资格门槛没有预设最小效应量或置信区间；通过门槛不等于改善稳健、统计显著或已确定最优学习率。')
            for other in eligible:
                if other['run_id'] != selected['run_id']:
                    lines.append(f"首选lr={selected['lr']:g}相对另一个合格lr={other['lr']:g}，最佳验证值差{selected['best_validation_wf1']-other['best_validation_wf1']:+.4f} pp、末10轮均值差{selected['last10_validation_wf1']-other['last10_validation_wf1']:+.4f} pp；排序来自预登记的最佳值优先规则，不代表每项统计都更优。")
    lines += ['', '**已证实事实：** 仅登记4个新增训练；完成项的完整轮数、选模规则、Loss有限性与加权身份、训练/验证轨迹由日志核验，候选依据只使用验证集。', '',
              '**尚未证实推断：** 学习率变化是否稳定改善其他Seed、是否解释论文F1差距、各辅助目标梯度竞争是否缓解，均不能由本轮单Seed历史对照确定。本报告不提供单Seed p值或显著性结论。', '',
              '**下一步建议：** 首先根据预登记验证证据判断是否值得开展第二Seed独立确认；2024仅预先保留，两数据集XP04同协议控制已存在，本批不自动启动第二Seed、Baseline或五Seed。弱候选先报告证据混合，不靠测试结果决定扩大训练。若两个lr都无最佳改善，停止此搜索。', '',
              '本阶段不需要追加Batch Size=8。若有合格学习率候选，优先保持batch16做独立Seed确认；若两项均无改善，则停止本轮搜索并分析现有曲线。暂缓Batch8是控制变量和保证解释性的实验顺序，并非证明Batch8无益。Batch8不是本轮变量，也不能由lr结果直接证明有益。未来若要比较，必须独立登记其对BN统计、HSCL正负对、CC尺度、每Epoch更新步数及算力预算的同时变化，不把改Batch伪装成单一优化超参数改善。', '',
              '## 曲线与可追溯证据', '',
              '[训练前登记](../reports/PREREGISTRATION.md)、[源码冻结清单](../provenance/frozen_execution.json)、[跳过最终测试的独立补丁](../provenance/skip_final_test.patch)、[首步学习率配对预检](../preflight/LR_first_step_parity.json)及[真实长度容量预检](../preflight/capacity_cpu.json)分别保存。首步初始化和RNG哈希完全相同，Loss/梯度仅在登记容差内匹配，不声称CUDA逐位确定性。', '',
              '[per_run.csv](../results/per_run.csv)、[comparison.csv](../results/comparison.csv)、[curves.csv](../results/curves.csv)、[loss_curves.csv](../results/loss_curves.csv)、[metrics_summary.json](../results/metrics_summary.json)。曲线仅train/validation，CSV不导出任何测试分数。', '',
              '| 数据集 | W-F1轨迹 | 分类Loss | 加权训练Loss |', '|---|---|---|---|']
    for dataset in ('IEMOCAP','MELD'):
        lines.append(f"| {dataset} | [PNG](../results/figures/{dataset}_wf1.png) | [PNG](../results/figures/{dataset}_classification_loss.png) | [PNG](../results/figures/{dataset}_weighted_loss.png) |")
    final = ROOT / 'provenance/final_verification.json'
    lines += ['', ('最终独立验收文件已生成：[final_verification.json](../provenance/final_verification.json)，通过范围以文件内容为准。' if final.exists() else '最终独立验收尚待执行；不将本脚本汇总检查等同于完整验收。'), '',
              f'实验根目录：`{ROOT}`。仓库根报告正文相同，链接按报告位置调整。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true')
    opts = parser.parse_args()
    registry = read(ROOT / 'configs/registry.json')
    references = read(ROOT / 'provenance/xp04_validation_references.json')
    assert len(registry['runs']) == registry['formal_run_count'] == 4
    per, comparison, curves, loss_curves, protocols = [], [], [], [], []
    for dataset, seed in [('IEMOCAP',1475),('MELD',67137)]:
        ref = references[f'{dataset}_{seed}']
        validate_epoch_rows(ref['epochs'], {'split_hash':ref['split']['hash']}, ref['args'], historical=True)
        old = {'dataset':dataset,'seed':seed,'lr':1e-4,'source':'XP04','complete':True,
               'epochs_completed':len(ref['epochs']),'epochs_planned':len(ref['epochs']),
               **basic_stats(ref['epochs']), 'delta_best_validation_wf1':0.,'delta_last10_validation_wf1':0.}
        assert old['best_epoch'] == ref['best_epoch'] and old['best_validation_wf1'] == ref['best_validation_wf1']
        assert math.isclose(old['last10_validation_wf1'],ref['last10_validation_wf1'],abs_tol=1e-12)
        protocols.append(old)
        series = [(f'{dataset}_XP04_lr1e-4_seed{seed}',1e-4,'XP04',ref['epochs'],ref['args'])]
        for run in registry['runs']:
            if run['dataset'] != dataset:
                continue
            row, epochs, watched, args = gather(run,ref,opts.allow_incomplete)
            per.append(row)
            protocols.append(row)
            comparison.append({key:row[key] for key in ('run_id','dataset','seed','lr','complete','best_epoch','best_validation_wf1','last10_validation_wf1','historical_best_validation_wf1','historical_last10_validation_wf1','delta_best_validation_wf1','delta_last10_validation_wf1','candidate_category') if key in row})
            series.append((run['run_id'],run['lr'],'new',epochs,args))
            for watch in watched:
                if watch['split'] == 'validation' and watch.get('loss_components'):
                    component=watch['loss_components']
                    loss_curves.append({'run_id':run['run_id'],'dataset':dataset,'lr':run['lr'],'source':'new','split':'validation','epoch':watch['epoch'],**{key:component[key] for key in LOSS_KEYS},**loss_weighted(component,args)})
        for run_id,lr,source,epochs,args in series:
            for epoch in epochs:
                curves.append({'run_id':run_id,'dataset':dataset,'seed':seed,'lr':lr,'source':source,'epoch':epoch['epoch'],**{key:epoch[key] for key in CURVE_KEYS}})
                loss_curves.append({'run_id':run_id,'dataset':dataset,'lr':lr,'source':source,'split':'train','epoch':epoch['epoch'],**{key:epoch[key] for key in LOSS_KEYS},**loss_weighted(epoch,args)})
    decisions={dataset:choose_candidate([r for r in per if r['dataset']==dataset]) for dataset in ('IEMOCAP','MELD')}
    summary={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':all(row['complete'] for row in per),
             'completed_runs':sum(row['complete'] for row in per),'planned_runs':4,'per_run':per,'all_protocols':protocols,
             'comparison':comparison,'dataset_decisions':decisions,'second_seed_reserved':2024,'second_seed_started_by_analysis':False,
             'scope':'training/validation only; no test artifacts opened; no GPU or model calls',
             'inputs_sha256':{name:sha(ROOT/name) for name in ['configs/registry.json','provenance/xp04_validation_references.json','scripts/analyze_lr.py']}}
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_run.csv',per),('comparison.csv',comparison),('curves.csv',curves),('loss_curves.csv',loss_curves)]:csv_write(output/name,data)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    prose=generate_report(summary,registry)
    batch=ROOT/'reports/CMERC_LEARNING_RATE_DIAGNOSIS.md';batch.parent.mkdir(parents=True,exist_ok=True);batch.write_text(prose)
    prefix=ROOT.relative_to(REPO).as_posix()
    for directory in ('results','provenance','configs','preflight','reports'):
        prose=prose.replace('../'+directory+'/', '../'+prefix+'/'+directory+'/')
    root_report=REPO/'reports/CMERC_LEARNING_RATE_DIAGNOSIS.md';root_report.parent.mkdir(parents=True,exist_ok=True);root_report.write_text(prose)
    print(json.dumps({'completed':summary['completed_runs'],'all_complete':summary['formal_training_complete'],'decisions':decisions},ensure_ascii=False))


if __name__=='__main__':
    main()
