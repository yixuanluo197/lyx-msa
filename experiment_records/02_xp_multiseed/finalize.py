"""Validate complete outputs and add evidence-grounded batch diagnostics."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from scripts.xp_multiseed import SEEDS, summarize

summary = summarize(ROOT)
state = json.loads((ROOT/'state.json').read_text())
manifest = json.loads((ROOT/'batch_manifest.json').read_text())
assert len(state['tasks']) == 12
assert all(task['status'] in ('completed','failed') for task in state['tasks'].values())
rows = list(csv.DictReader((ROOT/'results.csv').open()))
evidence = {'terminal_runs':12,'successful_runs':0,'failed_runs':[], 'epoch_checks':{},
            'official_checkpoint':{},'baseline_diagnostics':{},'cm_baseline_gains':{}}
historical_baseline = {'IEMOCAP':69.6256,'MELD':66.0644}
for task in manifest['tasks']:
    entry = state['tasks'][task['id']]
    if entry['status'] == 'failed':
        evidence['failed_runs'].append({'id':task['id'],'error':entry.get('error')})
        continue
    evidence['successful_runs'] += 1
    run = ROOT/'runs'/task['variant']/f"{task['dataset']}_official_test_seed{task['seed']}"
    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in records] == list(range(1,task['epochs']+1))
    assert records[-1]['sample_ratio'] == 1.0
    assert records[-1]['train_dialogues'] == (120 if task['dataset']=='IEMOCAP' else 1152)
    assert all(r['selection_protocol']=='official_test' and r['causal'] is False for r in records)
    evidence['epoch_checks'][task['id']] = len(records)

for dataset in SEEDS:
    official = json.loads((ROOT/'preflight/official_checkpoint'/dataset/'metrics.json').read_text())
    evidence['official_checkpoint'][dataset] = {'weighted_f1':official['weighted_f1'],'cel':official['cel']['total']}
    baseline = next(r for r in rows if r['dataset']==dataset and r['variant']=='baseline')
    anchor = next(r for r in rows if r['dataset']==dataset and r['variant']=='cmerc' and int(r['seed'])==SEEDS[dataset][0])
    if baseline['status']=='completed':
        xp = float(baseline['weighted_f1'])
        evidence['baseline_diagnostics'][dataset] = {'xp':xp,'v100':historical_baseline[dataset],
            'xp_minus_v100':xp-historical_baseline[dataset]}
        if anchor['status']=='completed':
            evidence['cm_baseline_gains'][dataset] = float(anchor['weighted_f1'])-xp
    completed = [float(r['weighted_f1']) for r in rows if r['dataset']==dataset and r['variant']=='cmerc' and r['status']=='completed']
    if len(completed)==5:
        assert abs(statistics.mean(completed)-summary['groups'][dataset]['mean']) < 1e-10
        assert abs(statistics.stdev(completed)-summary['groups'][dataset]['sample_std']) < 1e-10

for name,expected in manifest['file_sha256'].items():
    assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected, name
patch = manifest['environment_patch']
assert hashlib.sha256(Path(patch['path']).read_bytes()).hexdigest()==patch['after_sha256']
(ROOT/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')

lines = ['','## 实验链诊断与限制','',
    '| 数据集 | XP baseline 官方 seed | V100 baseline 官方 seed | XP−V100 | XP同 seed CMERC−baseline |',
    '| --- | ---: | ---: | ---: | ---: |']
for dataset in SEEDS:
    d = evidence['baseline_diagnostics'].get(dataset)
    if d:
        gain = evidence['cm_baseline_gains'].get(dataset)
        gain_text = f'{gain:+.4f}' if gain is not None else '缺失'
        lines.append(f"| {dataset} | {d['xp']:.4f} | {d['v100']:.4f} | {d['xp_minus_v100']:+.4f} | {gain_text} |")
    else:
        lines.append(f'| {dataset} | 失败/缺失 | {historical_baseline[dataset]:.4f} | — | — |')
lines += ['', '两个官方 checkpoint 的 XP W-F1/CEL 与历史评估一致，支持输入、推理和指标链对齐；它不能保证从头训练重现论文均值。',
    '默认非因果路径的模型初始化、CPU RNG、辅助同步、课程顺序和损失定义已与上游审计；未发现因果扩展污染当前路径的证据。见代码快照 reports/XP_PROTOCOL_AUDIT.md。',
    '全部 seed 使用固定作者难度文件。MELD 的公开训练集合为 train+valid 共1152对话，IEMOCAP为120对话；official_test 每轮按测试分数选模。论文的hold-out调参与完整五seed仍未公开，数值接近不能消除这些协议差异。',
    '本批为torch1.13.1+cu117/PyG2.4.0匹配环境。为保留UUID绑定，修复了PyTorch1.13的NVML设备计数问题，仅选择既有CUDA runtime回退；原源码、补丁、哈希及加载失败/通过证据已保存。',
    '默认缓存和max_split128的工程运行reserved峰值分别11736/11836MiB，未通过余量门槛。最终统一GC0.8，并以memory_fraction1.0启用torch1.13缓存回收，保持原显存容量、batch16、FP32和所有训练超参数；完整课程探测与同卡吞吐证据均保留。',
    'GPU散射数值差异、作者实际训练配置与初始化/seed集合仍未被完全排除。单seed的XP/V100差异和baseline诊断不足以确定因果，不直接归咎于显卡或随机种子。',
    '失败项进入清单，不替换seed、不挑最高seed，不因未达±0.5点自动追加实验。样本Std为ddof=1；n=5只能粗估当前固定curriculum条件下的波动，不作复现等价或显著提升声明。',
    '', '验证结果见 verification.json，环境与资源见 preflight/，运行状态、配置和日志分别见 state.json、runs/ 和 logs/。','']
with (ROOT/'REPORT.md').open('a') as handle:
    handle.write('\n'.join(lines))
source = ROOT.parents[2]/'reports/08_XP_MULTISEED_STATUS.md'
source.write_text((ROOT/'REPORT.md').read_text())
print(json.dumps({'successful_runs':evidence['successful_runs'],'failed_runs':evidence['failed_runs'],
    'groups':summary['groups'],'both_datasets_within_half_point':summary['both_datasets_within_half_point']},indent=2))
