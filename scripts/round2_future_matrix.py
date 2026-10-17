#!/usr/bin/env python3
"""Write gated future experiment commands; never execute training or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


OFFICIAL = {
    'IEMOCAP': dict(epochs=80, dropout=.5, num_K=4, rank=.005, mscl=.05, cscl=.05, ratio=.15),
    'MELD': dict(epochs=40, dropout=.4, num_K=3, rank=.002, mscl=.15, cscl=.15, ratio=.4),
}
SEEDS = (1937, 2025, 3407)
STAGES = ['baseline_training', 'strict_teacher_admission', 'difficulty_generation',
          'strict_difficulty_admission', 'full_training', 'completed_full_and_pair_admission',
          'validation_evaluation', 'validation_comparison_and_immutable_freeze',
          'one_test_evaluation_per_selected_best', 'paired_result_summary']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variable(name):
    return '"${' + name + ':?bind_verified_' + name + '}"'


def python_command(script, args='', gpu=True):
    prefix = 'CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" ' if gpu else ''
    return prefix + variable('CMERC_PYTHON') + ' -u ' + script + (' ' + args if args else '')


def architecture(dataset, seed):
    settings = OFFICIAL[dataset]
    return (f'--Dataset {dataset} --base-model GRU --graph_type hyper --graph_construct direct '
            '--multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 '
            f'--num_K {settings["num_K"]} --batch-size 16 --lr 0.0001 --l2 0.00003 '
            f'--dropout {settings["dropout"]} --rec-dropout 0.1 --seed {seed} '
            '--split-seed 1475 --causal --replacement-policy prefix_mean_v1')


def training(dataset, seed, variant):
    settings = OFFICIAL[dataset]
    cc = variant not in ('baseline', 'without_cc')
    hscl = variant not in ('baseline', 'without_hscl')
    cl = variant not in ('baseline', 'without_cl')
    policy = 'same_step_same_mode' if variant == 'same_step_same_mode' else 'legacy_epoch_snapshot'
    arguments = (architecture(dataset, seed) + f' --epochs {settings["epochs"]} '
        '--selection-protocol validation --defer-final-test --fail-on-nonfinite '
        f'--diagnostic-batches 1 --num-workers 0 --auxiliary-policy {policy} '
        '--output-dir ' + variable('FUTURE_OUTPUT_ROOT') + f'/{dataset}/seed{seed}/{variant}')
    if cc:
        arguments += f' --calibrate --rank_coff {settings["rank"]}'
    if hscl:
        arguments += f' --contrastlearning --mscl_coff {settings["mscl"]} --cscl_coff {settings["cscl"]}'
    if cl:
        arguments += f' --courselearning --scheduler_steps 1 --epoch_ratio {settings["ratio"]} --difficulty-path '
        arguments += variable(f'DIFFICULTY_{dataset}_{seed}')
    command = python_command('train_cm.py', arguments)
    if cl:
        check = ('import sys;from causal_provenance import sha256_file;'
                 'actual=sha256_file(sys.argv[1]);'
                 '\nif actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")')
        command = python_command('-c ' + shlex.quote(check),
            variable(f'DIFFICULTY_{dataset}_{seed}') + ' ' + variable(f'DIFFICULTY_SHA_{dataset}_{seed}'),
            gpu=False) + ' && ' + command
    return command


DIFFICULTY_CHECK = '''import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='cpu')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'best.pt'): raise ValueError('Teacher is not the admitted paired baseline best')
expected=json.loads(Path(inventory_path).read_text())['data_provenance']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['args']),expected_teacher_seed=int(seed))
source=artifact['manifest']['source_checkpoint']
if source['sha256']!=sha256_file(teacher_path) or source['epoch']!=source['best_epoch']: raise ValueError('Difficulty teacher is not the admitted baseline best')
print(json.dumps({'status':'passed','difficulty_sha256':artifact['sha256'],'teacher_sha256':source['sha256'],'train_count':len(values)}))'''


FREEZE_CODE = '''import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='validation',split_seed=1475,replacement_policy='prefix_mean_v1')
check(all(comparison.get(key)==value for key,value in identity.items()),'Validation comparison identity mismatch')
best_hashes={}
for name,path,directory in [('baseline',baseline_path,baseline_eval),('cmerc',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='cpu'); args=checkpoint['args']
    check(args.get('Dataset')==dataset and args.get('seed')==seed and args.get('split_seed')==1475,'Frozen dataset/training seed mismatch')
    check(args.get('causal') is True and args.get('selection_protocol')=='validation' and checkpoint['epoch']==checkpoint['best_epoch'],'Frozen checkpoint is not validation-selected causal best')
    metadata=json.loads((Path(directory)/'metrics.json').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'Validation evaluator identity mismatch')
    check(Path(metadata['checkpoint']).resolve()==Path(path).resolve(),'Validation evaluated a different checkpoint')
    for filename in ('metrics.json','predictions.npz'):
        check(comparison['input_sha256'][name][filename]==sha256_file(Path(directory)/filename),'Validation comparison input SHA256 changed')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='create':
    frozen={**comparison,'training_seed':seed,'validation_comparison_sha256':comparison_sha,'best_sha256':best_hashes,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('x') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'status':'frozen','validation_freeze_sha256':sha256_file(freeze_path),'training_seed':seed,'best_sha256':best_hashes}))
elif mode=='check':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'Frozen validation JSON SHA256 changed')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('training_seed')==seed,'Frozen validation identity mismatch')
    check(frozen.get('validation_comparison_sha256')==comparison_sha and frozen.get('best_sha256')==best_hashes,'Frozen comparison or selected best SHA256 changed')
    check(frozen.get('input_sha256')==comparison.get('input_sha256'),'Frozen validation input identities changed')
    print(json.dumps({'status':'passed','validation_freeze_sha256':extra[0]}))
else: raise ValueError('Unknown freeze operation')'''


def pair(dataset, seed):
    tag = f'{dataset}_{seed}'
    settings = OFFICIAL[dataset]
    teacher, run, inventory = (variable(prefix + tag) for prefix in
                               ('TEACHER_BEST_', 'BASELINE_RUN_', 'DATA_ENVIRONMENT_'))
    difficulty = variable('DIFFICULTY_' + tag)
    verification = (f'--dataset {dataset} --seed {seed} --split-seed 1475 --epochs {settings["epochs"]} '
                    '--expected-provenance ' + inventory)
    verification_root = variable('FUTURE_OUTPUT_ROOT') + f'/{dataset}/seed{seed}/verification'
    plan = dict(dataset=dataset, training_seed=seed, split_seed=1475, status='planned',
        official=settings, ordered_stages=STAGES,
        split_definition='official train/dev IDs; split seed recorded, no random holdout' if dataset == 'MELD'
            else 'fixed training/validation holdout using split seed 1475',
        baseline_training=training(dataset, seed, 'baseline'), full_training=training(dataset, seed, 'full'),
        strict_teacher_admission=python_command('scripts/round2_verify_run.py',
            verification + ' --variant baseline --run-dir ' + run +
            ' --output ' + verification_root + '/baseline.json', gpu=False),
        difficulty_generation=python_command('caldiff/cal_diff.py', architecture(dataset, seed) +
            ' --ckpt_path ' + teacher + ' --difficulty-output ' + difficulty),
        strict_difficulty_admission=python_command('-c ' + shlex.quote(DIFFICULTY_CHECK),
            f'{dataset} {seed} {difficulty} {teacher} {inventory} {run}', gpu=False),
        completed_full_and_pair_admission=python_command('scripts/round2_verify_run.py',
            verification + ' --variant cmerc --run-dir ' + variable('CMERC_RUN_' + tag) +
            ' --paired-run ' + run + ' --difficulty ' + difficulty +
            ' --expected-difficulty-sha256 ' + variable('DIFFICULTY_SHA_' + tag) +
            ' --output ' + verification_root + '/full.json', gpu=False),
        extension_training={name: training(dataset, seed, name) for name in
            ('without_cl', 'without_hscl', 'without_cc', 'same_step_same_mode')},
        existing_pair='completed round-two pair available for hash-verified reuse; do not overwrite or rerun implicitly'
            if dataset == 'IEMOCAP' and seed == 1937 else None)
    output_root = variable('FUTURE_OUTPUT_ROOT') + f'/{dataset}/seed{seed}'
    comparison = output_root + '/comparison_validation/comparison.json'
    freeze = variable('VALIDATION_FREEZE_' + tag)
    freeze_args = (f'{dataset} {seed} {comparison} {teacher} ' + variable('CMERC_BEST_' + tag) +
                   f' {output_root}/evaluation/validation/baseline {output_root}/evaluation/validation/full {freeze}')
    plan['validation_freeze_create'] = python_command('-c ' + shlex.quote(FREEZE_CODE),
        'create ' + freeze_args, gpu=False)
    plan['validation_freeze_check'] = python_command('-c ' + shlex.quote(FREEZE_CODE),
        'check ' + freeze_args + ' ' + variable('VALIDATION_FREEZE_SHA_' + tag), gpu=False)
    for split in ('validation', 'test'):
        commands = {}
        for variant, checkpoint in [('baseline', teacher), ('full', variable('CMERC_BEST_' + tag))]:
            arguments = (f'--dataset {dataset} --causal --checkpoint {checkpoint} --device cuda:0 '
                         f'--split {split} --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 '
                         '--output-dir ' + variable('FUTURE_OUTPUT_ROOT') + f'/{dataset}/seed{seed}/evaluation/{split}/{variant}')
            command = python_command('scripts/evaluate_checkpoint.py', arguments)
            if split == 'test':
                command = (plan['validation_freeze_check'] + ' && test ! -e ' + output_root +
                           f'/evaluation/test/{variant} && ' + command)
            commands[variant] = command
        plan[split + '_evaluation'] = commands
        analysis_args = (f'--baseline-eval {output_root}/evaluation/{split}/baseline '
            f'--cmerc-eval {output_root}/evaluation/{split}/full --baseline-training {run}/metrics.jsonl '
            '--cmerc-training ' + variable('CMERC_RUN_' + tag) + f'/metrics.jsonl --output-dir {output_root}/comparison_{split}')
        if split == 'test':
            analysis_args += ' --validation-freeze ' + freeze
        plan[split + '_comparison'] = python_command('scripts/round2_analysis.py', analysis_args, gpu=False)
        if split == 'test':
            plan[split + '_comparison'] = plan['validation_freeze_check'] + ' && ' + plan[split + '_comparison']
    plan['full_admission_required'] = [
        'Complete contiguous epoch budget; finite per-epoch losses/metrics and all-batch nonfinite guard enabled',
        'Best/latest/Adam/CUDA RNG/curriculum and earliest validation-max checkpoint identities verified',
        'Actual initial parameter SHA256, architecture, features and split identical to paired baseline',
        'Difficulty SHA256 pinned in every full checkpoint and exact admitted teacher best verified',
        'First-batch raw and weighted loss gradients finite; active parameters update; auxiliaries read only']
    return plan


def generate(completed_batch):
    batch = Path(completed_batch)
    gates = dict(PIPELINE_STATE='completed', BASELINE_VERIFICATION='passed', CMERC_VERIFICATION='passed')
    for name, expected in gates.items():
        if json.loads((batch / (name + '.json')).read_text()).get('status') != expected:
            raise ValueError(name + ' must be ' + expected + ' before future planning')
    targets = [batch / 'FUTURE_MATRIX.json', batch / 'FUTURE_EXPERIMENTS.md']
    if any(path.exists() for path in targets):
        raise FileExistsError('Future plan artifacts already exist; preserve the frozen plan')
    matrix = dict(schema_version=1, status='planned', launched=False,
        gate_sha256={name: sha(batch / (name + '.json')) for name in gates},
        official_wrapper_sha256=sha(Path(__file__).resolve().parent / 'run_training.sh'),
        priority='MELD seed1937 single paired loop first; inspect results/resources before remaining seeds/ablations',
        replacement_policy='prefix_mean_v1', pairs=[pair(dataset, seed)
            for dataset in ('IEMOCAP', 'MELD') for seed in SEEDS])
    lines = ['# 后续正式实验矩阵（全部 planned，未启动）', '',
        '先完成 MELD 单种子 1937 闭环，再决定配对训练种子 1937、2025、3407 的扩展。',
        '训练 seed 与 split seed 分开；IEMOCAP 固定 split seed=1475，MELD 使用官方 train/dev ID，1475 仅记录。',
        '已有 IEMOCAP/1937 成果只能校验哈希后复用；新运行使用新的 FUTURE_OUTPUT_ROOT，禁止覆盖历史。', '',
        '每次启动前重新核对服务器归属、GPU UUID、健康状态、空闲显存和其他进程，显式设置 GPU_UUID；禁止沿用旧物理编号。',
        '在已有可用环境指定 CMERC_PYTHON。每个数据集/seed 先冻结代码、真实特征 SHA256、训练/验证 ID 哈希和 DATA_ENVIRONMENT_*。',
        'TEACHER_BEST_*、BASELINE_RUN_*、CMERC_BEST_*、DIFFICULTY_* 均必须绑定对应真实产物；无默认旧难度文件。', '',
        '执行顺序：Baseline → 完整预算/验证选模教师严格验收 → 真实 difficulty → artifact 严格验收 → Full → 完整训练与配对验收 → validation 评估/比较 → 冻结分析及 best/数据/协议哈希 → 每个 best 一次 test。',
        'FUTURE_MATRIX.json 的 full_admission_required 是测试前强制验收条件。round2_verify_run.py 已提供 --dataset、--seed、--split-seed 与 --epochs 参数，下面给出各组真实接口的完整验收命令。',
        'B 阶段 strict_difficulty_admission 输出的 difficulty_sha256 必须固定到 DIFFICULTY_SHA_*；训练前检查该哈希，Full 验收也显式绑定该 B 阶段哈希。不得从 C 的 checkpoint 反推 B 的预期值。',
        'validation_comparison 完成后执行 validation_freeze_create，VALIDATION_FREEZE_* 为新冻结文件路径，输出的 validation_freeze_sha256 固定到 VALIDATION_FREEZE_SHA_*。每次 test 前检查冻结 JSON、比较内容、验证输入文件、两 best 的 SHA256 及 dataset/train seed/split。已有 test 输出目录安全拒绝重跑。禁止根据 test 结果改变 epoch、seed 或参数。', '',
        '所有训练命令直接调用 train_cm.py，并显式延后测试、启用逐批非有限值保护和首批梯度诊断。difficulty/evaluator 使用各自实际 CLI，不传不支持的训练参数。',
        'w/o CL、w/o HSCL、w/o CC 分别只关闭指定组件；其余系数与预算固定。same_step_same_mode 属于独立辅助策略实验、独立 run 身份，可复用同 seed、同 backbone、validation 来源的 baseline 与 prefix_mean_v1 difficulty。', '',
        '统一报告 W-F1、ECE、五路及总 CEL；prefix_mean_v1 是可逆前缀平滑视图，不是完全模态移除。单种子只报告观察差值，不宣称统计显著性。',
        '多种子按相同训练 seed 配对，固定 split seed、数据和 replacement policy，报告逐 seed 差值、配对均值/标准差和预定配对统计；禁止挑最好 seed。',
        '按预定相对位置 (utterance_index+0.5)/dialogue_length 的三个等宽区间报告前/中/后段分类与校准。', '']
    for item in matrix['pairs']:
        lines += [f'## {item["dataset"]} / training seed {item["training_seed"]}', '']
        commands = {key: item[key] for key in ('baseline_training', 'strict_teacher_admission',
            'difficulty_generation', 'strict_difficulty_admission', 'full_training',
            'completed_full_and_pair_admission')}
        commands.update({'validation_evaluation_' + variant: command
                         for variant, command in item['validation_evaluation'].items()})
        commands.update({key: item[key] for key in
                         ('validation_comparison', 'validation_freeze_create', 'validation_freeze_check')})
        commands.update({'test_evaluation_' + variant: command
                         for variant, command in item['test_evaluation'].items()})
        commands['test_comparison'] = item['test_comparison']
        commands.update({'extension_' + key: value for key, value in item['extension_training'].items()})
        for stage, command in commands.items():
            lines += [f'### {stage}', '', '```bash', command, '```', '']
        lines += ['按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。', '']
    with targets[0].open('x') as handle:
        json.dump(matrix, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    with targets[1].open('x') as handle:
        handle.write('\n'.join(lines))
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-batch', type=Path, required=True)
    args = parser.parse_args()
    matrix = generate(args.completed_batch)
    print(json.dumps(dict(status=matrix['status'], launched=False, pairs=len(matrix['pairs']))))


if __name__ == '__main__':
    main()
