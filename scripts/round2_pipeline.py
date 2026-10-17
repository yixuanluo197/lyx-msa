#!/usr/bin/env python3
"""Complete the frozen IEMOCAP pair after its separately launched baseline.

No implicit resume, parameter search, seed expansion, or test-driven selection.
Each subprocess and gate is recorded. A failed gate stops the pipeline.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    with (batch / 'PIPELINE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(batch)


def execute(batch):
    code = batch / 'code'
    manifest = json.loads((batch / 'RUN_MANIFEST.json').read_text())
    tools_manifest_path = batch / 'ORCHESTRATION_MANIFEST.json'
    tools_manifest_sha = sha(tools_manifest_path)
    tools_manifest = json.loads(tools_manifest_path.read_text())
    python = manifest['python']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CMERC_PYTHON=python, CMERC_OUTPUT_DIR=str(batch / 'runs'))
    state_path = batch / 'PIPELINE_STATE.json'
    if state_path.exists():
        raise FileExistsError('Existing pipeline state: inspect checkpoints; do not silently restart')
    state = {'pid': os.getpid(), 'status': 'running', 'started_utc': now(), 'stages': {}}
    write(state_path, state)

    def frozen():
        if sha(tools_manifest_path) != tools_manifest_sha:
            raise ValueError('Frozen orchestration manifest changed')
        for name, digest in tools_manifest['source_sha256'].items():
            if sha(code / name) != digest:
                raise ValueError('Frozen orchestration tool changed: ' + name)
        for name, digest in manifest['source_sha256'].items():
            if sha(code / name) != digest:
                raise ValueError('Frozen runtime changed: ' + name)
        if sha(batch / 'EXPERIMENT_PROTOCOL.md') != manifest['protocol_sha256']:
            raise ValueError('Frozen protocol changed')
        if sha(batch / 'DATA_ENVIRONMENT.json') != manifest['data_environment_sha256']:
            raise ValueError('Frozen data/ID/environment inventory changed')
        inventory = json.loads((batch / 'DATA_ENVIRONMENT.json').read_text())
        for name, digest in inventory['data_provenance']['feature_files'].items():
            if sha(code / 'data' / name) != digest:
                raise ValueError('Frozen feature bytes changed: ' + name)

    def available():
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        matching = [row for row in rows.splitlines() if row.split(',')[0].strip() == manifest['gpu_uuid']]
        if len(matching) != 1 or int(matching[0].split(',')[1].strip()) < 5000:
            raise RuntimeError('Frozen GPU unavailable or less than 5 GB free: ' + str(matching))
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                             '--format=csv,noheader'], text=True)
        if manifest['gpu_uuid'] in processes:
            raise RuntimeError('Another GPU process is active on the reserved device: ' + processes)
        return {'gpu_free_query': matching, 'compute_processes': processes}

    def run(stage, command, gpu=False, paired_initial=None):
        frozen()
        record = {'command': command, 'cwd': str(code), 'started_utc': now(), 'status': 'running'}
        if gpu:
            record['gpu_before'] = available()
        state['current_stage'] = stage
        state['stages'][stage] = record
        write(state_path, state)
        started = time.monotonic()
        with (batch / (stage + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            write(state_path, state)
            while child.poll() is None:
                if paired_initial is not None:
                    candidates = list((batch / 'runs/cmerc').glob('*/initial_state.json'))
                    if candidates:
                        actual = json.loads(candidates[0].read_text())['model_state_sha256']
                        if actual != paired_initial:
                            child.terminate()
                            child.wait()
                            raise ValueError('Actual paired initialization fingerprint mismatch')
                time.sleep(1)
        record.update(exit_code=child.returncode, ended_utc=now(), elapsed_seconds=time.monotonic()-started,
                      status='completed' if child.returncode == 0 else 'failed')
        write(state_path, state)
        if child.returncode:
            raise RuntimeError(stage + ' failed; see ' + str(batch / (stage + '.log')))
        frozen()

    def one_run(variant):
        candidates = list((batch / 'runs' / variant).glob('*/run_manifest.json'))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one run for ' + variant)
        return candidates[0].parent

    def verify(variant, paired=None, difficulty=None):
        run_dir = one_run(variant)
        verifier = Path(__file__).with_name('round2_verify_run.py')
        cmd = [python, str(verifier), '--run-dir', str(run_dir), '--expected-provenance',
               str(batch / 'DATA_ENVIRONMENT.json'), '--variant', variant, '--epochs', '80',
               '--output', str(batch / (variant.upper() + '_VERIFICATION.json'))]
        if paired:
            cmd += ['--paired-run', str(paired)]
        if difficulty:
            cmd += ['--difficulty', str(difficulty)]
        run(variant + '_verification', cmd)
        return run_dir

    try:
        # Baseline is already running under a persistent supervisor. Never
        # duplicate it or interpret an old/noncausal checkpoint as a teacher.
        state['current_stage'] = 'wait_baseline'
        write(state_path, state)
        while True:
            baseline_process = json.loads((batch / 'BASELINE_PROCESS.json').read_text())
            if baseline_process['status'] != 'running':
                break
            os.kill(baseline_process['supervisor_pid'], 0)
            time.sleep(2)
        if baseline_process.get('exit_code') != 0:
            raise RuntimeError('Formal baseline did not complete successfully')
        baseline = verify('baseline')
        sys.path.insert(0, str(code))
        import torch
        from causal_provenance import (architecture_from_args, data_provenance_from_files,
                                       resolve_causal_difficulty)
        teacher = torch.load(baseline / 'best.pt', map_location='cpu')
        if teacher['epoch'] != teacher['best_epoch']:
            raise ValueError('Only the actual validation-selected best teacher is admitted')
        teacher_sha = sha(baseline / 'best.pt')
        common = ['--causal', '--base-model', 'GRU', '--Dataset', 'IEMOCAP', '--graph_type', 'hyper',
                  '--graph_construct', 'direct', '--multi_modal', '--mm_fusion_mthd', 'concat_DHT',
                  '--modals', 'avl', '--norm', 'LN', '--num_L', '3', '--num_K', '4',
                  '--dropout', '.5', '--batch-size', '16', '--seed', '1937', '--split-seed', '1475',
                  '--replacement-policy', 'prefix_mean_v1']
        run('difficulty_generation', [python, '-u', 'caldiff/cal_diff.py', *common,
                                     '--ckpt_path', str(baseline / 'best.pt')], gpu=True)
        artifacts = list((code / 'caldiff').glob('IEMOCAP_causal_diff_*.pkl'))
        if len(artifacts) != 1:
            raise ValueError('Expected one newly generated immutable difficulty')
        target_dir = batch / 'difficulty'
        target_dir.mkdir(exist_ok=False)
        artifact = target_dir / artifacts[0].name
        for source in (artifacts[0], Path(str(artifacts[0]) + '.manifest.json')):
            shutil.copy2(source, target_dir / source.name)
        data = data_provenance_from_files('IEMOCAP', 1475, root=code / 'data')
        expected_data = json.loads((batch / 'DATA_ENVIRONMENT.json').read_text())['data_provenance']
        if data != expected_data or sha(baseline / 'best.pt') != teacher_sha:
            raise ValueError('Teacher or data changed during difficulty generation')
        values, admitted = resolve_causal_difficulty(artifact, 'IEMOCAP', data,
            architecture_from_args(teacher['args']), expected_teacher_seed=1937)
        if admitted['manifest']['source_checkpoint']['sha256'] != teacher_sha:
            raise ValueError('Difficulty teacher SHA mismatch')
        write(batch / 'DIFFICULTY_VERIFICATION.json', {'status': 'passed',
            'artifact': str(artifact), 'sha256': admitted['sha256'], 'teacher_sha256': teacher_sha,
            'teacher_epoch': teacher['epoch']+1, 'count': len(values),
            'min': min(values.values()), 'max': max(values.values()), 'manifest': admitted['manifest']})
        initial = json.loads((baseline / 'initial_state.json').read_text())['model_state_sha256']
        resolve_causal_difficulty(artifact, 'IEMOCAP', data, architecture_from_args(teacher['args']),
            expected_sha256=admitted['sha256'], expected_teacher_seed=1937)
        run('cmerc_training', ['bash', 'scripts/run_training.sh', 'IEMOCAP', 'cmerc', 'validation',
             '--causal', '--seed', '1937', '--split-seed', '1475', '--norm', 'LN', '--num-workers', '0',
             '--defer-final-test', '--diagnostic-batches', '1', '--fail-on-nonfinite',
             '--difficulty-path', str(artifact)], gpu=True, paired_initial=initial)
        resolve_causal_difficulty(artifact, 'IEMOCAP', data, architecture_from_args(teacher['args']),
            expected_sha256=admitted['sha256'], expected_teacher_seed=1937)
        for checkpoint_name in ('best.pt', 'latest.pt'):
            actual = torch.load(one_run('cmerc') / checkpoint_name, map_location='cpu')
            if actual.get('difficulty_sha256') != admitted['sha256']:
                raise ValueError('Training difficulty differs from the stage-B admitted SHA256')
        cmerc = verify('cmerc', baseline, artifact)
        checkpoint_hashes = {variant: sha(path/'best.pt') for variant,path in (('baseline',baseline),('cmerc',cmerc))}
        for split in ('validation', 'test'):
            if split == 'test':
                freeze = batch / 'VALIDATION_ANALYSIS_FREEZE.json'
                frozen_analysis = json.loads(freeze.read_text())
                if frozen_analysis['best_sha256'] != checkpoint_hashes or sha(batch/'comparison_validation/comparison.json') != frozen_analysis['comparison_sha256']:
                    raise ValueError('Frozen validation analysis or selected objects changed')
            for variant, path in (('baseline', baseline), ('cmerc', cmerc)):
                if sha(path/'best.pt') != checkpoint_hashes[variant]:
                    raise ValueError('Selected checkpoint changed before evaluation')
                output = batch / 'evaluation' / split / variant
                if output.exists():
                    raise FileExistsError('Refusing to repeat an existing evaluation')
                run(split+'_'+variant+'_evaluation', [python, '-u', 'scripts/evaluate_checkpoint.py',
                    '--dataset', 'IEMOCAP', '--causal', '--checkpoint', str(path/'best.pt'),
                    '--output-dir', str(output), '--device', 'cuda:0', '--split', split,
                    '--split-seed', '1475', '--batch-size', '16', '--num-workers', '0', '--n-bins', '15'], gpu=True)
            command = [python, 'scripts/round2_analysis.py', '--baseline-eval', str(batch/'evaluation'/split/'baseline'),
                '--cmerc-eval', str(batch/'evaluation'/split/'cmerc'), '--baseline-training', str(baseline/'metrics.jsonl'),
                '--cmerc-training', str(cmerc/'metrics.jsonl'), '--output-dir', str(batch/('comparison_'+split))]
            if split == 'test':
                command += ['--validation-freeze', str(batch/'comparison_validation/comparison.json')]
            run(split+'_comparison', command)
            if split == 'validation':
                write(batch/'VALIDATION_ANALYSIS_FREEZE.json', {'frozen_utc': now(),
                    'comparison_sha256': sha(batch/'comparison_validation/comparison.json'),
                    'best_sha256': checkpoint_hashes, 'protocol_sha256': manifest['protocol_sha256'],
                    'test_inference_started': False, 'no_post_test_reselection': True})
        frozen()
        state.update(status='completed', ended_utc=now(), current_stage='completed')
        write(state_path, state)
    except Exception as error:
        state.update(status='failed', ended_utc=now(), error_type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc())
        write(state_path, state)
        raise


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


if __name__ == '__main__':
    main()
