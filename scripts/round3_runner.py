#!/usr/bin/env python3
"""Execute one frozen causal pair; validation and authorized test are separate phases.

This supervisor never changes scientific parameters or silently retries training.
Failed attempts and their artifacts remain available for explicit checkpoint resume.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


OFFICIAL = {'IEMOCAP': {'epochs': 80, 'dropout': '.5', 'num_K': '4'},
            'MELD': {'epochs': 40, 'dropout': '.4', 'num_K': '3'}}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_identity(plan, dataset, seed, gpu_uuid):
    require(dataset in OFFICIAL, 'unsupported dataset')
    require(seed in plan['seeds'] and seed in (1937, 2025, 3407), 'seed outside frozen matrix')
    require(not (dataset == 'IEMOCAP' and seed == 1937),
            'IEMOCAP seed1937 must reuse the verified round-two pair; retraining forbidden')
    require(gpu_uuid in plan['gpu_allowlist'], 'device outside frozen healthy GPU allowlist')


def training_command(dataset, seed, variant, difficulty=None, resume=None):
    command = ['bash', 'scripts/run_training.sh', dataset, variant, 'validation',
        '--causal', '--seed', str(seed), '--split-seed', '1475', '--norm', 'LN',
        '--num-workers', '0', '--defer-final-test', '--diagnostic-batches', '1',
        '--fail-on-nonfinite', '--replacement-policy', 'prefix_mean_v1',
        '--auxiliary-policy', 'legacy_epoch_snapshot']
    if variant == 'cmerc':
        require(difficulty is not None, 'full training requires admitted real difficulty')
        command += ['--difficulty-path', str(difficulty)]
    if resume:
        command += ['--resume', str(Path(resume).resolve())]
    return command


def optimizer_statistics(checkpoint, records):
    """Recover executed Adam update counts without mistaking loss magnitudes for gradients."""
    import torch
    expected = sum(math.ceil(record['train_dialogues'] / checkpoint['args']['batch_size']) for record in records)
    states = checkpoint['optimizer_state_dict']['state']
    steps = [int(state['step'].item()) if torch.is_tensor(state['step']) else int(state['step'])
             for state in states.values() if 'step' in state]
    require(steps and max(steps) == expected and min(steps) > 0, 'Adam step count differs from executed batches')
    require(all(torch.isfinite(value).all().item() for state in states.values()
                for value in state.values() if torch.is_tensor(value)), 'nonfinite Adam state')
    rng = checkpoint.get('rng_state', {})
    require(all(key in rng for key in ('python', 'numpy', 'torch', 'cuda')), 'missing full GPU recovery RNG state')
    require(torch.isfinite(rng['torch']).all().item() and
            all(torch.isfinite(value).all().item() for value in rng['cuda']), 'nonfinite RNG state')
    return {'optimizer_steps_total': max(steps), 'expected_optimizer_steps_total': expected,
            'optimizer_parameter_states': len(states), 'per_parameter_step_min': min(steps),
            'per_parameter_step_max': max(steps), 'optimizer_tensors_finite': True,
            'full_rng_state_present': True, 'completed_epochs': checkpoint['epoch'] + 1}


def validate_test_authorization(batch, pair_id, expected_sha, best_paths):
    """Admission gate runs before any test subprocess, including resumed phases."""
    require(expected_sha is not None, 'test phase requires explicit authorization SHA256')
    path = batch / 'TEST_AUTHORIZATION.json'
    require(sha(path) == expected_sha, 'test authorization SHA256 changed')
    auth = json.loads(path.read_text())
    require(auth.get('mode') in ('pilot', 'remaining_matrix'), 'invalid test authorization mode')
    require(auth.get('protocol_sha256') == sha(batch / 'THIRD_ROUND_PROTOCOL.md'),
            'test authorization protocol mismatch')
    plan = json.loads((batch / 'PLAN.json').read_text())
    for field in ('source_sha256', 'data_inventories'):
        if field in auth:
            require(auth[field] == plan[field], 'test authorization frozen ' + field + ' mismatch')
    require(pair_id in auth.get('pairs', {}), 'pair not admitted for test')
    expected_pairs = ({'MELD_seed1937'} if auth['mode'] == 'pilot' else
        {'IEMOCAP_seed2025', 'IEMOCAP_seed3407', 'MELD_seed2025', 'MELD_seed3407'})
    require(set(auth['pairs']) == expected_pairs,
            'authorization must freeze the complete preregistered pilot or remaining matrix')
    # Before any remaining-matrix test, verify ALL four validation comparisons,
    # their actual inference inputs, and the corresponding selected objects.
    for admitted_pair, record in auth['pairs'].items():
        dataset, seed = admitted_pair.rsplit('_seed', 1)
        require(record.get('dataset') == dataset and record.get('seed', record.get('training_seed')) == int(seed),
                'test authorization dataset/seed mismatch')
        pair = batch / 'pairs' / admitted_pair
        comparison = pair / 'comparison_validation' / 'comparison.json'
        require(sha(comparison) == record['comparison_sha256'], 'validation comparison changed after freeze')
        values = json.loads(comparison.read_text())
        require(values.get('dataset') == dataset and values.get('split') == 'validation',
                'frozen comparison identity mismatch')
        for variant in ('baseline', 'cmerc'):
            if admitted_pair == pair_id:
                checkpoint = best_paths[variant]
            else:
                candidates = list((pair / 'runs' / variant).glob('*/best.pt'))
                require(len(candidates) == 1, 'frozen matrix selected object missing/ambiguous')
                checkpoint = candidates[0]
            require(sha(checkpoint) == record['best_sha256'][variant], 'selected best checkpoint changed')
            for filename, digest in values['input_sha256'][variant].items():
                require(sha(pair / 'evaluation' / 'validation' / variant / filename) == digest,
                        'frozen validation inference input changed')
    return auth


class Runner:
    def __init__(self, args):
        self.args = args
        self.batch = args.batch_dir.resolve()
        self.plan_path = self.batch / 'PLAN.json'
        self.plan_sha = sha(self.plan_path)
        self.plan = json.loads(self.plan_path.read_text())
        validate_identity(self.plan, args.dataset, args.seed, args.gpu_uuid)
        self.code = Path(self.plan['code_dir']).resolve()
        self.python = self.plan['python']
        self.pair_id = f'{args.dataset}_seed{args.seed}'
        self.pair = self.batch / 'pairs' / self.pair_id
        self.pair.mkdir(parents=True, exist_ok=True)
        self.state_path = self.pair / 'STATE.json'
        self.inventory_path = Path(self.plan['data_inventories'][args.dataset]['path'])
        self.inventory = json.loads(self.inventory_path.read_text())
        self.data = self.inventory.get('data_provenance', self.inventory)
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        CMERC_PYTHON=self.python, CMERC_OUTPUT_DIR=str(self.pair / 'runs'))
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists() else
                      {'pair_id': self.pair_id, 'dataset': args.dataset, 'seed': args.seed,
                       'created_utc': now(), 'stages': {}, 'phase_attempts': []})
        require(self.state.get('pair_id') == self.pair_id, 'state pair identity mismatch')
        require(self.state.get('plan_sha256', self.plan_sha) == self.plan_sha, 'frozen PLAN changed')
        for attempts in self.state['stages'].values():
            if attempts and attempts[-1].get('status') == 'running':
                pid = attempts[-1].get('pid')
                require(not pid or not Path('/proc', str(pid)).exists(),
                        'previous child still exists; inspect it instead of starting duplicate work')
        self.state.update(pid=os.getpid(), gpu_uuid=args.gpu_uuid, plan_sha256=self.plan_sha)

    def save(self):
        write(self.state_path, self.state)

    def frozen(self):
        require(sha(self.plan_path) == self.plan_sha, 'frozen PLAN changed')
        require(sha(self.batch / 'THIRD_ROUND_PROTOCOL.md') == self.plan['protocol_sha256'],
                'frozen protocol changed')
        for filename, digest in self.plan['source_sha256'].items():
            require(sha(self.code / filename) == digest, 'frozen source changed: ' + filename)
        for dataset, entry in self.plan['data_inventories'].items():
            path = Path(entry['path'])
            require(sha(path) == entry['sha256'], 'frozen data inventory changed: ' + dataset)
            inventory = json.loads(path.read_text())
            for filename, digest in inventory.get('data_provenance', inventory)['feature_files'].items():
                require(sha(self.code / 'data' / filename) == digest, 'real feature bytes changed: ' + filename)

    def available(self):
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                    '--format=csv,noheader,nounits'], text=True)
        matching = [row for row in rows.splitlines() if row.split(',')[0].strip() == self.args.gpu_uuid]
        require(len(matching) == 1 and int(matching[0].split(',')[1]) >= self.args.min_free_mb,
                'selected GPU unavailable or insufficient free memory: ' + repr(matching))
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                    '--format=csv,noheader,nounits'], text=True)
        active = [row for row in processes.splitlines() if row.split(',')[0].strip() == self.args.gpu_uuid]
        allowed = set()
        if self.args.allow_shared_managed:
            for path in (self.batch / 'pairs').glob('*/STATE.json'):
                other = json.loads(path.read_text())
                if other.get('gpu_uuid') != self.args.gpu_uuid or other.get('plan_sha256') != self.plan_sha:
                    continue
                # Only a currently recorded subprocess of this frozen batch is trusted.
                for attempts in other.get('stages', {}).values():
                    if attempts and attempts[-1].get('status') == 'running':
                        allowed.add(attempts[-1].get('pid'))
        require(all(int(row.split(',')[1]) in allowed for row in active),
                'unrelated GPU processes active on selected card: ' + repr(active))
        return {'gpu': matching, 'active_processes': active, 'allowed_managed_pids': sorted(allowed)}

    def one_run(self, variant):
        candidates = list((self.pair / 'runs' / variant).glob('*/run_manifest.json'))
        require(len(candidates) == 1, 'expected exactly one ' + variant + ' run; inspect preserved attempts')
        return candidates[0].parent

    def completed(self, stage):
        attempts = self.state['stages'].get(stage, [])
        if not attempts or attempts[-1]['status'] != 'completed':
            return False
        require(bool(attempts[-1].get('artifact_sha256')), 'completed stage missing artifact hashes')
        for path, digest in attempts[-1]['artifact_sha256'].items():
            require(sha(path) == digest, 'completed stage artifact changed: ' + path)
        return True

    def run(self, stage, command, *, gpu=False, artifacts=(), resume=None, paired_initial=None):
        self.frozen()
        if self.completed(stage):
            require(resume is None, 'cannot resume an already completed stage')
            return
        attempts = self.state['stages'].setdefault(stage, [])
        if attempts and not resume:
            raise RuntimeError(stage + ' has an incomplete/failed attempt; inspect STATE and explicitly resume its checkpoint')
        if resume:
            require(stage.endswith('_training') and Path(resume).is_file(), 'resume requires an existing full training checkpoint')
            require(Path(resume).resolve() == (self.one_run(stage.split('_')[0]) / 'latest.pt').resolve(),
                    'runner resume must use this run latest.pt; historical best resume requires a separate explicit branch')
        record = {'command': command, 'cwd': str(self.code), 'started_utc': now(),
                  'status': 'starting', 'gpu_uuid': self.args.gpu_uuid if gpu else None,
                  'resume_checkpoint': str(resume) if resume else None}
        attempts.append(record)
        self.state.update(status='running', current_stage=stage)
        self.save()
        log_path = self.pair / 'logs' / f'{stage}.attempt{len(attempts)}.log'
        log_path.parent.mkdir(exist_ok=True)
        record['log'] = str(log_path)
        started = time.monotonic()
        child = None
        try:
            if gpu:
                record['gpu_before'] = self.available()
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=self.code, env=self.env, stdout=log, stderr=subprocess.STDOUT)
                record.update(pid=child.pid, status='running')
                self.save()
                while child.poll() is None:
                    if paired_initial is not None:
                        candidates = list((self.pair / 'runs' / 'cmerc').glob('*/initial_state.json'))
                        if candidates:
                            require(len(candidates) == 1, 'ambiguous actual paired initialization')
                            try:
                                fingerprint = json.loads(candidates[0].read_text())['model_state_sha256']
                            except json.JSONDecodeError:
                                # The trainer writes this tiny file directly; a
                                # poll may race the write before training begins.
                                time.sleep(.05)
                                continue
                            require(fingerprint == paired_initial,
                                    'actual paired initialization mismatch')
                    time.sleep(1)
            record['exit_code'] = child.returncode
            require(child.returncode == 0, stage + ' failed; see ' + str(log_path))
            self.frozen()
            paths = artifacts() if callable(artifacts) else artifacts
            record['artifact_sha256'] = {str(path): sha(path) for path in paths}
            require(bool(record['artifact_sha256']), 'stage must pin at least one output artifact')
            record['status'] = 'completed'
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            record.update(status='failed', error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(), exit_code=child.returncode if child else None)
            raise
        finally:
            record.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
            self.save()

    def training_artifacts(self, variant):
        return [self.one_run(variant) / name for name in ('best.pt', 'latest.pt', 'metrics.jsonl',
                'run_manifest.json', 'initial_state.json', 'training_diagnostics.jsonl')]

    def verify(self, variant, difficulty=None, difficulty_sha=None):
        output = self.pair / 'verification' / (variant + '.json')
        command = [self.python, '-u', 'scripts/round2_verify_run.py', '--run-dir', str(self.one_run(variant)),
            '--expected-provenance', str(self.inventory_path), '--variant', variant,
            '--dataset', self.args.dataset, '--seed', str(self.args.seed), '--split-seed', '1475',
            '--epochs', str(OFFICIAL[self.args.dataset]['epochs']), '--output', str(output)]
        if variant == 'cmerc':
            command += ['--paired-run', str(self.one_run('baseline')), '--difficulty', str(difficulty),
                        '--expected-difficulty-sha256', difficulty_sha]
        self.run(variant + '_verification', command, artifacts=[output])
        statistics = self.pair / 'verification' / (variant + '_optimizer.json')
        if not statistics.exists():
            import torch
            run = self.one_run(variant)
            checkpoint = torch.load(run / 'latest.pt', map_location='cpu')
            records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
            result = optimizer_statistics(checkpoint, records)
            result.update(latest_sha256=sha(run / 'latest.pt'), verified_utc=now())
            write(statistics, result)

    def admit_difficulty(self, artifact, teacher_sha, expected_sha=None):
        sys.path.insert(0, str(self.code))
        import torch
        from causal_provenance import architecture_from_args, resolve_causal_difficulty
        teacher_path = self.one_run('baseline') / 'best.pt'
        teacher = torch.load(teacher_path, map_location='cpu')
        require(teacher['epoch'] == teacher['best_epoch'] and sha(teacher_path) == teacher_sha,
                'teacher must be exact admitted validation-selected paired baseline best')
        values, admitted = resolve_causal_difficulty(artifact, self.args.dataset, self.data,
            architecture_from_args(teacher['args']), expected_sha256=expected_sha,
            expected_teacher_seed=self.args.seed)
        require(admitted['manifest']['source_checkpoint']['sha256'] == teacher_sha,
                'difficulty source is not admitted paired best')
        return values, admitted

    def difficulty(self):
        folder = self.pair / 'difficulty'
        report = self.pair / 'verification' / 'difficulty.json'
        baseline_verified = json.loads((self.pair / 'verification' / 'baseline.json').read_text())
        teacher_sha = baseline_verified['best_sha256']
        if report.exists():
            saved = json.loads(report.read_text())
            artifact = Path(saved['artifact'])
            self.admit_difficulty(artifact, teacher_sha, saved['sha256'])
            return artifact, saved['sha256']
        folder.mkdir(exist_ok=True)
        # Unique per-pair staging prevents seed-dependent output collisions in shared code.
        staged = folder / 'generated.pkl'
        common = ['--causal', '--base-model', 'GRU', '--Dataset', self.args.dataset,
            '--graph_type', 'hyper', '--graph_construct', 'direct', '--multi_modal',
            '--mm_fusion_mthd', 'concat_DHT', '--modals', 'avl', '--norm', 'LN',
            '--num_L', '3', '--num_K', OFFICIAL[self.args.dataset]['num_K'],
            '--dropout', OFFICIAL[self.args.dataset]['dropout'], '--batch-size', '16',
            '--seed', str(self.args.seed), '--split-seed', '1475', '--replacement-policy', 'prefix_mean_v1']
        self.run('difficulty_generation', [self.python, '-u', 'caldiff/cal_diff.py', *common,
                 '--ckpt_path', str(self.one_run('baseline') / 'best.pt'), '--difficulty-output', str(staged)],
                 gpu=True, artifacts=[staged, Path(str(staged) + '.manifest.json')])
        values, admitted = self.admit_difficulty(staged, teacher_sha)
        artifact = folder / (self.args.dataset + '_causal_diff_' + admitted['sha256'] + '.pkl')
        for source in (staged, Path(str(staged) + '.manifest.json')):
            destination = artifact if source == staged else Path(str(artifact) + '.manifest.json')
            with source.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        self.admit_difficulty(artifact, teacher_sha, admitted['sha256'])
        write(report, {'status': 'passed', 'artifact': str(artifact), 'sha256': admitted['sha256'],
            'teacher_sha256': teacher_sha, 'count': len(values), 'min': min(values.values()),
            'max': max(values.values()), 'manifest': admitted['manifest']})
        return artifact, admitted['sha256']

    def evaluate(self, split):
        best = {v: self.one_run(v) / 'best.pt' for v in ('baseline', 'cmerc')}
        if split == 'test':
            validate_test_authorization(self.batch, self.pair_id, self.args.test_authorization_sha256, best)
        for variant, checkpoint in best.items():
            if split == 'test':
                validate_test_authorization(self.batch, self.pair_id, self.args.test_authorization_sha256, best)
            output = self.pair / 'evaluation' / split / variant
            stage = f'{split}_{variant}_evaluation'
            if not self.completed(stage):
                require(not output.exists(), 'refusing repeated or partial evaluation: ' + str(output))
            command = [self.python, '-u', 'scripts/evaluate_checkpoint.py', '--dataset', self.args.dataset,
                '--causal', '--checkpoint', str(checkpoint), '--output-dir', str(output), '--device', 'cuda:0',
                '--split', split, '--split-seed', '1475', '--batch-size', '16', '--num-workers', '0', '--n-bins', '15']
            self.run(stage, command, gpu=True, artifacts=[output / 'metrics.json', output / 'predictions.npz'])
            if split == 'test':
                validate_test_authorization(self.batch, self.pair_id, self.args.test_authorization_sha256, best)
        output = self.pair / ('comparison_' + split)
        command = [self.python, 'scripts/round2_analysis.py', '--baseline-eval', str(self.pair / 'evaluation' / split / 'baseline'),
            '--cmerc-eval', str(self.pair / 'evaluation' / split / 'cmerc'),
            '--baseline-training', str(self.one_run('baseline') / 'metrics.jsonl'),
            '--cmerc-training', str(self.one_run('cmerc') / 'metrics.jsonl'), '--output-dir', str(output)]
        if split == 'test':
            command += ['--validation-freeze', str(self.pair / 'comparison_validation' / 'comparison.json')]
        self.run(split + '_comparison', command, artifacts=[output / 'comparison.json', output / 'comparison.md'])

    def execute(self):
        self.state['phase_attempts'].append({'phase': self.args.phase, 'started_utc': now(), 'pid': os.getpid()})
        self.save()
        try:
            self.frozen()
            if self.args.phase == 'validation':
                self.run('baseline_training', training_command(self.args.dataset, self.args.seed, 'baseline',
                    resume=self.args.resume_baseline), gpu=True,
                    artifacts=lambda: self.training_artifacts('baseline'), resume=self.args.resume_baseline)
                self.verify('baseline')
                difficulty, digest = self.difficulty()
                initial = json.loads((self.one_run('baseline') / 'initial_state.json').read_text())['model_state_sha256']
                self.run('cmerc_training', training_command(self.args.dataset, self.args.seed, 'cmerc', difficulty,
                    self.args.resume_cmerc), gpu=True, artifacts=lambda: self.training_artifacts('cmerc'),
                    resume=self.args.resume_cmerc, paired_initial=initial)
                self.admit_difficulty(difficulty, sha(self.one_run('baseline') / 'best.pt'), digest)
                self.verify('cmerc', difficulty, digest)
                self.evaluate('validation')
                self.state.update(status='validation_completed', validation_completed_utc=now())
            else:
                require(self.completed('validation_comparison'), 'validation phase must complete before test')
                self.evaluate('test')
                self.state.update(status='completed', completed_utc=now())
            self.state['phase_attempts'][-1].update(status='completed', ended_utc=now())
        except BaseException as error:
            self.state.update(status='failed', error=str(error), error_type=type(error).__name__,
                              traceback=traceback.format_exc())
            recovery = {}
            for variant in ('baseline', 'cmerc'):
                attempts = self.state['stages'].get(variant + '_training', [])
                if attempts and attempts[-1].get('status') == 'completed':
                    continue
                candidates = list((self.pair / 'runs' / variant).glob('*/latest.pt'))
                if len(candidates) == 1:
                    recovery[variant] = {'latest_checkpoint': str(candidates[0]), 'sha256': sha(candidates[0]),
                        'command': [self.python, str(self.code / 'scripts/round3_runner.py'),
                            '--batch-dir', str(self.batch), '--dataset', self.args.dataset, '--seed', str(self.args.seed),
                            '--gpu-uuid', self.args.gpu_uuid, '--phase', 'validation',
                            '--resume-' + variant, str(candidates[0])]}
            self.state['recovery_candidates'] = recovery
            self.state['phase_attempts'][-1].update(status='failed', ended_utc=now(), error=str(error))
            raise
        finally:
            self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=tuple(OFFICIAL), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--phase', choices=('validation', 'test'), required=True)
    parser.add_argument('--test-authorization-sha256')
    parser.add_argument('--resume-baseline', type=Path)
    parser.add_argument('--resume-cmerc', type=Path)
    parser.add_argument('--allow-shared-managed', action='store_true')
    parser.add_argument('--min-free-mb', type=int, default=5000)
    args = parser.parse_args(argv)
    require(args.min_free_mb > 0, 'minimum free memory must be positive')
    pair = args.batch_dir.resolve() / 'pairs' / f'{args.dataset}_seed{args.seed}'
    pair.mkdir(parents=True, exist_ok=True)
    with (pair / 'RUNNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def terminate(signum, frame):
            raise KeyboardInterrupt('supervisor received signal ' + str(signum))
        signal.signal(signal.SIGTERM, terminate)
        Runner(args).execute()


if __name__ == '__main__':
    main()
