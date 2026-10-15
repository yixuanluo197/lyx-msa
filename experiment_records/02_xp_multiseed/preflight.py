"""Batch-local gates; no changes to training mathematics or random state."""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

BATCH = Path(__file__).resolve().parent
CODE = BATCH / 'code'
PREFIX = Path('/data2/yb/conda_envs/cmerc-xp-repro')
PYTHON = str(PREFIX / 'bin/python')
GPUS = ['GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248',
        'GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a',
        'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456']
EVENTS = []


def run(name, command, gpu, extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               CMERC_ENV_PREFIX=str(PREFIX), CMERC_PYTHON=PYTHON,
               PYTHONUNBUFFERED='1')
    env.update(extra or {})
    command = ['bash', 'scripts/with_cmerc_env.sh', *command]
    used = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().strip())
    if used > 100:
        raise RuntimeError(f'{name}: GPU busy ({used} MiB), do not overlap unrelated tasks')
    started = time.time()
    with (BATCH / 'logs' / (name + '.log')).open('x') as log:
        child = subprocess.Popen(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
        (BATCH / 'preflight' / (name + '_launch.json')).write_text(json.dumps({
            'command': command, 'pid': child.pid, 'gpu_uuid': gpu,
            'extra_env': extra, 'started_at': started}, indent=2))
        with (BATCH / 'preflight' / (name + '_gpu_samples.jsonl')).open('x') as samples:
            while child.poll() is None:
                resource = subprocess.check_output(['nvidia-smi', '-i', gpu,
                    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip()
                samples.write(json.dumps({'elapsed_seconds': time.time()-started, 'memory_used_utilization':resource})+'\n')
                samples.flush()
                time.sleep(5)
        returncode = child.returncode
    event = dict(name=name, command=command, gpu_uuid=gpu,
                 exit_code=returncode, elapsed_seconds=time.time()-started)
    EVENTS.append(event)
    (BATCH / 'preflight' / (name + '_exit.json')).write_text(json.dumps(event, indent=2))
    if returncode:
        raise RuntimeError(f'{name} failed with exit {returncode}; see log')


def official(dataset, gpu):
    run('official_' + dataset, [PYTHON, str(BATCH / 'profile_run.py'),
        'scripts/evaluate_checkpoint.py', '--dataset', dataset,
        '--checkpoint', 'checkpoints/' + dataset + '_checkpoint.pkl',
        '--output-dir', str(BATCH / 'preflight' / 'official_checkpoint' / dataset),
        '--device', 'cuda:0', '--batch-size', '16', '--num-workers', '0'], gpu,
        {'CMERC_PROFILE_OUTPUT': str(BATCH / 'preflight' / (dataset + '_official_memory.jsonl'))})
    metrics = json.loads((BATCH / 'preflight' / 'official_checkpoint' / dataset / 'metrics.json').read_text())
    expected = (71.9838, 1143.4928) if dataset == 'IEMOCAP' else (66.8484, 605.0503)
    if abs(metrics['weighted_f1']-expected[0]) > .05:
        raise RuntimeError(f'{dataset}: checkpoint W-F1 does not pass 0.05pp gate')
    if abs(metrics['cel']['total']-expected[1]) > .1:
        raise RuntimeError(f'{dataset}: checkpoint CEL differs by >0.1 absolute')
    if set(metrics['cel']['by_removal']) != {'text','audio','visual','intra_context','inter_context'}:
        raise RuntimeError('CEL removal definitions incomplete')


def engineering():
    run('tests', [PYTHON, '-m', 'pytest', '-q', 'tests'], GPUS[2])
    probe = ('import torch,torch_geometric,torch_scatter,torch_sparse,json; '
             'from torch_geometric.nn import GraphConv; '
             'from HypergraphConv import HypergraphConv; '
             'x=torch.randn(3,4,device="cuda",requires_grad=True); '
             'edge=torch.tensor([[0,1,1,2],[0,0,1,1]],device="cuda"); '
             'm=HypergraphConv(4,4).cuda(); y=m(x,edge); y.square().mean().backward(); '
             'assert torch.isfinite(y).all() and torch.isfinite(x.grad).all(); '
             'z=GraphConv(4,3).cuda()(x,edge); z.sum().backward(); '
             'assert torch.isfinite(z).all(); '
             'print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda, '
             '"pyg":torch_geometric.__version__,"scatter":torch_scatter.__version__, '
             '"sparse":torch_sparse.__version__,"gpu":torch.cuda.get_device_name(0),"passed":True}))')
    run('cuda_extensions', [PYTHON, '-c', probe], GPUS[2])
    for dataset in ['IEMOCAP','MELD']:
        run('smoke_' + dataset, [PYTHON, 'scripts/smoke_test.py', '--dataset', dataset,
            '--device', 'cuda:0', '--batch-size', '16',
            '--output-dir', str(BATCH / 'preflight' / 'smoke' / dataset)], GPUS[2])


def profile(dataset, gpu, epochs):
    out = BATCH / 'preflight' / 'profile_runs'
    run('profile_' + dataset, ['bash', 'scripts/run_training.sh', dataset, 'cmerc',
        'official_test', '--num-workers', '0', '--epochs', str(epochs)], gpu,
        {'CMERC_PYTHON':str(BATCH / 'profile_python'), 'CMERC_OUTPUT_DIR':str(out),
         'CMERC_PROFILE_OUTPUT':str(BATCH / 'preflight' / (dataset+'_training_memory.jsonl'))})
    seed = 1475 if dataset == 'IEMOCAP' else 67137
    rows = [json.loads(line) for line in (out / 'cmerc' / f'{dataset}_official_test_seed{seed}' / 'metrics.jsonl').read_text().splitlines()]
    memory = [json.loads(line) for line in (BATCH / 'preflight' / (dataset+'_training_memory.jsonl')).read_text().splitlines()]
    assert len(rows) == epochs and rows[-1]['sample_ratio'] == 1
    assert rows[-1]['train_dialogues'] == (120 if dataset == 'IEMOCAP' else 1152)
    peak = max(row['peak_reserved_mb'] for row in memory)
    if peak > 11200:
        raise RuntimeError(f'{dataset}: insufficient 12GiB headroom, reserved={peak}')
    full = rows[-1]['elapsed_seconds']
    estimate = sum(r['elapsed_seconds'] for r in rows) + ((80 if dataset == 'IEMOCAP' else 40)-epochs)*full
    report = dict(dataset=dataset, engineering_only=True, full_curriculum_epoch_seconds=full,
        cuda_allocator_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        peak_allocated_mb=max(r['peak_allocated_mb'] for r in memory), peak_reserved_mb=peak,
        estimated_seconds_per_run=estimate, estimated_five_run_gpu_hours=estimate*5/3600,
        official_test_validation_loader_empty=True)
    (BATCH / 'preflight' / (dataset+'_resource_profile.json')).write_text(json.dumps(report,indent=2))


def main():
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(official,'IEMOCAP',GPUS[0]),
                       pool.submit(official,'MELD',GPUS[1]), pool.submit(engineering)]
            for future in futures:
                future.result()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(profile,'IEMOCAP',GPUS[0],7),
                       pool.submit(profile,'MELD',GPUS[1],3)]
            for future in futures:
                future.result()
        subprocess.run([str(PREFIX/'bin/python'),str(BATCH/'concurrency_probe.py')],check=True)
        status = {'status':'passed','events':EVENTS, 'w_f1_tolerance_pp':.05, 'cel_tolerance_absolute':.1}
    except Exception as error:
        status = {'status':'failed','error':str(error),'events':EVENTS}
        raise
    finally:
        (BATCH / 'preflight' / 'status.json').write_text(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
