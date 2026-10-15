#!/usr/bin/env python3
"""Real two-epoch noncausal CMERC interrupt/resume check; never formal results."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    options = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = options.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    processes = []
    handles = []
    command = ['bash', 'scripts/run_training.sh', 'IEMOCAP', 'cmerc', 'validation',
               '--seed', '2024', '--split-seed', '1475', '--rng-policy', 'continuous',
               '--batch-policy', 'capacity_safe', '--edge-capacity', '1000',
               '--node-capacity', '5200', '--batch-max-retries', '10000',
               '--require-noncausal', '--epochs', '2', '--num-workers', '0']
    try:
        for gpu, name in ((1, 'uninterrupted'), (2, 'interrupted')):
            occupied = subprocess.check_output(
                ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
                text=True).strip()
            if occupied:
                raise RuntimeError(f'GPU {gpu} is occupied: {occupied}')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               CMERC_OUTPUT_DIR=str(root / name))
            handle = (root / f'{name}.log').open('x')
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=repo, env=environment,
                                               stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        interrupted = processes[1]
        checkpoint = root / 'interrupted/cmerc/IEMOCAP_validation_seed2024/latest.pt'
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if interrupted.poll() is not None:
                raise RuntimeError('interrupted run exited before interruption window')
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu')
                if saved['epoch'] == 0:
                    os.killpg(interrupted.pid, signal.SIGTERM)
                    interrupted.wait(timeout=30)
                    break
                raise RuntimeError('missed first-epoch interruption window')
            time.sleep(.5)
        else:
            raise RuntimeError('timeout waiting for first full checkpoint')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2',
                           CMERC_OUTPUT_DIR=str(root / 'interrupted'))
        resume_handle = (root / 'resumed.log').open('x')
        handles.append(resume_handle)
        resumed = subprocess.Popen(command + ['--resume', str(checkpoint)], cwd=repo,
                                   env=environment, stdout=resume_handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(resumed)
        assert processes[0].wait(timeout=600) == 0, 'uninterrupted run failed'
        assert resumed.wait(timeout=600) == 0, 'resumed run failed'
        reference = torch.load(root / 'uninterrupted/cmerc/IEMOCAP_validation_seed2024/latest.pt', map_location='cpu')
        actual = torch.load(checkpoint, map_location='cpu')
        assert reference['epoch'] == actual['epoch'] == 1
        assert reference['provenance'] == actual['provenance']
        differences = {name: float((tensor - actual['model_state_dict'][name]).abs().max())
                       for name, tensor in reference['model_state_dict'].items()
                       if tensor.is_floating_point()}
        for name, tensor in reference['model_state_dict'].items():
            torch.testing.assert_close(tensor, actual['model_state_dict'][name], rtol=1e-3, atol=1e-4)
        assert torch.equal(reference['rng_state']['torch'], actual['rng_state']['torch'])
        assert all(torch.equal(a, b) for a, b in zip(reference['rng_state']['cuda'], actual['rng_state']['cuda']))
        payload = {'status': 'passed', 'dataset': 'IEMOCAP', 'variant': 'cmerc',
                   'epochs': 2, 'seed': 2024, 'split_seed': 1475, 'causal': False,
                   'rng_policy': 'continuous', 'host_gpus': [1, 2],
                   'batch_policy': 'capacity_safe', 'edge_capacity': 1000,
                   'node_capacity': 5200, 'batch_max_retries': 10000,
                   'max_parameter_difference': max(differences.values()),
                   'tolerance': {'rtol': 1e-3, 'atol': 1e-4},
                   'rng_states_equal': True,
                   'note': 'Smoke verification only. CUDA atomic reductions need not be bit-identical.'}
        (root / 'verification.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
