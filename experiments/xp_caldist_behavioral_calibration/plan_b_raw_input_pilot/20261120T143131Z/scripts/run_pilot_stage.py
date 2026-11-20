#!/usr/bin/env python3
"""Persistent per-dataset pilot supervisor with explicit encoder/CMERC runtimes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
ENCODER = '/home/yangbin/miniconda3/envs/TEXT_repro/bin/python'
CMERC = '/home/yangbin/miniconda3/envs/multimodalerc310/bin/python'
FFMPEG = '/home/yangbin/miniconda3/envs/multimodalerc310/bin/ffmpeg'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    ds = args.dataset
    local = ROOT / 'local_only/pilot' / ds
    output = ROOT / 'results/pilot' / ds
    status_path = ROOT / 'provenance' / ('pilot_stage_' + ds + '.json')
    if status_path.exists():
        raise FileExistsError('Preserve existing pilot supervision record')
    review = local / 'text_reviewed.jsonl'
    if not review.is_file():
        raise FileNotFoundError('Reviewed text must be frozen before running this stage')
    scripts = {name: ROOT / 'scripts' / name for name in
               ['run_pilot_stage.py', 'raw_pilot.py', 'raw_path_smoke.py',
                'extract_pilot_features.py', 'text_encoder.py', 'audio_encoder.py', 'downstream.py']}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
        text=True).strip().split('\n')
    status = dict(status='RUNNING', dataset=ds, started_at=now(), server=socket.gethostname(),
                  supervisor_pid=os.getpid(), host_gpu=args.gpu, logical_gpu='cuda:0',
                  initial_gpu_inventory=inventory, steps=[],
                  code_hashes={str(p): digest(p) for p in scripts.values()},
                  text_review_sha256=digest(review),
                  audio_review_sha256=digest(local / 'audio_label_review.csv'),
                  scientific_scope='Frozen baseline, fixed 120 targets, one modality at a time; no fitting/Test',
                  automatic_audio_qc_is_not_emotion_label_review=True)
    save(status_path, status)
    common = ['--dataset', ds, '--dataset-dir', str(ROOT / 'local_only/data' / ds),
              '--text-features', str(ROOT / 'local_only/features' / ds / 'text_features.pt'),
              '--audio-features', str(ROOT / 'local_only/features' / ds / 'audio_features.pt'),
              '--checkpoint', str(ROOT / 'local_only/data' / ds / 'runs/cmerc/best.pt'),
              '--baseline-lock', str(ROOT / 'configs' / ('BASELINE_LOCK_' + ds + '.json')),
              '--text-model', str(ROOT / 'local_only/text_models' / ds),
              '--text-review', str(review), '--audio-review', str(local / 'audio_label_review.csv'),
              '--local-dir', str(local), '--device', 'cuda:0', '--ffmpeg', FFMPEG]
    commands = [
        ('input_text_qc', [ENCODER, str(scripts['raw_pilot.py']), 'validate-text',
                           *common, '--output', str(output)]),
        ('encoder', [ENCODER, str(scripts['extract_pilot_features.py']), '--mode', 'pilot',
                     *common, '--pilot-output', str(output), '--output', str(local / 'features.pt')]),
        ('cmerc', [CMERC, str(scripts['raw_pilot.py']), 'infer', *common,
                   '--output', str(output), '--precomputed-features', str(local / 'features.pt'),
                   '--modality', 'both'])]
    try:
        for name, command in commands:
            for path, expected in status['code_hashes'].items():
                if digest(path) != expected:
                    raise RuntimeError('Pilot code changed after launch: ' + path)
            log = ROOT / 'logs' / ('pilot_' + name + '_' + ds + '.log')
            step = dict(name=name, command=command, started_at=now(), log=str(log), status='RUNNING')
            status['steps'].append(step)
            save(status_path, status)
            print(json.dumps(dict(dataset=ds, phase=name, started_at=step['started_at'])), flush=True)
            with log.open('x') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                step['pid'] = process.pid
                save(status_path, status)
                returncode = process.wait()
            step.update(returncode=returncode, finished_at=now(), status='COMPLETE' if returncode == 0 else 'FAILED')
            save(status_path, status)
            if returncode:
                raise RuntimeError(name + ' failed; inspect retained log ' + str(log))
        inference = json.loads((output / 'inference_both.json').read_text())
        if inference['planned'] != 1200 or inference['successful'] + inference['failed'] != 1200:
            raise RuntimeError('All fixed 1200 variants must be accounted for, including retained failures')
        if not inference['model_state_bit_exact'] or not inference['precomputed_file_unchanged']:
            raise RuntimeError('Frozen state integrity failed')
        status.update(status='COMPLETE', finished_at=now(), inference=inference,
                      output_sha256=digest(output / 'perturbed_predictions_both.csv'))
        save(status_path, status)
        print(json.dumps(dict(dataset=ds, status='COMPLETE', successful=inference['successful'],
                              retained_failed_variants=inference['failed'])), flush=True)
    except Exception:
        status.update(status='FAILED', finished_at=now(), traceback=traceback.format_exc())
        save(status_path, status)
        raise


if __name__ == '__main__':
    main()
