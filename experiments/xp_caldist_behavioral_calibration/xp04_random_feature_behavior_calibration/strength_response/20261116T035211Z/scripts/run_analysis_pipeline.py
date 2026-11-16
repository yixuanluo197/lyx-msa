"""Run the fixed CPU analysis once all eight owned inference jobs complete."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data2/yb/conda_envs/cmerc-xp-repro/bin/python'


def record(value):
    value['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = ROOT / 'logs/analysis_pipeline.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(ROOT / 'logs/analysis_pipeline.json')


def main():
    record({'status': 'waiting_for_inference', 'pid': os.getpid()})
    gate = ROOT / 'logs/queue_complete.json'
    while not gate.is_file():
        time.sleep(5)
    if not json.loads(gate.read_text())['all_passed']:
        record({'status': 'blocked_by_inference_failure', 'pid': os.getpid()})
        return 1
    commands = [
        [PYTHON, '-m', 'unittest', 'discover', '-s', str(ROOT / 'scripts'), '-p', 'test_*.py', '-v'],
        [PYTHON, str(ROOT / 'scripts/analyze_strength.py')],
        [PYTHON, str(ROOT / 'scripts/summarize_decision.py')],
        [PYTHON, str(ROOT / 'scripts/verify_strength.py'), '--stage', 'full'],
    ]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for index, command in enumerate(commands):
        name = ['tests', 'analysis', 'decision', 'independent_verification'][index]
        record({'status': 'running', 'stage': name, 'pid': os.getpid(), 'command': command})
        with (ROOT / f'logs/{name}.log').open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        if result.returncode:
            record({'status': 'failed', 'stage': name, 'exit_code': result.returncode,
                    'pid': os.getpid(), 'command': command})
            return result.returncode
    record({'status': 'completed', 'pid': os.getpid(), 'all_passed': True})
    return 0


if __name__ == '__main__':
    sys.exit(main())
