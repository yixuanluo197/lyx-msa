"""Pinned XP allocator and read-only memory telemetry; no training math changes."""
import atexit
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
import torch

if torch.__version__ != '1.13.1+cu117':
    raise RuntimeError('XP allocator initialization is only validated for torch1.13.1+cu117')
if torch.cuda.device_count() != 1:
    raise RuntimeError('Every XP workload must see exactly one CUDA device')
if 'garbage_collection_threshold:0.8' not in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''):
    raise RuntimeError('Pinned XP allocator GC0.8 is required')
if 'TITAN Xp' not in torch.cuda.get_device_name(0):
    raise RuntimeError('XP workload resolved to a non-XP device')
torch.cuda.set_per_process_memory_fraction(1.0, 0)
started = time.monotonic()
output = Path(os.environ.get('CMERC_PROFILE_OUTPUT', str(Path(__file__).resolve().parents[1] / 'logs' / f'memory_{os.getpid()}.jsonl')))
output.parent.mkdir(parents=True, exist_ok=True)
original = torch.cuda.max_memory_allocated

def record(*args, **kwargs):
    allocated = original(*args, **kwargs)
    row = {'elapsed_seconds': time.monotonic()-started, 'peak_allocated_mib': allocated/2**20,
           'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
           'allocator_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
           'memory_fraction': 1.0, 'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'logical_device': 'cuda:0', 'pid': os.getpid()}
    query = subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        text=True,capture_output=True)
    row['whole_device_snapshot'] = query.stdout.strip()
    with output.open('a') as handle:handle.write(json.dumps(row)+'\n')
    return allocated

torch.cuda.max_memory_allocated = record
@atexit.register
def finish():
    if torch.cuda.is_initialized():record()
if os.environ.get('CMERC_GRADIENT_PROBE'):
    from gradient_observer import install
    install(Path(os.environ['CMERC_GRADIENT_PROBE']))
if os.environ.get('CMERC_STOP_AFTER_FIRST_BEST') == '1':
    # Engineering-only interruption after epoch1 metrics+latest+best atomic writes.
    import signal
    replace = os.replace
    def stop_at_boundary(src, dst, *args, **kwargs):
        result = replace(src, dst, *args, **kwargs)
        if Path(dst).name == 'best.pt':
            record()
            print('PREFLIGHT_EPOCH_BOUNDARY_INTERRUPT', flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
        return result
    os.replace = stop_at_boundary
script=sys.argv[1];sys.argv=sys.argv[1:]
sys.path.insert(0,str(Path(script).resolve().parent))
runpy.run_path(script,run_name='__main__')
