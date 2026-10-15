import atexit, json, os, runpy, sys, time
from pathlib import Path
import torch
if 'garbage_collection_threshold:' in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''):
    # Torch 1.13 only enables allocator GC after setMemoryFraction is called.
    # Fraction 1.0 preserves the original per-device capacity.
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
started = time.monotonic()
output = Path(os.environ["CMERC_PROFILE_OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)
original = torch.cuda.max_memory_allocated

def record(*args, **kwargs):
    allocated = original(*args, **kwargs)
    stats = torch.cuda.memory_stats()
    row = {"elapsed_seconds": time.monotonic()-started, "peak_allocated_mb": allocated/2**20, "peak_reserved_mb": torch.cuda.max_memory_reserved()/2**20, "inactive_split_peak_mb":stats.get('inactive_split_bytes.all.peak',0)/2**20, "inactive_split_current_mb":stats.get('inactive_split_bytes.all.current',0)/2**20, "allocator_conf":os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid()}
    with output.open("a") as handle:
        handle.write(json.dumps(row)+"\n")
    return allocated

torch.cuda.max_memory_allocated = record
@atexit.register
def finish():
    if torch.cuda.is_initialized():
        record()
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
