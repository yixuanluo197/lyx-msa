"""Resource-only overlay: preserve the frozen scheduler, exclude host GPU0/1."""
import importlib.util,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('xp_frozen_scheduler',B/'code/scripts/multiseed_scheduler.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
Original=m.Scheduler
class ExclusiveXP3(Original):
 def acquire_gpu(self,excluded=()):
  return super().acquire_gpu(tuple(set(excluded)|{0,1}))
 def attach(self,job):
  if job['status'].startswith('running_') and job.get('host_gpu_index')!=3:
   raise RuntimeError('Resource revision permits active tasks only on host GPU3')
  return super().attach(job)
m.Scheduler=ExclusiveXP3
if __name__=='__main__':raise SystemExit(m.main())
