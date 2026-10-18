"""Durable XP-only engineering preflight, one task per leased physical GPU."""
import csv,fcntl,json,os,socket,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parents[1];CODE=B/'code'; ENV='/data2/yb/conda_envs/cmerc-xp-repro'
UUIDS={1:'GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a',3:'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'}
assert socket.gethostname()=='user23'
state={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'lanes':{}}
mutex=threading.Lock()
def save():
 with mutex:
  tmp=B/'state/gpu_preflight_iemocap.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(B/'state/gpu_preflight_iemocap.json')
def gpu_rows():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(r[0]):r for r in csv.reader(out.splitlines(),skipinitialspace=True)}
def empty(gpu):
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return not any(r[0].strip()==UUIDS[gpu] for r in csv.reader(text.splitlines(),skipinitialspace=True))
def lane(gpu,dataset):
 lane_state={'status':'waiting_gpu','jobs':[]};state['lanes'][str(gpu)]=lane_state;save()
 lock_path=Path('/tmp/cmerc-gpu-locks')/f'cmerc-{UUIDS[gpu]}.lock';lock_path.parent.mkdir(exist_ok=True)
 lease=lock_path.open('a+')
 fcntl.flock(lease,fcntl.LOCK_EX)
 while True:
  row=gpu_rows()[gpu]
  assert 'TITAN Xp' in row[1] and row[2]==UUIDS[gpu]
  if empty(gpu) and int(row[3])-int(row[4])>=11000 and int(row[5])<=10:break
  time.sleep(15)
 lane_state.update(status='running',uuid=UUIDS[gpu]);save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu],CMERC_HOST_GPU_INDEX=str(gpu),CMERC_HOST_GPU_UUID=UUIDS[gpu],CMERC_ENV_PREFIX=ENV,CMERC_PYTHON=str(B/'tools/xp_python'),PYTORCH_CUDA_ALLOC_CONF='garbage_collection_threshold:0.8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 env['LD_LIBRARY_PATH']=ENV+'/lib/python3.10/site-packages/nvidia/cusparse/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 destination=B/'preflight'/f'gpu{gpu}_{dataset}';destination.mkdir(exist_ok=False)
 def run(name,command,extra=None,expected=0):
  for _ in range(20):
   if empty(gpu):break
   time.sleep(.5)
  if not empty(gpu):raise RuntimeError(f'GPU{gpu} acquired an unrelated process; do not share')
  log=B/'logs'/f'preflight_iemocap_gpu{gpu}_{name}.log'
  if log.exists():raise FileExistsError(log)
  child_env=dict(env,CMERC_PROFILE_OUTPUT=str(log)+'.memory.jsonl')
  if extra:child_env.update(extra)
  row={'name':name,'command':command,'gpu_uuid':UUIDS[gpu],'started_at':time.time(),'status':'running','log':str(log)}
  lane_state['jobs'].append(row);save()
  with log.open('x') as handle:
   child=subprocess.Popen(command,cwd=CODE,env=child_env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
   row['pid']=child.pid;save();code=child.wait()
  row.update(exit_code=code,ended_at=time.time(),status='completed' if code==expected else 'failed');save()
  if code!=expected:raise RuntimeError(f'{name} exit{code}, expected{expected}; see {log}')
  return row
 wrapper=str(B/'tools/xp_python')
 run('operators',[wrapper,str(B/'tools/gpu_probe.py'),str(destination/'gpu_probe.json')])
 run('smoke',[wrapper,str(B/'tools/aten_smoke.py'),'--dataset',dataset,'--device','cuda:0','--batch-size','16','--output-dir',str(destination/'smoke')])
 run('official',[wrapper,str(CODE/'scripts/evaluate_checkpoint.py'),'--dataset',dataset,'--checkpoint',str(CODE/'checkpoints'/f'{dataset}_checkpoint.pkl'),'--output-dir',str(destination/'official'),'--device','cuda:0','--batch-size','16','--num-workers','0','--require-noncausal'])
 values=json.loads((destination/'official/metrics.json').read_text())
 target={'IEMOCAP':(71.9838,1143.4928),'MELD':(66.8484,605.0503)}[dataset]
 assert abs(values['weighted_f1']-target[0])<=.05 and abs(values['cel']['total']-target[1])<=.1
 def training(variant,seed,epochs,out):
  return ['bash',str(CODE/'scripts/run_training.sh'),dataset,variant,'validation','--seed',str(seed),'--split-seed','1475','--rng-policy','continuous','--batch-policy','capacity_safe','--edge-capacity','1000','--node-capacity','5200','--batch-max-retries','10000','--num-workers','0','--require-noncausal','--epochs',str(epochs),'--output-dir',str(out/variant)]
 seed=1475 if dataset=='IEMOCAP' else 67137
 for variant,epochs in [('cmerc',7 if dataset=='IEMOCAP' else 3),('baseline',1)]:
  out=destination/'profile'
  extra={'CMERC_GRADIENT_PROBE':str(destination/'gradient_components.jsonl')} if variant=='cmerc' else {}
  run('profile_'+variant,training(variant,seed,epochs,out),extra)
  records=[json.loads(l) for l in (out/variant/f'{dataset}_validation_seed{seed}'/'metrics.jsonl').read_text().splitlines()]
  assert len(records)==epochs and records[-1]['sample_ratio']==1.0
  assert records[-1]['train_dialogues']==(108 if dataset=='IEMOCAP' else 1038)
  run('profile_'+variant+'_evaluate',[wrapper,str(CODE/'scripts/evaluate_checkpoint.py'),'--dataset',dataset,'--checkpoint',str(out/variant/f'{dataset}_validation_seed{seed}'/'best.pt'),'--output-dir',str(destination/'profile_evaluation'/variant),'--device','cuda:0','--batch-size','16','--num-workers','0','--require-noncausal'])
 if dataset=='MELD':
  # Sequential same-GPU comparison includes real model + optimizer + RNG + sampler.
  restore=destination/'resume';restore.mkdir()
  dataset='IEMOCAP'
  cmd=training('cmerc',2024,2,restore/'full')
  run('resume_uninterrupted',cmd)
  partial=training('cmerc',2024,2,restore/'resumed')
  run('resume_interrupted',partial,{'CMERC_STOP_AFTER_FIRST_BEST':'1'},expected=-15)
  checkpoint=restore/'resumed/cmerc/IEMOCAP_validation_seed2024/latest.pt'
  run('resume_restored',partial+['--resume',str(checkpoint)])
  run('resume_verify',[ENV+'/bin/python',str(B/'tools/resume_verify.py'),str(restore)])
  run('rnn_process_resume',[ENV+'/bin/python','-m','pytest','-q','tests/test_multiseed_training.py::test_cuda_rnn_dropout_resume_across_fresh_processes'],{'CMERC_GPU_RESUME_TEST':'1'})
 lane_state.update(status='passed',ended_at=time.time());save();lease.close()
 return dataset
try:
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(lane,gpu,dataset) for gpu,dataset in [(3,'IEMOCAP')]]
  for future in futures:future.result()
 state.update(status='passed',finished_at=time.time());save()
except BaseException as error:
 state.update(status='failed',error=repr(error),finished_at=time.time());save();raise
print(json.dumps({'status':state['status'],'state':str(B/'state/gpu_preflight_iemocap.json')}))
