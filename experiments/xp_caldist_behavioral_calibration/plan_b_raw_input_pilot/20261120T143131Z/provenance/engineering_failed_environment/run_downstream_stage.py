#!/usr/bin/env python3
"""Hash-locked fresh M3Net -> difficulty -> fresh CMERC -> frozen Validation."""
import argparse,hashlib,json,os,pathlib,socket,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parents[1]
PYTHON='/home/yangbin/miniconda3/envs/TEXT_repro/bin/python'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD']);parser.add_argument('--gpu',required=True);args=parser.parse_args();ds=args.dataset
 data=D/f'local_only/data/{ds}';features=D/f'local_only/features/{ds}';state_file=D/f'provenance/downstream_stage_{ds}.json';script=D/'scripts/downstream.py'
 if state_file.exists():raise FileExistsError('Stage already recorded; inspect explicit resume rather than overwrite')
 text_state=json.loads((D/f'provenance/text_stage_{ds}.json').read_text())
 assert text_state['status']=='COMPLETE'
 required=[data/'dataset.pt',features/'text_features.pt',features/'audio_features.pt',D/'configs/PLAN_B_PROTOCOL.json',D/'configs/DOWNSTREAM_CLASS_WEIGHT_AMENDMENT.json',script,*sorted((D/'scripts/core_snapshot').glob('*.py'))]
 for path in required:
  if not path.is_file():raise FileNotFoundError(path)
 hashes={str(p):sha(p) for p in required}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 gpu=subprocess.check_output(['nvidia-smi','--id='+args.gpu,'--query-gpu=index,uuid,name,memory.free','--format=csv,noheader'],text=True).strip()
 state=dict(dataset=ds,status='RUNNING',server=socket.gethostname(),pid=os.getpid(),gpu_host_index=args.gpu,gpu_inventory_at_start=gpu,started_at=time.time(),hashes=hashes,commands=[],supervisor_sha256=sha(__file__))
 write(state_file,state)
 try:
  for action in ['smoke','train_m3net','build_difficulty','train_cmerc','evaluate']:
   for path,value in hashes.items():
    if sha(path)!=value:raise ValueError('Locked code/input changed: '+path)
   if action=='evaluate':
    lock=dict(dataset=ds,baseline_frozen=True,checkpoint_sha256=sha(data/'runs/cmerc/best.pt'),dataset_sha256=sha(data/'dataset.pt'),text_features_sha256=sha(features/'text_features.pt'),audio_features_sha256=sha(features/'audio_features.pt'),frozen_at=time.time(),selection='Train inner-dev macro-F1 only',official_validation_used_for_selection=False)
    write(D/f'configs/BASELINE_LOCK_{ds}.json',lock)
   cmd=[PYTHON,str(script),action,'--dataset',ds,'--data-dir',str(data),'--device','cuda:0']
   state.update(stage=action);state['commands'].append(cmd);write(state_file,state)
   with (D/f'logs/downstream_{action}_{ds}.log').open('x') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);state['child_pid']=child.pid;write(state_file,state);code=child.wait()
   if code:raise RuntimeError(f'{action} failed exit {code}; retained log, no automatic repeated launch')
  state.update(status='COMPLETE',completed_at=time.time(),baseline_checkpoint_sha256=sha(data/'runs/cmerc/best.pt'));write(state_file,state)
 except BaseException:
  state.update(status='FAILED',failed_at=time.time(),traceback=traceback.format_exc());write(state_file,state);raise
if __name__=='__main__':main()
