#!/usr/bin/env python3
"""Persistent, hash-locked train then extract supervisor for one Plan B dataset."""
import argparse,hashlib,json,os,pathlib,socket,subprocess,sys,time,traceback
D=pathlib.Path(__file__).resolve().parents[1]
PYTHON='/home/yangbin/miniconda3/envs/TEXT_repro/bin/python'
PRETRAINED='/home/yangbin/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD']);a.add_argument('--gpu',required=True);args=a.parse_args()
 ds=args.dataset; script=D/'scripts/text_encoder.py'; data=D/f'local_only/data/{ds}/dataset.pt';out=D/f'local_only/text_models/{ds}';features=D/f'local_only/features/{ds}/text_features.pt'
 state_path=D/f'provenance/text_stage_{ds}.json'
 if state_path.exists():raise FileExistsError('Supervisor already recorded; inspect before explicit resume')
 smoke=json.loads((D/f'local_only/text_smoke/{ds}/smoke_completion.json').read_text())
 assert smoke['status']=='COMPLETE'
 for name in ['backbone','classifier']:
  v=smoke['gradient_verification'][name];assert v['gradient_present'] and v['gradient_finite'] and v['gradient_l2']>0 and v['max_parameter_update']>0
 features.parent.mkdir(parents=True,exist_ok=True)
 alias=data.parent/'features'
 if not alias.exists():alias.symlink_to(features.parent,target_is_directory=True)
 assert alias.resolve()==features.parent.resolve()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 state=dict(dataset=ds,server=socket.gethostname(),pid=os.getpid(),gpu_host_index=args.gpu,started_at=time.time(),status='RUNNING',stage='train',script_sha256=sha(script),data_sha256=sha(data),protocol_sha256=sha(D/'configs/PLAN_B_PROTOCOL.json'),supervisor_sha256=sha(__file__),commands=[])
 write(state_path,state)
 try:
  for mode in ['train','extract']:
   assert sha(script)==state['script_sha256'] and sha(data)==state['data_sha256'] and sha(D/'configs/PLAN_B_PROTOCOL.json')==state['protocol_sha256']
   command=[PYTHON,str(script),'--mode',mode,'--dataset',ds,'--data',str(data),'--output',str(out),'--pretrained',PRETRAINED]
   if mode=='extract':command+=['--features',str(features)]
   state.update(stage=mode);state['commands'].append(command);write(state_path,state)
   with (D/f'logs/text_{mode}_{ds}.log').open('x') as log:
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;write(state_path,state)
    code=child.wait()
   if code:raise RuntimeError(f'{mode} exited {code}; see preserved log')
  state.update(status='COMPLETE',completed_at=time.time(),feature_sha256=sha(features));write(state_path,state)
 except BaseException:
  state.update(status='FAILED',failed_at=time.time(),traceback=traceback.format_exc());write(state_path,state);raise
if __name__=='__main__':main()
