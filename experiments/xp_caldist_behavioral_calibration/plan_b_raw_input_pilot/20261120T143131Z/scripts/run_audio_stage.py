#!/usr/bin/env python3
"""Pinned official frozen audio encoder GPU smoke and Train/Validation extraction."""
import argparse,hashlib,json,os,pathlib,socket,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parents[1];PYTHON='/home/yangbin/miniconda3/envs/TEXT_repro/bin/python'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD']);parser.add_argument('--gpu',required=True);args=parser.parse_args();ds=args.dataset
 import torch
 data_path=D/f'local_only/data/{ds}/dataset.pt';data=torch.load(data_path,map_location='cpu',weights_only=False)
 source=next(p for dialogue in data['splits']['train'] for p in dialogue['audio_paths'] if p is not None)
 script=D/'scripts/audio_encoder.py';feature=D/f'local_only/features/{ds}/audio_features.pt';feature.parent.mkdir(parents=True,exist_ok=True)
 state_file=D/f'provenance/audio_stage_{ds}.json'
 if state_file.exists():raise FileExistsError('Already recorded stage; inspect before explicit compatible resume')
 cpu=json.loads((D/'provenance/audio_encoder_cpu_smoke.json').read_text());assert cpu['repeat_exact'] and cpu['encoder']['trainable_parameters']==0 and cpu['encoder']['loading_info']['all_base_state_tensors_exact']
 locked=[script,data_path,D/'provenance/audio_model_files.json',D/'configs/PLAN_B_PROTOCOL.json'];hashes={str(p):sha(p) for p in locked}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 state=dict(dataset=ds,status='RUNNING',server=socket.gethostname(),pid=os.getpid(),gpu_host_index=args.gpu,started_at=time.time(),hashes=hashes,commands=[],supervisor_sha256=sha(__file__))
 write(state_file,state)
 try:
  common=[PYTHON,str(script)]
  commands=[('smoke',common+['smoke','--device','cuda:0','--audio',source,'--cache-dir',str(D/'local_only/audio_cache'), '--report',str(D/f'provenance/audio_gpu_smoke_{ds}.json')]),('extract',common+['extract','--device','cuda:0','--dataset',ds.lower(),'--data',str(data_path),'--output',str(feature),'--cache-dir',str(D/'local_only/audio_cache')])]
  for mode,cmd in commands:
   for path,h in hashes.items():assert sha(path)==h
   state.update(stage=mode);state['commands'].append(cmd);write(state_file,state)
   with (D/f'logs/audio_{mode}_{ds}.log').open('x') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);state['child_pid']=p.pid;write(state_file,state);code=p.wait()
   if code:raise RuntimeError(f'{mode} exit {code}; preserved log')
  state.update(status='COMPLETE',completed_at=time.time(),feature_sha256=sha(feature));write(state_file,state)
 except BaseException:
  state.update(status='FAILED',failed_at=time.time(),traceback=traceback.format_exc());write(state_file,state);raise
if __name__=='__main__':main()
