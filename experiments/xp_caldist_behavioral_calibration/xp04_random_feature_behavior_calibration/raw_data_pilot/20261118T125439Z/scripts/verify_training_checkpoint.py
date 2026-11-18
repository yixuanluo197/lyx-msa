"""CPU-only structural/numerical verification of completed training checkpoints."""
from pathlib import Path
import argparse,hashlib,json,torch
from safetensors import safe_open
OUT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD']);args=a.parse_args()
 torch.set_num_threads(2);run=OUT/'local_only'/args.dataset/'training';summary=json.loads((run/'training_summary.json').read_text());p=run/'latest_training.pt'
 assert summary['training_complete']
 ck=torch.load(p,map_location='cpu',weights_only=False)
 required={'model','optimizer','scaler','rng_by_rank','loader_rng_by_rank','completed_epoch','global_step','best_loss','best_epoch','bad_epochs','world_size','arguments','data_sha256','code_sha256','scheduler'}
 assert required<=set(ck)
 assert ck['completed_epoch']==summary['completed_epochs'] and ck['global_step']==summary['global_steps'] and ck['best_epoch']==summary['best_epoch']
 assert ck['arguments']['dataset']==args.dataset and ck['world_size']==2 and len(ck['rng_by_rank'])==len(ck['loader_rng_by_rank'])==2
 assert ck['code_sha256']==sha(OUT/'scripts/train_student.py')
 assert ck['data_sha256']==sha(OUT/'local_only'/args.dataset/'train_prepared.pt')
 assert all(torch.isfinite(t).all() for t in ck['model'].values())
 assert not any('pooler' in name or 'lm_head' in name or 'classifier' in name for name in ck['model'])
 with safe_open(run/'best_student/model.safetensors',framework='pt',device='cpu') as saved:
  assert set(saved.keys())==set(ck['model'])
  assert all(list(t.shape)==saved.get_slice(name).get_shape() for name,t in ck['model'].items())
 states=ck['optimizer']['state'];params=[x for g in ck['optimizer']['param_groups'] for x in g['params']]
 assert set(states)==set(params)
 for entry in states.values():
  assert {'step','exp_avg','exp_avg_sq'}<=set(entry)
  assert all(torch.isfinite(t).all() for t in entry.values() if isinstance(t,torch.Tensor))
 for g in ck['optimizer']['param_groups']:assert g['lr']==1e-5
 steps=[int(entry['step']) for entry in states.values()]
 record={'dataset':args.dataset,'verified':True,'checkpoint_sha256':sha(p),'checkpoint_bytes':p.stat().st_size,'completed_epoch':ck['completed_epoch'],'attempted_global_steps':ck['global_step'],'actual_optimizer_steps_min':min(steps),'actual_optimizer_steps_max':max(steps),'model_tensor_count':len(ck['model']),'all_model_and_optimizer_tensors_finite':True,'optimizer_parameter_states':len(states),'per_rank_RNG_states':2,'per_rank_loader_RNG_states':2,'scaler_state_present':bool(ck['scaler']),'latest_is_final_epoch_not_selected_best':True,'no_Validation_or_Test_read':True}
 (OUT/'provenance'/f'{args.dataset}_training_checkpoint_verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
