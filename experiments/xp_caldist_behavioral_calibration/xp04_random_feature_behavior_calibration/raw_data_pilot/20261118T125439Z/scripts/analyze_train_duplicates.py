"""Train-only deterministic-input consistency diagnostic; never modifies training."""
from pathlib import Path
import json, torch, numpy as np
OUT=Path(__file__).resolve().parents[1]
for ds in ('IEMOCAP','MELD'):
 p=torch.load(OUT/'local_only'/ds/'train_prepared.pt',map_location='cpu',weights_only=False)
 result={'dataset':ds,'official_validation_used':False,'training_changed':False,'partitions':{}}
 for subset in ('fit_indices','dev_indices'):
  indices=[int(i) for i in p[subset]];groups={}
  for i in indices:groups.setdefault(tuple(p['input_ids'][i]),[]).append(i)
  repeat=[ids for ids in groups.values() if len(ids)>1];sse=0.;different=0;maxgap=0.;allrows=0
  for ids in repeat:
   y=p['targets'][ids].numpy().astype(np.float64);gap=float(np.abs(y-y[0]).max());maxgap=max(maxgap,gap)
   different+=int(not np.allclose(y,y[0],atol=1e-4,rtol=1e-4));sse+=float(np.square(y-y.mean(0)).sum());allrows+=len(ids)
  result['partitions'][subset]={'rows':len(indices),'unique_token_sequences':len(groups),'repeated_sequence_groups':len(repeat),'rows_in_repeated_groups':allrows,'repeated_groups_not_allclose_to_first':different,'maximum_within_sequence_target_abs_gap':maxgap,'empirical_deterministic_mse_lower_bound':sse/(len(indices)*4*1024),'definition':'Duplicate exact token sequences must share a deterministic encoder output; sum within-group squared deviations from group mean / total partition elements. Singletons contribute0. Train-only diagnostic, not Validation compatibility or model selection.'}
 path=OUT/'results'/ds;path.mkdir(parents=True,exist_ok=True);(path/'train_duplicate_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
