"""Read-only model-free diagnostic of completed rho1 conditions, before masks finish."""
import json
from pathlib import Path
import numpy as np
from analyze_evidence import dialogue_bootstrap_weights,summarize_condition
r=Path(__file__).resolve().parents[1]
c=json.loads((r/'configs/evidence_sensitivity_config.json').read_text());rows=[]
for ds,seed in c['anchors'].items():
 p=r/f'runs/{ds}/seed{seed}'
 with np.load(p/'baseline_validation.npz') as z:b={k:z[k] for k in z.files}
 di,w=dialogue_bootstrap_weights(b['dialogue_ids'],2000,20261108)
 for t in ['T-All','A','V']:
  with np.load(p/f'conditions/decay_{t}_rho1p00.npz') as z:q=z['probabilities']
  s=summarize_condition(b['labels'],b['probabilities'],q,np.ones(len(q),bool),di,w)
  rows.append(dict(dataset=ds,target=t,delta_accuracy=s['delta_accuracy'],delta_accuracy_ci=[s['delta_accuracy_ci_low'],s['delta_accuracy_ci_high']],true_probability_change=s['true_probability_change'],true_probability_change_ci=[s['true_probability_change_ci_low'],s['true_probability_change_ci_high']]))
print(json.dumps(dict(interim_only=True,source='completed rho1 attenuation validation predictions',rows=rows),indent=2))
