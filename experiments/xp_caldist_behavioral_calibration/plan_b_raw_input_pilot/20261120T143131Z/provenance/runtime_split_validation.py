import argparse,copy,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
D=Path('/data2/yb/paper/lyx-msa/experiments/xp_caldist_behavioral_calibration/plan_b_raw_input_pilot/20261120T143131Z')
sys.path.insert(0,str(D/'scripts'))
import raw_pilot as shared
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--dir',type=Path,required=True);a=p.parse_args();a.dir.mkdir(exist_ok=True,parents=True)
files={name:a.dir/name for name in ['dataset.pt','text.pt','audio.pt','best.pt']}
args=SimpleNamespace(dataset='IEMOCAP',dataset_dir=a.dir,text_features=files['text.pt'],audio_features=files['audio.pt'],checkpoint=files['best.pt'])
cache=a.dir/'features.pt'
def save(bundle):
 torch.save(bundle,cache)
 m=bundle['metadata'];shared.save_json(cache.with_suffix('.manifest.json'),dict(format=m['format'],mode=m['mode'],dataset=m['dataset'],status=m['status'],metadata=m,cache_sha256=shared.sha256(cache)))
if a.stage=='produce':
 import extract_pilot_features as producer
 shared.require_runtime('encoder')
 for k,v in files.items():v.write_text(k+' synthetic only')
 states={name:producer.state_fingerprint(torch.nn.Linear(2,2)) for name in ['text','audio']}
 controls={}
 for modality,shape in [('text',(4,1024)),('audio',(768,))]:
  feature=torch.zeros(shape,dtype=torch.float32)
  rec=dict(sample_id='train_smoke',dialogue_id='synthetic',utterance_index=0,modality=modality,status='EXTRACTED',feature=feature,target_feature_sha256=shared.array_hash(feature),original_feature_sha256=shared.array_hash(feature))
  if modality=='text':rec.update(actual_token_ids=[0,42,2],actual_token_ids_sha256=shared.array_hash(np.array([0,42,2],dtype=np.int64)))
  controls[modality]=rec
 variants=[]
 for modality,kind,version in [('text','neutral_append','neutral_append'),('audio','pitch_shift','pitch_plus_0.5')]:
  rec=copy.deepcopy(controls[modality]);rec.update(perturbation_type=kind,variant_id=version);variants.append(rec)
 metadata=dict(format='PLAN_B_PRECOMPUTED_FEATURES_V2',mode='smoke',dataset='IEMOCAP',status='COMPLETE',runtime=shared.require_runtime('encoder'),scientific_protocol_sha256=shared.sha256(D/'configs/PLAN_B_PROTOCOL.json'),encoder_states_unchanged=True,input_files_unchanged=True,encoder_states_before=states,encoder_states_after=copy.deepcopy(states),encoder_provenance={},input_sha256={str(v.resolve()):shared.sha256(v) for v in files.values()})
 bundle=dict(metadata=metadata,clean_controls={'train_smoke':controls},variants=variants)
 producer.atomic_bundle(cache,bundle)
 # Sidecar is the same explicit producer schema.
 m=bundle['metadata'];shared.save_json(cache.with_suffix('.manifest.json'),dict(format=m['format'],mode=m['mode'],dataset=m['dataset'],status=m['status'],metadata=m,cache_sha256=shared.sha256(cache)))
 print(json.dumps(dict(stage='produce',runtime=metadata['runtime'],synthetic_only=True,status='PASS')))
else:
 runtime=shared.require_runtime('cmerc');baseline=torch.load(cache,map_location='cpu');shared.load_precomputed_features(cache,args,'smoke');tests=['cross_environment_roundtrip']
 def rejected(name, mutate):
  bundle=copy.deepcopy(baseline);mutate(bundle);save(bundle)
  try:shared.load_precomputed_features(cache,args,'smoke')
  except (ValueError,TypeError,KeyError):tests.append(name);return
  raise AssertionError('Accepted '+name)
 rejected('unknown_modality',lambda b:b['variants'][0].update(modality='visual'))
 rejected('family_modality_mismatch',lambda b:b['variants'][0].update(perturbation_type='pitch_shift',variant_id='pitch_plus_0.5'))
 rejected('unknown_status',lambda b:b['variants'][0].update(status='PASS'))
 rejected('token_hash_mismatch',lambda b:b['variants'][0].update(actual_token_ids=[0,99,2]))
 rejected('feature_shape_mismatch',lambda b:b['variants'][0].update(feature=torch.zeros(1024)))
 rejected('control_shape_mismatch',lambda b:b['clean_controls']['train_smoke']['audio'].update(feature=torch.zeros(1)))
 rejected('target_identity_mismatch',lambda b:b['variants'][0].update(dialogue_id='other'))
 rejected('encoder_runtime_mismatch',lambda b:b['metadata']['runtime'].update(transformers='4.45.1'))
 rejected('encoder_state_mutation',lambda b:b['metadata']['encoder_states_after']['text'].update(sha256='bad'))
 save(baseline);m=json.loads(cache.with_suffix('.manifest.json').read_text());m['metadata']['status']='BAD';shared.save_json(cache.with_suffix('.manifest.json'),m)
 try:shared.load_precomputed_features(cache,args,'smoke')
 except ValueError:tests.append('sidecar_metadata_mismatch')
 else:raise AssertionError('sidecar mismatch accepted')
 save(baseline);files['dataset.pt'].write_text('mutated')
 try:shared.load_precomputed_features(cache,args,'smoke')
 except ValueError:tests.append('source_file_mutation')
 else:raise AssertionError('input mutation accepted')
 print(json.dumps(dict(stage='consume',runtime=runtime,status='PASS',tests=tests,synthetic_only=True,real_validation_accessed=False,real_models_loaded=False)))
