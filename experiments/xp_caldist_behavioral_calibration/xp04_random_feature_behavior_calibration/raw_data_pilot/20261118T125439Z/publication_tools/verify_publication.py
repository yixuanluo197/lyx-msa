"""Static publication checks only; never trains or opens raw feature/model files."""
from pathlib import Path
import argparse,ast,csv,datetime,hashlib,io,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--local-source',type=Path);args=parser.parse_args()
 checks=[]
 def check(name,ok):
  checks.append({'check':name,'passed':bool(ok)})
  if not ok:raise AssertionError(name)
 manifest=json.loads((ROOT/'publication_manifest.json').read_text())
 entries=manifest['files']+manifest.get('generated_files',[])
 for row in entries:
  path=ROOT/row['file'];check('public SHA '+row['file'],path.is_file() and sha(path)==row['public_sha256'])
  if args.local_source and row.get('local_source'):
   source=args.local_source/row['local_source'];check('original SHA '+row['file'],sha(source)==row['local_source_sha256'])
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
 check('complete publication coverage',actual-{r['file'] for r in entries}==({'publication_manifest.json','publication_verification.json'} & actual))
 counts={'python':0,'json':0,'jsonl':0,'csv':0,'binary_curves':0}
 forbidden_data={'original_text','processed_text','raw_text','sentences','input_ids','targets'}
 def inspect_json(value):
  if isinstance(value,dict):
   check('no raw data JSON keys',not forbidden_data.intersection(value))
   for v in value.values():inspect_json(v)
  elif isinstance(value,list):
   for v in value:inspect_json(v)
 for name in sorted(actual):
  p=ROOT/name
  check('no private asset '+name,'local_only' not in p.relative_to(ROOT).parts and p.suffix not in {'.pt','.bin','.pkl','.npz','.npy','.safetensors','.hdf5','.wav','.mp4','.zip','.gz','.bpe'} and p.name not in {'encoder.json','vocab.json','merges.txt','dict.txt','train_manifest.csv','validation_manifest.csv','train_texts.jsonl'})
  check('bounded public size '+name,p.stat().st_size<5*1024*1024)
  if p.suffix in {'.png','.pdf'}:
   counts['binary_curves']+=1;continue
  text=p.read_text()
  check('no original local identity '+name,not re.search(r'/(?:data[0-9]+|home)/|\buser[0-9]+\b|GPU-[0-9a-f]{8}-[0-9a-f-]{27}',text))
  check('no credentials '+name,not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text))
  check('no utterance IDs '+name,not re.search(r'Ses0[1-5][FM]_(?:impro|script)',text))
  if p.suffix=='.py':ast.parse(text);counts['python']+=1
  elif p.suffix=='.json':inspect_json(json.loads(text));counts['json']+=1
  elif p.suffix=='.jsonl':
   for line in text.splitlines():inspect_json(json.loads(line))
   counts['jsonl']+=1
  elif p.suffix=='.csv':list(csv.DictReader(io.StringIO(text)));counts['csv']+=1
 local_verify=json.loads((ROOT/'verification.json').read_text());check('local experiment verification passed',local_verify['verified'] is True)
 for ds,n in [('IEMOCAP',527),('MELD',1109)]:
  parity=json.loads((ROOT/'results'/ds/'feature_parity.json').read_text());summary=json.loads((ROOT/'training_records'/ds/'training_summary.json').read_text())
  check(ds+' completed ten epochs best eight',summary['completed_epochs']==10 and summary['best_epoch']==8 and summary['training_complete'] is True)
  check(ds+' all feature gates failed',parity['rows']==n and parity['shape']==[n,4,1024] and parity['feature_gate_passed'] is False and len(parity['layers'])==4 and all(r['passed'] is False for r in parity['layers']))
  check(ds+' probability unrun',parity['probability_status']=='NOT_RUN_FEATURE_GATE_FAILED' and parity['probability_metrics'] is None)
  prob=json.loads((ROOT/'results'/ds/'probability_parity.json').read_text());check(ds+' frozen CMERC not loaded',prob['CMERC_checkpoint_loaded'] is False and prob['metrics'] is None)
  hist=[json.loads(line) for line in (ROOT/'training_records'/ds/'history.jsonl').read_text().splitlines()]
  check(ds+' all ten epochs preserved',[x['epoch'] for x in hist]==list(range(1,11)))
  events=[json.loads(line) for line in (ROOT/'training_records'/ds/'formal_events.jsonl').read_text().splitlines()]
  check(ds+' epoch records equal logs',[x for x in events if x['event']=='epoch']==[dict(event='epoch',**x,improved=(x['inner_dev_mse']==min(v['inner_dev_mse'] for v in hist[:x['epoch']]))) for x in hist])
  if args.local_source:
   src=json.loads((args.local_source/'results'/ds/'feature_parity.json').read_text());check(ds+' all scientific metrics unchanged',src['layers']==parity['layers'])
   for csvname in ['per_layer_metrics.csv','per_coordinate_metrics.csv','loss_curve.csv']:
    check(ds+' CSV parsed fields identical '+csvname,list(csv.DictReader(io.StringIO((args.local_source/'results'/ds/csvname).read_text())))==list(csv.DictReader(io.StringIO((ROOT/'results'/ds/csvname).read_text()))))
 # Relative Markdown resource links in publication entry page must resolve.
 readme=(ROOT/'README.md').read_text()
 for link in re.findall(r'\]\(([^)]+)\)',readme):
  check('README link '+link,link=='publication_verification.json' or (ROOT/link).exists())
 result={'verified':True,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'publication_manifest_sha256':sha(ROOT/'publication_manifest.json'),'checks_passed':len(checks),'format_counts':counts,'local_source_compared':bool(args.local_source),'private_assets_uploaded':False,'scientific_result':'BOTH_DATASET_FEATURE_GATES_FAILED','probability_parity':'NOT_RUN_FEATURE_GATE_FAILED','training_or_validation_rerun_for_publication':False,'note':'Static publication verification is not feature compatibility success.'}
 (ROOT/'publication_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
