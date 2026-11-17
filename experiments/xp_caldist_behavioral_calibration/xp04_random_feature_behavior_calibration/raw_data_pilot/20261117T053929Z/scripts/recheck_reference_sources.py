"""Fetch small fixed primary source files, never execute their extraction loops."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json,urllib.request
OUT=Path(__file__).resolve().parents[1]
PRIOR=OUT.parent/'20261116T174410Z'

def sha(b):return hashlib.sha256(b).hexdigest()

def fetch(item):
 name,url,old=item
 record={'name':name,'url':url,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'XP04-text-provenance-audit'})
  with urllib.request.urlopen(req,timeout=25) as r:
   data=r.read(2_000_001);record.update(status=r.status,bytes=len(data),content_type=r.headers.get('Content-Type'))
  if len(data)>2_000_000:raise ValueError('Refuse unexpected large source download')
  target=OUT/'sources/reference_code'/name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(data)
  record.update(sha256=sha(data),stored_path=str(target.relative_to(OUT)),prior_sha256=sha(old.read_bytes()),matches_prior=sha(data)==sha(old.read_bytes()))
 except Exception as e:record.update(error=type(e).__name__,message=str(e))
 return record

def main():
 prior=json.loads((PRIOR/'provenance/roberta_sources.json').read_text())
 wanted={'roberta_init_iemocap.py','roberta_init_meld.py','roberta_train_iemocap.sh','roberta_train_meld.sh','roberta_preprocess_iemocap.sh','roberta_preprocess_meld.sh','roberta_feature_extract_iemocap.py','roberta_feature_extract_meld.py','multiprocessing_bpe_encoder.py','train.py','README.md'}
 items=[]
 for r in prior['files']:
  if r['file'] not in wanted or 'error' in r:continue
  url=r.get('source_url',r.get('url'))
  name='ROOT_README.md' if r['file']=='README.md' and '/COSMIC/feature-extraction/' not in url else r['file']
  items.append((name,url,PRIOR/'sources/roberta'/name))
 assert len({x[0] for x in items})==len(items), 'Duplicate destination names'
 repo=json.loads((PRIOR/'provenance/cmerc_m3net_sources.json').read_text())['HITSZ-HLT/CMERC']
 for name in ['README.md','dataloader.py','model_cm.py']:
  items.append(('CMERC_'+name,f"https://raw.githubusercontent.com/HITSZ-HLT/CMERC/{repo['commit']}/{name}",PRIOR/'sources/cmerc_m3net/CMERC'/name))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch,items))
 bpe=[]
 for name in ['encoder.json','vocab.bpe','dict.txt']:
  path=PRIOR/'sources/roberta'/name
  # Keep cached exact bytes; these are tokenizer resources, not corpus contents.
  if not path.is_file():path=Path('/LOCAL/PROJECTS/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction')/name
  bpe.append({'name':name,'local_path':str(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes()),'new_download':False})
 result={'cosmic_commit':prior['commit'],'cmerc_commit':repo['commit'],'records':records,'tokenizer_assets':bpe,'upstream_scripts_executed':False,'models_downloaded':0}
 with (OUT/'provenance/reference_source_recheck.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({'source_files':len(records),'byte_identical_to_previous':sum(r.get('matches_prior',False) for r in records),'failures':[r for r in records if 'error' in r],'tokenizer_assets_reused':len(bpe)}))
if __name__=='__main__':main()
