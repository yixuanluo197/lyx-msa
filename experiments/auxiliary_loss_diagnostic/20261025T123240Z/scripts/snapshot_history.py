from pathlib import Path
import json,hashlib,os,datetime
B=Path(__file__).resolve().parents[1];R=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for x in iter(lambda:s.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
roots=list(json.loads((R/'experiments/hscl_ablation_validation/20261025T080937Z/provenance/history_before.json').read_text())['roots'])+['experiment_records/12_hscl_ablation_validation','experiments/hscl_ablation_validation/20261025T080937Z']
out={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':{}}
for rel in roots:
 base=R/rel;files={};links={}
 for d,dirs,names in os.walk(base,followlinks=False):
  for n in names:
   p=Path(d)/n;k=str(p.relative_to(base))
   if p.is_symlink():links[k]=os.readlink(p)
   elif p.is_file():files[k]=sha(p)
 out['roots'][rel]={'files':files,'symlinks':links}
with (B/'provenance/history_before.json').open('x') as s:json.dump(out,s,indent=2);s.write('\n')
print(json.dumps({'protected_roots':len(roots),'files':sum(len(x['files']) for x in out['roots'].values())}))
