"""Capture protected04-10 archives and raw batches without changing them."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    roots=list(json.loads((REPO/'experiments/dropout_diagnosis/20261025T032711Z/provenance/history_before.json').read_text())['roots'])
    roots+=['experiment_records/10_dropout_diagnosis','experiments/dropout_diagnosis/20261025T032711Z']
    output={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'roots':{}}
    for relative in roots:
        base=REPO/relative
        files,symlinks={},{}
        for directory,dirs,names in os.walk(base,followlinks=False):
            for name in names:
                path=Path(directory)/name;key=str(path.relative_to(base))
                if path.is_symlink(): symlinks[key]=os.readlink(path)
                elif path.is_file(): files[key]=sha(path)
        output['roots'][relative]={'files':files,'symlinks':symlinks}
    path=ROOT/'provenance/history_before.json'
    with path.open('x') as stream: json.dump(output,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'captured','protected_roots':len(roots),'files':sum(len(v['files']) for v in output['roots'].values())}))
if __name__=='__main__':main()
