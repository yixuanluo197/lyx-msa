"""Regression for physical-LF JSONL record boundaries; synthetic text only."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from raw_pilot import read_jsonl

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows=[{'id':1,'text':'Original\u0085text\u2028with\u2029all separators; escaped newline\nkept.'},
      {'id':2,'text':'Second record: café and 中文.'}]
checks=[]
with tempfile.TemporaryDirectory(prefix='planb-jsonl-regression-') as temporary:
    for ascii_output in (False,True):
        for delimiter in ('\n','\r\n'):
            path=Path(temporary)/('records-'+str(ascii_output)+'-'+str(len(delimiter))+'.jsonl')
            payload=delimiter.join(json.dumps(row,ensure_ascii=ascii_output) for row in rows)+delimiter+delimiter
            path.write_bytes(payload.encode('utf-8'))
            before=digest(path)
            assert read_jsonl(path)==rows,'JSONL content or boundaries changed'
            assert digest(path)==before,'Input bytes changed during read'
            checks.append(dict(ensure_ascii=ascii_output,physical_delimiter='LF' if delimiter=='\n' else 'CRLF',records=2,all_unicode_content_preserved=True,input_unchanged=True))
    old_parser_reproduced=False
    try:
        [json.loads(line) for line in json.dumps(rows[0],ensure_ascii=False).splitlines() if line.strip()]
    except json.JSONDecodeError:
        old_parser_reproduced=True
    assert old_parser_reproduced,'Old splitlines failure was not reproduced'
print(json.dumps(dict(status='PASS',checks=checks,old_splitlines_failure_reproduced=True,synthetic_only=True,validation_accessed=False,script_sha256=digest(ROOT/'scripts/raw_pilot.py')),ensure_ascii=True))

from types import SimpleNamespace
from raw_path_smoke import verify_baseline_complete
with tempfile.TemporaryDirectory(prefix='planb-completion-regression-') as temporary:
    directory=Path(temporary).resolve();checkpoint=directory/'best.pt'
    data=directory/'dataset.pt';text=directory/'text.pt';audio=directory/'audio.pt'
    for path in [checkpoint,data,text,audio]:path.write_text(path.name+' synthetic only')
    inputs={str(path):digest(path) for path in [checkpoint,data,text,audio]}
    complete=directory/'completion.json'
    complete.write_text(json.dumps(dict(status='COMPLETE',stage='cmerc',best_checkpoint_sha256=digest(checkpoint),input_hashes=inputs,completed_at_utc='synthetic-time')))
    args=SimpleNamespace(checkpoint=checkpoint,dataset_dir=directory,text_features=text,audio_features=audio)
    result=verify_baseline_complete(args,inputs)
    assert result['path']==str(complete)
    assert result['sha256']==digest(complete)
    assert result['sha256']!=digest(audio)
    assert result['completed_at']=='synthetic-time'
    # The guard still rejects changed baseline sources.
    corrupted=dict(inputs);corrupted[str(audio)]='wrong'
    try:verify_baseline_complete(args,corrupted)
    except ValueError:pass
    else:raise AssertionError('Changed baseline inputs were accepted')
print(json.dumps(dict(status='PASS',check='completion_provenance_path_and_hash_and_input_guard',synthetic_only=True,validation_accessed=False,script_sha256=digest(ROOT/'scripts/raw_path_smoke.py'))))
