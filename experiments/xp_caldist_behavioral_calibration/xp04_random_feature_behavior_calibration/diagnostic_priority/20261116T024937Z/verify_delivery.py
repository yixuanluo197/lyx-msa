"""Portable read-only diagnostic artifact integrity check."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
manifest=json.loads((r/'provenance/delivery_hashes.json').read_text())
for key,digest in manifest.items():
    assert hashlib.sha256((r/key).read_bytes()).hexdigest()==digest,key
assert json.loads((r/'verification.json').read_text())['all_passed']
audit=json.loads((r/'results/independent_verification.json').read_text())
assert audit['status']=='passed' and audit['failed_checks']==0
print(json.dumps({'all_passed':True,'verified_files':len(manifest),'scope':'existing-repeat statistical diagnostics only'}))
