"""Portable read-only check of the completed interaction experiment delivery."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
expected=json.loads((root/'provenance/delivery_hashes.json').read_text())
for name,digest in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert json.loads((root/'verification.json').read_text())['all_passed']
assert json.loads((root/'results/independent_verification.json').read_text())['all_passed']
print(json.dumps({'all_passed':True,'verified_files':len(expected),'experiment':'confidence_interaction'}))
