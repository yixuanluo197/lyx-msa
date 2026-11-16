"""Portable standard-library delivery audit; never loads models or original features."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    manifest_path = root / 'provenance/delivery_hashes.json'
    manifest = json.loads(manifest_path.read_text())
    if 'files' in manifest:
        manifest = manifest['files']
    assert isinstance(manifest, dict) and manifest, 'Empty delivery manifest'
    checked = set()
    for relative, digest in manifest.items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts, relative
        full = root / path
        assert full.is_file() and not full.is_symlink(), relative
        assert full.resolve().is_relative_to(root.resolve()), relative
        assert full.suffix.lower() not in {'.pt', '.pth', '.pkl', '.pickle', '.ckpt', '.safetensors', '.bin'}, relative
        assert 'features_raw' not in full.name.lower(), relative
        assert hashlib.sha256(full.read_bytes()).hexdigest() == digest, relative
        checked.add(path.as_posix())
    excluded = {'provenance/delivery_hashes.json', 'results/delivery_verification.json'}
    existing = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    assert existing - checked <= excluded, 'Unmanifested artifacts: ' + repr(sorted(existing-checked-excluded))
    audit = json.loads((root/'results/independent_verification.json').read_text())
    assert audit.get('status') == 'passed' and audit.get('stage') == 'full', 'Full independent audit required'
    assert audit.get('all_passed') is True
    return {'status':'passed', 'all_passed':True, 'verified_files':len(checked),
            'relative_paths_only':True, 'artifact_closure':True,
            'no_checkpoint_or_original_feature_payloads':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
