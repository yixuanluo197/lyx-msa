"""Verify delivered artifact bytes and completion evidence without ML dependencies."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'provenance/artifact_hashes.json').read_text())
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4194304), b''):
                digest.update(chunk)
        assert digest.hexdigest() == expected, relative
    overall = json.loads((ROOT / 'verification.json').read_text())
    assert overall['status'] == 'completed' and overall['all_passed']
    assert overall['seven_method_oof_rows'] == 57260
    assert overall['joint_perturbed_rows'] == 163600
    for relative in ['baseline/verification.json', 'provenance/joint_verification.json']:
        assert json.loads((ROOT / relative).read_text())['all_passed']
    behavior = json.loads((ROOT / 'behavior/verification.json').read_text())
    assert not behavior['independent_verification_pending']
    assert behavior['independent_verification']['all_passed']
    print(json.dumps({'all_passed': True, 'artifact_files_verified': len(manifest),
                      'scope': 'delivered artifact hashes and completion evidence; '
                               'numerical reconstruction requires verify_behavior.py'}, indent=2))


if __name__ == '__main__':
    main()
