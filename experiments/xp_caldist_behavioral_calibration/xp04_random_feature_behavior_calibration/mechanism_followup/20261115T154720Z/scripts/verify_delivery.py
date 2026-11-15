"""Read-only verification of the current delivery, using Python's standard library."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    hashes = json.loads((ROOT / 'provenance/artifact_hashes.json').read_text())
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4194304), b''):
                digest.update(chunk)
        assert digest.hexdigest() == expected, relative
    for relative in ['verification.json', 'diagnostic/verification.json',
                     'calibration/verification.json', 'calibration/independent_verification.json']:
        assert json.loads((ROOT / relative).read_text())['all_passed'], relative
    overall = json.loads((ROOT / 'verification.json').read_text())
    assert overall['calibration']['oof_rows'] == 73620
    assert overall['calibration']['bootstrap_rows'] == 60
    assert not any(overall['meets_continuation_criteria'].values())
    print(json.dumps({'all_passed': True, 'artifact_files_verified': len(hashes),
                      'scope': 'current delivery bytes and completion evidence; '
                               'numerical audit is recorded separately'}, indent=2))


if __name__ == '__main__':
    main()
