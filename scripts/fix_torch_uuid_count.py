"""Use CUDA's own UUID-aware device count for the pinned torch 1.13.1 env."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

MARKER = '    visible_devices = _parse_visible_devices()\n'
PATCH = ('    # CMERC XP: torch 1.13 NVML parser only accepts integer IDs.\n'
         '    # Returning -1 selects the existing CUDA-runtime fallback.\n'
         '    if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith(("GPU-", "MIG-")):\n'
         '        return -1\n')


def patched_source(source):
    if PATCH in source:
        return source
    if source.count(MARKER) != 1 or 'def _device_count_nvml()' not in source:
        raise ValueError('unrecognized torch CUDA source; refusing patch')
    return source.replace(MARKER, PATCH + MARKER, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-prefix', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    args = parser.parse_args()
    package = args.env_prefix.resolve()/'lib/python3.10/site-packages/torch'
    if "1.13.1+cu117" not in (package/'version.py').read_text():
        raise ValueError('this patch is restricted to torch 1.13.1+cu117')
    path = package/'cuda/__init__.py'
    before = path.read_text()
    after = patched_source(before)
    args.record_dir.mkdir(parents=True,exist_ok=True)
    backup = args.record_dir/'torch_cuda_uuid_original.py'
    if not backup.exists():
        backup.write_text(before)
    path.write_text(after)
    (args.record_dir/'torch_uuid.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile=str(path),tofile=str(path))))
    (args.record_dir/'torch_uuid_patch.json').write_text(json.dumps({
        'path':str(path),'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
        'after_sha256':hashlib.sha256(after.encode()).hexdigest(),
        'scope':'only UUID NVML enumeration; existing CUDA runtime fallback; no model/RNG change',
        'source':'https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/__init__.py'},indent=2)+'\n')


if __name__ == '__main__':
    main()
