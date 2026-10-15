#!/usr/bin/env python3
"""Archive final executable sources and record the actual experiment environment."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('snapshots'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    candidates = list(root.glob('*.py')) + list((root / 'scripts').glob('*.py'))
    candidates += list((root / 'scripts').glob('*.sh')) + list((root / 'configs').glob('*'))
    candidates += [root / 'source_origin.json', root / 'MULTISEED_PLAN.md', root / 'AGENTS.md']
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(candidates) if path.is_file()}
    payload = {'created_at': datetime.now(timezone.utc).isoformat(),
               'origin': json.loads((root / 'source_origin.json').read_text()),
               'files': files,
               'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    (destination / 'source_snapshot.json').write_text(json.dumps(payload, indent=2) + '\n')
    with tarfile.open(destination / 'code_snapshot.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(root / name, arcname=name, recursive=False)
    import torch
    packages = {}
    for name in ('torch', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'numpy',
                 'scikit-learn', 'pandas', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    environment = {'python': sys.version, 'python_executable': sys.executable,
                   'platform': platform.platform(), 'packages': packages,
                   'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'nvidia_smi': subprocess.check_output(['nvidia-smi',
                       '--query-gpu=index,uuid,name,memory.total,driver_version',
                       '--format=csv'], text=True),
                   'source_snapshot_sha256': payload['sha256']}
    (destination / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps({'snapshot': str(destination), 'files': len(files), 'sha256': payload['sha256']}))


if __name__ == '__main__':
    main()
