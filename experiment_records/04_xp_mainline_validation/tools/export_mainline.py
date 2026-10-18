#!/usr/bin/env python3
"""Back up accepted local artifacts, then publish this batch's lightweight record.

No GPU imports, third-party dependencies, Git operations, or source-data copies.
Run with no arguments only after results/acceptance.json reports status=passed.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


BATCH = Path(__file__).absolute().parents[1]
CMERC = BATCH.parents[2]
BACKUP = Path('/data2/yb/paper/experiment_backups/CMERC/04_xp_mainline_validation') / BATCH.name
PUBLIC = CMERC / 'experiment_records/04_xp_mainline_validation'
TEXT = {'.py', '.sh', '.json', '.jsonl', '.txt', '.md', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.patch', '.diff'}
SMALL_TEXT = {'.json', '.txt', '.md'}
BANNED_DIRS = {'data', 'caldiff', 'checkpoints', '__pycache__', '.pytest_cache', '.mypy_cache',
               '.ruff_cache', '.git', '.cache', 'node_modules', 'site-packages', 'lib', 'lib64',
               'logs', 'raw_logs', 'predictions'}
PREDICTION_NAMES = ('prediction', 'per_sample', 'per-sample', 'sample_prediction')
MAX_PUBLIC_BYTES = 32 * 1024 * 1024
CSV_PREFIXES = ('aggregate', 'per_seed', 'paired', 'registry', 'summary', 'comparison',
                'curve', 'convergence', 'selected_epoch', 'official')
SAMPLE_COLUMNS = {'dialogue_id', 'utterance_id', 'utterance_index', 'sample_id', 'y_true', 'y_pred',
                  'prediction', 'predicted_label', 'true_label', 'label', 'target'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def regular_source(path):
    """Reject symlinks at any level rather than following external inputs."""
    relative = path.relative_to(BATCH)
    current = BATCH
    require(not BATCH.is_symlink(), 'Batch root must not be a symlink')
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), 'Refusing symlink source: ' + str(current))
    require(stat.S_ISREG(path.stat().st_mode), 'Expected regular source: ' + str(path))
    require(path.resolve().is_relative_to(BATCH.resolve()), 'Source escaped this batch')


def tree_files(root, exclude_public=False):
    if not root.exists():
        return
    require(not root.is_symlink(), 'Refusing symlink root: ' + str(root))
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if not exclude_public:
            require(not any((base / name).is_symlink() for name in dirs),
                    'Refusing symlink artifact directory: ' + str(base))
        dirs[:] = sorted(name for name in dirs if not (base / name).is_symlink()
                         and (not exclude_public or (name not in BANNED_DIRS and not name.startswith('.'))))
        for name in sorted(files):
            path = base / name
            if path.is_symlink():
                if exclude_public:
                    continue
                raise RuntimeError('Refusing symlink artifact: ' + str(path))
            regular_source(path)
            yield path


def verified_copy(source, target):
    regular_source(source)
    require(not target.exists() and not target.is_symlink(), 'Refusing overwrite: ' + str(target))
    before = source.stat()
    source_sha = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    require(not target.is_symlink(), 'Copy became a symlink')
    require(not os.path.samefile(source, target), 'Artifact backup must not be a hardlink')
    target_sha = sha256(target)
    after = source.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'Source changed during copy: ' + str(source))
    require(source_sha == target_sha == sha256(source), 'Copy SHA verification failed: ' + str(source))
    return {'source': str(source), 'source_sha256': source_sha, 'target_sha256': target_sha,
            'size_bytes': after.st_size, 'copy_method': 'shutil.copy2 (independent inode, SHA-256 verified)'}


def artifact_sources(run_dirs):
    sources = set()
    for run in run_dirs:
        for name in ('best.pt', 'latest.pt'):
            path = run / name
            regular_source(path)
            sources.add(path)
    for path in tree_files(BATCH / 'results'):
        if path.suffix.lower() in {'.npz', '.csv'}:
            sources.add(path)
    logs = list(tree_files(BATCH / 'logs'))
    require(logs, 'No raw batch logs found; refusing an incomplete backup')
    sources.update(logs)
    require(any(path.suffix.lower() == '.npz' for path in sources), 'No per-sample NPZ results found')
    require(any(path.suffix.lower() == '.csv' and 'prediction' in path.name.lower() for path in sources),
            'No per-sample prediction CSV results found')
    return sorted(sources)


def make_backup(sources, acceptance_sha):
    """An existing complete backup may only be reused after full read-only checks."""
    backup_manifest = BACKUP / 'backup_manifest.json'
    if BACKUP.exists() or BACKUP.is_symlink():
        require(not BACKUP.is_symlink() and backup_manifest.is_file(), 'Existing backup is incomplete or a symlink')
        record = read_json(backup_manifest)
        require(record.get('batch') == str(BATCH) and record.get('acceptance_sha256') == acceptance_sha,
                'Existing backup belongs to a different batch/acceptance')
        require({item['source'] for item in record['files']} == {str(path) for path in sources},
                'Existing backup source set differs')
        for item in record['files']:
            source = Path(item['source'])
            target = BACKUP / item['path']
            require(target.resolve().is_relative_to(BACKUP.resolve()) and not target.is_symlink(),
                    'Unsafe existing backup path')
            require(not os.path.samefile(source, target), 'Existing backup is a hardlink')
            require(sha256(source) == sha256(target) == item['source_sha256'] == item['target_sha256'],
                    'Existing backup SHA mismatch')
        actual = {str(path.relative_to(BACKUP)) for path in BACKUP.rglob('*') if path.is_file()}
        require(actual == {item['path'] for item in record['files']} | {'backup_manifest.json'},
                'Existing backup has unmanifested files')
        return record
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + BACKUP.name + '.', dir=BACKUP.parent))
    record = {'schema_version': 1, 'status': 'verified', 'batch': str(BATCH), 'backup_root': str(BACKUP),
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'acceptance_sha256': acceptance_sha,
              'scope': 'Only this batch: all formal best/latest checkpoints, result NPZ/CSV and raw batch logs',
              'same_filesystem': BATCH.stat().st_dev == BACKUP.parent.stat().st_dev,
              'disaster_recovery_limit': 'Independent directory on the same /data2 filesystem; not an independent-disk or off-site disaster-recovery backup.',
              'source_data_policy': 'Read-only references only; source datasets/features/difficulty files are not copied.',
              'files': []}
    try:
        for source in sources:
            relative = source.relative_to(BATCH)
            item = verified_copy(source, stage / relative)
            item.update({'path': str(relative), 'backup_path': str(BACKUP / relative)})
            record['files'].append(item)
        write_json(stage / 'backup_manifest.json', record)
        require(not BACKUP.exists() and not BACKUP.is_symlink(), 'Backup destination appeared during copy')
        os.rename(stage, BACKUP)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return record


def public_allowed(source, section):
    relative = source.relative_to(BATCH / section)
    lower_parts = [part.lower() for part in relative.parts]
    if any(part in BANNED_DIRS or part.startswith('.') for part in lower_parts):
        return False
    if any(token in source.name.lower() for token in PREDICTION_NAMES):
        return False
    if (source.name.lower().endswith(('.log', '.out', '.err', '.log.txt'))
            or source.stem.lower() in {'stdout', 'stderr', 'raw_log', 'raw_logs', 'training_log'}):
        return False
    if source.stat().st_size > MAX_PUBLIC_BYTES:
        return False
    suffix = source.suffix.lower()
    if section in {'preflight', 'reports'}:
        return suffix in SMALL_TEXT
    if section in {'code', 'tools', 'configs', 'provenance'}:
        return suffix in TEXT or (section == 'tools' and source.name == 'xp_python')
    if section == 'results':
        stem = source.stem.lower()
        if suffix == '.csv':
            if not stem.startswith(CSV_PREFIXES):
                return False
            with source.open(newline='', encoding='utf-8-sig') as handle:
                header = next(csv.reader(handle), [])
            return not ({column.strip().lower() for column in header} & SAMPLE_COLUMNS)
        if suffix == '.json':
            return source.name in {'metrics.json', 'acceptance.json'} or stem.startswith(CSV_PREFIXES)
        if suffix in {'.md', '.txt', '.svg', '.png', '.pdf'}:
            return (stem.startswith(CSV_PREFIXES) or any(part in {'figures', 'curves'} for part in lower_parts)
                    or 'convergence' in stem or 'curve' in stem or stem == 'readme')
    return False


def export_sources(run_dirs):
    selected, excluded = set(), []
    for section in ('code', 'tools', 'configs', 'provenance', 'reports', 'preflight', 'results'):
        for source in tree_files(BATCH / section, exclude_public=True):
            if public_allowed(source, section):
                selected.add(source)
            else:
                excluded.append(str(source.relative_to(BATCH)))
    run_names = {'run_manifest.json', 'metrics.jsonl', 'final_test.json', 'config.json',
                 'effective_config.json', 'effective_args.json', 'effective_configuration.json'}
    for run in run_dirs:
        for name in run_names:
            source = run / name
            if source.exists():
                regular_source(source)
                require(source.stat().st_size <= MAX_PUBLIC_BYTES, 'Run metadata exceeds lightweight limit')
                selected.add(source)
    return sorted(selected), sorted(excluded)


def publish(run_dirs, backup_record, acceptance_sha):
    require(not PUBLIC.exists() and not PUBLIC.is_symlink(), '04 public target exists; refusing overwrite')
    selected, excluded = export_sources(run_dirs)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.04_xp_mainline_validation.', dir=PUBLIC.parent))
    files = []
    try:
        for source in selected:
            relative = source.relative_to(BATCH)
            item = verified_copy(source, stage / relative)
            item['path'] = str(relative)
            files.append(item)
        readme = (
            '# 04 XP mainline validation\n\n'
            f'Batch: `{BATCH.name}`. Source: `{BATCH}`. Export was gated by '
            '`results/acceptance.json` with `status: passed`.\n\n'
            f'Formal runs included: {len(run_dirs)}. This is a lightweight evidence package: frozen '
            'source/scripts/config/tests, batch tools, provenance, reports, JSON/TXT/MD preflight '
            'evidence, formal run manifests/effective configuration/epoch metrics/final tests, '
            'aggregate metrics/tables/summaries/curves. Full file provenance and SHA-256 values '
            'are in `export_manifest.json`. Its own hash is excluded to avoid self-reference.\n\n'
            f'Large artifacts were copied with `shutil.copy2` and SHA-256 verified at `{BACKUP}`. '
            'Each copy has an independent inode; no hardlinks are used. This is an independent '
            'directory on the same /data2 filesystem, not independent-disk disaster recovery. '
            'See `provenance/large_artifact_backup.json`.\n\n'
            'Checkpoints/weights, per-sample prediction CSV/NPZ, raw logs, datasets/features/'
            'difficulty files, symlinks, environment libraries and package caches are excluded. '
            'Source input paths and hashes are retained only as read-only provenance references. '
            'Excluded inputs are not evidence of absent local artifacts.\n\n'
            'Acceptance does not establish unknown upstream feature/difficulty generation '
            'provenance. See input-audit limitations. No older experiment record was modified. '
            'No Git commit or push is performed by this helper.\n'
        )
        (stage / 'README.md').write_text(readme, encoding='utf-8')
        files.append({'path': 'README.md', 'source': str(Path(__file__)) + '::generated README',
                      'source_sha256': sha256(Path(__file__)), 'target_sha256': sha256(stage / 'README.md'),
                      'size_bytes': (stage / 'README.md').stat().st_size, 'transformation': 'generated export documentation'})
        manifest = {'schema_version': 1, 'status': 'exported', 'batch': str(BATCH), 'source_root': str(BATCH),
                    'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'acceptance_sha256': acceptance_sha,
                    'formal_runs': [str(run.relative_to(BATCH)) for run in run_dirs],
                    'backup_root': str(BACKUP), 'backup_file_count': len(backup_record['files']),
                    'filter_policy': {'max_public_file_bytes': MAX_PUBLIC_BYTES,
                                      'no_symlinks': True, 'no_checkpoints_or_predictions_or_raw_logs': True,
                                      'preflight_and_reports_extensions': sorted(SMALL_TEXT),
                                      'manifest_self_hash': 'excluded to avoid self-reference'},
                    'excluded_regular_files': excluded, 'files': sorted(files, key=lambda item: item['path'])}
        write_json(stage / 'export_manifest.json', manifest)
        require(sha256(BATCH / 'results/acceptance.json') == acceptance_sha, 'Acceptance changed during export')
        for item in files:
            require(sha256(stage / item['path']) == item['target_sha256'], 'Public stage SHA mismatch')
        require(not PUBLIC.exists() and not PUBLIC.is_symlink(), '04 public target appeared during export')
        os.rename(stage, PUBLIC)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return len(files)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require(BATCH.name == '20261016T050823Z' and BATCH.parent.name == 'xp_mainline_validation',
            'Helper is bound to the designated batch')
    acceptance_path = BATCH / 'results/acceptance.json'
    regular_source(acceptance_path)
    require(read_json(acceptance_path).get('status') == 'passed', 'Acceptance has not passed; no backup/export performed')
    acceptance_sha = sha256(acceptance_path)
    require(not PUBLIC.exists() and not PUBLIC.is_symlink(), '04 public target already exists; refusing overwrite')
    provenance = BATCH / 'provenance'
    provenance.mkdir(parents=True, exist_ok=True)
    with (provenance / 'export_mainline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not PUBLIC.exists() and not PUBLIC.is_symlink(), '04 public target already exists')
        run_dirs = sorted({path.parent for path in tree_files(BATCH / 'runs') if path.name == 'run_manifest.json'})
        require(run_dirs, 'No formal runs found')
        for run in run_dirs:
            for name in ('run_manifest.json', 'metrics.jsonl', 'final_test.json'):
                regular_source(run / name)
        backup_record = make_backup(artifact_sources(run_dirs), acceptance_sha)
        provenance_path = provenance / 'large_artifact_backup.json'
        if provenance_path.exists():
            require(read_json(provenance_path) == backup_record, 'Existing backup provenance differs; refusing overwrite')
        else:
            write_json(provenance_path, backup_record)
        count = publish(run_dirs, backup_record, acceptance_sha)
        print(json.dumps({'status': 'exported', 'batch': str(BATCH), 'backup': str(BACKUP),
                          'public': str(PUBLIC), 'backup_files': len(backup_record['files']),
                          'public_files': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
