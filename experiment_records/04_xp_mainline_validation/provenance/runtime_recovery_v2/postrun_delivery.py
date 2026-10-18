#!/usr/bin/env python3
"""Supplement this accepted batch's existing backup/public record, on CPU only.

No training, inference, GPU query, Git command, or original archive rewrite.
Use --check-only to inspect readiness; --self-test uses temporary fixtures only.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

BATCH = Path(__file__).resolve().parents[2]
EXPECTED_BATCH = Path('/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z')
CMERC = BATCH.parents[2]
BACKUP = Path('/data2/yb/paper/experiment_backups/CMERC/04_xp_mainline_validation') / BATCH.name
PUBLIC = CMERC / 'experiment_records/04_xp_mainline_validation'
SEEDS = [1475, 67137, 2024, 3407, 42]
MAX_PUBLIC_BYTES = 32 * 1024 * 1024
BANNED_DIRS = {'data', 'caldiff', 'checkpoints', 'weights', 'logs', 'raw_logs', 'predictions',
               '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.cache', '.git',
               'node_modules', 'site-packages', 'lib', 'lib64'}
BANNED_SUFFIXES = {'.pt', '.pth', '.pkl', '.pickle', '.npz', '.npy', '.bin', '.safetensors',
                   '.onnx', '.log', '.out', '.err', '.pyc', '.pyo'}
SAMPLE_COLUMNS = {'dialogue_id', 'utterance_id', 'utterance_index', 'sample_id', 'y_true', 'y_pred',
                  'prediction', 'predicted_label', 'true_label', 'label', 'target'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_path(root, relative):
    root, relative = Path(root).absolute(), Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            f'Unsafe relative path: {relative}')
    current = root
    require(not current.is_symlink(), f'Symlink root: {current}')
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f'Symlink path: {current}')
    require(current.resolve().is_relative_to(root.resolve()), f'Path escapes root: {current}')
    return current


def regular_source(path):
    path = Path(path).absolute()
    require(path.is_relative_to(BATCH), f'Source outside this batch: {path}')
    safe_path(BATCH, path.relative_to(BATCH))
    require(stat.S_ISREG(path.stat().st_mode), f'Not a regular source: {path}')
    return path


def files_in(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), f'Not a regular directory: {root}')
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        require(not any((base / name).is_symlink() for name in dirs + names), f'Symlink in {base}')
        dirs.sort()
        for name in sorted(names):
            path = base / name
            require(stat.S_ISREG(path.stat().st_mode), f'Not a regular file: {path}')
            yield path


def temporary_for(path):
    path = Path(path)
    return path.with_name('.' + path.name + '.postrun.tmp')


def discard_owned_temporary(path):
    temporary = temporary_for(path)
    if os.path.lexists(temporary):
        require(not temporary.is_symlink() and stat.S_ISREG(temporary.stat().st_mode)
                and temporary.stat().st_nlink == 1, f'Unsafe interrupted temporary: {temporary}')
        temporary.unlink()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    discard_owned_temporary(path)
    temporary = temporary_for(path)
    try:
        with temporary.open('xb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if not Path(path).exists() or Path(path).read_bytes() != payload:
        atomic_bytes(path, payload)


def owned_process_alive(pid):
    if not pid:
        return False
    try:
        proc = Path('/proc') / str(int(pid))
        status = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().decode(errors='replace')
        return status != 'Z' and str(BATCH) in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def terminal_gate():
    queue = read_json(regular_source(BATCH / 'state/queue.json'))
    pipeline = read_json(regular_source(BATCH / 'state/pipeline.json'))
    acceptance = read_json(regular_source(BATCH / 'results/acceptance.json'))
    jobs = queue.get('jobs', [])
    expected = {(dataset, variant, seed) for dataset in ('IEMOCAP', 'MELD')
                for variant in ('baseline', 'cmerc') for seed in SEEDS}
    require(queue.get('phase') == 'completed' and len(jobs) == 20
            and all(job.get('status') == 'completed' for job in jobs)
            and {(job['dataset'], job['variant'], job['seed']) for job in jobs} == expected,
            'All 20 registered tasks must be completed before delivery')
    require(all(job.get('training_exit_code') == 0 and job.get('evaluation_exit_code') == 0 for job in jobs),
            'Nonzero/missing final run exit code')
    require(pipeline.get('status') == 'completed', 'Continuation pipeline has not completed')
    require(acceptance.get('status') == 'passed' and acceptance.get('runs_completed') == 20,
            'The complete batch acceptance has not passed')
    require(acceptance.get('batch') == str(BATCH), 'Acceptance belongs to another batch')
    require(acceptance.get('queue_sha256') == sha256(BATCH / 'state/queue.json'), 'Accepted queue changed')
    require(acceptance.get('summary_sha256') == sha256(regular_source(BATCH / 'results/summary.json')),
            'Accepted summary changed')
    require(read_json(regular_source(BATCH / 'configs/xp_mainline_validation.json')) == queue['config'],
            'Effective queue configuration changed')
    require(acceptance.get('immutable_manifest_sha256') == sha256(regular_source(BATCH / 'provenance/immutable_inputs.json')),
            'Immutable manifest changed after acceptance')
    for name, state in [('pipeline', pipeline),
                        ('manager', read_json(regular_source(BATCH / 'state/recovery/manager.json'))),
                        ('resource guard', read_json(regular_source(BATCH / 'state/resource_guard.json')))]:
        require(state.get('status') in ('completed', 'released'), f'{name} is not terminal')
        require(not owned_process_alive(state.get('pid')), f'{name} still runs; logs are not terminal yet')
    old = read_json(regular_source(BATCH / 'state/recovery/pipeline_v1.json'))
    require(old.get('phases', {}).get('formal', {}).get('exit_code') == 130,
            'Missing preserved original controller exit 130')
    require(not owned_process_alive(old.get('pid')), 'Original pipeline is still alive')
    snapshot = read_json(regular_source(BATCH / 'code/snapshots/source_snapshot.json'))
    require(hashlib.sha256(json.dumps(snapshot['files'], sort_keys=True).encode()).hexdigest() == snapshot['sha256']
            == acceptance.get('source_snapshot_sha256'), 'Source snapshot identity differs from acceptance')
    for name, digest in snapshot['files'].items():
        require(sha256(safe_path(BATCH / 'code', name)) == digest, f'Frozen source changed: {name}')
    return queue, pipeline, acceptance


def verify_history():
    recorded = read_json(regular_source(BATCH / 'provenance/history_trees.json'))
    require(set(recorded) == {'01_v100_single_seed', '02_xp_multiseed', '03_v100_multiseed'},
            'Unexpected history tree registry')
    counts = {}
    for name, expected in recorded.items():
        root = CMERC / 'experiment_records' / name
        actual = {str(path.relative_to(root)): sha256(path) for path in files_in(root)}
        require(actual == expected, f'Historical archive changed: {name}')
        counts[name] = len(actual)
    return counts


def item_map(record):
    files = record.get('files', [])
    require(isinstance(files, list), 'Manifest files must be a list')
    result = {item['path']: item for item in files}
    require(len(result) == len(files), 'Duplicate manifest paths')
    return result


def associated_manifests(acceptance_sha):
    backup = read_json(safe_path(BACKUP, 'backup_manifest.json'))
    public = read_json(safe_path(PUBLIC, 'export_manifest.json'))
    for label, record in [('backup', backup), ('public', public)]:
        require(record.get('schema_version') == 1 and record.get('batch') == str(BATCH)
                and record.get('acceptance_sha256') == acceptance_sha,
                f'Existing {label} is not associated with this accepted batch')
    require(backup.get('backup_root') == str(BACKUP) and public.get('backup_root') == str(BACKUP)
            and public.get('source_root') == str(BATCH), 'Manifest root association mismatch')
    require(backup.get('status') == 'verified' and public.get('status') == 'exported', 'Existing export is incomplete')
    return backup, public


def is_prefix(prefix, whole):
    with Path(prefix).open('rb') as old, Path(whole).open('rb') as new:
        for block in iter(lambda: old.read(1024 * 1024), b''):
            if new.read(len(block)) != block:
                return False
    return True


def sync_file(source, target, old_item=None, allow_source_change=False):
    """Atomic independent copy; safely reuse files from an interrupted supplement."""
    source = regular_source(source)
    before = source.stat()
    source_sha = sha256(source)
    discard_owned_temporary(target)
    if old_item is not None and not allow_source_change:
        require(source_sha == old_item['source_sha256'], f'Immutable archived source changed: {source}')
    if target.exists():
        require(not target.is_symlink() and stat.S_ISREG(target.stat().st_mode), f'Unsafe target: {target}')
        require(not os.path.samefile(source, target), f'Independent inode required: {target}')
        current_sha = sha256(target)
        if current_sha != source_sha:
            require(old_item is not None and current_sha == old_item['target_sha256'], f'Unrecognized target bytes: {target}')
            if allow_source_change and source.relative_to(BATCH).parts[0] == 'logs':
                require(is_prefix(target, source), f'Controller log was rewritten instead of appended: {source}')
        else:
            return {**(old_item or {}), 'source': str(source), 'source_sha256': source_sha,
                    'target_sha256': source_sha, 'size_bytes': before.st_size,
                    'copy_method': 'shutil.copy2 (independent inode, SHA-256 verified)'}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_for(target)
    try:
        with temporary.open('xb'):
            pass
        shutil.copy2(source, temporary, follow_symlinks=False)
        require(sha256(temporary) == source_sha and not os.path.samefile(source, temporary), 'Copy verification failed')
        after = source.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                and sha256(source) == source_sha, f'Source changed while copying: {source}')
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {**(old_item or {}), 'source': str(source), 'source_sha256': source_sha,
            'target_sha256': sha256(target), 'size_bytes': before.st_size,
            'copy_method': 'shutil.copy2 (independent inode, SHA-256 verified)'}


def active_output_logs():
    paths = set()
    for descriptor in (1, 2):
        try:
            path = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
            if path.is_absolute() and path.is_relative_to(BATCH / 'logs'):
                paths.add(path)
        except OSError:
            pass
    return paths


def supplement_backup(record):
    discard_owned_temporary(BACKUP / 'backup_manifest.json')
    old = item_map(record)
    sources = {Path(item['source']) for item in old.values()}
    sources.update(files_in(BATCH / 'attempts'))
    sources.update(files_in(BATCH / 'logs'))
    sources.difference_update(active_output_logs())
    # Preserve control metadata as well as the raw controller/manager logs.
    sources.update(path for path in files_in(BATCH / 'state')
                   if path.suffix in {'.json', '.py'} and '__pycache__' not in path.parts)
    records = {}
    for source in sorted(sources):
        relative = source.relative_to(BATCH)
        path = str(relative)
        previous = old.get(path)
        require(previous is None or previous['source'] == str(source), f'Backup source path changed: {path}')
        mutable_log = relative.parts[0] == 'logs' and len(relative.parts) == 2
        item = sync_file(source, safe_path(BACKUP, relative), previous, allow_source_change=mutable_log)
        item.update(path=path, backup_path=str(BACKUP / relative))
        records[path] = item
    require(set(old) <= set(records), 'Supplement would drop a backed-up artifact')
    expected = set(records) | {'backup_manifest.json'}
    actual = {str(path.relative_to(BACKUP)) for path in files_in(BACKUP)}
    require(actual == expected, 'Backup contains unexpected/unmanifested files')
    record = {**record, 'files': [records[path] for path in sorted(records)],
              'scope': 'This batch: formal best/latest, result NPZ/CSV, terminal logs, all failed attempts and final control metadata',
              'postrun_supplement': {'terminal_logs': True, 'all_failed_attempts': True,
                                     'active_delivery_stdout_excluded': sorted(map(str, active_output_logs()))}}
    atomic_json(BACKUP / 'backup_manifest.json', record)
    atomic_json(BATCH / 'provenance/large_artifact_backup.json', record)
    return record


def check_public_file(path, relative):
    relative = Path(relative)
    require(not any(part.lower() in BANNED_DIRS or part.startswith('.') for part in relative.parts),
            f'Banned public directory: {relative}')
    lower = path.name.lower()
    require(path.suffix.lower() not in BANNED_SUFFIXES and '.log.' not in lower
            and path.stem.lower() not in {'stdout', 'stderr', 'raw_log', 'raw_logs', 'training_log'}
            and not any(token in lower for token in ('prediction', 'per_sample', 'per-sample')),
            f'Banned public artifact: {relative}')
    require(path.stat().st_size <= MAX_PUBLIC_BYTES, f'Public file too large: {relative}')
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as handle:
            header = next(csv.reader(handle), [])
        require(not ({column.strip().lower() for column in header} & SAMPLE_COLUMNS), f'Per-sample public CSV: {relative}')


def verify_public(record, source_checks=True):
    indexed = item_map(record)
    actual = {str(path.relative_to(PUBLIC)) for path in files_in(PUBLIC)}
    require(actual == set(indexed) | {'export_manifest.json'}, 'Public files do not match export_manifest.json')
    for relative, item in indexed.items():
        target = safe_path(PUBLIC, relative)
        check_public_file(target, relative)
        require(sha256(target) == item['target_sha256'], f'Public target SHA mismatch: {relative}')
        if source_checks:
            source = item['source']
            if '::generated README' in source:
                source = source.split('::', 1)[0]
            source = regular_source(source)
            require(sha256(source) == item['source_sha256'], f'Public source SHA mismatch: {relative}')
            if relative != 'README.md':
                require(not os.path.samefile(source, target), f'Public file is a hardlink: {relative}')


def registry_block():
    return (f'<!-- XP_MAINLINE_04_BATCH:{BATCH} -->\n'
            f'本批绝对目录：`{BATCH}`。\n\n'
            f'公开档案：`{PUBLIC}`；独立副本：`{BACKUP}`。\n\n'
            '验收证据：批次 `results/acceptance.json`（passed、20/20）和 '
            '`reports/XP_MAINLINE_ACCEPTANCE.md`；结果见 `reports/XP_MAINLINE_REPRODUCTION.md`。\n'
            '<!-- END_XP_MAINLINE_04_BATCH -->\n')


def update_registry(path):
    if path.exists():
        require(not path.is_symlink(), f'Symlink registry: {path}')
        text = path.read_text(encoding='utf-8')
    else:
        text = ''
    marker = f'<!-- XP_MAINLINE_04_BATCH:{BATCH} -->'
    require(text.count(marker) <= 1, f'Duplicate batch registry section: {path}')
    if marker in text:
        start = text.index(marker)
        end_marker = '<!-- END_XP_MAINLINE_04_BATCH -->'
        end = text.index(end_marker, start) + len(end_marker)
        new = text[:start] + registry_block().rstrip('\n') + text[end:]
    else:
        separator = '' if not text or text.endswith('\n\n') else '\n' if text.endswith('\n') else '\n\n'
        new = text + separator + registry_block()
    if new != text:
        atomic_bytes(path, new.encode())


def public_readme():
    return (f'# 04 XP 主研究基线\n\n'
            f'批次：`{BATCH}`。20个预注册任务均完成，完整验收passed；五seed配对结果见 '
            '`reports/XP_MAINLINE_REPRODUCTION.md`，逐项技术验收见 `results/acceptance.json`。\n\n'
            '唯一批准的正式调度入口是 `provenance/runtime_recovery_v2/xp3_scheduler.py` 对应的原批次 '
            '`state/recovery/xp3_scheduler.py`：在冻结调度器之上强制排除GPU0/1，只分配健康、空闲的XP3。'
            '完整续跑控制器为同目录 `continue_pipeline.py`，验收使用 `finalize_mainline_v2.py`（4锚点+16任务）。'
            '本批已完成，不应重跑或覆盖当前run；新实验另建批次。\n\n'
            '冻结 `tools/formal_pipeline.py` 和 `tools/finalize_mainline.py` 保留原初控制流程的历史证据，'
            '不再建议直接运行；资源竞争的失败attempt、原pipeline退出130和恢复审计均保留，未按分数选择重试。'
            '对应失败权重/日志位于批次与独立备份的 `attempts/`；公开包只含恢复元数据。\n\n'
            'CPU测试排除及原因见 `preflight/test_scope.json`：`test_multiseed_scheduler.py` 为V100专用规则；'
            '`test_caldiff_batching.py` 及个例 `test_causal_checkpoint_roundtrip_and_architecture_rejection` '
            '依赖快照未带的caldiff生成helper，均未被冒充为通过。原始失败见独立备份的 `logs/cpu_tests*.log`。'
            'XP工程回归入口为 `code/tests/test_xp_scheduler.py`；CPU测试不证明GPU数值或跨卡逐位等价。\n\n'
            '`curriculum_source=fixed_author`：共用作者外部difficulty。生成过程未公开且文件包含当前validation键，'
            '不能称为train-only curriculum，也不能证明完整预处理validation-free。测试集未用于选择epoch/seed/超参数。\n\n'
            f'大型产物的独立副本在 `{BACKUP}`，逐文件SHA256相同且inode独立；'
            '该目录仍在同一/data2文件系统，不是异盘或异地灾难恢复备份。'
            '备份清单：`provenance/large_artifact_backup.json`。\n\n'
            '公开包不含权重、逐样本预测、原始日志、数据/特征/难度文件、symlink或缓存。'
            '路径与哈希仅作来源引用，不等于上游特征生成独立性已经证明。'
            '01/02/03历史树逐项核对 `provenance/history_trees.json`，未覆盖或修改。'
            '每个公开文件的source/target SHA256见 `export_manifest.json`；清单自身不自引用。\n')


def public_sources(record):
    sources = {}
    for relative, item in item_map(record).items():
        safe_path(PUBLIC, relative)
        if relative != 'README.md':
            sources[relative] = regular_source(item['source'])
    for name in ('batch_manifest.json', 'BASELINE_REGISTRY.md', 'state/queue.json', 'state/pipeline.json',
                 'state/resource_guard.json', 'provenance/large_artifact_backup.json', 'provenance/resource_recovery.json'):
        sources[name] = regular_source(BATCH / name)
    for path in files_in(BATCH / 'state/recovery'):
        if path.suffix == '.json':
            sources[str(path.relative_to(BATCH))] = regular_source(path)
    for path in files_in(BATCH / 'provenance/runtime_recovery_v2'):
        if path.suffix in {'.py', '.json', '.md', '.diff'}:
            sources[str(path.relative_to(BATCH))] = regular_source(path)
    sources['provenance/postrun_delivery.py'] = regular_source(Path(__file__))
    for relative, source in sources.items():
        check_public_file(source, relative)
    return sources


def supplement_public(record, backup_record, history_counts):
    old = item_map(record)
    sources = public_sources(record)
    for relative in set(sources) | {'README.md', 'export_manifest.json'}:
        discard_owned_temporary(safe_path(PUBLIC, relative))
    allowed_new = set(sources) | {'README.md', 'export_manifest.json'}
    require({str(path.relative_to(PUBLIC)) for path in files_in(PUBLIC)} <= set(old) | allowed_new,
            'Unexpected public files; refusing to remove or absorb them')
    files = []
    for relative, source in sorted(sources.items()):
        item = sync_file(source, safe_path(PUBLIC, relative), old.get(relative), allow_source_change=True)
        item['path'] = relative
        files.append(item)
    readme_path = safe_path(PUBLIC, 'README.md')
    content = public_readme().encode()
    if readme_path.exists():
        known = old.get('README.md', {}).get('target_sha256')
        require(sha256(readme_path) in {known, hashlib.sha256(content).hexdigest()}, 'Unexpected public README mutation')
    if not readme_path.exists() or readme_path.read_bytes() != content:
        atomic_bytes(readme_path, content)
    files.append({'path': 'README.md', 'source': str(Path(__file__)) + '::generated README',
                  'source_sha256': sha256(Path(__file__)), 'target_sha256': sha256(readme_path),
                  'size_bytes': readme_path.stat().st_size, 'transformation': 'generated export documentation'})
    result = {**record, 'files': sorted(files, key=lambda item: item['path']),
              'backup_file_count': len(backup_record['files']),
              'postrun_supplement': {'terminal_control_metadata': True, 'historical_tree_files_verified': history_counts,
                                     'approved_scheduler': 'provenance/runtime_recovery_v2/xp3_scheduler.py',
                                     'acceptance_adapter': 'provenance/runtime_recovery_v2/finalize_mainline_v2.py'}}
    atomic_json(PUBLIC / 'export_manifest.json', result)
    verify_public(result)
    return result


def deliver(check_only=False):
    require(BATCH == EXPECTED_BATCH, 'This helper is bound to one designated batch')
    require(BACKUP == Path('/data2/yb/paper/experiment_backups/CMERC/04_xp_mainline_validation') / BATCH.name
            and PUBLIC == CMERC / 'experiment_records/04_xp_mainline_validation', 'Unexpected delivery target')
    for root in (BATCH, BACKUP, PUBLIC):
        require(not any(path.is_symlink() for path in (root, *root.parents)), f'Symlink target ancestry: {root}')
    queue, pipeline, acceptance = terminal_gate()
    histories = verify_history()
    acceptance_sha = sha256(BATCH / 'results/acceptance.json')
    backup_record, public_record = associated_manifests(acceptance_sha)
    if check_only:
        # Initial export records can legitimately precede final controller log
        # writes. The mutating path verifies old bytes before each atomic update.
        for relative in item_map(public_record):
            check_public_file(safe_path(PUBLIC, relative), relative)
        return {'status': 'ready', 'batch': str(BATCH), 'historical_files': histories,
                'runs': len(queue['jobs']), 'public_sources': len(public_sources(public_record))}
    lock_path = safe_path(BATCH, 'state/recovery/postrun_delivery.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal_gate()
        update_registry(safe_path(BATCH, 'BASELINE_REGISTRY.md'))
        require(not (CMERC / 'BASELINE_REGISTRY.md').is_symlink(), 'Root registry is a symlink')
        update_registry(CMERC / 'BASELINE_REGISTRY.md')
        backup_record = supplement_backup(backup_record)
        public_record = supplement_public(public_record, backup_record, histories)
        require(sha256(BATCH / 'results/acceptance.json') == acceptance_sha, 'Acceptance changed during supplement')
        require(verify_history() == histories, 'Historical trees changed during delivery')
        return {'status': 'supplemented', 'batch': str(BATCH), 'backup': str(BACKUP), 'public': str(PUBLIC),
                'backup_files': len(backup_record['files']), 'public_files': len(public_record['files']),
                'backup_manifest_sha256': sha256(BACKUP / 'backup_manifest.json'),
                'export_manifest_sha256': sha256(PUBLIC / 'export_manifest.json')}


def self_test():
    """Exercise copy/reentry/filter/history guards only in disposable fixtures."""
    import unittest
    from unittest.mock import patch

    class Checks(unittest.TestCase):
        def setUp(self):
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
            self.batch = self.root / 'batch'
            self.batch.mkdir()
            replacement = patch.dict(globals(), BATCH=self.batch)
            replacement.start()
            self.addCleanup(replacement.stop)

        def test_independent_copy_reentry_append_and_bad_target(self):
            source = self.batch / 'logs/controller.log'
            source.parent.mkdir()
            source.write_text('started\n')
            target = self.root / 'backup/logs/controller.log'
            first = sync_file(source, target)
            self.assertFalse(os.path.samefile(source, target))
            self.assertEqual(sync_file(source, target, first)['target_sha256'], first['target_sha256'])
            source.write_text('started\ncompleted\n')
            updated = sync_file(source, target, first, allow_source_change=True)
            self.assertEqual(updated['source_sha256'], sha256(target))
            self.assertEqual(sync_file(source, target, first, allow_source_change=True)['target_sha256'], updated['target_sha256'])
            target.write_text('unexpected bytes')
            with self.assertRaises(RuntimeError):
                sync_file(source, target, updated, allow_source_change=True)

        def test_hardlink_and_symlink_rejected(self):
            source = self.batch / 'plain.json'
            source.write_text('{}')
            linked = self.root / 'hardlink.json'
            os.link(source, linked)
            with self.assertRaisesRegex(RuntimeError, 'inode'):
                sync_file(source, linked)
            symlink = self.batch / 'symlink.json'
            symlink.symlink_to(source)
            with self.assertRaisesRegex(RuntimeError, 'Symlink'):
                regular_source(symlink)
            with self.assertRaises(RuntimeError):
                safe_path(self.batch, '../outside')

        def test_public_binary_raw_sample_and_cache_rejected(self):
            for name in ('best.pt', 'predictions.npz', 'run.log', 'run.log.memory.jsonl', '__pycache__/file.py'):
                path = self.batch / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('x')
                with self.assertRaises(RuntimeError):
                    check_public_file(path, name)
            sample = self.batch / 'sample.csv'
            sample.write_text('dialogue_id,weighted_f1\na,1\n')
            with self.assertRaises(RuntimeError):
                check_public_file(sample, sample.name)

        def test_registry_preserves_other_batches_and_is_idempotent(self):
            path = self.batch / 'registry.md'
            path.write_text('Other batch stays here\n')
            update_registry(path)
            first = path.read_text()
            update_registry(path)
            self.assertEqual(path.read_text(), first)
            self.assertTrue(first.startswith('Other batch stays here\n'))

        def test_manifest_duplicate_paths_rejected(self):
            with self.assertRaises(RuntimeError):
                item_map({'files': [{'path': 'x'}, {'path': 'x'}]})

        def test_backup_supplements_attempts_final_logs_and_reenters(self):
            backup = self.root / 'backup'
            backup.mkdir()
            (self.batch / 'attempts/failed_run').mkdir(parents=True)
            (self.batch / 'attempts/failed_run/latest.pt').write_bytes(b'failed checkpoint kept')
            (self.batch / 'logs').mkdir()
            log = self.batch / 'logs/controller.log'
            log.write_text('started\n')
            (self.batch / 'state').mkdir()
            atomic_json(self.batch / 'state/pipeline.json', {'status': 'completed'})
            with patch.dict(globals(), BACKUP=backup):
                old = sync_file(log, backup / 'logs/controller.log')
                old.update(path='logs/controller.log', backup_path=str(backup / 'logs/controller.log'))
                record = {'schema_version': 1, 'batch': str(self.batch), 'files': [old]}
                atomic_json(backup / 'backup_manifest.json', record)
                log.write_text('started\ncompleted\n')
                # Simulate interruption leaving a partial copy temporary.
                target = backup / 'attempts/failed_run/latest.pt'
                target.parent.mkdir(parents=True)
                temporary_for(target).write_bytes(b'incomplete copy')
                first = supplement_backup(record)
                self.assertEqual(first, supplement_backup(first))
                self.assertEqual(read_json(self.batch / 'provenance/large_artifact_backup.json'), first)
                self.assertEqual(target.read_bytes(), b'failed checkpoint kept')
                for item in first['files']:
                    self.assertEqual(sha256(Path(item['source'])), sha256(backup / item['path']))
                    self.assertFalse(os.path.samefile(item['source'], backup / item['path']))

        def test_public_supplement_preserves_schema_and_reenters_without_raw_artifacts(self):
            public = self.root / 'public'
            public.mkdir()
            script = self.batch / 'state/recovery/postrun_delivery.py'
            script.parent.mkdir(parents=True)
            script.write_text('# fixture source\n')
            for relative in ('batch_manifest.json', 'state/queue.json', 'state/pipeline.json',
                             'state/resource_guard.json', 'state/recovery/pipeline_v1.json',
                             'state/recovery/manager.json', 'provenance/large_artifact_backup.json',
                             'provenance/resource_recovery.json', 'provenance/runtime_recovery_v2/manifest.json'):
                atomic_json(self.batch / relative, {'fixture': True})
            (self.batch / 'BASELINE_REGISTRY.md').write_text('Accepted fixture\n')
            (public / 'README.md').write_text('old README\n')
            record = {'schema_version': 1, 'status': 'exported', 'batch': str(self.batch),
                      'source_root': str(self.batch), 'backup_root': str(self.root / 'backup'),
                      'files': [{'path': 'README.md', 'source': str(script) + '::generated README',
                                 'source_sha256': sha256(script), 'target_sha256': sha256(public / 'README.md')}]}
            atomic_json(public / 'export_manifest.json', record)
            with patch.dict(globals(), PUBLIC=public, __file__=str(script)):
                first = supplement_public(record, {'files': [1, 2]}, {'history': 3})
                self.assertEqual(first, supplement_public(first, {'files': [1, 2]}, {'history': 3}))
                verify_public(first)
                self.assertEqual(first['schema_version'], 1)
                self.assertIn('state/recovery/pipeline_v1.json', item_map(first))
                (public / 'unexpected.pt').write_bytes(b'forbidden')
                with self.assertRaises(RuntimeError):
                    verify_public(first)

        def test_history_tree_hashes_detect_changed_or_extra_files(self):
            cm = self.root / 'cmerc'
            names = ('01_v100_single_seed', '02_xp_multiseed', '03_v100_multiseed')
            recorded = {}
            for name in names:
                path = cm / 'experiment_records' / name / 'README.md'
                path.parent.mkdir(parents=True)
                path.write_text(name)
                recorded[name] = {'README.md': sha256(path)}
            atomic_json(self.batch / 'provenance/history_trees.json', recorded)
            with patch.dict(globals(), CMERC=cm):
                self.assertEqual(verify_history(), {name: 1 for name in names})
                (cm / 'experiment_records' / names[0] / 'extra.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError, 'Historical archive changed'):
                    verify_history()

        def test_terminal_gate_does_not_write_on_incomplete_queue(self):
            (self.batch / 'state').mkdir()
            (self.batch / 'results').mkdir()
            atomic_json(self.batch / 'state/queue.json', {'phase': 'additional_seeds', 'jobs': []})
            atomic_json(self.batch / 'state/pipeline.json', {'status': 'running'})
            atomic_json(self.batch / 'results/acceptance.json', {'status': 'not_ready'})
            before = sorted(str(path) for path in self.batch.rglob('*'))
            with self.assertRaisesRegex(RuntimeError, '20 registered'):
                terminal_gate()
            self.assertEqual(before, sorted(str(path) for path in self.batch.rglob('*')))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='read-only terminal readiness checks')
    parser.add_argument('--self-test', action='store_true', help='temporary-fixture CPU self-tests only')
    options = parser.parse_args()
    if options.self_test:
        return self_test()
    print(json.dumps(deliver(options.check_only), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
