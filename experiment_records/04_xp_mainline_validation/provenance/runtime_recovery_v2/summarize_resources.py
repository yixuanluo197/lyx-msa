#!/usr/bin/env python3
"""CPU-only terminal resource accounting; this is not scientific acceptance.

Writes only provenance/resource_outcome.json and reports/XP_RESOURCE_OUTCOME.md
after the registered queue, continuation pipeline and resource observer finish.
--self-test exercises synthetic fixtures without touching batch outputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

BATCH = Path(__file__).resolve().parents[2]
EXPECTED_BATCH = Path('/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z')
DATASETS = ('IEMOCAP', 'MELD')
VARIANTS = ('baseline', 'cmerc')
SEEDS = (1475, 67137, 2024, 3407, 42)
PHASES = ('training', 'evaluation')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f'Invalid {name}: {value!r}')
    return float(value)


def timestamp(value):
    if type(value) in (int, float):
        return finite(value, 'Unix timestamp')
    require(isinstance(value, str), f'Invalid timestamp: {value!r}')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timestamp must include a timezone')
    return parsed.timestamp()


def iso(value):
    return datetime.fromtimestamp(timestamp(value), timezone.utc).isoformat()


def elapsed(start, end):
    result = timestamp(end) - timestamp(start)
    require(result >= 0, 'End precedes start')
    return result


def describe(values):
    values = [finite(value, 'duration') for value in values]
    require(values, 'Empty duration group')
    return {'count': len(values), 'total_seconds': math.fsum(values),
            'mean_seconds': statistics.mean(values), 'minimum_seconds': min(values),
            'maximum_seconds': max(values)}


def read_source(path, sources):
    path = Path(path).absolute()
    require(path.is_relative_to(BATCH) and path.resolve().is_relative_to(BATCH.resolve())
            and not path.is_symlink(), f'Unapproved source: {path}')
    current = BATCH
    for part in path.relative_to(BATCH).parts:
        current = current / part
        require(not current.is_symlink(), f'Symlink resource source: {current}')
    content = path.read_bytes()
    sources[str(path.relative_to(BATCH))] = hashlib.sha256(content).hexdigest()
    return content.decode('utf-8')


def read_json(path, sources):
    return json.loads(read_source(path, sources))


def read_jsonl(path, sources):
    return [json.loads(line) for line in read_source(path, sources).splitlines() if line.strip()]


def profile_summary(rows, job, phase):
    require(rows, f'Empty memory profile: {job["id"]}/{phase}')
    expected_pid = job[f'{phase}_pid']
    expected_uuid = job[f'{phase}_gpu_uuid']
    for row in rows:
        require(row.get('pid') == expected_pid and row.get('visible_device') == expected_uuid,
                f'Memory profile PID/UUID mismatch: {job["id"]}/{phase}')
        finite(row.get('peak_allocated_mib'), 'allocated MiB')
        finite(row.get('peak_reserved_mib'), 'reserved MiB')
        require(row['peak_reserved_mib'] >= row['peak_allocated_mib'], 'Reserved memory below allocated memory')
    return {'samples': len(rows), 'pid': expected_pid, 'gpu_uuid': expected_uuid,
            'peak_allocated_mib': max(row['peak_allocated_mib'] for row in rows),
            'peak_reserved_mib': max(row['peak_reserved_mib'] for row in rows)}


def observer_summary(rows, state, gpu_uuid, formal_start, formal_end):
    require(state.get('status') == 'completed', 'Resource observer has not completed')
    require(state.get('interval_seconds') == 5, 'Expected registered 5-second observer interval')
    require(rows and state.get('samples') == len(rows), 'Observer state/sample count mismatch')
    times = [timestamp(row['timestamp']) for row in rows]
    require(times == sorted(times), 'Observer timestamps are not chronological')
    sampled, errors, unexpected = [], [], []
    for number, row in enumerate(rows, 1):
        if row.get('sampling_error'):
            errors.append({'sample': number, 'timestamp': row['timestamp'], 'error': row['sampling_error']})
        if row.get('unexpected_pids'):
            unexpected.append({'sample': number, 'timestamp': row['timestamp'], 'pids': row['unexpected_pids']})
        if 'gpu_csv' not in row:
            continue
        try:
            fields = next(csv.reader(io.StringIO(row['gpu_csv']), skipinitialspace=True))
            require(len(fields) == 6, 'Malformed observer GPU CSV')
            index, uuid, total, free, used, utilization = [value.strip() for value in fields]
            require(int(index) == 3 and uuid == gpu_uuid, 'Observer covered a different GPU')
            item = {'total_mib': finite(float(total), 'whole-card total MiB'),
                    'free_mib': finite(float(free), 'whole-card free MiB'),
                    'used_mib': finite(float(used), 'whole-card used MiB'),
                    'utilization_percent': finite(float(utilization), 'utilization percent'),
                    'timestamp': row['timestamp']}
            require(item['free_mib'] <= item['total_mib'] and item['used_mib'] <= item['total_mib']
                    and item['utilization_percent'] <= 100, 'Invalid whole-card sample range')
            sampled.append(item)
        except (ValueError, RuntimeError, StopIteration) as error:
            errors.append({'sample': number, 'timestamp': row['timestamp'], 'parse_error': str(error)})
    require(sampled, 'Observer has no usable whole-card memory samples')
    reported_unexpected = state.get('unexpected_pid_samples')
    require(reported_unexpected == len(unexpected), 'Unexpected-PID sample count mismatch')
    overlap = max(0.0, min(times[-1], timestamp(formal_end)) - max(times[0], timestamp(formal_start)))
    gaps = [right - left for left, right in zip(times, times[1:])]
    return {'status': 'completed', 'interval_seconds_requested': 5,
            'coverage_start_utc': iso(times[0]), 'coverage_end_utc': iso(times[-1]),
            'observer_ended_at_utc': state.get('ended_at'),
            'coverage_window_seconds': times[-1] - times[0],
            'formal_wall_window_overlap_seconds': overlap,
            'formal_wall_window_overlap_fraction': overlap / elapsed(formal_start, formal_end),
            'recorded_samples': len(rows), 'usable_gpu_samples': len(sampled),
            'sampling_error_count': sum(bool(row.get('sampling_error')) for row in rows),
            'gpu_csv_parse_error_count': sum('parse_error' in error for error in errors),
            'errors': errors, 'unexpected_pid_samples': len(unexpected),
            'unexpected_pid_details': unexpected,
            'actual_sample_gap_seconds': describe(gaps) if gaps else None,
            'whole_card_peak_used_mib': max(item['used_mib'] for item in sampled),
            'whole_card_minimum_free_mib': min(item['free_mib'] for item in sampled),
            'whole_card_maximum_free_mib': max(item['free_mib'] for item in sampled),
            'maximum_sampled_utilization_percent': max(item['utilization_percent'] for item in sampled),
            'scope': 'XP3 only, observer-start onward; discrete samples over a partial batch interval',
            'exclusivity_limitation': 'No unexpected PID in sampled records does not prove exclusive CUDA contexts for the entire run; processes can start/end between samples or outside this coverage.'}


def build_outcome():
    require(BATCH == EXPECTED_BATCH, 'This helper is bound to the designated XP batch')
    sources = {}
    queue = read_json(BATCH / 'state/queue.json', sources)
    pipeline = read_json(BATCH / 'state/pipeline.json', sources)
    acceptance = read_json(BATCH / 'results/acceptance.json', sources)
    observer_state = read_json(BATCH / 'state/recovery/resource_observer.json', sources)
    jobs = queue.get('jobs', [])
    expected = {(dataset, variant, seed) for dataset in DATASETS for variant in VARIANTS for seed in SEEDS}
    require(queue.get('phase') == 'completed' and len(jobs) == 20
            and all(job.get('status') == 'completed' for job in jobs)
            and {(job['dataset'], job['variant'], job['seed']) for job in jobs} == expected,
            'Resource report requires the complete registered 20-run matrix')
    require(pipeline.get('status') == 'completed' and pipeline.get('phases', {}).get('formal', {}).get('exit_code') == 0,
            'Continuation pipeline/formal phase has not completed')
    require(acceptance.get('status') == 'passed' and acceptance.get('runs_completed') == 20
            and acceptance.get('batch') == str(BATCH), 'Complete scientific acceptance has not passed')
    require(observer_state.get('status') == 'completed', 'Resource observer has not completed')
    require(acceptance.get('queue_sha256') == sources['state/queue.json'], 'Queue differs from accepted terminal queue')
    old = read_json(BATCH / 'state/recovery/pipeline_v1.json', sources)
    require(old.get('phases', {}).get('formal', {}).get('exit_code') == 130, 'Missing original controller exit 130')
    # The original controller started directly with formal training, after
    # preflight; follow its start through recovery to the new formal finish.
    formal_start = old['started_at']
    formal_end = pipeline['phases']['formal']['finished_at']
    require(elapsed(formal_start, formal_end) > 0, 'Empty formal batch wall-clock interval')
    runs, epoch_groups = [], {(dataset, variant): [] for dataset in DATASETS for variant in VARIANTS}
    starts, evaluations = set(), set()
    for job in jobs:
        require(job.get('training_exit_code') == 0 and job.get('evaluation_exit_code') == 0,
                f'Incomplete successful phases: {job["id"]}')
        metrics_path = Path(job['run_dir']) / 'metrics.jsonl'
        epochs = read_jsonl(metrics_path, sources)
        budget = 80 if job['dataset'] == 'IEMOCAP' else 40
        require([row.get('epoch') for row in epochs] == list(range(1, budget + 1)),
                f'Incomplete/duplicate epoch records: {job["id"]}')
        epoch_times = [finite(row.get('elapsed_seconds'), 'epoch elapsed_seconds') for row in epochs]
        epoch_groups[job['dataset'], job['variant']].extend(epoch_times)
        run = {'id': job['id'], 'dataset': job['dataset'], 'variant': job['variant'], 'seed': job['seed'],
               'epoch_time': describe(epoch_times), 'phases': {}}
        for phase, log_name, identities in (('training', 'train_log', starts), ('evaluation', 'evaluation_log', evaluations)):
            pid = job[f'{phase}_pid']
            require(type(pid) is int and pid > 0, 'Invalid phase PID')
            identity = (pid, job[f'{phase}_started_at'])
            require(identity not in identities, 'Duplicate formal phase start')
            identities.add(identity)
            path = Path(str(job[log_name]) + '.memory.jsonl')
            memory = profile_summary(read_jsonl(path, sources), job, phase)
            run['phases'][phase] = {'started_at_utc': iso(job[f'{phase}_started_at']),
                                    'ended_at_utc': iso(job[f'{phase}_ended_at']),
                                    'wall_seconds': elapsed(job[f'{phase}_started_at'], job[f'{phase}_ended_at']),
                                    'memory_profile': str(path.relative_to(BATCH)), **memory}
        runs.append(run)
    require(sum(len(values) for values in epoch_groups.values()) == 1200, 'Expected 1200 valid formal epochs')
    require(len(starts) == len(evaluations) == 20, 'Expected 20 valid training and 20 independent evaluation starts')
    attempt_files = sorted((BATCH / 'attempts').glob('**/failed_task.json'))
    require(len(attempt_files) == 1, 'Expected exactly one archived invalid partial attempt')
    partial = read_json(attempt_files[0], sources)
    require(partial.get('id') == 'MELD_cmerc_validation_seed3407' and partial.get('training_exit_code') == -15,
            'Unexpected invalid partial task or exit code')
    require((partial['training_pid'], partial['training_started_at']) not in starts, 'Invalid attempt counted as a valid training start')
    assessment = read_json(attempt_files[0].parent / 'retry_assessment.json', sources)
    require(assessment.get('score_used_to_decide_retry') is False
            and assessment.get('model_or_scientific_protocol_changes') is False, 'Unexpected retry rationale')
    original_eval = read_json(BATCH / 'state/recovery/iemocap2024_original_resource_failure.json', sources)
    require(original_eval.get('training_exit_code') == 0 and not original_eval.get('evaluation_pid')
            and not original_eval.get('evaluation_started_at'), 'Original IEMOCAP2024 evaluation was unexpectedly started')
    observer_path = BATCH / 'logs/xp3_resource_observer.jsonl'
    observer = observer_summary(read_jsonl(observer_path, sources), observer_state,
                                queue['config']['allowed_gpu_uuids']['3'], formal_start, formal_end)
    groups = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            matching = [run for run in runs if run['dataset'] == dataset and run['variant'] == variant]
            require(len(matching) == 5, 'Resource group must contain all five seeds')
            group = {'dataset': dataset, 'variant': variant, 'runs': 5,
                     'epoch_time': describe(epoch_groups[dataset, variant]), 'phases': {}}
            for phase in PHASES:
                phase_rows = [run['phases'][phase] for run in matching]
                group['phases'][phase] = {'process_wall_time': describe(row['wall_seconds'] for row in phase_rows),
                                          'peak_allocated_mib': max(row['peak_allocated_mib'] for row in phase_rows),
                                          'peak_reserved_mib': max(row['peak_reserved_mib'] for row in phase_rows)}
            groups.append(group)
    sources[str(Path(__file__).relative_to(BATCH))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {'schema_version': 1, 'status': 'completed', 'batch': str(BATCH),
            'purpose': 'Terminal resource accounting only; this does not rerun or replace scientific acceptance',
            'scientific_source_snapshot_sha256_unchanged': acceptance['source_snapshot_sha256'],
            'units': {'duration': 'seconds', 'memory': 'MiB (2**20 bytes)', 'utilization': 'percent'},
            'counts': {'registered_valid_runs': 20, 'valid_epoch_records': 1200, 'valid_training_starts': 20,
                       'archived_invalid_partial_training_starts': 1, 'total_formal_training_starts': 21,
                       'independent_evaluation_starts': 20, 'memory_profile_files': 40,
                       'original_iemocap2024_blocked_evaluation_starts': 0},
            'formal_batch_wall': {'start_utc': iso(formal_start), 'end_utc': iso(formal_end),
                                  'total_seconds': elapsed(formal_start, formal_end),
                                  'start_basis': 'original pipeline started_at (after preflight)',
                                  'end_basis': 'continuation pipeline formal phase finished_at',
                                  'includes': 'initialization, IO, queue/resource waits, drain/recovery and controller polling; excludes preflight and final acceptance/export'},
            'valid_epoch_time': describe(value for values in epoch_groups.values() for value in values),
            'valid_phase_wall_time': {phase: describe(run['phases'][phase]['wall_seconds'] for run in runs) for phase in PHASES},
            'duration_limitations': ['Epoch elapsed_seconds is the rounded training JSONL timing, including loader/train/validation work inside its timer; checkpoint saving and final standalone evaluation are outside that epoch timer.',
                                     'Phase wall times include process initialization, IO, and scheduler poll-tail delay; summed concurrent process wall times are not batch wall time or GPU-kernel hours.',
                                     'Allocated/reserved peaks are process allocator measurements; whole-card used/free includes contexts and other allocations and is a separate discrete observation.'],
            'recovery_accounting': {'old_pipeline_formal_exit_code': 130, 'old_pipeline_status': old['status'],
                                     'invalid_partial_task': partial['id'], 'invalid_partial_exit_code': -15,
                                     'invalid_partial_training_started_at': partial['training_started_at'],
                                     'invalid_partial_training_ended_at': partial['training_ended_at'],
                                     'invalid_partial_process_wall_seconds': elapsed(partial['training_started_at'], partial['training_ended_at']),
                                     'invalid_partial_excluded_from_valid_epoch_and_phase_totals': True,
                                     'retry_basis': assessment['checkpoint_assessment'], 'score_used_to_decide_retry': False,
                                     'iemocap2024_recovery': 'Existing valid 80-epoch training retained; its fixed validation-selected best was independently evaluated once on exclusive XP3.'},
            'groups': groups, 'runs': runs, 'observer': observer,
            'source_files_sha256': dict(sorted(sources.items()))}


def report_text(payload):
    wall, observer = payload['formal_batch_wall'], payload['observer']
    lines = ['# XP 主线资源终态汇总', '',
             '本报告只汇总已验收批次的资源证据，不再次进行科学验收，不改变训练、配置、队列或登记。', '',
             f"科学源码摘要沿用验收：`{payload['scientific_source_snapshot_sha256_unchanged']}`。", '',
             f"20个有效run共1200条epoch记录、20次有效训练、20次独立评估、40份进程显存侧车。"
             '另保留1次因GPU竞争失效的MELD3407 partial，因此正式训练实际共21次启动。', '',
             f"正式批次墙钟：{wall['start_utc']} → {wall['end_utc']}，{wall['total_seconds']:.2f} 秒"
             f"（{wall['total_seconds']/3600:.4f} 小时）。起点为旧pipeline开始，终点为新formal完成；含资源等待、drain和恢复，排除预检与最终验收/导出。", '',
             '下表elapsed_seconds来自训练JSONL的epoch计时，包含计时范围内的数据加载、训练和validation；'
             'checkpoint保存及最终独立评估不在该epoch计时内。', '',
             '| Dataset / Variant | epoch总数 | 总秒数 | 每epoch均值（秒） | 最短 / 最长（秒） |',
             '| --- | ---: | ---: | ---: | ---: |']
    for group in payload['groups']:
        item = group['epoch_time']
        lines.append(f"| {group['dataset']} / {group['variant']} | {item['count']} | {item['total_seconds']:.2f} | "
                     f"{item['mean_seconds']:.4f} | {item['minimum_seconds']:.2f} / {item['maximum_seconds']:.2f} |")
    lines += ['', '阶段起止wall time含模型初始化、IO及调度器poll尾延迟。'
              '下表相加的是进程墙钟；并发进程会重叠，不能称为GPU kernel hours，也不能替代整批墙钟。', '',
              '| Dataset / Variant / Phase | 进程墙钟总秒数 | 每run均值（秒） | allocated峰值 MiB | reserved峰值 MiB |',
              '| --- | ---: | ---: | ---: | ---: |']
    for group in payload['groups']:
        for phase in PHASES:
            item = group['phases'][phase]
            lines.append(f"| {group['dataset']} / {group['variant']} / {phase} | {item['process_wall_time']['total_seconds']:.2f} | "
                         f"{item['process_wall_time']['mean_seconds']:.2f} | {item['peak_allocated_mib']:.2f} | {item['peak_reserved_mib']:.2f} |")
    lines += ['', '原pipeline退出130是drain后有意停止旧控制器。归档MELD3407旧attempt退出-15，'
              '因同卡竞争不纳入有效epoch/阶段合计，保留原权重与日志，原seed/配置从头重试。'
              'IEMOCAP2024原训练成功，第一次独立评估在启动前被资源门阻止，恢复时只评估同一best一次。', '',
              f"整卡离散观测仅覆盖XP3：{observer['coverage_start_utc']} → {observer['coverage_end_utc']}；"
              f"请求间隔5秒，实际样本间隔见JSON，共{observer['recorded_samples']}条、有效GPU记录{observer['usable_gpu_samples']}条。",
              f"采样错误{observer['sampling_error_count']}条，CSV解析错误{observer['gpu_csv_parse_error_count']}条，"
              f"unexpected PID样本{observer['unexpected_pid_samples']}条；异常明细保留在JSON。",
              f"采样整卡used峰值 {observer['whole_card_peak_used_mib']:.0f} MiB，free最小 {observer['whole_card_minimum_free_mib']:.0f} MiB，"
              f"free最大 {observer['whole_card_maximum_free_mib']:.0f} MiB。",
              '此observer在正式批次后段才启动，且是离散采样；即使unexpected PID为0，也不证明未覆盖时段或两次采样之间全程CUDA context独占。'
              '整卡used/free与进程allocated/reserved含义不同，不能互相替代。', '',
              '逐run阶段起止、40份显存峰值、时间/单位定义及全部来源SHA256见 `../provenance/resource_outcome.json`。', '']
    return '\n'.join(lines)


def atomic_text(path, content):
    path = Path(path)
    require(path in {BATCH / 'provenance/resource_outcome.json', BATCH / 'reports/XP_RESOURCE_OUTCOME.md'},
            'Unexpected resource output path')
    require(not path.is_symlink(), 'Resource output is a symlink')
    current = BATCH
    for part in path.relative_to(BATCH).parts:
        current = current / part
        require(not current.is_symlink(), f'Symlink resource output ancestry: {current}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.resource-outcome-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def self_test():
    import unittest

    class Checks(unittest.TestCase):
        def test_iso_and_unix_duration_units(self):
            self.assertEqual(elapsed('2026-10-16T00:00:00+00:00', '2026-10-16T00:01:30Z'), 90)
            self.assertEqual(elapsed(1000, 1090), 90)
            with self.assertRaises(RuntimeError):
                elapsed(1090, 1000)

        def test_group_statistics_are_seconds_not_hours(self):
            result = describe([1, 2, 3, 4])
            self.assertEqual(result, {'count': 4, 'total_seconds': 10, 'mean_seconds': 2.5,
                                      'minimum_seconds': 1.0, 'maximum_seconds': 4.0})
            for invalid in (float('nan'), float('inf'), -1, True):
                with self.assertRaises(RuntimeError):
                    describe([invalid])

        def test_memory_peaks_keep_mib_and_phase_identity(self):
            job = {'id': 'fixture', 'training_pid': 12, 'training_gpu_uuid': 'GPU-test'}
            rows = [{'pid': 12, 'visible_device': 'GPU-test', 'peak_allocated_mib': 1024,
                     'peak_reserved_mib': 1536},
                    {'pid': 12, 'visible_device': 'GPU-test', 'peak_allocated_mib': 1100,
                     'peak_reserved_mib': 1500}]
            result = profile_summary(rows, job, 'training')
            self.assertEqual(result['peak_allocated_mib'], 1100)
            self.assertEqual(result['peak_reserved_mib'], 1536)
            with self.assertRaises(RuntimeError):
                profile_summary([{**rows[0], 'pid': 13}], job, 'training')

        def test_observer_partial_coverage_errors_and_unexpected_samples(self):
            rows = [{'timestamp': '2026-10-16T00:00:10Z', 'gpu_csv': '3, GPU-test, 12288, 1000, 11184, 80', 'unexpected_pids': [77]},
                    {'timestamp': '2026-10-16T00:00:15Z', 'sampling_error': 'query timeout'},
                    {'timestamp': '2026-10-16T00:00:20Z', 'gpu_csv': '3, GPU-test, 12288, 2000, 10184, 50', 'unexpected_pids': []}]
            state = {'status': 'completed', 'interval_seconds': 5, 'samples': 3, 'unexpected_pid_samples': 1}
            result = observer_summary(rows, state, 'GPU-test', '2026-10-16T00:00:00Z', '2026-10-16T00:00:30Z')
            self.assertEqual(result['whole_card_peak_used_mib'], 11184)
            self.assertEqual(result['whole_card_minimum_free_mib'], 1000)
            self.assertEqual(result['sampling_error_count'], 1)
            self.assertEqual(result['unexpected_pid_samples'], 1)
            self.assertAlmostEqual(result['formal_wall_window_overlap_fraction'], 1/3)
            self.assertEqual(result['actual_sample_gap_seconds']['mean_seconds'], 5)
            with self.assertRaises(RuntimeError):
                observer_summary(rows, {**state, 'status': 'running'}, 'GPU-test', 1, 2)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='pure synthetic-fixture CPU checks only')
    parser.add_argument('--check-only', action='store_true', help='compute terminal resource summary without writing reports')
    options = parser.parse_args()
    if options.self_test:
        return self_test()
    payload = build_outcome()
    # Protect the report from changing source files, including the completed
    # observer's final flush, between collection and atomic publication.
    for relative, digest in payload['source_files_sha256'].items():
        require(hashlib.sha256((BATCH / relative).read_bytes()).hexdigest() == digest,
                f'Resource source changed during accounting: {relative}')
    if not options.check_only:
        atomic_text(BATCH / 'provenance/resource_outcome.json',
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
        atomic_text(BATCH / 'reports/XP_RESOURCE_OUTCOME.md', report_text(payload))
    print(json.dumps({'status': 'completed', 'check_only': options.check_only,
                      'valid_runs': 20, 'valid_epochs': 1200,
                      'formal_wall_seconds': payload['formal_batch_wall']['total_seconds']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
