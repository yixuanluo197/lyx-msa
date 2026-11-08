#!/usr/bin/env python
"""Independently verify saved Validation evidence-sensitivity observations.

No model, runner, evidence_core, or analyzer is imported. Test prediction files
are never deserialized; their opaque bytes are used only for SHA256 integrity.
The audit's independently recomputed training statistics are checked by SHA;
this verifier independently recomputes every per-repeat CSV observation, every
summary/class point estimate, and every saved paired dialogue bootstrap CI.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from scipy.special import rel_entr
from sklearn.metrics import accuracy_score, f1_score

NAMES = ('flip_rate', 'true_probability_change', 'confidence_change', 'js_divergence',
         'correct_to_wrong_rate', 'wrong_to_correct_rate')
META = ('labels', 'dialogue_ids', 'utterance_indices', 'speaker_ids', 'batch_indices',
        'dialogue_lengths')
GROUPS = ('all', 'original_correct', 'original_wrong', 'visual_present', 'visual_missing')
FIGURES = ('modality_acc_curves', 'trueprob_curves', 'text_layer_comparison',
           'flip_curves', 'class_heatmap', 'meld_visual_missingness')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_npz(path):
    path = Path(path)
    require('test' not in path.name.lower(), 'Forbidden Test prediction read: ' + str(path))
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def same_value(observed, expected, context):
    if isinstance(expected, (bool, np.bool_)):
        require(str(observed) == str(bool(expected)), context)
    elif isinstance(expected, (int, float, np.number)):
        actual = float(observed)
        target = float(expected)
        require((math.isnan(target) and math.isnan(actual)) or
                math.isclose(actual, target, abs_tol=2e-11, rel_tol=2e-10),
                f'{context}: observed={actual}, independently computed={target}')
    else:
        require(str(observed) == str(expected), f'{context}: {observed!r} != {expected!r}')


def same_row(observed, expected, context):
    require(set(observed) == set(expected), context + ' columns disagree')
    for key, value in expected.items():
        same_value(observed[key], value, context + '/' + key)


def specifications(config):
    expected_phases = {
        'overall_attenuation': ['T-All', 'A', 'V'],
        'layer_attenuation': ['T1', 'T2', 'T3', 'T4'],
        'random_masking': ['T1', 'T2', 'T3', 'T4', 'T-All', 'A', 'V']}
    require(config['phases'] == expected_phases, 'Unexpected phases/targets')
    require(list(config['phases']) == list(expected_phases), 'Phase order changed')
    require(config['attenuation']['rho'] == [.25, .5, .75, 1.] and
            config['attenuation']['repeats'] == 1, 'Attenuation configuration changed')
    require(config['random_masking']['ratios'] == [.1, .3, .5, .8] and
            config['random_masking']['repeats'] == 5, 'Masking configuration changed')
    result = []
    for phase, targets in expected_phases.items():
        mask = phase == 'random_masking'
        for target in targets:
            for strength in ([.1, .3, .5, .8] if mask else [.25, .5, .75, 1.]):
                suffix = f'{strength:.2f}'.replace('.', 'p')
                result.append(dict(phase=phase,
                                   condition_id=f'mask_{target}_ratio{suffix}' if mask else f'decay_{target}_rho{suffix}',
                                   target=target, kind='random_masking' if mask else 'attenuation',
                                   strength=strength, repeats=5 if mask else 1))
    require(len(result) == 56, 'Condition count')
    return result


def independent_observations(y, p, q):
    """Use SciPy relative entropy and direct indexing, independently of core."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    y = np.asarray(y, np.int64)
    n, repeats, _ = q.shape
    predicted = np.argmax(p, axis=1)
    altered = np.argmax(q, axis=2)
    confidence = np.max(p, axis=1)
    altered_confidence = np.max(q, axis=2)
    true_before = p[np.arange(n), y]
    true_after = np.take_along_axis(q, np.broadcast_to(y[:, None, None], (n, repeats, 1)), axis=2)[..., 0]
    midpoint = (q + p[:, None, :]) * .5
    js = .5 * (rel_entr(p[:, None, :], midpoint).sum(2) + rel_entr(q, midpoint).sum(2))
    correct = predicted == y
    altered_correct = altered == y[:, None]
    metrics = np.stack((altered != predicted[:, None], true_after - true_before[:, None],
                        altered_confidence - confidence[:, None], js,
                        correct[:, None] & ~altered_correct,
                        ~correct[:, None] & altered_correct), axis=2)
    return dict(before=predicted, after=altered, confidence=confidence,
                after_confidence=altered_confidence, true_before=true_before,
                true_after=true_after, correct=correct, altered_correct=altered_correct,
                metrics=metrics)


def bootstrap_weights(dialogues, repeats=2000, seed=20261108):
    ordered = np.array(sorted(set(str(x) for x in dialogues)))
    lookup = {name: i for i, name in enumerate(ordered)}
    codes = np.array([lookup[str(name)] for name in dialogues], dtype=np.int64)
    random = np.random.Generator(np.random.PCG64(seed))
    draws = random.integers(len(ordered), size=(repeats, len(ordered)))
    # Count each resampled dialogue independently, without analyzer's add.at.
    weights = np.stack([np.bincount(draw, minlength=len(ordered)) for draw in draws])
    return ordered, codes, weights


def divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, float),
                                                 np.asarray(denominator, float))
    output = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def f1_from_counts(counts):
    """Union-of-present-labels F1 on arbitrary leading bootstrap/repeat axes."""
    truth = counts.sum(axis=-1)
    called = counts.sum(axis=-2)
    hits = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = truth + called
    per_class = np.divide(2 * hits, denominator, out=np.zeros_like(hits), where=denominator != 0)
    return dict(weighted_f1=divide((truth * per_class).sum(-1), truth.sum(-1)),
                macro_f1=divide(per_class.sum(-1), np.count_nonzero(denominator, axis=-1)))


def add_interval(row, name, estimate, samples):
    finite = np.asarray(samples)[np.isfinite(samples)]
    endpoints = np.percentile(finite, [2.5, 97.5]) if finite.size else [np.nan, np.nan]
    row[name] = float(estimate)
    row[name + '_ci_low'], row[name + '_ci_high'] = map(float, endpoints)
    row[name + '_valid_bootstraps'] = int(finite.size)


def independent_summary(y, p, q, selected, codes, weights, observations, include_f1=True):
    """Recompute all CIs from fixed paired weights and dialogue sufficient sums.

    Classification points are checked via sklearn; bootstrap F1 independently
    derives confusion histograms with bincount. Random repeat metrics are
    calculated separately before averaging, never from ensembled probabilities.
    Empty subgroup bootstrap draws remain NaN and are excluded only at quantiles.
    """
    selected = np.asarray(selected, bool)
    repeats, classes = q.shape[1:]
    n_dialogues = weights.shape[1]
    count = np.bincount(codes[selected], minlength=n_dialogues)
    denominator = weights @ count
    row = dict(n_utterances=int(selected.sum()), n_dialogues_present=int(np.count_nonzero(count)),
               repeats=repeats, bootstrap_unit='original_dialogue_pool',
               metric_scale='fraction; natural-log JS',
               aggregation='per-repeat classification metric then mean; no probability ensemble')
    metrics = observations['metrics']
    per_dialogue = np.zeros((n_dialogues, repeats, len(NAMES)))
    for d in np.flatnonzero(count):
        per_dialogue[d] = metrics[selected & (codes == d)].sum(0)
    boot_metrics = np.einsum('bd,drk->brk', weights, per_dialogue, optimize=True)
    boot_metrics = divide(boot_metrics, denominator[:, None, None]).mean(1)
    point = metrics[selected].mean((0, 1)) if selected.any() else np.full(len(NAMES), np.nan)
    for j, name in enumerate(NAMES):
        add_interval(row, name, point[j], boot_metrics[:, j])
    before, after = observations['before'], observations['after']
    correct = observations['correct']
    changed_correct = observations['altered_correct']
    accuracy_sum = np.zeros((n_dialogues, 1 + repeats))
    for d in np.flatnonzero(count):
        chosen = selected & (codes == d)
        accuracy_sum[d, 0] = correct[chosen].sum()
        accuracy_sum[d, 1:] = changed_correct[chosen].sum(0)
    sampled_accuracy = divide(weights @ accuracy_sum, denominator[:, None])
    original_accuracy = accuracy_score(y[selected], before[selected]) if selected.any() else np.nan
    perturbed_accuracy = np.mean([accuracy_score(y[selected], after[selected, r]) for r in range(repeats)]) if selected.any() else np.nan
    add_interval(row, 'original_accuracy', original_accuracy, sampled_accuracy[:, 0])
    add_interval(row, 'perturbed_accuracy', perturbed_accuracy, sampled_accuracy[:, 1:].mean(1))
    add_interval(row, 'delta_accuracy', perturbed_accuracy - original_accuracy,
                 sampled_accuracy[:, 1:].mean(1) - sampled_accuracy[:, 0])
    if include_f1:
        counts = np.zeros((n_dialogues, 1 + repeats, classes, classes))
        all_predictions = np.column_stack((before, after))
        for d in np.flatnonzero(count):
            chosen = selected & (codes == d)
            for r in range(1 + repeats):
                counts[d, r] = np.bincount(classes * y[chosen] + all_predictions[chosen, r],
                                           minlength=classes * classes).reshape(classes, classes)
        sampled = f1_from_counts(np.einsum('bd,drij->brij', weights, counts, optimize=True))
        for name, averaging in [('weighted_f1', 'weighted'), ('macro_f1', 'macro')]:
            original = f1_score(y[selected], before[selected], average=averaging, zero_division=0) if selected.any() else np.nan
            perturbed = np.mean([f1_score(y[selected], after[selected, r], average=averaging,
                                         zero_division=0) for r in range(repeats)]) if selected.any() else np.nan
            add_interval(row, 'original_' + name, original, sampled[name][:, 0])
            add_interval(row, 'perturbed_' + name, perturbed, sampled[name][:, 1:].mean(1))
            add_interval(row, 'delta_' + name, perturbed - original,
                         sampled[name][:, 1:].mean(1) - sampled[name][:, 0])
    return row


def independent_correctness_contrasts(observations, codes, weights):
    """Direct differences within identical dialogue draws, with masks fixed."""
    correct = observations['correct']
    values = observations['metrics'].mean(axis=1)
    group_points, group_draws = {}, {}
    for name, selected in [('correct', correct), ('wrong', ~correct)]:
        counts = np.array([np.count_nonzero(selected & (codes == d))
                           for d in range(weights.shape[1])])
        sums = np.array([values[selected & (codes == d)].sum(0)
                         for d in range(weights.shape[1])])
        group_points[name] = values[selected].mean(0) if selected.any() else np.full(len(NAMES), np.nan)
        group_draws[name] = divide(np.einsum('bd,dk->bk', weights, sums, optimize=True),
                                  (weights @ counts)[:, None])
    result = {}
    differences = group_draws['wrong'] - group_draws['correct']
    for name in ('flip_rate', 'js_divergence', 'confidence_change', 'true_probability_change'):
        index = NAMES.index(name)
        samples = differences[:, index]
        samples = samples[np.isfinite(samples)]
        low, high = np.percentile(samples, [2.5, 97.5]) if samples.size else (np.nan, np.nan)
        result[name] = dict(correct_mean=float(group_points['correct'][index]),
                            wrong_mean=float(group_points['wrong'][index]),
                            difference_wrong_minus_correct=float(group_points['wrong'][index] - group_points['correct'][index]),
                            ci_low=float(low), ci_high=float(high), valid_bootstraps=int(samples.size),
                            original_correct_count=int(correct.sum()), original_wrong_count=int((~correct).sum()),
                            interpretation='unadjusted association; paired original-dialogue draws; fixed masks')
    return result


def independent_previous_comparison(previous_rows, dataset, seed, target, base, observations):
    """Exact Validation identity join, independently checking all three controls."""
    modality = {'T-All': 'text_representation', 'A': 'audio', 'V': 'visual'}[target]
    by_utterance = {}
    for row in previous_rows:
        if row['dataset'] != dataset or row['modality'] != modality:
            continue
        require(int(row['seed']) == seed, 'Previous pilot seed mismatch')
        identity = (row['dialogue_id'], int(row['utterance_index']))
        kind = row['perturbation_type']
        require(kind in ('gaussian', 'dropout', 'attenuation'), 'Unexpected prior control')
        require(float(row['perturbation_strength']) == (.95 if kind == 'attenuation' else .05) and
                int(row['repeats']) == (1 if kind == 'attenuation' else 5), 'Previous pilot control configuration')
        require(kind not in by_utterance.setdefault(identity, {}), 'Duplicate prior identity/control')
        by_utterance[identity][kind] = row
    identities = [(str(d), int(u)) for d, u in zip(base['dialogue_ids'], base['utterance_indices'])]
    require(len(set(identities)) == len(identities) and set(by_utterance) == set(identities),
            'Previous/current exact identity join not one-to-one')
    stable = np.zeros(len(identities), bool)
    for index, identity in enumerate(identities):
        controls = by_utterance[identity]
        require(set(controls) == {'gaussian', 'dropout', 'attenuation'}, 'Previous control missing')
        for row in controls.values():
            require(int(row['label']) == int(base['labels'][index]) and
                    int(row['original_prediction']) == int(base['predictions'][index]) and
                    int(row['speaker_id']) == int(base['speaker_ids'][index]), 'Joined previous baseline metadata')
            same_value(row['original_confidence'], float(observations['confidence'][index]), 'Joined original confidence')
            same_value(row['correct'], bool(observations['correct'][index]), 'Joined original correctness')
        stable[index] = all(float(row['flip_rate']) == 0 for row in controls.values())
    strong_flip = observations['metrics'][:, 0, 0].astype(bool)
    true_drop = observations['metrics'][:, 0, 1] <= -.1
    wrong = ~observations['correct']
    return dict(dataset=dataset, target=target, n_utterances=len(identities),
                weak_stable_count=int(np.count_nonzero(stable)),
                strong_flip_among_weak_stable=int(np.count_nonzero(stable & strong_flip)),
                strong_trueprob_drop_ge_point1_among_weak_stable=int(np.count_nonzero(stable & true_drop)),
                weak_stable_wrong_count=int(np.count_nonzero(stable & wrong)),
                strong_flip_among_weak_stable_wrong=int(np.count_nonzero(stable & wrong & strong_flip)),
                weak_definition='zero flips in all three prior 0.05/0.05/0.95 conditions; 5/5/1 draws',
                strong_definition='current overall target-only mean contraction rho1; other utterances unchanged')


def check_distribution(probability, log_probability, context):
    require(probability.dtype == np.float32 and log_probability.dtype == np.float32,
            context + ' not FP32')
    require(np.isfinite(probability).all() and np.isfinite(log_probability).all(), context + ' nonfinite')
    require((probability >= 0).all(), context + ' negative probability')
    require(np.allclose(probability.astype(float).sum(-1), 1, atol=1e-6, rtol=1e-5), context + ' normalization')
    require(np.array_equal(np.exp(log_probability), probability), context + ' exp(logp) exact mismatch')


def indexed_rows(path, keys):
    rows = read_csv(path)
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(rows) == len(result), 'Duplicate CSV key: ' + str(path))
    return result


def verify_sources(root, config):
    audit = json.loads((root / 'provenance/audit.json').read_text())
    source = json.loads((root / 'provenance/source_hashes.json').read_text())
    require(audit['status'] == 'passed', 'Input audit failed')
    require(sha(root / 'configs/evidence_sensitivity_config.json') == source['config_sha256'], 'Config SHA')
    require(sha(root / 'provenance/audit.json') == source['audit_sha256'], 'Audit SHA')
    previous = Path(config['previous_pilot'])
    require(len(audit['previous_files_sha256']) == 152, 'Prior file count')
    actual_files = {str(p.relative_to(previous)) for p in previous.rglob('*') if p.is_file()}
    require(actual_files == set(audit['previous_files_sha256']), 'Prior tree files added/deleted')
    for relative, digest in audit['previous_files_sha256'].items():
        require(sha(previous / relative) == digest, 'Prior immutable file changed: ' + relative)
    digests = {}
    def scan(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name.startswith('/') and isinstance(item, str) and re.fullmatch('[0-9a-f]{64}', item):
                    require(name not in digests or item == digests[name], 'Conflicting source digest: ' + name)
                    digests[name] = item
                else:
                    scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
    scan(source)
    for name, digest in digests.items():
        require(sha(name) == digest, 'Final source digest changed: ' + name)
    for script in (root / 'scripts').glob('*.py'):
        require(str(script) in digests, 'New script missing final source SHA: ' + str(script))
    for name, digest in audit['frozen_source_sha256'].items():
        require(digests.get(name) == digest, 'Frozen science source absent/different: ' + name)
    for dataset, item in audit['datasets'].items():
        for key in ('checkpoint', 'effective_config', 'train_statistics', 'validation_baseline'):
            name, digest = item[key + '_path'], item[key + '_sha256']
            require(sha(name) == digest and digests.get(name) == digest, dataset + ' ' + key + ' SHA')
        for name, digest in item['data_sha256'].items():
            require(sha(name) == digest and digests.get(name) == digest, dataset + ' data SHA')
        require(all(c['mean_bitwise_equal'] and c['std_ddof0_bitwise_equal'] and c['count_equal']
                    for c in item['training_statistics_exact_checks']), 'Training audit evidence')
    return audit, len(digests)


def verify_parity(root):
    cache = json.loads((root / 'provenance/graph_cache_parity.json').read_text())
    require(cache['passed'] and cache['fp32'] and cache['exit_code'] == 0, 'Cached eager parity failed')
    require(cache['model_unchanged'] == {'IEMOCAP': True, 'MELD': True}, 'Parity model changed')
    seen = set()
    for check in cache['checks']:
        require(check['passed'] and check['full_inputs_unchanged'], 'Parity check failed')
        require(np.isfinite(check['max_log_error']) and np.isfinite(check['max_probability_error']), 'Parity error nonfinite')
        require(check['max_probability_error'] <= 1e-6 + 1e-5, 'Parity probability error exceeds conservative bound')
        seen.add((check['dataset'], int(check['batch_index']), check['target'], check['kind'], check['value']))
    for dataset, batches in [('IEMOCAP', 1), ('MELD', 8)]:
        for batch in range(batches):
            require((dataset, batch, 'original', None, None) in seen, 'Missing original parity')
            # The preregistered preflight covered one stored text array and the
            # three overall targets. Every actual condition additionally has
            # per-batch full-vector parity in its phase completion records.
            for target in ('T1', 'T-All', 'A', 'V'):
                for kind, strength in [('attenuation', .25), ('attenuation', 1.), ('random_masking', .8)]:
                    require((dataset, batch, target, kind, strength) in seen, 'Missing intervention parity ' + str((dataset, batch, target, kind, strength)))
    acceleration = json.loads((root / 'provenance/acceleration_benchmark.json').read_text())
    require(acceleration['validation_only'] and acceleration['original_batch_size'] == 16 and
            acceleration['fp32'] and not acceleration['gradients'], 'Acceleration probe protocol')
    require(acceleration['adopted'] is False and acceleration['final_cached_eager_proof'] == 'provenance/graph_cache_parity.json',
            'Unverified acceleration adopted')
    return len(cache['checks'])


def verify_execution_jobs(root, config):
    jobs = json.loads((root / 'provenance/execution_jobs.json').read_text())
    require(jobs['status'] == 'complete', 'Execution supervisor not complete')
    expected = {(dataset, seed, phase) for dataset, seed in config['anchors'].items()
                for phase in config['phases']}
    identities = [(job['dataset'], job['seed'], job['phase']) for job in jobs['jobs']]
    require(len(identities) == 6 and len(set(identities)) == 6 and set(identities) == expected,
            'Exactly six registered execution jobs required')
    for job in jobs['jobs']:
        require(job['status'] == 'complete' and job['exit_code'] == 0, 'Execution job not successful')
        dataset, seed, phase = job['dataset'], job['seed'], job['phase']
        path = root / f'runs/{dataset}/seed{seed}/{phase}_complete.json'
        require(Path(job['complete_artifact']).resolve() == path, 'Job completion artifact identity')
        done = json.loads(path.read_text())
        for key in ('target_forwards', 'stored_vectors', 'skipped_visual_missing'):
            require(job[key] == done[key], 'Job/completion count disagreement: ' + key)
        command = job['command']
        require(str(root / 'scripts/run_evidence.py') in command and
                command[command.index('--dataset') + 1] == dataset and
                command[command.index('--phase') + 1] == phase, 'Job command scope')
        require(Path(job['log']).is_file(), 'Job run log absent')
        require(int(job['pid']) > 0 and job['finished_at'] >= job['started_at'], 'Job lifecycle metadata')
        for condition_id, digest in done['conditions'].items():
            require(sha(root / f'runs/{dataset}/seed{seed}/conditions/{condition_id}.npz') == digest,
                    'Job condition manifest SHA: ' + condition_id)
    # Concurrent jobs can legitimately share the same GPU. Historical resource
    # snapshots need not show an empty device; no such requirement is imposed.
    return len(identities)


def verify_figures(root):
    manifest = json.loads((root / 'figures/figure_manifest.json').read_text())
    require(manifest['real_saved_validation_data_only'] is True, 'Figure data provenance')
    figures = {figure['name']: figure for figure in manifest['figures']}
    require(set(figures) == set(FIGURES), 'Six scientific figures required')
    for name, item in figures.items():
        require(set(item['outputs']) == {f'figures/{name}.{extension}' for extension in ('svg', 'pdf', 'png')}, 'Figure formats')
        for source in item['source']:
            require(source.startswith('results/') and (root / source).is_file(), 'Figure input absent')
        for extension in ('svg', 'pdf', 'png'):
            path = root / 'figures' / (name + '.' + extension)
            require(path.stat().st_size > 1000, 'Figure file empty: ' + str(path))
            with path.open('rb') as file:
                header = file.read(24)
                if extension == 'png':
                    require(header[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid PNG')
                    dimensions = list(struct.unpack('>II', header[16:24]))
                    require(dimensions == item['png_dimensions'] and min(dimensions) >= 200, 'PNG dimensions')
                elif extension == 'pdf':
                    require(header.startswith(b'%PDF'), 'Invalid PDF')
                else:
                    require(ET.parse(path).getroot().tag.endswith('svg'), 'Invalid SVG')
    return 18


def self_test():
    """Temporary in-memory tests only; never generate research output/figures."""
    p = np.array([[.8, .2], [.2, .8], [.9, .1], [.4, .6]], np.float32)
    q = np.array([[[.7, .3], [.4, .6]], [[.1, .9], [.8, .2]],
                  [[.9, .1], [.7, .3]], [[.6, .4], [.3, .7]]], np.float32)
    y = np.array([0, 1, 1, 0])
    dialogue = np.array(['b', 'a', 'a', 'c'])
    names, codes, weights = bootstrap_weights(dialogue, 2000, 20261108)
    observed = independent_observations(y, p, q)
    require(np.array_equal(observed['metrics'][:, :, 0], [[0, 1], [0, 1], [0, 0], [1, 0]]), 'Synthetic flips')
    unchanged = independent_observations(y, p, p[:, None])
    require(np.array_equal(unchanged['metrics'], np.zeros((4, 1, 6))), 'Unchanged distributions must have zero sensitivity')
    expected_true_change = np.stack([q[i, :, int(y[i])].astype(float) - float(p[i, int(y[i])]) for i in range(4)])
    require(np.array_equal(observed['metrics'][:, :, 1], expected_true_change), 'True-class probability change')
    for selected in (np.ones(4, bool), np.array([1, 0, 0, 0], bool), np.zeros(4, bool)):
        answer = independent_summary(y, p, q, selected, codes, weights, observed)
        explicit = {name: [] for name in ('original_weighted_f1', 'perturbed_weighted_f1',
                                         'delta_weighted_f1', 'original_macro_f1',
                                         'perturbed_macro_f1', 'delta_macro_f1', 'flip_rate')}
        for counts in weights:
            chosen = np.concatenate([np.flatnonzero(selected & (codes == d)).repeat(int(counts[d]))
                                     for d in range(len(names))])
            if not len(chosen):
                continue
            for averaging in ('weighted', 'macro'):
                before = f1_score(y[chosen], p[chosen].argmax(1), average=averaging, zero_division=0)
                after = np.mean([f1_score(y[chosen], q[chosen, r].argmax(1), average=averaging, zero_division=0) for r in range(2)])
                prefix = 'weighted_f1' if averaging == 'weighted' else 'macro_f1'
                explicit['original_' + prefix].append(before)
                explicit['perturbed_' + prefix].append(after)
                explicit['delta_' + prefix].append(after - before)
            explicit['flip_rate'].append(observed['metrics'][chosen, :, 0].mean())
        for metric, samples in explicit.items():
            ends = np.percentile(samples, [2.5, 97.5]) if samples else [np.nan, np.nan]
            same_value(answer[metric + '_ci_low'], ends[0], 'Synthetic lower CI ' + metric)
            same_value(answer[metric + '_ci_high'], ends[1], 'Synthetic upper CI ' + metric)
            same_value(answer[metric + '_valid_bootstraps'], len(samples), 'Synthetic CI count')
    contrasts = independent_correctness_contrasts(observed, codes, weights)
    all_correct = independent_observations(p.argmax(1), p, q)
    empty_contrasts = independent_correctness_contrasts(all_correct, codes, weights)
    require(all(value['valid_bootstraps'] == 0 and np.isnan(value['ci_low'])
                for value in empty_contrasts.values()), 'Empty correctness contrast stays NA')
    for name, answer in contrasts.items():
        j = NAMES.index(name)
        explicit = []
        for counts in weights:
            chosen = np.concatenate([np.flatnonzero(codes == d).repeat(int(counts[d])) for d in range(len(names))])
            flags = observed['correct'][chosen]
            if not flags.any() or flags.all():
                continue
            values = observed['metrics'][chosen, :, j].mean(1)
            explicit.append(values[~flags].mean() - values[flags].mean())
        low, high = np.percentile(explicit, [2.5, 97.5])
        same_value(answer['ci_low'], low, 'Synthetic paired contrast lower')
        same_value(answer['ci_high'], high, 'Synthetic paired contrast upper')
        same_value(answer['valid_bootstraps'], len(explicit), 'Synthetic paired contrast count')
    base = dict(labels=y, predictions=p.argmax(1), dialogue_ids=dialogue,
                utterance_indices=np.arange(4), speaker_ids=np.zeros(4, int))
    prior = []
    for i in range(4):
        for kind in ('gaussian', 'dropout', 'attenuation'):
            prior.append(dict(dataset='IEMOCAP', seed='1475', modality='audio',
                              dialogue_id=str(dialogue[i]), utterance_index=str(i),
                              perturbation_type=kind, perturbation_strength='.95' if kind == 'attenuation' else '.05',
                              repeats='1' if kind == 'attenuation' else '5', label=str(y[i]),
                              original_prediction=str(p[i].argmax()), speaker_id='0',
                              original_confidence=str(float(p[i].max())), correct=str(bool(observed['correct'][i])),
                              flip_rate='0.2' if i == 1 else '0'))
    joined = independent_previous_comparison(prior, 'IEMOCAP', 1475, 'A', base, observed)
    require(joined['weak_stable_count'] == 3 and joined['weak_stable_wrong_count'] == 2 and
            joined['strong_flip_among_weak_stable'] == 1 and
            joined['strong_flip_among_weak_stable_wrong'] == 1, 'Synthetic prior exact join counts')
    duplicate_rejected = False
    try:
        independent_previous_comparison(prior + [prior[0]], 'IEMOCAP', 1475, 'A', base, observed)
    except AssertionError:
        duplicate_rejected = True
    require(duplicate_rejected, 'Prior join must reject duplicate identity/control')
    caught = False
    try:
        same_row({'value': '0.1'}, {'value': .2}, 'Synthetic tampering')
    except AssertionError:
        caught = True
    require(caught, 'Verifier must reject changed values')
    return {'status': 'passed', 'synthetic_self_test_only': True,
            'bootstrap_draws_checked_per_group': 2000, 'groups': 3,
            'direct_correctness_contrast_cis_and_empty_groups_checked': True,
            'previous_identity_join_counts_and_duplicate_rejection_checked': True,
            'no_research_results_or_files_generated': True}


def verify(root, report):
    config = json.loads((root / 'configs/evidence_sensitivity_config.json').read_text())
    require(config['stage'] == 'validation_modality_evidence_sensitivity' and
            config['anchors'] == {'IEMOCAP': 1475, 'MELD': 67137}, 'Research scope changed')
    require(config['inference']['split'] == 'validation' and config['inference']['eval'] is True and
            config['inference']['dtype'] == 'float32' and config['inference']['backend'] == 'aten' and
            config['inference']['batch_size'] == 16 and config['inference']['one_target_utterance_at_a_time'] and
            config['inference']['preserve_original_batch_order_padding_masks'], 'Inference protocol')
    require(config['analysis']['bootstrap_repeats'] == 2000 and config['analysis']['bootstrap_seed'] == 20261108 and
            config['analysis']['bootstrap_unit'] == 'dialogue_id' and config['analysis']['high_confidence_threshold'] == .9,
            'Analysis protocol')
    require(set(config['prohibited']) >= {'CMERC_training', 'test_split_inference_or_metrics',
            'joint_modalities', 'extra_seeds', 'reliability_score_fitting', 'temperature_scaling',
            'historical_writes', 'automatic_formal_experiments'}, 'Missing prohibited operations')
    specs = specifications(config)
    audit, source_count = verify_sources(root, config)
    parity_count = verify_parity(root)
    execution_jobs = verify_execution_jobs(root, config)
    report('Prior 152 immutable files and all final source/config/input SHA checked; train statistics audit preserved.')
    output = root / 'results'
    require(sha(output / 'modality_sensitivity_summary.csv') == sha(output / 'sensitivity_summary.csv'), 'Summary alias not byte-identical')
    summaries = indexed_rows(output / 'modality_sensitivity_summary.csv', ('dataset', 'condition_id', 'subgroup'))
    classes = indexed_rows(output / 'classwise_sensitivity.csv', ('dataset', 'condition_id', 'class_basis', 'class_index'))
    missing_table = indexed_rows(output / 'visual_missingness_analysis.csv', ('dataset', 'subgroup'))
    monotonic_table = indexed_rows(output / 'monotonicity_analysis.csv', ('dataset', 'target', 'subgroup'))
    contrast_table = indexed_rows(output / 'correctness_group_contrasts.csv', ('dataset', 'condition_id', 'metric'))
    previous_table = indexed_rows(output / 'previous_pilot_sensitivity_comparison.csv', ('dataset', 'target'))
    prior_rows = read_csv(Path(config['previous_pilot']) / 'results/utterance_behavior_metrics.csv')
    high_table = read_csv(output / 'high_confidence_wrong_sensitive_cases.csv')
    require(len(summaries) == 560 and len(classes) == 1456 and len(missing_table) == 4 and len(monotonic_table) == 70,
            'CSV summary/class/missing/monotonicity row counts')
    require(len(contrast_table) == 448 and len(previous_table) == 6 and len(prior_rows) == 14724,
            'Correctness contrast/prior comparison/source row counts')
    expected_high = []
    independent_rows = []
    totals = dict(condition_files=0, stored_vectors=0, actual_target_forwards=0, skipped_visual_missing_vectors=0,
                  per_utterance_rows=0, summary_rows=0, classwise_rows=0, phase_completion_records=0,
                  correctness_contrast_rows=0, previous_pilot_comparison_rows=0)
    require({p.name for p in (root / 'runs').iterdir() if p.is_dir()} == set(config['anchors']), 'Extra dataset directories')
    require(not any('test' in p.name.lower() for p in (root / 'runs').rglob('*.npz')), 'Forbidden new Test prediction artifacts')
    with (output / 'per_utterance_sensitivity.csv').open(newline='', encoding='utf-8') as stream:
        detail_rows = iter(csv.DictReader(stream))
        for dataset, seed in config['anchors'].items():
            directory = root / 'runs' / dataset / f'seed{seed}'
            require({p.name for p in (root / 'runs' / dataset).iterdir() if p.is_dir()} == {f'seed{seed}'}, 'Extra seed directory')
            base = read_npz(directory / 'baseline_validation.npz')
            previous = read_npz(Path(config['previous_pilot']) / f'runs/{dataset}/seed{seed}/validation_predictions.npz')
            require(set(base) == set(previous) | {'visual_missing'}, 'Baseline columns')
            for key, value in previous.items():
                require(np.array_equal(base[key], value), 'Baseline changed: ' + dataset + '/' + key)
            y, p = base['labels'], base['probabilities']
            check_distribution(p, base['log_probabilities'], dataset + ' baseline')
            n, c = p.shape
            require(n == config['validation_sizes'][dataset]['utterances'] and c == (6 if dataset == 'IEMOCAP' else 7), 'Validation size/classes')
            visual = base['visual_missing']
            require(visual.dtype == bool and visual.shape == (n,), 'Baseline visual mask')
            recorded_missing = {(x['dialogue_id'], x['utterance_index']) for x in audit['datasets'][dataset]['all_zero_features']['visual']}
            expected_missing = np.array([(str(d), int(u)) in recorded_missing for d, u in zip(base['dialogue_ids'], base['utterance_indices'])])
            require(np.array_equal(visual, expected_missing) and int(visual.sum()) == (1 if dataset == 'IEMOCAP' else 95), 'Missing visual identity')
            dialogue_names, codes, weights = bootstrap_weights(base['dialogue_ids'])
            saved_weights = read_npz(output / f'{dataset}_paired_bootstrap_weights.npz')
            require(np.array_equal(dialogue_names, saved_weights['dialogue_ids']) and
                    np.array_equal(weights, saved_weights['weights']), 'Shared bootstrap weights not seed-exact')
            correct = base['predictions'] == y
            masks = dict(all=np.ones(n, bool), original_correct=correct, original_wrong=~correct,
                         visual_present=~visual, visual_missing=visual)
            conditions = directory / 'conditions'
            require({f.name for f in conditions.glob('*.npz')} == {s['condition_id'] + '.npz' for s in specs}, 'Condition files unexpected/missing')
            completion = {}
            for phase in config['phases']:
                done = json.loads((directory / f'{phase}_complete.json').read_text())
                require(done['passed'] and done['dataset'] == dataset and done['seed'] == seed and done['phase'] == phase,
                        'Phase completion identity')
                for key in ('inputs_restored_exact', 'parameters_and_buffers_unchanged', 'gradients_absent'):
                    require(done[key] is True, 'Phase invariant proof: ' + key)
                require(done['test_inference'] is False and done['graph_guard_observations'] > 0, 'Phase inference scope/graph guard')
                require(done['config_sha256'] == sha(root / 'configs/evidence_sensitivity_config.json') and
                        done['runner_sha256'] == sha(root / 'scripts/run_evidence.py'), 'Phase source/config SHA')
                selected_specs = [s for s in specs if s['phase'] == phase]
                require(set(done['conditions']) == {s['condition_id'] for s in selected_specs}, 'Phase condition membership')
                vectors = n * sum(s['repeats'] for s in selected_specs)
                skipped = int(visual.sum()) * sum(s['repeats'] for s in selected_specs if s['target'] == 'V')
                require(done['stored_vectors'] == vectors and done['skipped_visual_missing'] == skipped and
                        done['target_forwards'] == vectors - skipped, 'Phase forward/vector counts')
                checks = done['complete_probability_parity_checks']
                require(checks and all(np.isfinite(r['log_error']) and np.isfinite(r['probability_error'])
                                       and r['probability_error'] <= 1.1e-5 for r in checks), 'Phase numerical parity records')
                require({r['batch'] for r in checks if r['kind'] == 'original'} == set(range((len(dialogue_names)+15)//16)),
                        'Every original batch needs recorded full-vector parity')
                expected_checks = {(batch, 'original') for batch in np.unique(base['batch_indices'])}
                for batch in np.unique(base['batch_indices']):
                    first = np.flatnonzero(base['batch_indices'] == batch)[0]
                    for spec in selected_specs:
                        if spec['target'] != 'V' or not visual[first]:
                            expected_checks.add((batch, spec['condition_id']))
                require({(r['batch'], r['kind']) for r in checks} == expected_checks and
                        len(checks) == len(expected_checks), 'Every active per-batch condition needs full-vector parity')
                completion[phase] = done
                totals['phase_completion_records'] += 1
                totals['actual_target_forwards'] += done['target_forwards']
            for spec in specs:
                identifier, repeats, target = spec['condition_id'], spec['repeats'], spec['target']
                path = conditions / (identifier + '.npz')
                require(sha(path) == completion[spec['phase']]['conditions'][identifier], 'Condition SHA changed')
                payload = read_npz(path)
                tag = dict(dataset=dataset, seed=seed, **spec)
                for key, value in spec.items():
                    same_value(payload[key].item(), value, identifier + '/' + key)
                for key in META:
                    require(np.array_equal(payload[key], base[key]), 'Condition metadata: ' + key)
                q, logq = payload['probabilities'], payload['log_probabilities']
                require(q.shape == (n, repeats, c) and logq.shape == q.shape, 'Condition array shape')
                check_distribution(q, logq, identifier)
                require(np.array_equal(payload['original_probabilities'], p) and
                        np.array_equal(payload['original_predictions'], base['predictions']) and
                        np.array_equal(payload['perturbed_predictions'], q.argmax(2)), 'Condition prediction/original invariants')
                active = ~visual if target == 'V' else np.ones(n, bool)
                require(payload['active'].dtype == bool and np.array_equal(payload['active'], active) and
                        np.array_equal(payload['visual_missing'], visual), 'Active/missing mask')
                if target == 'V':
                    require(np.array_equal(q[visual], np.repeat(p[visual, None], repeats, 1)) and
                            np.array_equal(logq[visual], np.repeat(base['log_probabilities'][visual, None], repeats, 1)),
                            'Missing V probabilities/logp not baseline-exact')
                observed = independent_observations(y, p, q)
                for metric, values in independent_correctness_contrasts(observed, codes, weights).items():
                    row = dict(tag, metric=metric, **values)
                    same_row(contrast_table[(dataset, identifier, metric)], row,
                             f'Paired correctness contrast {dataset}/{identifier}/{metric}')
                    totals['correctness_contrast_rows'] += 1
                if spec['phase'] == 'overall_attenuation' and spec['strength'] == 1:
                    row = independent_previous_comparison(prior_rows, dataset, seed, target, base, observed)
                    same_row(previous_table[(dataset, target)], row, f'Previous pilot exact join {dataset}/{target}')
                    totals['previous_pilot_comparison_rows'] += 1
                for i in range(n):
                    for repeat in range(repeats):
                        row = dict(tag, dialogue_id=str(base['dialogue_ids'][i]),
                                   utterance_index=int(base['utterance_indices'][i]), speaker_id=int(base['speaker_ids'][i]),
                                   batch_index=int(base['batch_indices'][i]), dialogue_length=int(base['dialogue_lengths'][i]),
                                   repeat=repeat, label=int(y[i]), active=bool(active[i]), visual_missing=bool(visual[i]),
                                   original_correct=bool(correct[i]), original_prediction=int(observed['before'][i]),
                                   perturbed_prediction=int(observed['after'][i, repeat]), original_confidence=float(observed['confidence'][i]),
                                   perturbed_confidence=float(observed['after_confidence'][i, repeat]),
                                   original_true_probability=float(observed['true_before'][i]),
                                   perturbed_true_probability=float(observed['true_after'][i, repeat]))
                        for j, key in enumerate(('flip', 'true_probability_change', 'confidence_change', 'js_divergence', 'correct_to_wrong', 'wrong_to_correct')):
                            row[key] = float(observed['metrics'][i, repeat, j])
                        actual = next(detail_rows, None)
                        require(actual is not None, 'Per-utterance CSV ended prematurely')
                        same_row(actual, row, f'{dataset}/{identifier}/{i}/{repeat}')
                        totals['per_utterance_rows'] += 1
                        if spec['kind'] == 'attenuation' and spec['strength'] == 1 and not correct[i] and row['original_confidence'] >= .9 and (row['flip'] or row['true_probability_change'] <= -.1):
                            expected_high.append(row)
                for subgroup, mask in masks.items():
                    row = dict(tag, subgroup=subgroup, active_utterances=int(active[mask].sum()))
                    row.update(independent_summary(y, p, q, mask, codes, weights, observed))
                    same_row(summaries[(dataset, identifier, subgroup)], row, 'Summary ' + dataset + '/' + identifier + '/' + subgroup)
                    independent_rows.append(row)
                    totals['summary_rows'] += 1
                for basis, labels in [('true_class', y), ('original_prediction_class', base['predictions'])]:
                    for class_index in range(c):
                        row = dict(tag, class_basis=basis, class_index=class_index)
                        row.update(independent_summary(y, p, q, labels == class_index, codes, weights, observed, include_f1=False))
                        same_row(classes[(dataset, identifier, basis, str(class_index))], row,
                                 f'Class {dataset}/{identifier}/{basis}/{class_index}')
                        totals['classwise_rows'] += 1
                totals['condition_files'] += 1
                totals['stored_vectors'] += n * repeats
                totals['skipped_visual_missing_vectors'] += int((~active).sum()) * repeats
                if totals['condition_files'] % 7 == 0:
                    report(f'Independently verified {totals["condition_files"]}/112 conditions, {totals["per_utterance_rows"]} per-repeat rows; all subgroup/class CIs recomputed.')
            for subgroup in ('visual_present', 'visual_missing'):
                selected = masks[subgroup]
                observations = independent_observations(y, p, p[:, None])
                points = independent_summary(y, p, p[:, None], selected, codes, weights, observations, include_f1=False)
                row = dict(dataset=dataset, seed=seed, subgroup=subgroup, n_total=n, n_utterances=int(selected.sum()),
                           original_correct_count=int(correct[selected].sum()), original_accuracy=points['original_accuracy'],
                           original_accuracy_ci_low=points['original_accuracy_ci_low'], original_accuracy_ci_high=points['original_accuracy_ci_high'],
                           visual_intervention_applied=subgroup != 'visual_missing',
                           handling='missing rows retained; V missing inputs/probabilities unchanged')
                same_row(missing_table[(dataset, subgroup)], row, 'Missingness ' + dataset + '/' + subgroup)
        require(next(detail_rows, None) is None, 'Per-utterance CSV contains extra rows')
    require(len(expected_high) == len(high_table), 'High-confidence case count')
    for i, row in enumerate(expected_high):
        same_row(high_table[i], row, f'High-confidence case {i}')
    for dataset in config['anchors']:
        for target in ('T-All', 'A', 'V', 'T1', 'T2', 'T3', 'T4'):
            for subgroup in GROUPS:
                curve = sorted([r for r in independent_rows if r['dataset'] == dataset and r['target'] == target and
                                r['subgroup'] == subgroup and r['kind'] == 'attenuation'], key=lambda r: r['strength'])
                accuracy = [curve[0]['original_accuracy']] + [r['perturbed_accuracy'] for r in curve]
                truechange = [0.] + [r['true_probability_change'] for r in curve]
                flips = [0.] + [r['flip_rate'] for r in curve]
                row = dict(dataset=dataset, target=target, subgroup=subgroup,
                           rho=json.dumps([0.] + [r['strength'] for r in curve]), accuracy=json.dumps(accuracy),
                           true_probability_change=json.dumps(truechange), flip_rate=json.dumps(flips),
                           accuracy_increases=int(np.count_nonzero(np.diff(accuracy) > 1e-12)),
                           true_probability_increases=int(np.count_nonzero(np.diff(truechange) > 1e-12)),
                           flip_rate_decreases=int(np.count_nonzero(np.diff(flips) < -1e-12)),
                           interpretation='descriptive reversals, no monotonicity assumption; exploratory')
                original = monotonic_table[(dataset, target, subgroup)]
                require(set(original) == set(row), 'Monotonicity columns')
                for key, value in row.items():
                    if key in ('rho', 'accuracy', 'true_probability_change', 'flip_rate'):
                        require(np.allclose(json.loads(original[key]), json.loads(value), atol=2e-11, rtol=2e-10, equal_nan=True), 'Monotonicity curve ' + key)
                    else:
                        same_value(original[key], value, 'Monotonicity ' + key)
    require(totals == dict(condition_files=112, stored_vectors=274848, actual_target_forwards=272544,
                          skipped_visual_missing_vectors=2304, per_utterance_rows=274848,
                          summary_rows=560, classwise_rows=1456, phase_completion_records=6,
                          correctness_contrast_rows=448, previous_pilot_comparison_rows=6), 'Final exact counts')
    figures = verify_figures(root)
    # Repeat integrity checks AFTER all computation, to catch concurrent changes.
    verify_sources(root, config)
    return dict(status='passed', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **totals, high_confidence_wrong_sensitive_rows=len(expected_high),
                all_per_utterance_csv_fields_independently_recomputed=True,
                all_summary_and_classwise_point_metrics_independently_recomputed=True,
                all_summary_and_classwise_2000_dialogue_bootstrap_intervals_independently_recomputed=True,
                direct_correctness_contrast_paired_bootstrap_intervals_independently_recomputed=True,
                previous_pilot_exact_identity_join_and_counts_independently_recomputed=True,
                successful_execution_jobs_verified=execution_jobs,
                bootstrap_seed_exact_shared_weights_verified=True, immutable_previous_files_verified=152,
                final_source_hashes_verified=source_count, original_and_intervention_parity_records=parity_count,
                missing_visual_utterances=96, missing_visual_input_policy='not intervened; complete probabilities and logp baseline-exact; retained',
                exact_input_and_parameters_invariants='verified runner proof records; no fresh model forward in independent verifier',
                training_statistics='independent audit float64->float32 mean/std/count evidence plus immutable SHA checked',
                figure_files_verified=figures, no_model_forward=True, no_gpu=True,
                no_test_prediction_arrays_deserialized=True, no_test_metrics=True,
                no_training_or_calibration_or_joint_modalities=True,
                verifier_sha256=sha(Path(__file__)),
                limitations=['Parameter/input equality and numerical parity are supported by saved inference proof records; verifier does not rerun GPU inference.',
                             'Bootstrap verifies fixed paired Validation dialogue resampling; no claim of population significance or causal emotion evidence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    root = args.root.resolve()
    log_path = root / 'logs/independent_verification.log'
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        def report(message):
            print(message, flush=True)
            log.write(message + '\n')
        report('Independent CPU Validation-only verifier started; Test prediction files are byte-hashed only, never deserialized.')
        try:
            result = verify(root, report)
        except Exception as error:
            import traceback
            log.write(traceback.format_exc())
            (root / 'verification.json').write_text(json.dumps(dict(
                status='failed', failed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                error=repr(error), verifier_sha256=sha(Path(__file__))), indent=2) + '\n')
            raise
        (root / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        report('All independent checks passed. Verification saved; artifact manifest will include this completed log.')
    excluded = root / 'provenance/artifact_manifest.json'
    files = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
             if path.is_file() and path != excluded}
    excluded.write_text(json.dumps(dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                        files=files, excludes_self=True), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
