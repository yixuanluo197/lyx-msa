"""Label-free representation interventions and post-hoc sensitivity metrics.

All interventions return fresh FP32 vectors. Labels enter only the retrospective
sensitivity_metrics function. Probabilities remain unnormalized observations;
metrics use float64 and JS divergence is measured in nats.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _fraction(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + ' must be finite and within [0, 1]')
    return value


def _vector_pairs(vectors, means):
    if not len(vectors) or len(vectors) != len(means):
        raise ValueError('vectors and means must be nonempty lists with equal lengths')
    pairs = []
    for vector, mean in zip(vectors, means):
        x, m = np.asarray(vector, dtype=np.float64), np.asarray(mean, dtype=np.float64)
        if x.ndim != 1 or not x.size or m.shape != x.shape:
            raise ValueError('Each vector and mean must be matching nonempty 1D arrays')
        if not np.isfinite(x).all() or not np.isfinite(m).all():
            raise ValueError('Features and means must be finite')
        pairs.append((x, m))
    return pairs


def _fp32(value):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(value).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Intervened features exceed finite FP32 range')
    return result


def attenuate(vectors, means, rho):
    """Move toward train means: mean + (1-rho)*(x-mean).

    Explicit endpoints avoid cancellation so rho=0 preserves FP32 inputs exactly
    and rho=1 returns the FP32 train means exactly.
    """
    rho = _fraction(rho, 'rho')
    result = []
    for x, mean in _vector_pairs(vectors, means):
        changed = x if rho == 0 else mean if rho == 1 else mean + (1-rho)*(x-mean)
        result.append(_fp32(changed))
    return result


def mask_rng(dataset, seed, dialogue_id, utterance_index, target, ratio, repeat):
    """PCG64 from first128 SHA256 bits, interpreted unsigned big-endian.

    Canonical input: UTF-8 JSON without spaces, preserving Unicode, excluding
    nonfinite numbers. T-All layer streams consume independent sequential draws.
    """
    key = [str(dataset), int(seed), str(dialogue_id), int(utterance_index),
           str(target), _fraction(ratio, 'ratio'), int(repeat)]
    canonical = json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    value = int.from_bytes(hashlib.sha256(canonical.encode('utf-8')).digest()[:16], 'big')
    return np.random.Generator(np.random.PCG64(value))


def mask_vectors(vectors, means, ratio, rng):
    """Replace exactly floor(ratio*D+.5) coordinates per layer by train means.

    Sampling is without replacement. All unselected dimensions retain their
    original FP32 values, with no rescaling. Selected indices follow RNG order.
    """
    ratio = _fraction(ratio, 'ratio')
    outputs, selected = [], []
    for x, mean in _vector_pairs(vectors, means):
        count = int(np.floor(ratio * x.size + .5))
        indices = np.asarray(rng.choice(x.size, size=count, replace=False), dtype=np.int64)
        changed = x.copy()
        changed[indices] = mean[indices]
        outputs.append(_fp32(changed))
        selected.append(indices)
    return outputs, selected


def target_indices(target):
    """Indices of stored arrays in the unchanged XP04 dataloader tuple."""
    mapping = {'T1': [0], 'T2': [1], 'T3': [2], 'T4': [3],
               'T-All': [0, 1, 2, 3], 'A': [5], 'V': [4]}
    if target not in mapping:
        raise ValueError('Unknown intervention target: ' + str(target))
    return mapping[target].copy()


def condition_specs(phase):
    """Enumerate only conditions fixed by this run's independent preregistration."""
    config_path = Path(__file__).resolve().parent.parent / 'configs/evidence_sensitivity_config.json'
    config = json.loads(config_path.read_text())
    if phase not in config['phases']:
        raise ValueError('Unknown preregistered phase: ' + str(phase))
    masking = phase == 'random_masking'
    settings = config['random_masking' if masking else 'attenuation']
    values = settings['ratios' if masking else 'rho']
    specs = []
    for target in config['phases'][phase]:
        target_indices(target)
        for strength in values:
            strength = _fraction(strength, 'strength')
            suffix = f'{strength:.2f}'.replace('.', 'p')
            identifier = f'mask_{target}_ratio{suffix}' if masking else f'decay_{target}_rho{suffix}'
            specs.append(dict(id=identifier, target=target,
                              kind='random_masking' if masking else 'attenuation',
                              strength=strength, repeats=int(settings['repeats'])))
    return specs


def _probabilities(value, ndim, name):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != ndim or any(size == 0 for size in p.shape) or p.shape[-1] < 2:
        raise ValueError(name + ' has an invalid probability shape')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError(name + ' must contain finite nonnegative probabilities')
    if not np.allclose(p.sum(axis=-1), 1., atol=1e-6, rtol=1e-5):
        raise ValueError(name + ' probabilities are not normalized')
    return p


def _kl(p, midpoint):
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive] / midpoint[positive])
    return terms.sum(axis=-1)


def sensitivity_metrics(labels, original_p, pert_p):
    """Per-target/repeat changes; positive changes mean perturbed minus original.

    Labels serve only retrospective evaluation here and never influence
    intervention generation. No repeat averaging or probability ensemble occurs.
    """
    p = _probabilities(original_p, 2, 'original_p')
    q = _probabilities(pert_p, 3, 'pert_p')
    if p.shape[0] != q.shape[0] or p.shape[1] != q.shape[2]:
        raise ValueError('Original and intervened sample/class dimensions disagree')
    y = np.asarray(labels)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.number):
        raise ValueError('Labels must have shape [N] with numeric class indices')
    if not np.isfinite(y).all() or np.any(y != np.floor(y)) or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError('Labels must be integer class indices within [0,C)')
    y = y.astype(np.int64)
    original_prediction, perturbed_prediction = p.argmax(-1), q.argmax(-1)
    original_confidence, perturbed_confidence = p.max(-1), q.max(-1)
    original_true_probability = p[np.arange(len(p)), y]
    perturbed_true_probability = q[np.arange(len(p))[:, None], np.arange(q.shape[1])[None, :], y[:, None]]
    expanded = np.broadcast_to(p[:, None, :], q.shape)
    midpoint = (expanded + q)/2
    original_correct = original_prediction == y
    perturbed_correct = perturbed_prediction == y[:, None]
    return dict(
        original_prediction=original_prediction,
        perturbed_prediction=perturbed_prediction,
        original_confidence=original_confidence,
        perturbed_confidence=perturbed_confidence,
        original_true_probability=original_true_probability,
        perturbed_true_probability=perturbed_true_probability,
        flip=perturbed_prediction != original_prediction[:, None],
        confidence_change=perturbed_confidence-original_confidence[:, None],
        true_probability_change=perturbed_true_probability-original_true_probability[:, None],
        js_divergence=(_kl(expanded, midpoint)+_kl(q, midpoint))/2,
        correct_to_wrong=original_correct[:, None] & ~perturbed_correct,
        wrong_to_correct=~original_correct[:, None] & perturbed_correct,
    )
