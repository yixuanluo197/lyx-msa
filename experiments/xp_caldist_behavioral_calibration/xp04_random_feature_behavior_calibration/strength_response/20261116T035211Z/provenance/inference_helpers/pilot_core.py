"""Pure NumPy perturbation and behavioral metrics for the validation-only pilot.

Probabilities are never renormalized or clipped: saved FP32 model outputs remain
the observations. Calculations use float64 and tolerate small normalization
roundoff. JS divergence uses natural logarithms (nats).
"""

import hashlib
import json

import numpy as np


def rng_for(dataset, seed, dialogue_id, utterance_index, modality, kind, strength, repeat):
    """Replayable PCG64 stream from the preregistered canonical JSON key.

    SHA256 input is UTF-8 JSON without spaces, with Unicode preserved and no
    nonfinite numbers. The first 16 digest bytes form an unsigned big-endian seed.
    Text layers consume independent sequential draws in their original order.
    """
    key = [str(dataset), int(seed), str(dialogue_id), int(utterance_index),
           str(modality), str(kind), float(strength), int(repeat)]
    canonical = json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    seed_value = int.from_bytes(hashlib.sha256(canonical.encode('utf-8')).digest()[:16], 'big')
    return np.random.Generator(np.random.PCG64(seed_value))


def perturb_vectors(vectors, stds, kind, strength, rng):
    """Return new FP32 feature vectors; never modify inputs or any other target.

    Gaussian strength is a multiplier of each dimension's training population
    standard deviation. Dropout zeros coordinates without rescaling survivors.
    Attenuation strength is the retained fraction, e.g. 0.95.
    """
    strength = float(strength)
    if not np.isfinite(strength):
        raise ValueError('Perturbation strength must be finite')
    if kind not in {'gaussian', 'dropout', 'attenuation'}:
        raise ValueError('Unknown perturbation kind: ' + str(kind))
    if strength < 0 or (kind != 'gaussian' and strength > 1):
        raise ValueError('Invalid perturbation strength')
    if len(vectors) != len(stds) or not len(vectors):
        raise ValueError('vectors and stds must be nonempty lists of equal length')
    result = []
    for vector, std in zip(vectors, stds):
        x = np.asarray(vector, dtype=np.float64)
        s = np.asarray(std, dtype=np.float64)
        if x.ndim != 1 or s.shape != x.shape:
            raise ValueError('Each vector and std must be matching 1D arrays')
        if not np.isfinite(x).all() or not np.isfinite(s).all() or np.any(s < 0):
            raise ValueError('Vectors and standard deviations must be finite; std >= 0')
        if kind == 'gaussian':
            changed = x + strength * s * rng.normal(size=x.shape)
        elif kind == 'dropout':
            changed = x * (rng.random(size=x.shape) >= strength)
        else:
            changed = x * strength
        with np.errstate(over='ignore', invalid='ignore'):
            changed = changed.astype(np.float32)
        if not np.isfinite(changed).all():
            raise ValueError('Perturbation overflowed FP32 features')
        result.append(changed)
    return result


def _validate_probabilities(prob, ndim, name):
    values = np.asarray(prob, dtype=np.float64)
    if values.ndim != ndim or any(size == 0 for size in values.shape) or values.shape[-1] < 2:
        raise ValueError(name + ' has an invalid probability shape')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(name + ' must contain finite nonnegative probabilities')
    if not np.allclose(values.sum(axis=-1), 1., atol=1e-6, rtol=1e-5):
        raise ValueError(name + ' probabilities are not normalized')
    return values


def _kl_terms(p, midpoint):
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive] / midpoint[positive])
    return terms.sum(axis=-1)


def behavior_metrics(original_prob, perturbed_prob):
    """Compare N original distributions with N x R perturbed distributions.

    Returns per-utterance float64 arrays. mu weights flips by the perturbed
    maximum confidence. delta_original is abs(c - mean(c')), confidence_mad is
    mean(abs(c - c')), and confidence_change is mean(c') - c.
    """
    p = _validate_probabilities(original_prob, 2, 'original_prob')
    q = _validate_probabilities(perturbed_prob, 3, 'perturbed_prob')
    if p.shape[0] != q.shape[0] or p.shape[1] != q.shape[2]:
        raise ValueError('Original and perturbed sample/class dimensions disagree')
    confidence = p.max(axis=-1)
    perturbed_confidence = q.max(axis=-1)
    flips = q.argmax(axis=-1) != p.argmax(axis=-1)[:, None]
    expanded = np.broadcast_to(p[:, None, :], q.shape)
    midpoint = (expanded + q) / 2
    js = (_kl_terms(expanded, midpoint) + _kl_terms(q, midpoint)) / 2
    change = perturbed_confidence.mean(axis=1) - confidence
    return {
        'flip_rate': flips.mean(axis=1, dtype=np.float64),
        'mu': (flips * perturbed_confidence).mean(axis=1),
        'delta_original': np.abs(change),
        'confidence_mad': np.abs(confidence[:, None] - perturbed_confidence).mean(axis=1),
        'js_divergence': js.mean(axis=1),
        'confidence_change': change,
    }
