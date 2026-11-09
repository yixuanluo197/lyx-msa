#!/usr/bin/env python
"""Shared deterministic utilities for Global Temperature Scaling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "labels", "dialogue_ids", "utterance_indices", "speaker_ids",
    "batch_indices", "dialogue_lengths", "log_probabilities",
    "probabilities", "predictions",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"missing keys in {path}: {missing}")
    validate_prediction_payload(payload, path)
    return payload


def validate_prediction_payload(payload: dict[str, np.ndarray], path: Path | str = "payload") -> None:
    labels = np.asarray(payload["labels"], dtype=int)
    probabilities = np.asarray(payload["probabilities"], dtype=float)
    log_probabilities = np.asarray(payload["log_probabilities"], dtype=float)
    predictions = np.asarray(payload["predictions"], dtype=int)
    n = len(labels)
    if probabilities.ndim != 2 or probabilities.shape != log_probabilities.shape or probabilities.shape[0] != n:
        raise ValueError(f"invalid probability shape: {path}")
    for key in ("dialogue_ids", "utterance_indices", "speaker_ids", "batch_indices", "dialogue_lengths", "predictions"):
        if len(payload[key]) != n:
            raise ValueError(f"metadata length mismatch for {key}: {path}")
    if not np.isfinite(probabilities).all() or not np.isfinite(log_probabilities).all():
        raise ValueError(f"non-finite probabilities: {path}")
    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-6, rtol=1e-5):
        raise ValueError(f"probabilities do not sum to one: {path}")
    if not np.array_equal(predictions, probabilities.argmax(1)):
        raise ValueError(f"saved predictions disagree with argmax: {path}")
    if not np.allclose(np.exp(log_probabilities), probabilities, atol=1e-6, rtol=1e-5):
        raise ValueError(f"log probabilities disagree with probabilities: {path}")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= probabilities.shape[1]:
        raise ValueError(f"labels outside class range: {path}")


def identity_records(payload: dict[str, np.ndarray]) -> list[list[object]]:
    return [[str(d), int(u), int(y)] for d, u, y in zip(
        payload["dialogue_ids"], payload["utterance_indices"], payload["labels"]
    )]


def identity_hash(payload: dict[str, np.ndarray]) -> str:
    return canonical_sha256(identity_records(payload))


def stable_log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = values.max(axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def temperature_log_probabilities(log_probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return stable_log_softmax(np.asarray(log_probabilities, dtype=np.float64) / float(temperature))


def nll_from_log_probabilities(log_probabilities: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    return float(-np.asarray(log_probabilities, dtype=np.float64)[np.arange(len(labels)), labels].mean())


def nll_at_temperature(log_probabilities: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    return nll_from_log_probabilities(temperature_log_probabilities(log_probabilities, temperature), labels)


def fit_temperature(log_probabilities: np.ndarray, labels: np.ndarray, bounds=(0.05, 20.0), grid_size=2001,
                    refinement_iterations=128) -> tuple[dict, list[dict]]:
    lower, upper = map(float, bounds)
    if not (0 < lower < 1 < upper) or grid_size < 1001 or grid_size % 2 != 1:
        raise ValueError("invalid preregistered grid")
    log_grid = np.linspace(np.log(lower), np.log(upper), int(grid_size), dtype=np.float64)
    temperatures = np.exp(log_grid)
    if not np.isclose(temperatures[grid_size // 2], 1.0, atol=1e-15, rtol=0):
        raise ValueError("grid does not contain T=1 exactly enough")
    temperatures[grid_size // 2] = 1.0
    nlls = np.asarray([nll_at_temperature(log_probabilities, labels, value) for value in temperatures])
    best = int(np.argmin(nlls))
    boundary = best in (0, len(temperatures)-1)
    if boundary:
        optimum_log = log_grid[best]
        status = "boundary_lower" if best == 0 else "boundary_upper"
    else:
        left, right = float(log_grid[best-1]), float(log_grid[best+1])
        ratio = (np.sqrt(5.0)-1.0)/2.0
        c = right-ratio*(right-left); d = left+ratio*(right-left)
        fc = nll_at_temperature(log_probabilities, labels, np.exp(c))
        fd = nll_at_temperature(log_probabilities, labels, np.exp(d))
        for _ in range(int(refinement_iterations)):
            if fc <= fd:
                right, d, fd = d, c, fc
                c = right-ratio*(right-left)
                fc = nll_at_temperature(log_probabilities, labels, np.exp(c))
            else:
                left, c, fc = c, d, fd
                d = left+ratio*(right-left)
                fd = nll_at_temperature(log_probabilities, labels, np.exp(d))
        candidates = [
            (float(np.exp((left+right)/2)), nll_at_temperature(log_probabilities, labels, np.exp((left+right)/2))),
            (float(temperatures[best]), float(nlls[best])),
            (1.0, nll_at_temperature(log_probabilities, labels, 1.0)),
        ]
        candidates.sort(key=lambda item: (item[1], abs(np.log(item[0])), item[0]))
        optimum_log = float(np.log(candidates[0][0])); status = "converged_interior"
    temperature = float(np.exp(optimum_log))
    before = nll_at_temperature(log_probabilities, labels, 1.0)
    after = nll_at_temperature(log_probabilities, labels, temperature)
    curve = [{"grid_index": i, "temperature": float(t), "log_temperature": float(lt),
              "validation_nll": float(value), "point_type": "grid",
              "is_grid_minimum": i == best}
             for i, (t, lt, value) in enumerate(zip(temperatures, log_grid, nlls))]
    curve.append({"grid_index": -1, "temperature": temperature, "log_temperature": optimum_log,
                  "validation_nll": after, "point_type": "refined_optimum",
                  "is_grid_minimum": False})
    result = {
        "temperature": temperature,
        "validation_nll_before": before,
        "validation_nll_after": after,
        "nll_improvement": before-after,
        "optimization_method": "2001-point log grid plus deterministic golden-section refinement",
        "search_bounds": [lower, upper],
        "search_grid_size": int(grid_size),
        "grid_best_index": best,
        "grid_best_temperature": float(temperatures[best]),
        "grid_best_nll": float(nlls[best]),
        "convergence_status": status,
        "boundary_optimum": bool(boundary),
        "refinement_iterations": int(refinement_iterations),
    }
    return result, curve


def calibration_bins(probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> list[dict]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    indices = np.minimum((confidence * n_bins).astype(int), n_bins-1)
    rows = []
    for index in range(n_bins):
        selected = indices == index
        count = int(selected.sum())
        if count:
            mean_conf = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            gap = mean_conf-accuracy
        else:
            mean_conf = accuracy = gap = np.nan
        rows.append({"bin": index, "lower": index/n_bins, "upper": (index+1)/n_bins,
                     "count": count, "mean_confidence": mean_conf, "accuracy": accuracy,
                     "signed_gap": gap, "absolute_gap": abs(gap) if count else np.nan})
    return rows


def classification_metrics(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    predictions, labels = np.asarray(predictions, int), np.asarray(labels, int)
    f1s, supports = [], []
    for cls in range(n_classes):
        tp = int(((predictions == cls) & (labels == cls)).sum())
        fp = int(((predictions == cls) & (labels != cls)).sum())
        fn = int(((predictions != cls) & (labels == cls)).sum())
        denominator = 2*tp+fp+fn
        f1s.append(0.0 if denominator == 0 else 2*tp/denominator)
        supports.append(int((labels == cls).sum()))
    return {"accuracy": float((predictions == labels).mean()),
            "weighted_f1": float(np.average(f1s, weights=supports)),
            "macro_f1": float(np.mean(f1s))}


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(1)
    confidence = probabilities.max(1)
    result = classification_metrics(predictions, labels, probabilities.shape[1])
    onehot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    result.update({
        "mean_confidence": float(confidence.mean()),
        "brier": float(np.square(probabilities-onehot).sum(1).mean()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean()),
    })
    for n_bins, suffix in ((15, "15"), (10, "10")):
        bins = calibration_bins(probabilities, labels, n_bins)
        weights = np.asarray([row["count"]/len(labels) for row in bins])
        gaps = np.asarray([0.0 if row["count"] == 0 else row["absolute_gap"] for row in bins])
        result[f"ece{suffix}"] = float(np.sum(weights*gaps))
        if n_bins == 15:
            result["mce15"] = float(max(row["absolute_gap"] for row in bins if row["count"]))
            result["rmsce15"] = float(np.sqrt(np.sum(weights*np.square(gaps))))
    return result


def bootstrap_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
