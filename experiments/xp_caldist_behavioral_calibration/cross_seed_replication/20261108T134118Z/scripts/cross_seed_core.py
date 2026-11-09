"""Pure helpers for preregistered cross-seed CMERC representation interventions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_INDICES = {
    "T1": (0,), "T2": (1,), "T3": (2,), "T4": (3,),
    "T-All": (0, 1, 2, 3), "V": (4,), "A": (5,),
}


def fraction(value: float, name: str = "value") -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return value


def target_indices(target: str) -> list[int]:
    if target not in TARGET_INDICES:
        raise ValueError(f"unknown target: {target}")
    return list(TARGET_INDICES[target])


def native_key_map(keys):
    """Map serialized string IDs back to unique native dataset keys."""
    mapping = {}
    for key in keys:
        serialized = str(key)
        if serialized in mapping and mapping[serialized] != key:
            raise ValueError(f"dataset keys collide after string serialization: {serialized}")
        mapping[serialized] = key
    if len(mapping) != len(keys):
        raise ValueError("dataset keys are not unique")
    return mapping


def contract(config: dict) -> tuple[list[dict], list[dict]]:
    """Return the fixed ten main conditions and three five-repeat donor conditions."""
    main_cfg = config.get("main_conditions", config.get("mean_reference_conditions", {}))
    rho = main_cfg.get("rho", main_cfg.get("overall_rho", main_cfg.get("strengths", [0.5, 1.0])))
    overall = main_cfg.get("targets", main_cfg.get("overall_targets", ["T-All", "A", "V"]))
    arrays = main_cfg.get("text_arrays", main_cfg.get("text_array_targets", ["T1", "T2", "T3", "T4"]))
    array_values = main_cfg.get("text_array_rho", [1.0])
    array_rho = fraction(array_values[0] if isinstance(array_values, list) else array_values, "text_array_rho")
    main = []
    for target in overall:
        target_indices(target)
        for value in rho:
            value = fraction(value, "rho")
            main.append({"id": f"decay_{target}_rho{value:.2f}".replace(".", "p"),
                         "target": target, "kind": "train_mean_contraction",
                         "strength": value, "repeats": 1})
    for target in arrays:
        target_indices(target)
        main.append({"id": f"decay_{target}_rho{array_rho:.2f}".replace(".", "p"),
                     "target": target, "kind": "train_mean_contraction",
                     "strength": array_rho, "repeats": 1})
    if len(main) != 10 or len({x["id"] for x in main}) != 10:
        raise ValueError("preregistered main contract must contain exactly ten unique conditions")
    donor_cfg = config.get("donor", config.get("donor_reference", config.get("donor_reference_conditions", {})))
    donor_targets = donor_cfg.get("targets", ["T-All", "A", "V"])
    donor_value = donor_cfg.get("rho", 1.0)
    donor_rho = fraction(donor_value[0] if isinstance(donor_value, list) else donor_value, "donor rho")
    repeats = int(donor_cfg.get("repeats", 5))
    if donor_rho != 1 or repeats != 5:
        raise ValueError("donor contract requires rho=1 and five fixed donors")
    donor = [{"id": f"donor_{target}_rho1p00", "target": target,
              "kind": "random_training_donor", "strength": 1.0, "repeats": 5}
             for target in donor_targets]
    if len(donor) != 3:
        raise ValueError("donor contract requires T-All, A and V")
    return main, donor


def _vectors(vectors, references):
    if not vectors or len(vectors) != len(references):
        raise ValueError("vectors/references must be nonempty and aligned")
    result = []
    for vector, reference in zip(vectors, references):
        x = np.asarray(vector, dtype=np.float64)
        r = np.asarray(reference, dtype=np.float64)
        if x.ndim != 1 or x.shape != r.shape or not np.isfinite(x).all() or not np.isfinite(r).all():
            raise ValueError("invalid intervention vector/reference")
        result.append((x, r))
    return result


def interpolate(vectors, references, rho):
    rho = fraction(rho, "rho")
    output = []
    for x, ref in _vectors(vectors, references):
        changed = x if rho == 0 else ref if rho == 1 else (1-rho)*x + rho*ref
        changed = np.asarray(changed, dtype=np.float32)
        if not np.isfinite(changed).all():
            raise ValueError("intervention exceeded finite FP32 range")
        output.append(changed)
    return output


def deterministic_rng(dataset, dialogue_id, utterance_index, target, rho, repeat):
    """Donor RNG shared across model seeds and independent of all labels.

    Canonical key is frozen by preregistration. Sharing it across model seeds
    prevents donor identity from becoming a seed-specific nuisance variable.
    """
    key = [str(dataset), str(dialogue_id), int(utterance_index), str(target),
           fraction(rho, "rho"), int(repeat), "donor"]
    canonical = json.dumps(key, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    seed = int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(seed))


def donor_index(n_train, dataset, dialogue_id, utterance_index, target, rho, repeat):
    n_train = int(n_train)
    if n_train <= 0:
        raise ValueError("training donor pool is empty")
    return int(deterministic_rng(dataset, dialogue_id, utterance_index,
                                 target, rho, repeat).integers(n_train))


def donor_modality_missing(target, donor_vectors):
    """All-zero status for the donor representation actually being replaced."""
    expected = len(target_indices(target))
    if len(donor_vectors) != expected:
        raise ValueError("donor vector count does not match target")
    vectors = [np.asarray(value) for value in donor_vectors]
    if any(value.ndim != 1 for value in vectors):
        raise ValueError("donor representations must be vectors")
    # T-All is missing only when all four stored arrays contain no signal.
    return bool(all(np.all(value == 0) for value in vectors))


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def js_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    midpoint = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p*np.log(p/midpoint), 0).sum(-1)
        right = np.where(q > 0, q*np.log(q/midpoint), 0).sum(-1)
    return (left + right) / 2


def behavior(original, perturbed):
    """Label-free, per-sample behavior averaged over repeats."""
    p, q = np.asarray(original, float), np.asarray(perturbed, float)
    if p.ndim != 2 or q.ndim != 3 or q.shape[::2] != p.shape:
        raise ValueError("expected original [N,C] and perturbed [N,R,C]")
    if not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=1e-5) or not np.allclose(q.sum(-1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError("probabilities are not normalized")
    op, pp = p.argmax(-1), q.argmax(-1)
    oc, pc = p.max(-1), q.max(-1)
    flip = pp != op[:, None]
    return {
        "flip_rate": flip.mean(1),
        "prediction_instability_mu": (flip*pc).mean(1),
        "absolute_confidence_change": np.abs(pc.mean(1)-oc),
        "confidence_mad": np.abs(pc-oc[:, None]).mean(1),
        "js_divergence": js_divergence(p[:, None, :], q).mean(1),
    }
