#!/usr/bin/env python3
"""Fit frozen CMERC Calibration probabilities; this command has no Evaluation role.

E2 lambda is selected by grouped, utterance-weighted OOF NLL. Its OOF score is
a selection diagnostic, not an unbiased estimate of generalization. Final
parameters are refitted on all supervision-eligible Calibration utterances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar


BOUNDS = (0.05, 20.0)
LAMBDA_CANDIDATES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
METHODS = {"E0": "vanilla", "E1": "global_ts", "E2": "hierarchical_predicted_class_ts",
           "E3": "unregularized_predicted_class_ts"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def probabilities(logits, temperature=1.0):
    temperature = np.asarray(temperature, dtype=np.float64)
    if not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("temperatures must be finite and positive")
    if temperature.ndim == 1:
        temperature = temperature[:, None]
    return np.exp(log_softmax(np.asarray(logits) / temperature))


def nll(logits, labels, temperature=1.0):
    temperature = np.asarray(temperature)
    if temperature.ndim == 1:
        temperature = temperature[:, None]
    lp = log_softmax(logits / temperature)
    return float(-lp[np.arange(len(labels)), labels].mean())


def fit_global(logits, labels, maxiter=1000):
    lo, hi = np.log(BOUNDS)
    fit_curve = []

    def objective(theta):
        value = nll(logits, labels, np.exp(theta))
        fit_curve.append({"evaluation": len(fit_curve), "theta": float(theta),
                          "temperature": float(np.exp(theta)), "objective": value})
        return value
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-12, "maxiter": maxiter})
    theta = min((float(result.x), float(lo), float(hi), 0.0),
                key=lambda t: (objective(t), abs(t), t))
    temperature = float(np.clip(np.exp(theta), *BOUNDS))
    return {"temperature": temperature, "theta": float(np.log(temperature)),
            "objective": objective(theta), "success": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit),
            "boundary": bool(np.isclose(theta, lo, atol=1e-7) or np.isclose(theta, hi, atol=1e-7)),
            "fit_curve": fit_curve}


def fit_predicted_class(logits, labels, selector, global_theta, regularization, maxiter=1000):
    """Mean NLL + lambda/K sum_k (log T_k - log T_global)^2.

    The selector is always the original predicted class. Classes with no
    Calibration support use the global fit, including for lambda=0.
    """
    classes = logits.shape[1]
    counts = np.bincount(selector, minlength=classes)
    present = np.flatnonzero(counts)
    lo, hi = np.log(BOUNDS)
    fit_curve = []

    def objective(values):
        theta = np.full(classes, global_theta)
        theta[present] = values
        t = np.exp(theta[selector])
        lp = log_softmax(logits / t[:, None])
        p = np.exp(lp)
        loss = -lp[np.arange(len(labels)), labels].mean()
        loss += regularization / classes * np.square(values - global_theta).sum()
        row_gradient = (logits[np.arange(len(labels)), labels] - (p * logits).sum(1)) / t
        gradient = np.bincount(selector, weights=row_gradient, minlength=classes)[present] / len(labels)
        gradient += 2 * regularization / classes * (values - global_theta)
        fit_curve.append({"evaluation": len(fit_curve), "objective": float(loss),
                          "gradient_norm": float(np.linalg.norm(gradient)),
                          "log_temperatures": np.asarray(values, float).tolist(),
                          "temperatures": np.exp(np.asarray(values, float)).tolist()})
        return float(loss), gradient

    start = np.full(len(present), global_theta)
    result = minimize(objective, start, method="L-BFGS-B", jac=True,
                      bounds=[(lo, hi)] * len(present),
                      options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50})
    values = result.x
    retained_global = not np.isfinite(values).all() or objective(values)[0] > objective(start)[0]
    if retained_global:
        values = start
    theta = np.full(classes, global_theta)
    theta[present] = values
    temps = np.clip(np.exp(theta), *BOUNDS)
    return {"temperatures": temps.tolist(), "log_temperatures": np.log(temps).tolist(),
            "support": counts.tolist(), "fallback_global": (counts == 0).tolist(),
            "lambda": float(regularization), "global_theta": float(global_theta),
            "objective": objective(values)[0], "success": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit),
            "retained_global_solution": bool(retained_global),
            "boundary": (np.isclose(theta, lo, atol=1e-7) | np.isclose(theta, hi, atol=1e-7)).tolist(),
            "fit_curve": fit_curve}


def metrics(p, labels, log_probabilities=None):
    labels = np.asarray(labels, dtype=int)
    conf, pred = p.max(1), p.argmax(1)
    ids = np.minimum((conf * 15).astype(int), 14)
    ece = sum(float((ids == b).mean()) * abs(float(conf[ids == b].mean()) -
              float((pred[ids == b] == labels[ids == b]).mean())) for b in range(15) if (ids == b).any())
    lp = np.log(np.maximum(p, 1e-300)) if log_probabilities is None else log_probabilities
    return {"n": len(labels), "nll": float(-lp[np.arange(len(labels)), labels].mean()),
            "brier": float(np.square(p - np.eye(p.shape[1])[labels]).sum(1).mean()),
            "ece15": ece, "accuracy": float((pred == labels).mean())}


def grouping_ids(dialogue_ids, dataset, episode_mapping):
    if dataset == "IEMOCAP":
        return np.asarray(dialogue_ids, dtype=str)
    if dataset != "MELD":
        raise ValueError("dataset must be IEMOCAP or MELD")
    if not episode_mapping or set(map(str, dialogue_ids)) - episode_mapping.keys():
        raise ValueError("MELD requires a complete dialogue-to-episode mapping")
    groups = [episode_mapping[str(d)] for d in dialogue_ids]
    if any(value is None or not str(value).strip() for value in groups):
        raise ValueError("invalid MELD episode mapping")
    return np.asarray(groups, dtype=str)


def grouped_folds(groups, folds=5, seed=20261109):
    unique = np.asarray(sorted(set(groups)))
    if len(unique) < 2 or folds < 2:
        raise ValueError("grouped CV requires at least two groups and two folds")
    np.random.default_rng(seed).shuffle(unique)
    return [part for part in np.array_split(unique, min(folds, len(unique))) if len(part)]


def fit_calibrators(logits, labels, dialogue_ids, supervision_mask, *, dataset,
                    episode_mapping=None, cv_folds=5, seed=20261109, maxiter=1000):
    logits = np.asarray(logits, dtype=np.float64)
    labels, dialogue_ids, mask = np.asarray(labels), np.asarray(dialogue_ids, dtype=str), np.asarray(supervision_mask)
    if logits.ndim != 2 or logits.shape[1] < 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite N x K matrix")
    n = len(logits)
    if any(value.shape != (n,) for value in (labels, dialogue_ids, mask)):
        raise ValueError("unaligned labels, dialogue IDs, or mask")
    if mask.dtype != np.bool_:
        raise ValueError("supervision mask must be boolean")
    if not mask.any() or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("no eligible labels or labels are not integer")
    y = labels[mask].astype(int)
    if (y < 0).any() or (y >= logits.shape[1]).any():
        raise ValueError("eligible labels outside class range")
    all_groups = grouping_ids(dialogue_ids, dataset, episode_mapping)
    groups, z = all_groups[mask], logits[mask]
    selector = z.argmax(1)
    folds = grouped_folds(groups, cv_folds, seed)
    candidate_oof = {lam: np.empty_like(z) for lam in LAMBDA_CANDIDATES}
    candidate_logp = {lam: np.empty_like(z) for lam in LAMBDA_CANDIDATES}
    oof = {m: np.empty_like(z) for m in METHODS}
    oof_logp = {m: np.empty_like(z) for m in METHODS}
    oof["E0"] = probabilities(z)
    oof_logp["E0"] = log_softmax(z)
    assignments = np.full(len(y), -1, int)
    cv_details = []
    for fold, held in enumerate(folds):
        val = np.isin(groups, held)
        train = ~val
        g = fit_global(z[train], y[train], maxiter)
        u = fit_predicted_class(z[train], y[train], selector[train], g["theta"], 0.0, maxiter)
        oof["E1"][val] = probabilities(z[val], g["temperature"])
        oof["E3"][val] = probabilities(z[val], np.asarray(u["temperatures"])[selector[val]])
        oof_logp["E1"][val] = log_softmax(z[val] / g["temperature"])
        oof_logp["E3"][val] = log_softmax(z[val] / np.asarray(u["temperatures"])[selector[val], None])
        fits = {}
        for lam in LAMBDA_CANDIDATES:
            fitted = fit_predicted_class(z[train], y[train], selector[train], g["theta"], lam, maxiter)
            candidate_oof[lam][val] = probabilities(z[val], np.asarray(fitted["temperatures"])[selector[val]])
            candidate_logp[lam][val] = log_softmax(z[val] / np.asarray(fitted["temperatures"])[selector[val], None])
            fits[str(lam)] = fitted
        assignments[val] = fold
        cv_details.append({"fold": fold, "train_groups": sorted(set(groups[train])),
                           "held_groups": sorted(map(str, held)), "train_n": int(train.sum()),
                           "held_n": int(val.sum()), "global_fit": g, "unregularized_fit": u,
                           "hierarchical_fits": fits})
    scores = {lam: metrics(p, y, candidate_logp[lam])["nll"] for lam, p in candidate_oof.items()}
    best = min(scores.values())
    selected = max(lam for lam, score in scores.items() if score <= best + 1e-12)
    oof["E2"] = candidate_oof[selected]
    oof_logp["E2"] = candidate_logp[selected]
    g = fit_global(z, y, maxiter)
    params = {"E0": {"temperature": 1.0}, "E1": g,
              "E2": fit_predicted_class(z, y, selector, g["theta"], selected, maxiter),
              "E3": fit_predicted_class(z, y, selector, g["theta"], 0.0, maxiter)}
    full_selector = logits.argmax(1)
    temperatures = {"E0": np.ones(n), "E1": np.full(n, g["temperature"])}
    temperatures.update({m: np.asarray(params[m]["temperatures"])[full_selector] for m in ("E2", "E3")})
    full = {m: probabilities(logits, temperatures[m]) for m in METHODS}
    full_logp = {m: log_softmax(logits / temperatures[m][:, None]) for m in METHODS}
    if any(not np.array_equal(p.argmax(1), full_selector) for p in full.values()):
        raise RuntimeError("positive scalar temperature changed original argmax")
    if any(not np.array_equal(p.argmax(1), selector) for p in oof.values()):
        raise RuntimeError("OOF temperature changed original argmax")
    full_oof = {m: np.full_like(logits, np.nan) for m in METHODS}
    for m in METHODS:
        full_oof[m][mask] = oof[m]
    full_assignments = np.full(n, -1, int)
    full_assignments[mask] = assignments
    return {"parameters": params, "probabilities": full, "temperatures": temperatures,
            "calibrated_log_probabilities": full_logp,
            "input_scores": logits, "log_probabilities": log_softmax(logits),
            "metrics": {m: metrics(p[mask], y, full_logp[m][mask]) for m, p in full.items()},
            "oof_probabilities": full_oof, "oof_fold": full_assignments, "group_ids": all_groups,
            "oof_metrics": {m: metrics(p, y, oof_logp[m]) for m, p in oof.items()},
            "cv": {"lambda_oof_nll": {str(k): v for k, v in scores.items()}, "selected_lambda": selected,
                   "folds": cv_details, "diagnostic_only": True,
                   "selection_bias_note": "E2 uses the same grouped OOF predictions for lambda selection and diagnostics; not unbiased evaluation."},
            "config": {"methods": METHODS, "bounds": list(BOUNDS), "lambda_candidates": list(LAMBDA_CANDIDATES),
                       "cv_folds_requested": cv_folds, "cv_folds_actual": len(folds), "cv_seed": seed,
                       "maxiter": maxiter, "dataset": dataset, "grouping": "dialogue" if dataset == "IEMOCAP" else "episode",
                       "fit_role": "calibration", "fit_n": int(mask.sum()), "masked_n": int((~mask).sum()),
                       "regularizer": "lambda / K * sum((log(T_class) - log(T_global)) ** 2)",
                       "lambda_tie_break": "largest lambda within 1e-12 of minimum utterance-weighted OOF NLL"}}


def validate_role_artifact(payload, manifest=None, manifest_path=None):
    if payload.get("role") != "calibration":
        raise ValueError("only the Calibration role artifact is accepted")
    if payload.get("dataset") not in ("IEMOCAP", "MELD") or not payload.get("fold"):
        raise ValueError("Calibration artifact requires dataset and fold")
    ids, records = payload.get("dialogue_ids", []), payload.get("dialogues", {})
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(records):
        raise ValueError("Calibration artifact must exactly cover unique declared dialogues")
    for dialogue in ids:
        record = records[dialogue]
        labels = np.asarray(record["labels"])
        mask = np.asarray(record["supervision_mask"])
        if labels.ndim != 1 or len(labels) == 0 or mask.shape != labels.shape or mask.dtype != np.bool_:
            raise ValueError("invalid Calibration labels or boolean mask")
        if record.get("dialogue_id") != dialogue:
            raise ValueError("Calibration record ID mismatch")
    if manifest is not None:
        if manifest.get("status") != "approved" or canonical_sha256({k: v for k, v in manifest.items() if k != "content_sha256"}) != manifest.get("content_sha256"):
            raise ValueError("approved manifest status/content hash mismatch")
        role = manifest["protocols"][payload["dataset"]]["folds"][payload["fold"]]["roles"]["calibration"]
        if canonical_sha256({k: v for k, v in role.items() if k != "sha256"}) != role.get("sha256"):
            raise ValueError("Calibration role hash mismatch")
        if payload.get("approved_role") != role or ids != role["dialogue_ids"]:
            raise ValueError("artifact differs from approved Calibration role")
        provenance = payload.get("provenance", {})
        if provenance.get("role_sha256") != role["sha256"]:
            raise ValueError("Calibration role provenance hash mismatch")
        if manifest_path and (payload.get("approved_manifest_sha256") != sha256(manifest_path)
                              or provenance.get("approved_manifest_sha256") != sha256(manifest_path)):
            raise ValueError("Calibration artifact manifest file hash mismatch")
        unknown = {(r["dialogue_id"], r["utterance_index"]) for r in manifest["unknown_label_mask"]["utterances"]} if payload["dataset"] == "IEMOCAP" else set()
        total = eligible = 0
        for dialogue in ids:
            record = records[dialogue]
            labels, mask = np.asarray(record["labels"]), np.asarray(record["supervision_mask"])
            expected = np.asarray([(dialogue, i) not in unknown for i in range(len(labels))])
            if not np.array_equal(mask, expected) or np.any(labels[~mask] != -100):
                raise ValueError("Calibration unknown mask differs from approved manifest")
            total += len(labels)
            eligible += int(mask.sum())
        if (total, eligible, total - eligible) != (role["utterance_count"], role["supervision_eligible_count"], role["unknown_masked_count"]):
            raise ValueError("Calibration artifact counts differ from approved role")
    return payload


def validate_checkpoint(checkpoint, artifact=None, manifest_path=None):
    metadata = {**checkpoint.get("metadata", {}), **checkpoint}
    if "model_state_dict" not in checkpoint and "model_state" not in checkpoint:
        raise ValueError("checkpoint has no model state")
    if not ((metadata.get("is_best") is True or metadata.get("best_checkpoint") is True)
            and isinstance(metadata.get("selected_epoch"), int) and metadata["selected_epoch"] > 0
            and metadata.get("training_complete") is True and metadata.get("complete") is True
            and metadata.get("epoch") == metadata["selected_epoch"]
            and metadata.get("selection_metric") == "development_weighted_f1"
            and metadata.get("selection_role") == "development"):
        raise ValueError("checkpoint must be selected as best before Calibration starts")
    for key in ("dataset", "fold"):
        if artifact is not None and metadata.get(key) != artifact[key]:
            raise ValueError(f"checkpoint {key} differs from Calibration artifact")
    if manifest_path and checkpoint.get("bindings", {}).get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("checkpoint approved manifest hash mismatch")
    return {key: metadata.get(key) for key in ("dataset", "fold", "selected_epoch", "is_best", "best_checkpoint", "training_complete", "complete")}


def artifact_rows(artifact):
    ids, indices, labels, masks = [], [], [], []
    for dialogue in artifact["dialogue_ids"]:
        r = artifact["dialogues"][dialogue]
        y, mask = np.asarray(r["labels"]), np.asarray(r["supervision_mask"])
        ids.extend([dialogue] * len(y))
        indices.extend(range(len(y)))
        labels.extend(y.tolist())
        masks.extend(mask.tolist())
    return {"dialogue_ids": np.asarray(ids, str), "utterance_indices": np.asarray(indices, int),
            "labels": np.asarray(labels, int), "supervision_mask": np.asarray(masks, bool)}


def load_prediction_npz(path, artifact, checkpoint_path):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if str(data.get("role", "")) != "calibration":
        raise ValueError("prediction NPZ must declare Calibration role")
    if str(data.get("checkpoint_sha256", "")) != sha256(checkpoint_path):
        raise ValueError("prediction checkpoint hash differs from selected checkpoint")
    expected = artifact_rows(artifact)
    for key, value in expected.items():
        if key not in data or not np.array_equal(data[key], value):
            raise ValueError(f"prediction rows differ from Calibration artifact: {key}")
    if "logits" not in data and "log_probabilities" not in data:
        raise ValueError("prediction NPZ requires logits or log_probabilities")
    return np.asarray(data.get("logits", data.get("log_probabilities")), float)


def save_results(output, result, rows, provenance):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen calibration output: {output}")
    output.mkdir(parents=True)
    payload = {**rows, "original_predictions": result["probabilities"]["E0"].argmax(1),
               "input_scores": result["input_scores"], "log_probabilities": result["log_probabilities"],
               "group_ids": result["group_ids"], "oof_fold": result["oof_fold"], "role": np.asarray("calibration")}
    for method in METHODS:
        payload[f"{method}_probabilities"] = result["probabilities"][method]
        payload[f"{method}_log_probabilities"] = result["calibrated_log_probabilities"][method]
        payload[f"{method}_temperatures"] = result["temperatures"][method]
        payload[f"{method}_predictions"] = result["probabilities"][method].argmax(1)
        payload[f"{method}_oof_probabilities"] = result["oof_probabilities"][method]
    np.savez_compressed(output / "calibration_probabilities.npz", **payload)
    for name, value in (("parameters", result["parameters"]), ("calibration_metrics", result["metrics"]),
                        ("oof_diagnostic", {"metrics": result["oof_metrics"], **result["cv"]}),
                        ("effective_config", result["config"]), ("provenance", provenance)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Approved Protocol B manifest binding the Calibration role")
    parser.add_argument("--prediction-npz", type=Path, help="Previously exported Calibration-only predictions bound to checkpoint hash")
    parser.add_argument("--episode-mapping", type=Path, help="JSON object mapping MELD canonical dialogue ID to episode ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--maxiter", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen calibration output: {args.output}")
    # Torch and frozen model imports are intentionally absent from fitting APIs.
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Refuse unfinished/unselected checkpoints before opening Calibration labels.
    validate_checkpoint(checkpoint, manifest_path=args.manifest)
    manifest = json.loads(args.manifest.read_text())
    artifact = validate_role_artifact(torch.load(args.calibration_artifact, map_location="cpu", weights_only=False), manifest, args.manifest)
    checkpoint_metadata = validate_checkpoint(checkpoint, artifact, args.manifest)
    del checkpoint
    from cmerc_runner import validate_artifact
    validate_artifact(artifact, manifest, "calibration", args.manifest)
    rows = artifact_rows(artifact)
    if args.prediction_npz:
        logits = load_prediction_npz(args.prediction_npz, artifact, args.checkpoint)
    else:
        from cmerc_runner import predict_role_artifact
        predicted = predict_role_artifact(args.checkpoint, args.calibration_artifact, args.manifest, device=args.device)
        for key in ("labels", "dialogue_ids", "supervision_mask"):
            if not np.array_equal(predicted[key], rows[key]):
                raise ValueError(f"predictor row order differs from Calibration artifact: {key}")
        logits = predicted["logits"]
    if np.asarray(logits).shape != (len(rows["labels"]), 6 if artifact["dataset"] == "IEMOCAP" else 7):
        raise ValueError("prediction dimensions differ from the Calibration dataset")
    mapping = json.loads(args.episode_mapping.read_text()) if args.episode_mapping else artifact.get("dialogue_groups")
    if args.episode_mapping and artifact["dataset"] == "MELD" and artifact.get("dialogue_groups"):
        if any(mapping.get(d) != artifact["dialogue_groups"][d] for d in artifact["dialogue_ids"]):
            raise ValueError("supplied episode mapping differs from Calibration artifact")
    result = fit_calibrators(logits, rows["labels"], rows["dialogue_ids"], rows["supervision_mask"],
                            dataset=artifact["dataset"], episode_mapping=mapping,
                            cv_folds=args.cv_folds, seed=args.seed, maxiter=args.maxiter)
    provenance = {"checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
                  "checkpoint_metadata": checkpoint_metadata,
                  "calibration_artifact": str(args.calibration_artifact.resolve()),
                  "calibration_artifact_sha256": sha256(args.calibration_artifact),
                  "approved_manifest_sha256": sha256(args.manifest),
                  "prediction_npz_sha256": sha256(args.prediction_npz) if args.prediction_npz else None,
                  "episode_mapping_sha256": sha256(args.episode_mapping) if args.episode_mapping else None,
                  "episode_mapping_source": "explicit_json" if args.episode_mapping else "calibration_artifact.dialogue_groups",
                  "script_sha256": sha256(__file__), "role": "calibration"}
    save_results(args.output, result, rows, provenance)
    print(json.dumps({"output": str(args.output), "selected_lambda": result["cv"]["selected_lambda"],
                      "fit_n": result["config"]["fit_n"], "role": "calibration"}))


if __name__ == "__main__":
    main()
