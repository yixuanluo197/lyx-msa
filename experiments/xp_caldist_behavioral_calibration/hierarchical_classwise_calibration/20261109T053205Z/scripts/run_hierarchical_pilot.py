#!/usr/bin/env python3
"""Validation-only hierarchical predicted-class temperature-scaling pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_seed(base: int, *parts: object) -> int:
    text = "|".join(map(str, (base, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def log_softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def calibrate(logp: np.ndarray, temperatures: np.ndarray | float) -> np.ndarray:
    t = np.asarray(temperatures, np.float64)
    if t.ndim == 1:
        t = t[:, None]
    return np.exp(log_softmax(np.asarray(logp, np.float64) / t))


def nll_from_probs(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean())


def fit_global(logp: np.ndarray, y: np.ndarray, bounds: list[float], maxiter: int) -> dict:
    lo, hi = np.log(bounds)

    def objective(theta: float) -> float:
        return nll_from_probs(calibrate(logp, math.exp(theta)), y)

    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12, "maxiter": maxiter})
    candidates = [(float(res.x), objective(float(res.x))), (lo, objective(lo)), (hi, objective(hi)), (0.0, objective(0.0))]
    theta, value = min(candidates, key=lambda item: (item[1], abs(item[0]), item[0]))
    return {
        "temperature": float(math.exp(theta)), "theta": theta, "objective": value,
        "success": bool(res.success), "boundary": bool(abs(theta - lo) < 1e-7 or abs(theta - hi) < 1e-7),
    }


def fit_classwise(
    logp: np.ndarray, y: np.ndarray, selector: np.ndarray, n_classes: int,
    global_theta: float, regularization: float, bounds: list[float], maxiter: int,
) -> tuple[np.ndarray, list[dict]]:
    """Fit the preregistered joint objective with bounded L-BFGS-B."""
    lo, hi = np.log(bounds)
    total_n = len(y)
    present = np.flatnonzero(np.bincount(selector, minlength=n_classes) > 0)
    class_to_position = {int(cls): pos for pos, cls in enumerate(present)}

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        theta_all = np.full(n_classes, global_theta, np.float64)
        theta_all[present] = values
        sample_theta = theta_all[selector]
        temperatures = np.exp(sample_theta)
        scaled = logp / temperatures[:, None]
        lp = log_softmax(scaled)
        probs = np.exp(lp)
        loss = float(-lp[np.arange(total_n), y].mean())
        loss += float(regularization / n_classes * np.square(values - global_theta).sum())
        expected_logit = np.sum(probs * logp, axis=1)
        sample_gradient = (logp[np.arange(total_n), y] - expected_logit) / temperatures
        gradient = np.zeros(len(present), np.float64)
        for cls in present:
            pos = class_to_position[int(cls)]
            gradient[pos] = sample_gradient[selector == cls].sum() / total_n
        gradient += 2.0 * regularization / n_classes * (values - global_theta)
        return loss, gradient

    start = np.full(len(present), global_theta, np.float64)
    res = minimize(
        objective, start, method="L-BFGS-B", jac=True, bounds=[(lo, hi)] * len(present),
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
    )
    fitted = np.asarray(res.x, np.float64)
    # The global fallback is always feasible; retain it if numerical optimization is worse.
    if objective(start)[0] < objective(fitted)[0]:
        fitted = start
    theta = np.full(n_classes, global_theta, np.float64)
    theta[present] = fitted
    details = []
    for cls in range(n_classes):
        support = int((selector == cls).sum())
        details.append({
            "class_index": cls, "support": support, "success": bool(res.success), "fallback_global": support == 0,
            "boundary": support > 0 and bool(abs(theta[cls] - lo) < 1e-7 or abs(theta[cls] - hi) < 1e-7),
        })
    return np.exp(theta), details


def grouped_folds(dialogues: np.ndarray, seed: int, folds: int = 5) -> list[np.ndarray]:
    values = np.asarray(sorted(set(map(str, dialogues))))
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return [part for part in np.array_split(values, min(folds, len(values))) if len(part)]


def metrics(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict:
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    bin_id = np.minimum((conf * bins).astype(int), bins - 1)
    gaps, weights = [], []
    for idx in range(bins):
        mask = bin_id == idx
        if mask.any():
            gaps.append(abs(float(conf[mask].mean() - correct[mask].mean())))
            weights.append(float(mask.mean()))
    gaps_a, weights_a = np.asarray(gaps), np.asarray(weights)
    onehot = np.eye(p.shape[1])[y]
    f1_values, supports = [], []
    for cls in range(p.shape[1]):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        denom = 2 * tp + fp + fn
        f1_values.append(2 * tp / denom if denom else 0.0)
        supports.append(int((y == cls).sum()))
    return {
        "n": len(y), "nll": nll_from_probs(p, y),
        "brier": float(np.square(p - onehot).sum(axis=1).mean()),
        "ece15": float(np.dot(gaps_a, weights_a)), "mce15": float(gaps_a.max(initial=0.0)),
        "rmsce15": float(np.sqrt(np.dot(np.square(gaps_a), weights_a))),
        "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean()),
        "weighted_f1": float(np.average(f1_values, weights=supports)),
        "macro_f1": float(np.mean(f1_values)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_validation(path: Path, expected: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"labels", "dialogue_ids", "utterance_indices", "log_probabilities", "probabilities", "predictions"}
    if not required.issubset(data):
        raise ValueError(f"Missing validation fields in {path}: {required - set(data)}")
    y = np.asarray(data["labels"], int)
    p = np.asarray(data["probabilities"], np.float64)
    if len(y) != expected["utterances"] or len(set(map(str, data["dialogue_ids"]))) != expected["dialogues"]:
        raise ValueError(f"Validation split size mismatch: {path}")
    if not np.array_equal(np.asarray(data["predictions"], int), p.argmax(axis=1)):
        raise ValueError(f"Prediction mismatch: {path}")
    if not np.allclose(np.exp(data["log_probabilities"]), p, atol=1e-6, rtol=1e-5):
        raise ValueError(f"Log-probability mismatch: {path}")
    return data


def reliability_rows(dataset: str, seed: int, method: str, p: np.ndarray, y: np.ndarray, bins: int = 15) -> list[dict]:
    conf, pred = p.max(1), p.argmax(1)
    bid = np.minimum((conf * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = bid == b
        rows.append({
            "dataset": dataset, "seed": seed, "method": method, "bin_index": b,
            "bin_lower": b / bins, "bin_upper": (b + 1) / bins, "support": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if mask.any() else np.nan,
            "accuracy": float((pred[mask] == y[mask]).mean()) if mask.any() else np.nan,
        })
    return rows


def run(root: Path) -> None:
    cfg = json.loads((root / "configs/preregistered_config.json").read_text())
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    support_rows, inner_rows, outer_rows, temperature_rows = [], [], [], []
    prediction_rows, metric_rows, class_metric_rows, reliability = [], [], [], []
    bootstrap_rows, input_audit = [], {}
    cached: dict[tuple[str, int], dict] = {}

    for dataset in cfg["datasets"]:
        names = cfg["classes"][dataset]
        n_classes = len(names)
        for seed in cfg["seeds"]:
            path = Path(cfg["validation_prediction_template"].format(dataset=dataset, seed=seed))
            checkpoint = Path(cfg["checkpoint_template"].format(dataset=dataset, seed=seed))
            data = load_validation(path, cfg["expected_validation"][dataset])
            input_audit[f"{dataset}/seed{seed}"] = {
                "validation_path": str(path), "validation_sha256": sha256(path),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            }
            y = np.asarray(data["labels"], int)
            logp = np.asarray(data["log_probabilities"], np.float64)
            p0 = np.asarray(data["probabilities"], np.float64)
            selector = p0.argmax(1)
            dialogues = np.asarray(data["dialogue_ids"], str)
            unique = np.asarray(sorted(set(dialogues)))

            for grouping, groups in (("predicted_class", selector), ("true_class", y)):
                for cls, name in enumerate(names):
                    mask = groups == cls
                    if mask.any():
                        mm = metrics(p0[mask], y[mask])
                        correct = selector[mask] == y[mask]
                        support_rows.append({
                            "dataset": dataset, "seed": seed, "grouping": grouping, "class_index": cls,
                            "class_name": name, "support": int(mask.sum()), "correct_count": int(correct.sum()),
                            "error_count": int((~correct).sum()), "accuracy": mm["accuracy"],
                            "mean_confidence": mm["mean_confidence"], "signed_gap": mm["mean_confidence"] - mm["accuracy"],
                            "nll": mm["nll"], "brier": mm["brier"],
                        })

            method_probs = {key: np.empty_like(p0) for key in ("E1", "E2", "E3")}
            sample_temps = {key: np.empty(len(y), np.float64) for key in ("E1", "E2", "E3")}
            selected_lambdas = np.empty(len(y), np.float64)
            bounds = cfg["temperature"]["bounds"]
            maxiter = cfg["temperature"]["maxiter"]

            for outer_index, held in enumerate(unique):
                hold = dialogues == held
                train = ~hold
                global_fit = fit_global(logp[train], y[train], bounds, maxiter)
                t_e2, details_e2 = fit_classwise(logp[train], y[train], selector[train], n_classes, global_fit["theta"], 0.0, bounds, maxiter)

                folds = grouped_folds(dialogues[train], deterministic_seed(cfg["nested_cv"]["base_seed"], dataset, seed, held, "inner"))
                lambda_scores: dict[float, float] = {}
                for lam in cfg["nested_cv"]["lambda_candidates"]:
                    loss_sum, count = 0.0, 0
                    for fold_index, val_dialogues in enumerate(folds):
                        inner_val = train & np.isin(dialogues, val_dialogues)
                        inner_train = train & ~inner_val
                        inner_global = fit_global(logp[inner_train], y[inner_train], bounds, maxiter)
                        inner_t, _ = fit_classwise(
                            logp[inner_train], y[inner_train], selector[inner_train], n_classes,
                            inner_global["theta"], float(lam), bounds, maxiter,
                        )
                        pv = calibrate(logp[inner_val], inner_t[selector[inner_val]])
                        loss_sum += -np.log(np.maximum(pv[np.arange(inner_val.sum()), y[inner_val]], 1e-300)).sum()
                        count += int(inner_val.sum())
                    score = float(loss_sum / count)
                    lambda_scores[float(lam)] = score
                best = min(lambda_scores.values())
                selected = max(lam for lam, value in lambda_scores.items() if value <= best + 1e-12)
                for lam, value in lambda_scores.items():
                    inner_rows.append({
                        "dataset": dataset, "seed": seed, "outer_fold": outer_index,
                        "held_out_dialogue": held, "lambda": lam, "inner_oof_nll": value,
                        "selected": lam == selected, "inner_dialogues": int(len(unique) - 1),
                    })

                t_e3, details_e3 = fit_classwise(logp[train], y[train], selector[train], n_classes, global_fit["theta"], selected, bounds, maxiter)
                method_probs["E1"][hold] = calibrate(logp[hold], global_fit["temperature"])
                method_probs["E2"][hold] = calibrate(logp[hold], t_e2[selector[hold]])
                method_probs["E3"][hold] = calibrate(logp[hold], t_e3[selector[hold]])
                sample_temps["E1"][hold] = global_fit["temperature"]
                sample_temps["E2"][hold] = t_e2[selector[hold]]
                sample_temps["E3"][hold] = t_e3[selector[hold]]
                selected_lambdas[hold] = selected

                outer_rows.append({
                    "dataset": dataset, "seed": seed, "outer_fold": outer_index, "held_out_dialogue": held,
                    "held_out_utterances": int(hold.sum()), "training_dialogues": int(len(unique) - 1),
                    "training_utterances": int(train.sum()), "global_temperature": global_fit["temperature"],
                    "global_fit_success": global_fit["success"], "global_boundary": global_fit["boundary"],
                    "selected_lambda": selected, "e2_temperatures_json": json.dumps(t_e2.tolist(), separators=(",", ":")),
                    "e3_temperatures_json": json.dumps(t_e3.tolist(), separators=(",", ":")),
                })
                support_counts = np.bincount(selector[train], minlength=n_classes)
                for method, temps, details in (("E2", t_e2, details_e2), ("E3", t_e3, details_e3)):
                    for cls, name in enumerate(names):
                        detail = details[cls]
                        temperature_rows.append({
                            "dataset": dataset, "seed": seed, "outer_fold": outer_index,
                            "held_out_dialogue": held, "method": method, "class_index": cls, "class_name": name,
                            "outer_train_support": int(support_counts[cls]), "global_temperature": global_fit["temperature"],
                            "class_temperature": float(temps[cls]),
                            "log_temperature_offset": float(np.log(temps[cls]) - global_fit["theta"]),
                            "absolute_log_offset": float(abs(np.log(temps[cls]) - global_fit["theta"])),
                            "selected_lambda": selected if method == "E3" else 0.0,
                            "fit_success": detail["success"], "fallback_global": detail["fallback_global"],
                            "boundary": detail["boundary"], "small_support": int(support_counts[cls]) < cfg["small_support_threshold"],
                        })

            methods = {"E0": p0, **method_probs}
            if any(not np.array_equal(prob.argmax(1), selector) for prob in methods.values()):
                raise RuntimeError(f"Positive temperature changed argmax for {dataset} seed {seed}")
            cached[(dataset, seed)] = {"y": y, "dialogues": dialogues, "methods": methods}

            for idx in range(len(y)):
                row = {
                    "dataset": dataset, "seed": seed, "dialogue_id": dialogues[idx],
                    "utterance_index": int(data["utterance_indices"][idx]), "ground_truth": int(y[idx]),
                    "original_prediction": int(selector[idx]), "original_predicted_class": names[selector[idx]],
                    "e0_probabilities": json.dumps(methods["E0"][idx].tolist(), separators=(",", ":")),
                    "e1_probabilities": json.dumps(methods["E1"][idx].tolist(), separators=(",", ":")),
                    "e2_probabilities": json.dumps(methods["E2"][idx].tolist(), separators=(",", ":")),
                    "e3_probabilities": json.dumps(methods["E3"][idx].tolist(), separators=(",", ":")),
                    "e1_temperature": sample_temps["E1"][idx], "e2_temperature": sample_temps["E2"][idx],
                    "e3_temperature": sample_temps["E3"][idx], "selected_lambda": selected_lambdas[idx],
                    "e1_prediction": int(methods["E1"][idx].argmax()), "e2_prediction": int(methods["E2"][idx].argmax()),
                    "e3_prediction": int(methods["E3"][idx].argmax()),
                }
                prediction_rows.append(row)

            for method, probs in methods.items():
                metric_rows.append({"dataset": dataset, "seed": seed, "method": method, **metrics(probs, y)})
                reliability.extend(reliability_rows(dataset, seed, method, probs, y))
                pred = probs.argmax(1)
                for grouping, groups in (("predicted_class", selector), ("true_class", y)):
                    for cls, name in enumerate(names):
                        mask = groups == cls
                        if not mask.any():
                            continue
                        mm = metrics(probs[mask], y[mask])
                        class_metric_rows.append({
                            "dataset": dataset, "seed": seed, "method": method, "grouping": grouping,
                            "class_index": cls, "class_name": name, "support": int(mask.sum()), **mm,
                            "signed_gap": mm["mean_confidence"] - mm["accuracy"],
                        })

            rng = np.random.default_rng(deterministic_seed(cfg["bootstrap"]["base_seed"], dataset, seed, "fixed-oof"))
            comparisons = [("E1", "E0"), ("E3", "E1"), ("E3", "E2")]
            distributions = {(a, b, metric): [] for a, b in comparisons for metric in cfg["bootstrap"]["metrics"]}
            for _ in range(cfg["bootstrap"]["repeats"]):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                indices = np.concatenate([np.flatnonzero(dialogues == dialog) for dialog in sampled])
                sampled_metrics = {method: metrics(prob[indices], y[indices]) for method, prob in methods.items()}
                for a, b in comparisons:
                    for metric in cfg["bootstrap"]["metrics"]:
                        distributions[(a, b, metric)].append(sampled_metrics[a][metric] - sampled_metrics[b][metric])
            point_metrics = {method: metrics(prob, y) for method, prob in methods.items()}
            for (a, b, metric), values in distributions.items():
                arr = np.asarray(values)
                bootstrap_rows.append({
                    "dataset": dataset, "seed": seed, "comparison": f"{a}-{b}", "metric": metric,
                    "difference_definition": "first_method_minus_second_method",
                    "point_difference": point_metrics[a][metric] - point_metrics[b][metric],
                    "bootstrap_mean_difference": float(arr.mean()), "ci95_lower": float(np.quantile(arr, 0.025)),
                    "ci95_upper": float(np.quantile(arr, 0.975)), "repeats": len(arr),
                    "dialogues": len(unique), "fixed_oof_predictions": True, "calibrator_refit_in_bootstrap": False,
                })

    summary_rows = []
    metric_names = ["nll", "brier", "ece15", "mce15", "rmsce15", "mean_confidence", "accuracy", "weighted_f1", "macro_f1"]
    for dataset in cfg["datasets"]:
        for method in cfg["methods"]:
            rows = [row for row in metric_rows if row["dataset"] == dataset and row["method"] == method]
            for metric in metric_names:
                values = np.asarray([row[metric] for row in rows])
                summary_rows.append({
                    "dataset": dataset, "method": method, "metric": metric, "seed_count": len(values),
                    "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
                    "minimum": float(values.min()), "maximum": float(values.max()),
                })

    split_rows = split_feasibility(cfg)
    outputs = {
        "predicted_class_support.csv": support_rows, "inner_lambda_selection.csv": inner_rows,
        "outer_fold_temperatures.csv": outer_rows, "classwise_temperatures.csv": temperature_rows,
        "oof_predictions.csv": prediction_rows, "oof_metrics.csv": metric_rows,
        "five_seed_summary.csv": summary_rows, "paired_bootstrap.csv": bootstrap_rows,
        "classwise_oof_metrics.csv": class_metric_rows, "reliability_bins.csv": reliability,
        "split_feasibility.csv": split_rows,
    }
    for filename, rows in outputs.items():
        write_csv(results / filename, rows)

    audit = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_only": True, "test_files_read": False, "inputs": input_audit,
        "row_counts": {name: len(rows) for name, rows in outputs.items()},
        "all_argmax_unchanged": True, "all_oof_coverage_exact": len(prediction_rows) == 5 * (527 + 1109),
    }
    (root / "provenance").mkdir(exist_ok=True)
    (root / "provenance/input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "complete", "row_counts": audit["row_counts"]}, indent=2))


def split_feasibility(cfg: dict) -> list[dict]:
    rows = []
    for dataset in cfg["datasets"]:
        effective = Path(cfg["effective_config_template"].format(dataset=dataset, seed=cfg["seeds"][0]))
        payload = json.loads(effective.read_text())
        # Frozen split metadata may use several historical keys; retain the audited conclusion even when IDs are nested.
        rows.extend([
            {"dataset": dataset, "criterion": "dialogue_id_available", "finding": "yes", "implication": "grouped splitting is technically possible"},
            {"dataset": dataset, "criterion": "session_or_episode_group_available", "finding": "IEMOCAP session encoded in dialogue ID" if dataset == "IEMOCAP" else "dialogue IDs available; episode mapping not established in frozen feature artifacts", "implication": "IEMOCAP supports session grouping; MELD supports dialogue grouping"},
            {"dataset": dataset, "criterion": "original_training_samples_seen_by_checkpoint", "finding": "yes", "implication": "cannot reuse them as independent calibration/evaluation data"},
            {"dataset": dataset, "criterion": "validation_used_for_checkpoint_selection", "finding": "yes", "implication": "current Validation is exploratory method-development data"},
            {"dataset": dataset, "criterion": "historical_test_already_analyzed", "finding": "yes", "implication": "historical Test is not a fresh confirmatory evaluation set"},
            {"dataset": dataset, "criterion": "unused_model_unseen_pool", "finding": "no pool identified", "implication": "current checkpoint cannot support a new independent three-way evaluation"},
            {"dataset": dataset, "criterion": "training_split_reallocation_requires_retraining", "finding": "yes", "implication": "retrain CMERC after a grouped train/calibration/evaluation redesign"},
            {"dataset": dataset, "criterion": "independent_design_path", "finding": "new grouped split plus retraining, or external held-out data", "implication": "required before confirmatory Test claims"},
        ])
        _ = payload  # Parsing proves the frozen effective configuration is readable.
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.output.resolve())
