#!/usr/bin/env python
"""Independent integrity and metric verifier for the cross-seed experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cross_seed_core import contract, sha256


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def binary(y, score):
    y, score = np.asarray(y, int), np.clip(np.asarray(score, float), 1e-7, 1-1e-7)
    return {"auroc_success": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "auprc_success": average_precision_score(y, score) if y.sum() else np.nan,
            "binary_brier": np.mean((score-y)**2),
            "binary_nll": -np.mean(y*np.log(score)+(1-y)*np.log(1-score))}


def close(actual, expected, name, atol=1e-7):
    # Saved probabilities are float32, while the analyzer and verifier use
    # slightly different NumPy reduction orders.  Keep this many orders below
    # the preregistered inference tolerance (1e-6) without demanding bitwise
    # identity from floating-point means.
    if np.isnan(expected) and np.isnan(actual): return
    if not np.isclose(actual, expected, atol=atol, rtol=1e-9):
        raise ValueError(f"{name}: {actual} != {expected}")


def condition_path(root, cfg, dataset, seed, spec, phase):
    if phase == "main" and seed == cfg["anchors_reused_without_rerun"].get(dataset):
        return Path(cfg["historical_evidence_sensitivity"]) / f"runs/{dataset}/seed{seed}/conditions/{spec['id']}.npz"
    return root / f"runs/{dataset}/seed{seed}/conditions/{spec['id']}.npz"


def run(root):
    cfg_path = root / "configs/preregistered_config.json"
    cfg = json.loads(cfg_path.read_text())
    main, donor = contract(cfg)
    summary = csv_rows(root / "results/per_seed_sensitivity.csv")
    donor_rows = csv_rows(root / "results/donor_reference_results.csv")
    oof = csv_rows(root / "results/correctness_oof_predictions.csv")
    oof_metrics = csv_rows(root / "results/correctness_oof_metrics.csv")
    bootstrap = csv_rows(root / "results/paired_bootstrap_results.csv")
    checks = {"conditions": 0, "condition_vectors": 0, "summary_rows_recomputed": 0,
              "oof_metric_rows_recomputed": 0, "bootstrap_rows_recomputed": 0,
              "donor_identity_cross_seed_checks": 0}
    donor_reference = {}
    condition_hashes = {}
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            base_path = root / f"runs/{dataset}/seed{seed}/baseline_validation.npz"
            if not base_path.exists():
                base_path = Path(cfg["historical_pilot"]) / f"runs/{dataset}/seed{seed}/validation_predictions.npz"
            base = npz(base_path); p = base["probabilities"]; y = base["labels"].astype(int)
            for phase, specs in (("main", main), ("donor", donor)):
                for spec in specs:
                    path = condition_path(root, cfg, dataset, seed, spec, phase)
                    digest = sha256(path)
                    condition_hashes[str(path)] = digest
                    is_reused_anchor = phase == "main" and seed == cfg["anchors_reused_without_rerun"].get(dataset)
                    if not is_reused_anchor:
                        completion = json.loads((root / f"runs/{dataset}/seed{seed}/{phase}_complete.json").read_text())
                        if not completion["passed"] or completion["conditions"].get(spec["id"]) != digest:
                            raise ValueError(f"condition completion/hash mismatch: {path}")
                    data = npz(path); q = data["probabilities"]
                    if q.shape != (len(p), spec["repeats"], p.shape[1]) or not np.isfinite(q).all():
                        raise ValueError(f"invalid condition probabilities: {path}")
                    if not np.allclose(q.sum(-1), 1, atol=1e-6, rtol=1e-5):
                        raise ValueError(f"unnormalized condition: {path}")
                    for key in ("labels", "dialogue_ids", "utterance_indices"):
                        if not np.array_equal(data[key], base[key]): raise ValueError(f"metadata mismatch {path}: {key}")
                    missing = data["visual_missing"]
                    active = ~missing if spec["target"] == "V" else np.ones(len(p), bool)
                    if not np.array_equal(data["active"], active): raise ValueError(f"active mismatch: {path}")
                    if not np.array_equal(q[~active], np.repeat(p[~active, None], spec["repeats"], 1)):
                        raise ValueError(f"inactive visual outputs changed: {path}")
                    checks["conditions"] += 1; checks["condition_vectors"] += q.shape[0]*q.shape[1]
                    op, pp = p.argmax(1), q.argmax(2)
                    original_correct, changed_correct = op == y, pp == y[:, None]
                    midpoint = (p[:, None]+q)/2
                    with np.errstate(divide="ignore", invalid="ignore"):
                        js = (np.where(p[:, None] > 0, p[:, None]*np.log(p[:, None]/midpoint), 0).sum(-1)
                              + np.where(q > 0, q*np.log(q/midpoint), 0).sum(-1))/2
                    computed = {"original_accuracy": original_correct.mean(), "perturbed_accuracy": changed_correct.mean(),
                                "delta_accuracy": changed_correct.mean()-original_correct.mean(),
                                "flip_rate": (pp != op[:, None]).mean(),
                                "prediction_instability_mu": ((pp != op[:, None])*q.max(-1)).mean(),
                                "absolute_confidence_change": np.abs(q.max(-1).mean(1)-p.max(1)).mean(),
                                "confidence_mad": np.abs(q.max(-1)-p.max(1)[:, None]).mean(),
                                "js_divergence": js.mean(),
                                "true_probability_change": (q[np.arange(len(y))[:, None], np.arange(q.shape[1])[None], y[:, None]]-p[np.arange(len(y)), y][:, None]).mean(),
                                "maximum_confidence_change": (q.max(-1)-p.max(1)[:, None]).mean(),
                                "correct_to_wrong_rate": (original_correct[:, None] & ~changed_correct).mean(),
                                "wrong_to_correct_rate": (~original_correct[:, None] & changed_correct).mean()}
                    row = next(item for item in summary if item["dataset"] == dataset and int(item["seed"]) == seed
                               and item["condition_id"] == spec["id"] and item["subgroup"] == "all")
                    for name, value in computed.items(): close(float(row[name]), float(value), f"{dataset}/{seed}/{spec['id']}/{name}")
                    checks["summary_rows_recomputed"] += 1
                    if phase == "donor":
                        identity = (data["donor_flat_indices"], data["donor_dialogue_ids"], data["donor_utterance_indices"], data["donor_modality_missing"])
                        key = (dataset, spec["id"])
                        if key not in donor_reference: donor_reference[key] = identity
                        else:
                            if not all(np.array_equal(a, b) for a, b in zip(donor_reference[key], identity)):
                                raise ValueError(f"donor identities differ across model seeds: {key}")
                            checks["donor_identity_cross_seed_checks"] += 1
                        if spec["target"] == "V":
                            for subgroup, mask in (("all", np.ones_like(data["donor_modality_missing"], bool)),
                                                   ("donor_visual_present", ~data["donor_modality_missing"]),
                                                   ("donor_visual_missing", data["donor_modality_missing"])):
                                mask &= active[:, None]
                                found = next(item for item in donor_rows if item["dataset"] == dataset and int(item["seed"]) == seed and item["subgroup"] == subgroup)
                                close(float(found["flip_rate"]), float((pp != op[:, None])[mask].mean()), "donor flip")
                                close(float(found["js_divergence"]), float(js[mask].mean()), "donor js")

    # Recompute every fixed-OOF metric from public OOF rows.
    for row in oof_metrics:
        selected = [item for item in oof if item["dataset"] == row["dataset"] and item["seed"] == row["seed"] and item["model"] == row["model"]]
        y = np.asarray([int(item["correctness"]) for item in selected]); score = np.asarray([float(item["oof_success_probability"]) for item in selected])
        values = binary(y, score)
        for name, value in values.items(): close(float(row[name]), float(value), f"OOF {row['dataset']}/{row['seed']}/{row['model']}/{name}")
        checks["oof_metric_rows_recomputed"] += 1

    # Recompute each reported fixed-score paired bootstrap CI with the frozen seed.
    repeats, seed = cfg["statistics"]["bootstrap_repeats"], cfg["statistics"]["bootstrap_seed"]
    for group in {(r["dataset"], r["seed"], r["model"], r["comparator"]) for r in bootstrap}:
        dataset, model_seed, model, comparator = group
        candidate = [r for r in oof if r["dataset"] == dataset and r["seed"] == model_seed and r["model"] == model]
        baseline = [r for r in oof if r["dataset"] == dataset and r["seed"] == model_seed and r["model"] == comparator]
        if [(r["dialogue_id"], r["utterance_index"], r["correctness"]) for r in candidate] != [(r["dialogue_id"], r["utterance_index"], r["correctness"]) for r in baseline]:
            raise ValueError("OOF identity/order mismatch")
        y = np.asarray([int(r["correctness"]) for r in candidate]); dialogues = np.asarray([r["dialogue_id"] for r in candidate])
        cscore = np.asarray([float(r["oof_success_probability"]) for r in candidate]); bscore = np.asarray([float(r["oof_success_probability"]) for r in baseline])
        unique = np.unique(dialogues); rng = np.random.default_rng(seed); draws = {name: [] for name in binary(y, bscore)}
        for _ in range(repeats):
            indices = np.concatenate([np.flatnonzero(dialogues == value) for value in rng.choice(unique, len(unique), replace=True)])
            before, after = binary(y[indices], bscore[indices]), binary(y[indices], cscore[indices])
            for name in draws:
                value = after[name]-before[name]
                if np.isfinite(value): draws[name].append(value)
        for name, values in draws.items():
            row = next(r for r in bootstrap if (r["dataset"], r["seed"], r["model"], r["comparator"], r["metric"]) == (*group, name))
            low, high = np.quantile(values, [.025, .975])
            close(float(row["ci_low"]), low, "bootstrap low"); close(float(row["ci_high"]), high, "bootstrap high")
            if int(row["valid_bootstraps"]) != len(values): raise ValueError("bootstrap valid count mismatch")
            checks["bootstrap_rows_recomputed"] += 1

    scripts = sorted((root / "scripts").glob("*.py"))
    historical = [Path(cfg["historical_pilot"]) / "verification.json",
                  Path(cfg["historical_evidence_sensitivity"]) / "verification.json"]
    checkpoints = [Path(cfg["xp04"]) / f"runs/cmerc/{dataset}_validation_seed{seed}/best.pt"
                   for dataset in cfg["datasets"] for seed in cfg["seeds"]]
    baselines = [Path(cfg["historical_pilot"]) / f"runs/{dataset}/seed{seed}/validation_predictions.npz"
                 for dataset in cfg["datasets"] for seed in cfg["seeds"]]
    statistics = [Path(cfg["historical_pilot"]) / f"provenance/{dataset}_train_statistics.npz"
                  for dataset in cfg["datasets"]]
    historical_anchor_conditions = [condition_path(root, cfg, dataset,
        cfg["anchors_reused_without_rerun"][dataset], spec, "main")
        for dataset in cfg["datasets"] for spec in main]
    source_hashes = {str(path): sha256(path) for path in [cfg_path, *scripts, *historical,
                                                          *checkpoints, *baselines, *statistics,
                                                          *historical_anchor_conditions]}
    provenance = root / "provenance"; provenance.mkdir(exist_ok=True)
    (provenance / "source_hashes.json").write_text(json.dumps({"sha256": source_hashes}, indent=2) + "\n")
    results = sorted((root / "results").glob("*.csv"))
    figures = sorted((root / "figures").glob("*")) if (root / "figures").exists() else []
    verification = {"status": "passed", "validation_only": True, "test_split_used": False,
                    "models_retrained": False, "oof_models_refit_inside_bootstrap": False,
                    "checks": checks, "source_hashes_sha256": sha256(provenance / "source_hashes.json"),
                    "condition_sha256": condition_hashes,
                    "result_sha256": {str(path.relative_to(root)): sha256(path) for path in results},
                    "figure_sha256": {str(path.relative_to(root)): sha256(path) for path in figures if path.is_file()},
                    "limitations": ["Dialogue bootstrap resamples fixed OOF predictions; logistic models are not refit inside bootstrap.",
                                    "Five model seeds share the same Validation utterances and are not treated as independent datasets."]}
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args(); run(arguments.root.resolve())
