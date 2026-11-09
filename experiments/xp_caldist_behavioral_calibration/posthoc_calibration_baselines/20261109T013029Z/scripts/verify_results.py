#!/usr/bin/env python
"""Independent recomputation of saved temperatures, probabilities, metrics, and provenance."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from calibration_core import bootstrap_seed, canonical_sha256, fit_temperature, identity_hash, metrics, read_npz, sha256, temperature_log_probabilities  # noqa: E402


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    cfg = json.loads((ROOT / "configs/calibration_config.json").read_text())
    audit = json.loads((ROOT / "provenance/input_audit.json").read_text())
    frozen = json.loads((ROOT / "FROZEN_TEMPERATURES.json").read_text())
    completion = json.loads((ROOT / "TEST_EVALUATION_COMPLETE.json").read_text())
    parameter_rows = csv_rows(ROOT / "results/temperature_parameters.csv")
    test_rows = csv_rows(ROOT / "results/per_seed_test_metrics.csv")
    bootstrap_rows = csv_rows(ROOT / "results/paired_bootstrap_results.csv")
    checks = {
        "input_audit_passed": audit["status"] == "passed",
        "frozen_manifest_hash_valid": frozen["parameter_payload_sha256"] == canonical_sha256(frozen["parameters"]),
        "temperature_count_is_ten": len(frozen["parameters"]) == 10,
        "fit_declares_no_test_read": frozen["test_files_read_by_fit_script"] is False,
        "test_evaluation_once": completion["test_evaluation_passes"] == 1,
    }
    max_errors = {"temperature": 0.0, "validation_nll": 0.0, "probabilities": 0.0, "test_metric": 0.0,
                  "bootstrap_summary": 0.0}
    all_argmax = True
    all_classification = True
    for parameter in frozen["parameters"]:
        dataset, seed = parameter["dataset"], int(parameter["seed"])
        fmt = {"historical_prediction_root": cfg["paths"]["historical_prediction_root"], "dataset": dataset, "seed": seed}
        validation = read_npz(Path(cfg["paths"]["validation_template"].format(**fmt)))
        fit, _ = fit_temperature(validation["log_probabilities"], validation["labels"],
                                 cfg["temperature_fitting"]["bounds"], cfg["temperature_fitting"]["grid_size"],
                                 cfg["temperature_fitting"]["refinement_iterations"])
        max_errors["temperature"] = max(max_errors["temperature"], abs(fit["temperature"]-float(parameter["temperature"])))
        max_errors["validation_nll"] = max(max_errors["validation_nll"], abs(fit["validation_nll_after"]-float(parameter["validation_nll_after"])))
        test = read_npz(Path(cfg["paths"]["test_template"].format(**fmt)))
        recal_log = temperature_log_probabilities(test["log_probabilities"], float(parameter["temperature"]))
        recal = np.exp(recal_log)
        output_path = ROOT / "runs" / dataset / f"seed{seed}" / "calibrated_test_predictions.npz"
        with np.load(output_path, allow_pickle=False) as saved:
            max_errors["probabilities"] = max(max_errors["probabilities"], float(np.max(np.abs(recal-saved["calibrated_probabilities"]))))
            all_argmax &= np.array_equal(saved["original_predictions"], saved["calibrated_predictions"])
            all_argmax &= np.array_equal(test["probabilities"].argmax(1), recal.argmax(1))
            all_argmax &= str(saved["test_split_hash"]) == identity_hash(test)
        row = next(r for r in test_rows if r["dataset"] == dataset and int(r["seed"]) == seed)
        before, after = metrics(test["probabilities"], test["labels"]), metrics(recal, test["labels"])
        for metric in ("ece15", "brier", "nll", "mce15", "rmsce15", "ece10", "mean_confidence", "accuracy", "weighted_f1", "macro_f1"):
            max_errors["test_metric"] = max(max_errors["test_metric"], abs(before[metric]-float(row[f"{metric}_before"])), abs(after[metric]-float(row[f"{metric}_after"])))
        all_classification &= all(abs(before[m]-after[m]) < 1e-14 for m in ("accuracy", "weighted_f1", "macro_f1"))
        # Reproduce every paired dialogue bootstrap draw independently from the preregistered seed.
        dialogue_ids = np.asarray(test["dialogue_ids"], str)
        unique_dialogues = np.unique(dialogue_ids)
        rng = np.random.default_rng(bootstrap_seed(cfg["statistics"]["bootstrap_seed"], dataset, seed, "paired-test"))
        distributions = {name: [] for name in ("ece15", "brier", "nll", "mce15", "rmsce15", "ece10", "mean_confidence")}
        for _ in range(int(cfg["statistics"]["bootstrap_repeats"])):
            sampled = rng.choice(unique_dialogues, size=len(unique_dialogues), replace=True)
            indices = np.concatenate([np.flatnonzero(dialogue_ids == dialogue) for dialogue in sampled])
            m0, m1 = metrics(test["probabilities"][indices], test["labels"][indices]), metrics(recal[indices], test["labels"][indices])
            for name in distributions: distributions[name].append(m1[name]-m0[name])
        for name, values in distributions.items():
            saved = next(r for r in bootstrap_rows if r["dataset"] == dataset and int(r["seed"]) == seed and r["metric"] == name)
            values = np.asarray(values)
            recomputed = (values.mean(), np.quantile(values, .025), np.quantile(values, .975))
            recorded = tuple(float(saved[k]) for k in ("bootstrap_mean_difference", "ci95_lower", "ci95_upper"))
            max_errors["bootstrap_summary"] = max(max_errors["bootstrap_summary"], *(abs(a-b) for a,b in zip(recomputed, recorded)))
    checks.update({
        "independent_temperature_recomputation_exact": max_errors["temperature"] < 1e-12,
        "independent_validation_nll_recomputation_exact": max_errors["validation_nll"] < 1e-12,
        "saved_calibrated_probabilities_exact": max_errors["probabilities"] < 1e-14,
        "independent_test_metrics_match": max_errors["test_metric"] < 1e-12,
        "independent_dialogue_bootstrap_recomputation_exact": max_errors["bootstrap_summary"] < 1e-12,
        "all_argmax_unchanged": bool(all_argmax), "all_classification_metrics_unchanged": bool(all_classification),
        "all_temperatures_in_bounds": all(.05 <= float(r["temperature"]) <= 20 for r in frozen["parameters"]),
        "no_boundary_temperatures": all(not bool(r["boundary_optimum"]) for r in frozen["parameters"]),
    })
    source_hashes = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "inputs": {}, "outputs": {}}
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            fmt = {"xp04_root": cfg["paths"]["xp04_root"], "historical_prediction_root": cfg["paths"]["historical_prediction_root"], "dataset": dataset, "seed": seed}
            for name, template in (("checkpoint", "checkpoint_template"), ("validation_predictions", "validation_template"), ("test_predictions", "test_template")):
                path = Path(cfg["paths"][template].format(**fmt))
                source_hashes["inputs"][f"{dataset}/seed{seed}/{name}"] = {"path": str(path), "sha256": sha256(path)}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"verification.json", "source_hashes.json"} and "logs" not in path.relative_to(ROOT).parts and "__pycache__" not in path.relative_to(ROOT).parts:
            source_hashes["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    (ROOT / "provenance/source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    verification = {"status": "passed" if all(checks.values()) else "failed",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks, "maximum_absolute_errors": max_errors,
        "limitations": ["The dialogue bootstrap conditions on the fixed fitted temperature and fixed model predictions; it does not refit temperature inside each resample.",
                        "Five seeds share the same Test utterances and are summarized as model-seed replications, not independent datasets."]}
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    if verification["status"] != "passed":
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
