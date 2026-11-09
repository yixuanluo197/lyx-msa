#!/usr/bin/env python
"""Apply frozen validation temperatures once to Test and compute paired statistics."""
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
from calibration_core import bootstrap_seed, calibration_bins, canonical_sha256, identity_hash, metrics, read_npz, sha256, temperature_log_probabilities  # noqa: E402

METRICS = ["ece15", "brier", "nll", "mce15", "rmsce15", "ece10", "mean_confidence",
           "accuracy", "weighted_f1", "macro_f1"]
BOOTSTRAP_METRICS = ["ece15", "brier", "nll", "mce15", "rmsce15", "ece10", "mean_confidence"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    config = json.loads((ROOT / "configs/calibration_config.json").read_text())
    frozen = json.loads((ROOT / "FROZEN_TEMPERATURES.json").read_text())
    if frozen["status"] != "frozen" or frozen["parameter_payload_sha256"] != canonical_sha256(frozen["parameters"]):
        raise SystemExit("frozen temperature manifest is invalid")
    temperatures = {(r["dataset"], int(r["seed"])): float(r["temperature"]) for r in frozen["parameters"]}
    paths = config["paths"]
    per_seed, per_utterance, bins_rows, bootstrap_rows = [], [], [], []
    output_hashes = {}
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            fmt = {"historical_prediction_root": paths["historical_prediction_root"], "dataset": dataset, "seed": seed}
            source_path = Path(paths["test_template"].format(**fmt))
            payload = read_npz(source_path)
            temperature = temperatures[(dataset, seed)]
            calibrated_log = temperature_log_probabilities(payload["log_probabilities"], temperature)
            calibrated = np.exp(calibrated_log)
            original = np.asarray(payload["probabilities"], dtype=np.float64)
            original_predictions = original.argmax(1)
            calibrated_predictions = calibrated.argmax(1)
            if not np.array_equal(original_predictions, calibrated_predictions):
                raise RuntimeError(f"argmax changed for {dataset} seed {seed}")
            before, after = metrics(original, payload["labels"]), metrics(calibrated, payload["labels"])
            row = {"dataset": dataset, "seed": seed, "temperature": temperature,
                   "test_utterances": len(payload["labels"]), "test_dialogues": len(set(map(str, payload["dialogue_ids"]))),
                   "test_split_hash": identity_hash(payload), "argmax_unchanged": True,
                   "changed_prediction_count": 0}
            for metric in METRICS:
                row[f"{metric}_before"] = before[metric]
                row[f"{metric}_after"] = after[metric]
                row[f"{metric}_difference"] = after[metric] - before[metric]
            per_seed.append(row)
            for method, probabilities in (("vanilla", original), ("global_ts", calibrated)):
                for bin_row in calibration_bins(probabilities, payload["labels"], 15):
                    bins_rows.append({"dataset": dataset, "seed": seed, "method": method, **bin_row})
            for i in range(len(payload["labels"])):
                per_utterance.append({
                    "dataset": dataset, "seed": seed, "dialogue_id": str(payload["dialogue_ids"][i]),
                    "utterance_index": int(payload["utterance_indices"][i]), "ground_truth": int(payload["labels"][i]),
                    "prediction": int(original_predictions[i]), "temperature": temperature,
                    "original_confidence": float(original[i].max()), "calibrated_confidence": float(calibrated[i].max()),
                    "original_probabilities": json.dumps(original[i].tolist(), separators=(",", ":")),
                    "calibrated_probabilities": json.dumps(calibrated[i].tolist(), separators=(",", ":")),
                    "original_log_probabilities": json.dumps(np.asarray(payload["log_probabilities"][i], float).tolist(), separators=(",", ":")),
                    "calibrated_log_probabilities": json.dumps(calibrated_log[i].tolist(), separators=(",", ":")),
                })
            run_dir = ROOT / "runs" / dataset / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "calibrated_test_predictions.npz"
            np.savez_compressed(output_path,
                dataset=np.asarray(dataset), seed=np.asarray(seed), temperature=np.asarray(temperature),
                labels=payload["labels"], dialogue_ids=payload["dialogue_ids"], utterance_indices=payload["utterance_indices"],
                speaker_ids=payload["speaker_ids"], dialogue_lengths=payload["dialogue_lengths"],
                original_probabilities=original, original_log_probabilities=np.asarray(payload["log_probabilities"], np.float64),
                calibrated_probabilities=calibrated, calibrated_log_probabilities=calibrated_log,
                original_predictions=original_predictions, calibrated_predictions=calibrated_predictions,
                checkpoint_sha256=np.asarray(next(r["checkpoint_sha256"] for r in frozen["parameters"] if r["dataset"] == dataset and int(r["seed"]) == seed)),
                test_split_hash=np.asarray(identity_hash(payload)), source_prediction_sha256=np.asarray(sha256(source_path)))
            output_hashes[f"{dataset}/seed{seed}"] = sha256(output_path)

            dialogues = np.asarray(payload["dialogue_ids"], str)
            unique_dialogues = np.unique(dialogues)
            rng = np.random.default_rng(bootstrap_seed(config["statistics"]["bootstrap_seed"], dataset, seed, "paired-test"))
            distributions = {metric: [] for metric in BOOTSTRAP_METRICS}
            for _ in range(int(config["statistics"]["bootstrap_repeats"])):
                sampled = rng.choice(unique_dialogues, size=len(unique_dialogues), replace=True)
                indices = np.concatenate([np.flatnonzero(dialogues == dialogue) for dialogue in sampled])
                m0, m1 = metrics(original[indices], payload["labels"][indices]), metrics(calibrated[indices], payload["labels"][indices])
                for metric in BOOTSTRAP_METRICS:
                    distributions[metric].append(m1[metric] - m0[metric])
            for metric, values in distributions.items():
                values = np.asarray(values)
                point = after[metric] - before[metric]
                bootstrap_rows.append({"dataset": dataset, "seed": seed, "metric": metric,
                    "difference_definition": "global_ts_minus_vanilla", "point_difference": point,
                    "bootstrap_mean_difference": float(values.mean()), "ci95_lower": float(np.quantile(values, .025)),
                    "ci95_upper": float(np.quantile(values, .975)), "bootstrap_repeats": len(values),
                    "dialogues": len(unique_dialogues), "ci_excludes_zero": bool(np.quantile(values, .025) > 0 or np.quantile(values, .975) < 0)})

    aggregate = []
    lower_better = {"ece15", "brier", "nll", "mce15", "rmsce15", "ece10"}
    for dataset in config["datasets"]:
        rows = [r for r in per_seed if r["dataset"] == dataset]
        for metric in METRICS:
            before = np.asarray([r[f"{metric}_before"] for r in rows])
            after = np.asarray([r[f"{metric}_after"] for r in rows])
            difference = after-before
            improved = difference < -1e-15 if metric in lower_better else np.isclose(difference, 0, atol=1e-14)
            aggregate.append({"dataset": dataset, "metric": metric, "seed_count": len(rows),
                "vanilla_mean": before.mean(), "vanilla_sample_std": before.std(ddof=1),
                "global_ts_mean": after.mean(), "global_ts_sample_std": after.std(ddof=1),
                "paired_difference_mean": difference.mean(), "paired_difference_sample_std": difference.std(ddof=1),
                "improved_seeds": int(improved.sum()), "direction_consistency": f"{int(improved.sum())}/{len(rows)}",
                "difference_definition": "global_ts_minus_vanilla"})
    write_csv(ROOT / "results/per_seed_test_metrics.csv", per_seed)
    write_csv(ROOT / "results/aggregate_test_metrics.csv", aggregate)
    write_csv(ROOT / "results/paired_bootstrap_results.csv", bootstrap_rows)
    write_csv(ROOT / "results/per_utterance_calibrated_probabilities.csv", per_utterance)
    write_csv(ROOT / "results/reliability_bins.csv", bins_rows)
    completion = {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_parameter_payload_sha256": frozen["parameter_payload_sha256"], "evaluated_checkpoints": len(per_seed),
        "test_evaluation_passes": 1, "all_argmax_unchanged": all(r["argmax_unchanged"] for r in per_seed),
        "changed_prediction_count": sum(r["changed_prediction_count"] for r in per_seed), "output_npz_sha256": output_hashes}
    (ROOT / "TEST_EVALUATION_COMPLETE.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps({"status": "complete", "all_argmax_unchanged": completion["all_argmax_unchanged"],
                      "primary": [{"dataset": r["dataset"], "seed": r["seed"],
                                   "ece15_delta": r["ece15_difference"], "brier_delta": r["brier_difference"],
                                   "nll_delta": r["nll_difference"]} for r in per_seed]}, indent=2))


if __name__ == "__main__":
    main()
