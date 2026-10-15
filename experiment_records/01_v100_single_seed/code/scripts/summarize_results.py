#!/usr/bin/env python3
"""Combine training records and independent evaluations into one JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("IEMOCAP", "MELD")
VARIANTS = ("baseline", "cmerc")
PROTOCOLS = ("official_test", "validation")
SEEDS = {"IEMOCAP": 1475, "MELD": 67137}
PAPER_REFERENCES = {
    ("IEMOCAP", "baseline"): {"weighted_f1": 69.61, "cel": 1772.54},
    ("IEMOCAP", "cmerc"): {"weighted_f1": 71.98, "cel": 1143.50},
    ("MELD", "baseline"): {"weighted_f1": 65.37, "cel": 799.58},
    ("MELD", "cmerc"): {"weighted_f1": 66.85, "cel": 605.05},
}


def paper_reference(dataset: str, variant: str) -> dict[str, float]:
    return dict(PAPER_REFERENCES[(dataset, variant)])


def best_training_record(
    records: Iterable[dict[str, Any]], protocol: str
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol}")
    records = list(records)
    if not records:
        raise ValueError("training records are empty")
    return max(records, key=lambda record: float(record["selection_weighted_f1"]))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_run(
    run_root: Path,
    result_root: Path,
    dataset: str,
    variant: str,
    protocol: str,
) -> dict[str, Any]:
    seed = SEEDS[dataset]
    run_dir = run_root / variant / f"{dataset}_{protocol}_seed{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    evaluation_path = result_root / variant / dataset / protocol / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)

    records = read_jsonl(metrics_path)
    selected = best_training_record(records, protocol)
    evaluation = read_json(evaluation_path)
    reference = paper_reference(dataset, variant)
    summary = {
        "dataset": dataset,
        "variant": variant,
        "selection_protocol": protocol,
        "seed": seed,
        "epochs_completed": len(records),
        "selected_epoch": int(selected["epoch"]),
        "selection_weighted_f1": float(selected["selection_weighted_f1"]),
        "accuracy": float(evaluation["accuracy"]),
        "weighted_f1": float(evaluation["weighted_f1"]),
        "macro_f1": float(evaluation["macro_f1"]),
        "cel": float(evaluation["cel"]["total"]),
        "calibration": evaluation["calibration"],
        "paper_reference": reference,
    }
    summary["weighted_f1_gap_to_paper"] = summary["weighted_f1"] - reference["weighted_f1"]
    summary["cel_gap_to_paper"] = summary["cel"] - reference["cel"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/formal"))
    parser.add_argument("--result-root", type=Path, default=Path("results/formal"))
    parser.add_argument("--output", type=Path, default=Path("results/formal/summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [
        summarize_run(args.run_root, args.result_root, dataset, variant, protocol)
        for dataset in DATASETS
        for variant in VARIANTS
        for protocol in PROTOCOLS
    ]
    by_key = {
        (run["dataset"], run["variant"], run["selection_protocol"]): run
        for run in runs
    }
    for run in runs:
        baseline = by_key[(run["dataset"], "baseline", run["selection_protocol"])]
        run["weighted_f1_gain_over_baseline"] = run["weighted_f1"] - baseline["weighted_f1"]
        run["cel_reduction_from_baseline"] = baseline["cel"] - run["cel"]

    payload = {"runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"saved {len(runs)} run summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
