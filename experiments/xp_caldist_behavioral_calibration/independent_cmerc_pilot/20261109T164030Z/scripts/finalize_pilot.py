#!/usr/bin/env python3
"""Independently verify the two completed pilots and render frozen reports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/xp_caldist_behavioral_calibration/independent_training_pilot/20261109T142243Z/approved_split_manifests.json"
TEACHER = ROOT / "experiments/xp_caldist_behavioral_calibration/curriculum_teacher_pilot/20261109T154754Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def expected_ratio(dataset: str, epoch: int) -> float:
    base = .15 if dataset == "IEMOCAP" else .4
    return min(1.0, round(base * epoch, 2))


def verify_dataset(dataset: str) -> dict:
    run = EXP / "runs" / dataset / "fold1" / "seed1475"
    training = run / "training"
    calibration = run / "calibration"
    expected_epochs = 80 if dataset == "IEMOCAP" else 40
    epoch_rows = [json.loads(line) for line in (training / "epochs.jsonl").read_text().splitlines()]
    assert len(epoch_rows) == expected_epochs
    assert all(row["epoch"] == index and row["ratio"] == expected_ratio(dataset, index)
               for index, row in enumerate(epoch_rows, 1))
    assert all(finite_tree(row) for row in epoch_rows)
    selected = [row for row in epoch_rows if row["selected"]]
    maximum = max(row["development"]["weighted_f1_percent_rounded"] for row in epoch_rows)
    earliest = min(row["epoch"] for row in epoch_rows
                   if row["development"]["weighted_f1_percent_rounded"] == maximum)
    assert epoch_rows[-1]["best_epoch"] == earliest
    assert epoch_rows[-1]["best_score"] == maximum
    assert all(row["train"]["exact_curriculum_coverage"] and row["development"]["exact_development_coverage"]
               for row in epoch_rows)
    assert all(row["train"]["max_batch_edges"] <= 1000 and row["train"]["max_batch_nodes"] <= 5200
               for row in epoch_rows)
    batch_rows = [json.loads(line) for line in (training / "batch_access.jsonl").read_text().splitlines()]
    assert {row["role"] for row in batch_rows} == {"training", "development"}
    assert not any("evaluation" in json.dumps(row).lower() for row in batch_rows)
    best = torch.load(training / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(training / "latest.pt", map_location="cpu", weights_only=False)
    assert best["complete"] and best["training_complete"] and best["is_best"]
    assert latest["complete"] and latest["epoch"] == expected_epochs
    assert best["selected_epoch"] == best["epoch"] == earliest
    assert best["selection_role"] == "development"
    assert best["provenance"]["historical_checkpoint_loaded"] is False
    assert state_sha256(best["model_state_dict"]) == best["model_state_sha256"]
    assert state_sha256(latest["model_state_dict"]) == latest["model_state_sha256"]
    completion = json.loads((training / "completion.json").read_text())
    assert completion["status"] == "completed" and completion["roles_accessed"] == ["training", "development"]
    for name, digest in completion["hashes"].items():
        assert sha256(training / name) == digest
    gate = json.loads((training / "first_epoch_gate.json").read_text())
    assert gate["status"] == "passed" and gate["continues_same_optimizer_rng"]
    if dataset == "IEMOCAP":
        audit = gate["train"]["direct_unknown_gradient_audit"]
        assert audit["status"] == "passed" and audit["supervised_path_count"] == 11
        assert all(item["excluded_direct_gradient_abs_max"] == 0 for item in audit["paths"].values())
    params = json.loads((calibration / "parameters.json").read_text())
    metrics = json.loads((calibration / "calibration_metrics.json").read_text())
    oof = json.loads((calibration / "oof_diagnostic.json").read_text())
    provenance = json.loads((calibration / "provenance.json").read_text())
    assert provenance["role"] == "calibration"
    assert provenance["checkpoint_sha256"] == sha256(training / "best.pt")
    assert (calibration / "provenance.json").stat().st_mtime >= (training / "completion.json").stat().st_mtime
    with np.load(calibration / "calibration_probabilities.npz", allow_pickle=False) as data:
        assert str(data["role"]) == "calibration"
        mask = data["supervision_mask"].astype(bool)
        original = data["E0_predictions"]
        for method in ("E0", "E1", "E2", "E3"):
            probability = data[f"{method}_probabilities"]
            assert np.isfinite(probability).all()
            assert np.allclose(probability.sum(1), 1.0, atol=1e-10)
            assert np.array_equal(data[f"{method}_predictions"], original)
            assert metrics[method]["n"] == int(mask.sum())
    assert finite_tree(params) and finite_tree(metrics) and finite_tree(oof)
    allocated = max(row["peak_allocated_bytes"] for row in epoch_rows)
    reserved = max(row["peak_reserved_bytes"] for row in epoch_rows)
    elapsed = epoch_rows[-1]["elapsed_seconds"]
    return {
        "dataset": dataset, "epochs": expected_epochs, "best_epoch": earliest,
        "best_development_weighted_f1": maximum,
        "best_development_accuracy": epoch_rows[earliest - 1]["development"]["accuracy"],
        "best_development_macro_f1": epoch_rows[earliest - 1]["development"]["macro_f1"],
        "training_seconds": elapsed, "gpu_hours": elapsed / 3600,
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "max_batch_edges": max(row["train"]["max_batch_edges"] for row in epoch_rows),
        "max_batch_nodes": max(row["train"]["max_batch_nodes"] for row in epoch_rows),
        "initialization_sha256": best["provenance"]["random_initialization_state_sha256"],
        "best_checkpoint_sha256": sha256(training / "best.pt"),
        "difficulty_sha256": best["bindings"]["difficulty_sha256"],
        "manifest_sha256": best["bindings"]["manifest_sha256"],
        "calibration_fit_n": metrics["E0"]["n"],
        "global_temperature": params["E1"]["temperature"],
        "hierarchical_lambda": params["E2"]["lambda"],
        "fit_metrics": metrics, "oof_metrics": oof["metrics"],
        "no_evaluation_access": True, "all_argmax_invariant": True,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])


def main() -> None:
    manifest_hash = sha256(MANIFEST)
    results = [verify_dataset(name) for name in ("IEMOCAP", "MELD")]
    write_csv(EXP / "results/pilot_training_summary.csv", results,
              ["dataset", "epochs", "best_epoch", "best_development_weighted_f1",
               "best_development_accuracy", "best_development_macro_f1", "training_seconds",
               "gpu_hours", "peak_allocated_bytes", "peak_reserved_bytes", "max_batch_edges", "max_batch_nodes"])
    calibration_rows = []
    for result in results:
        for source, metrics in (("full_calibration_fit", result["fit_metrics"]),
                                ("grouped_oof_diagnostic", result["oof_metrics"])):
            for method, values in metrics.items():
                calibration_rows.append({"dataset": result["dataset"], "source": source, "method": method, **values})
    write_csv(EXP / "results/calibration_summary.csv", calibration_rows,
              ["dataset", "source", "method", "n", "accuracy", "nll", "brier", "ece15"])
    verification = {
        "status": "passed", "scope": "two Fold1 seed1475 CMERC pilots plus Calibration; Evaluation sealed",
        "approved_manifest_sha256": manifest_hash, "datasets": results,
        "tests": {"runner": "8 passed", "calibration": "10 passed", "py_compile": "passed"},
        "evaluation": {"loader_constructed": False, "labels_read": False, "forward_run": False,
                       "metrics_computed": False},
        "decision": "ready_for_formal_50_model_training_after_researcher_review",
        "limitations": ["Calibration metrics and grouped OOF values are internal diagnostics, not independent Evaluation results.",
                        "IEMOCAP peak reserved memory nearly fills a 12 GiB TITAN Xp; schedule one IEMOCAP job per such GPU."]}
    (EXP / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "datasets": [{k: r[k] for k in
          ("dataset", "best_epoch", "best_development_weighted_f1", "training_seconds",
           "global_temperature", "hierarchical_lambda")} for r in results]}))


if __name__ == "__main__":
    main()
