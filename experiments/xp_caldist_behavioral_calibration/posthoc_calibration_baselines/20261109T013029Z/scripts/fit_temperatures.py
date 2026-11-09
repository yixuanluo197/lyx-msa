#!/usr/bin/env python
"""Fit and freeze one validation-NLL temperature for each checkpoint; never reads Test."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from calibration_core import canonical_sha256, fit_temperature, identity_hash, metrics, read_npz, sha256, temperature_log_probabilities  # noqa: E402


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    config = json.loads((ROOT / "configs/calibration_config.json").read_text())
    audit = json.loads((ROOT / "provenance/input_audit.json").read_text())
    if audit["status"] != "passed":
        raise SystemExit("read-only audit is not passed")
    paths, fitting = config["paths"], config["temperature_fitting"]
    parameters, validation_rows, curve_rows = [], [], []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            fmt = {"historical_prediction_root": paths["historical_prediction_root"], "dataset": dataset, "seed": seed}
            prediction_path = Path(paths["validation_template"].format(**fmt))
            if "test" in prediction_path.name.lower():
                raise RuntimeError("fit script received a Test-named file")
            payload = read_npz(prediction_path)
            result, curve = fit_temperature(payload["log_probabilities"], payload["labels"],
                                             fitting["bounds"], fitting["grid_size"], fitting["refinement_iterations"])
            calibrated_log = temperature_log_probabilities(payload["log_probabilities"], result["temperature"])
            before = metrics(payload["probabilities"], payload["labels"])
            after = metrics(__import__("numpy").exp(calibrated_log), payload["labels"])
            checkpoint_hash = next(r["checkpoint_sha256"] for r in audit["datasets"][dataset] if r["seed"] == seed)
            record = {"dataset": dataset, "seed": seed, **result,
                      "checkpoint_sha256": checkpoint_hash, "validation_split_hash": identity_hash(payload),
                      "validation_prediction_sha256": sha256(prediction_path)}
            parameters.append(record)
            for method, values in (("vanilla", before), ("global_ts", after)):
                validation_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                        "temperature": 1.0 if method == "vanilla" else result["temperature"], **values})
            for row in curve:
                curve_rows.append({"dataset": dataset, "seed": seed, **row})
            run_dir = ROOT / "runs" / dataset / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "temperature_fit.json").write_text(json.dumps(record, indent=2) + "\n")
    write_csv(ROOT / "results/temperature_parameters.csv", parameters)
    write_csv(ROOT / "results/validation_metrics.csv", validation_rows)
    write_csv(ROOT / "results/validation_nll_temperature_curves.csv", curve_rows)
    payload = [{k: r[k] for k in ("dataset", "seed", "temperature", "validation_nll_before", "validation_nll_after",
                                  "convergence_status", "boundary_optimum", "checkpoint_sha256", "validation_split_hash")}
               for r in parameters]
    frozen = {"status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "fit_source": "Validation only", "test_files_read_by_fit_script": False,
              "parameter_count": len(payload), "parameters": payload, "parameter_payload_sha256": canonical_sha256(payload)}
    (ROOT / "FROZEN_TEMPERATURES.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"status": "frozen", "parameter_payload_sha256": frozen["parameter_payload_sha256"],
                      "temperatures": {f"{r['dataset']}/{r['seed']}": r["temperature"] for r in parameters}}, indent=2))


if __name__ == "__main__":
    main()
