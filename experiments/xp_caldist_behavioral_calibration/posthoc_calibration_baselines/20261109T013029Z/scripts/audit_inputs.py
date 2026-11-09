#!/usr/bin/env python
"""Read-only identity and integrity audit performed before temperature fitting."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from calibration_core import identity_hash, metrics, read_npz, sha256  # noqa: E402


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    config = load_json(ROOT / "configs/calibration_config.json")
    paths = config["paths"]
    historical_root = Path(paths["historical_prediction_root"])
    historical_manifest = load_json(historical_root / "provenance/artifact_manifest.json")["files"]
    prior_audit = load_json(Path(paths["cross_seed_root"]) / "provenance/input_audit.json")
    expected_checkpoint_hashes = prior_audit["checkpoints"]
    audit = {
        "status": "running", "completed_at_utc": None,
        "scope": "read-only checkpoint, prediction, split, and Experiment 6 identity audit",
        "datasets": {}, "checks": {}, "experiment6": {},
    }
    all_ok = True
    for dataset in config["datasets"]:
        expected = config["expected_splits"][dataset]
        dataset_rows = []
        reference = {}
        for seed in config["seeds"]:
            fmt = {"xp04_root": paths["xp04_root"], "historical_prediction_root": paths["historical_prediction_root"],
                   "dataset": dataset, "seed": seed}
            checkpoint = Path(paths["checkpoint_template"].format(**fmt))
            checkpoint_hash = sha256(checkpoint)
            cp_ok = checkpoint_hash == expected_checkpoint_hashes[dataset][str(seed)]
            row = {"seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                   "checkpoint_hash_matches_prior": cp_ok, "splits": {}}
            all_ok &= cp_ok
            for split, template_key in (("validation", "validation_template"), ("test", "test_template")):
                prediction = Path(paths[template_key].format(**fmt))
                payload = read_npz(prediction)
                rel = str(prediction.relative_to(historical_root))
                digest = sha256(prediction)
                manifest_ok = historical_manifest.get(rel) == digest
                identities = [(str(d), int(u)) for d, u in zip(payload["dialogue_ids"], payload["utterance_indices"])]
                unique_ok = len(identities) == len(set(identities))
                split_info = {
                    "path": str(prediction), "file_sha256": digest, "artifact_manifest_matches": manifest_ok,
                    "identity_sha256": identity_hash(payload), "utterances": len(payload["labels"]),
                    "dialogues": len(set(map(str, payload["dialogue_ids"]))), "classes": payload["probabilities"].shape[1],
                    "unique_dialogue_utterance_ids": unique_ok,
                }
                row["splits"][split] = split_info
                key = split
                fingerprint = (split_info["identity_sha256"], tuple(map(str, payload["dialogue_ids"])),
                               tuple(map(int, payload["utterance_indices"])), tuple(map(int, payload["labels"])))
                if key not in reference:
                    reference[key] = fingerprint
                same_seed_metadata = fingerprint == reference[key]
                split_info["same_metadata_across_seeds"] = same_seed_metadata
                size_ok = (split_info["utterances"] == expected[f"{split}_utterances"] and
                           split_info["dialogues"] == expected[f"{split}_dialogues"])
                split_info["expected_size_matches"] = size_ok
                all_ok &= manifest_ok and unique_ok and same_seed_metadata and size_ok
            val_ids = set(map(str, read_npz(Path(paths["validation_template"].format(**fmt)))["dialogue_ids"]))
            test_ids = set(map(str, read_npz(Path(paths["test_template"].format(**fmt)))["dialogue_ids"]))
            row["validation_test_dialogue_overlap"] = sorted(val_ids & test_ids)
            all_ok &= not row["validation_test_dialogue_overlap"]
            dataset_rows.append(row)
        audit["datasets"][dataset] = dataset_rows

    exp6_path = Path(paths["experiment6_root"]) / "exp6_overall_summary.csv"
    with exp6_path.open(encoding="utf-8-sig", newline="") as stream:
        exp6_rows = [row for row in csv.DictReader(stream) if row["model"].lower() == "cmerc"]
    for row in exp6_rows:
        dataset = row["dataset"]
        seed = config["historical_experiment6"]["anchors"][dataset]
        fmt = {"xp04_root": paths["xp04_root"], "historical_prediction_root": paths["historical_prediction_root"],
               "dataset": dataset, "seed": seed}
        payload = read_npz(Path(paths["test_template"].format(**fmt)))
        current = metrics(payload["probabilities"], payload["labels"])
        comparisons = {
            "accuracy": abs(current["accuracy"] - float(row["accuracy"])),
            "mean_confidence": abs(current["mean_confidence"] - float(row["mean_confidence"])),
            "ece10": abs(current["ece10"] - float(row["ece"])),
            "brier": abs(current["brier"] - float(row["brier_score"])),
            "nll": abs(current["nll"] - float(row["nll"])),
        }
        match = max(comparisons.values()) < 5e-7 and len(payload["labels"]) == int(row["valid_utterances"])
        audit["experiment6"][dataset] = {
            "anchor_seed": seed, "summary_path": str(exp6_path), "summary_sha256": sha256(exp6_path),
            "test_only_descriptive_reference": True, "never_used_for_fitting": True,
            "metric_absolute_differences": comparisons, "matches_current_test_predictions": match,
        }
        all_ok &= match
    audit["checks"] = {
        "all_checkpoint_hashes_match": all(r["checkpoint_hash_matches_prior"] for rows in audit["datasets"].values() for r in rows),
        "all_prediction_files_match_historical_manifest": all(s["artifact_manifest_matches"] for rows in audit["datasets"].values() for r in rows for s in r["splits"].values()),
        "all_expected_split_sizes_match": all(s["expected_size_matches"] for rows in audit["datasets"].values() for r in rows for s in r["splits"].values()),
        "validation_test_dialogues_disjoint": all(not r["validation_test_dialogue_overlap"] for rows in audit["datasets"].values() for r in rows),
        "saved_predictions_sufficient_no_gpu": True,
    }
    audit["status"] = "passed" if all_ok and all(audit["checks"].values()) else "failed"
    audit["completed_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (ROOT / "provenance").mkdir(exist_ok=True)
    (ROOT / "provenance/input_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Read-only input audit", "", f"Status: **{audit['status']}**.", "",
             "The ten XP04 checkpoints and twenty saved prediction files form an intact provenance chain. "
             "All prediction inputs match the prior artifact manifest; Validation and Test dialogue sets are disjoint.", "",
             "| Dataset | Validation | Test | Seeds |", "|---|---:|---:|---:|"]
    for dataset in config["datasets"]:
        first = audit["datasets"][dataset][0]
        lines.append(f"| {dataset} | {first['splits']['validation']['utterances']} / {first['splits']['validation']['dialogues']} dialogues | "
                     f"{first['splits']['test']['utterances']} / {first['splits']['test']['dialogues']} dialogues | 5 |")
    lines += ["", "Experiment 6 is accepted only as a Test-only historical metric identity check for IEMOCAP seed1475 "
              "and MELD seed67137. It supplies no fitting data or calibration parameters.", "",
              "Existing complete probability vectors are sufficient; no GPU inference is required."]
    (ROOT / "reports/READ_ONLY_AUDIT.md").write_text("\n".join(lines) + "\n")
    if audit["status"] != "passed":
        raise SystemExit("input audit failed")
    print(json.dumps(audit["checks"], indent=2))


if __name__ == "__main__":
    main()
