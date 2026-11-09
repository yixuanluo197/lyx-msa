#!/usr/bin/env python3
"""Freeze the researcher-approved Protocol B assignments without resampling."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/xp_caldist_behavioral_calibration/independent_calibration_protocol/20261109T100307Z"
CANDIDATE = PROTOCOL / "candidate_split_manifests.json"
LINEAGE = PROTOCOL / "results/sample_lineage.csv"
MASK_SOURCE = ROOT / "experiments/xp_caldist_behavioral_calibration/independent_training_pilot/20261109T115346Z/results/iemocap_xxx_anomalies.csv"
ROLES = ("training", "development", "calibration", "evaluation")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    candidate = json.loads(CANDIDATE.read_text())
    if candidate["status"] != "candidate_not_approved":
        raise RuntimeError("unexpected candidate status")
    lineage = read_csv(LINEAGE)
    mask_rows = read_csv(MASK_SOURCE)
    if len(mask_rows) != 53:
        raise RuntimeError("Unknown-label mask must contain exactly 53 utterances")
    masked_uids = {row["utterance_id"] for row in mask_rows}
    if len(masked_uids) != 53:
        raise RuntimeError("Unknown-label mask contains duplicate utterance IDs")

    utterances = defaultdict(list)
    dialogue_group = {}
    for row in lineage:
        key = (row["dataset"], row["dialogue_id"])
        utterances[key].append(row)
        group = row["session_id"] if row["dataset"] == "IEMOCAP" else row["episode_id"]
        if key in dialogue_group and dialogue_group[key] != group:
            raise RuntimeError(f"inconsistent group for {key}")
        dialogue_group[key] = group

    protocols = {}
    verification = {"dialogue_overlap_zero": True, "group_isolation_zero": True,
                    "mask_exact": True, "class_coverage_complete": True}
    expected_classes = {
        "IEMOCAP": {"happy", "sad", "neutral", "angry", "excited", "frustrated"},
        "MELD": {"neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"},
    }
    for dataset in ("IEMOCAP", "MELD"):
        source = candidate["protocols"][f"B_{dataset}"]
        frozen_folds = {}
        for fold_name, assignments in source["folds"].items():
            role_ids = {role: sorted(dialogue for dialogue, assigned in assignments.items() if assigned == role)
                        for role in ROLES}
            if set().union(*(set(ids) for ids in role_ids.values())) != set(assignments):
                raise RuntimeError(f"{dataset}/{fold_name}: incomplete assignment coverage")
            for index, left in enumerate(ROLES):
                for right in ROLES[index + 1:]:
                    if set(role_ids[left]) & set(role_ids[right]):
                        verification["dialogue_overlap_zero"] = False

            role_groups = {role: sorted({dialogue_group[(dataset, dialogue)] for dialogue in ids})
                           for role, ids in role_ids.items()}
            if dataset == "IEMOCAP":
                for isolated in ("calibration", "evaluation"):
                    if len(role_groups[isolated]) != 1:
                        verification["group_isolation_zero"] = False
                    if any(set(role_groups[isolated]) & set(role_groups[other])
                           for other in ROLES if other != isolated):
                        verification["group_isolation_zero"] = False
            else:
                for index, left in enumerate(ROLES):
                    for right in ROLES[index + 1:]:
                        if set(role_groups[left]) & set(role_groups[right]):
                            verification["group_isolation_zero"] = False

            role_payload = {}
            for role, ids in role_ids.items():
                counts = Counter()
                utterance_count = 0
                masked_count = 0
                for dialogue in ids:
                    for row in utterances[(dataset, dialogue)]:
                        utterance_count += 1
                        is_masked = dataset == "IEMOCAP" and row["source_utterance_id"] in masked_uids
                        masked_count += int(is_masked)
                        if not is_masked:
                            counts[row["emotion_label"]] += 1
                if set(counts) != expected_classes[dataset]:
                    verification["class_coverage_complete"] = False
                item = {
                    "dialogue_ids": ids,
                    "group_ids": role_groups[role],
                    "dialogue_count": len(ids),
                    "utterance_count": utterance_count,
                    "supervision_eligible_count": utterance_count - masked_count,
                    "unknown_masked_count": masked_count,
                    "class_support_supervision_eligible": dict(sorted(counts.items())),
                }
                item["sha256"] = canonical_sha256(item)
                role_payload[role] = item
            fold = {"roles": role_payload}
            fold["sha256"] = canonical_sha256(fold)
            frozen_folds[fold_name] = fold

        protocol = {
            "group_unit": source["group_unit"],
            "block_assignment": source.get("block_assignment"),
            "folds": frozen_folds,
        }
        protocol["sha256"] = canonical_sha256(protocol)
        protocols[dataset] = protocol

    if set(masked_uids) != {row["source_utterance_id"] for row in lineage
                            if row["dataset"] == "IEMOCAP" and row["source_utterance_id"] in masked_uids}:
        verification["mask_exact"] = False
    if not all(verification.values()):
        raise RuntimeError(f"manifest verification failed: {verification}")

    mask_payload = [{
        "dialogue_id": row["dialogue_id"],
        "utterance_index": int(row["utterance_index"]),
        "utterance_id": row["utterance_id"],
        "session_id": row["session_id"],
        "raw_primary_label": "xxx",
        "frozen_label": "happy",
    } for row in mask_rows]
    mask_payload.sort(key=lambda row: (row["dialogue_id"], row["utterance_index"]))
    approved = {
        "status": "approved",
        "protocol": "B_group_level_rotating_cross_fitting",
        "researcher_decisions": {
            "protocol_b": "approved_2026-11-09",
            "iemocap_label_policy": "unknown_label_mask",
            "legacy_policy_role": "optional_future_sensitivity_only",
        },
        "source": {
            "candidate_path": str(CANDIDATE.relative_to(ROOT)),
            "candidate_sha256": file_sha256(CANDIDATE),
            "sample_lineage_path": str(LINEAGE.relative_to(ROOT)),
            "sample_lineage_sha256": file_sha256(LINEAGE),
            "mask_source_path": str(MASK_SOURCE.relative_to(ROOT)),
            "mask_source_sha256": file_sha256(MASK_SOURCE),
        },
        "unknown_label_mask": {
            "count": 53,
            "policy": "retain_features_sequence_graph_qmask_umask;exclude_all_label_supervision_selection_calibration_evaluation_metrics",
            "utterances": mask_payload,
            "sha256": canonical_sha256(mask_payload),
        },
        "protocols": protocols,
        "verification": verification,
    }
    approved["content_sha256"] = canonical_sha256(approved)
    output = OUT / "approved_split_manifests.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(approved, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "content_sha256": approved["content_sha256"],
                      "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
