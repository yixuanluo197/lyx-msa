#!/usr/bin/env python3
"""Extract only approved Training records; the runner never opens source pickles."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch


MANIFEST_RELATIVE = "experiments/xp_caldist_behavioral_calibration/independent_training_pilot/20261109T142243Z/approved_split_manifests.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def approved_training(manifest, dataset, fold):
    if manifest.get("status") != "approved":
        raise ValueError("manifest is not approved")
    body = {k: v for k, v in manifest.items() if k != "content_sha256"}
    if canonical_sha256(body) != manifest.get("content_sha256"):
        raise ValueError("approved manifest content hash mismatch")
    role = manifest["protocols"][dataset]["folds"][fold]["roles"]["training"]
    if canonical_sha256({k: v for k, v in role.items() if k != "sha256"}) != role["sha256"]:
        raise ValueError("Training role hash mismatch")
    if len(set(role["dialogue_ids"])) != role["dialogue_count"]:
        raise ValueError("duplicate or missing Training IDs")
    return role


def meld_source_id(dialogue_id):
    match = re.fullmatch(r"MELD_(train|validation|test)_(\d{4})", dialogue_id)
    if match is None:
        raise ValueError(f"invalid canonical MELD ID: {dialogue_id}")
    split, raw = match.groups()
    return int(raw) + {"train": 0, "validation": 1039, "test": 1153}[split]


def build_artifact(manifest_path, data_root, dataset, fold):
    manifest_path, data_root = Path(manifest_path), Path(data_root)
    manifest = json.loads(manifest_path.read_text())
    role = approved_training(manifest, dataset, fold)
    filenames = (["IEMOCAP_features.pkl", "iemocap_features_roberta.pkl"] if dataset == "IEMOCAP"
                 else ["MELD_features_raw1.pkl", "meld_features_roberta.pkl"])
    paths = [data_root / name for name in filenames]
    # These legacy containers are monolithic. Only this extraction process opens
    # them; no held-out arrays or labels are retained in the role-scoped output.
    with paths[0].open("rb") as handle:
        source = pickle.load(handle, encoding="latin1")
    with paths[1].open("rb") as handle:
        roberta = pickle.load(handle, encoding="latin1")
    ids, speakers, labels, _, audio, visual = source[:6]
    text = roberta[2:6] if dataset == "IEMOCAP" else roberta[3:7]
    mask_rows = {}
    if dataset == "IEMOCAP":
        for row in manifest["unknown_label_mask"]["utterances"]:
            mask_rows.setdefault(row["dialogue_id"], []).append(row)
    records = {}
    for dialogue_id in role["dialogue_ids"]:
        key = dialogue_id if dataset == "IEMOCAP" else meld_source_id(dialogue_id)
        n = len(labels[key])
        supervision = torch.ones(n, dtype=torch.bool)
        for row in mask_rows.get(dialogue_id, []):
            index = row["utterance_index"]
            if ids[key][index] != row["utterance_id"] or labels[key][index] != 0:
                raise ValueError(f"unknown-label alignment mismatch: {dialogue_id}/{index}")
            supervision[index] = False
        qmask = ([[1, 0] if item == "M" else [0, 1] for item in speakers[key]]
                 if dataset == "IEMOCAP" else speakers[key])
        target = torch.as_tensor(np.asarray(labels[key]), dtype=torch.long).clone()
        target[~supervision] = -100
        record = {"dialogue_id": dialogue_id, "source_dialogue_id": key,
                  "text": [torch.as_tensor(np.asarray(item[key]), dtype=torch.float32).clone() for item in text],
                  "audio": torch.as_tensor(np.asarray(audio[key]), dtype=torch.float32).clone(),
                  "visual": torch.as_tensor(np.asarray(visual[key]), dtype=torch.float32).clone(),
                  "qmask": torch.as_tensor(np.asarray(qmask), dtype=torch.float32),
                  "umask": torch.ones(n), "labels": target, "supervision_mask": supervision}
        arrays = record["text"] + [record[k] for k in ("audio", "visual", "qmask", "umask")]
        if n == 0 or any(len(item) != n for item in arrays):
            raise ValueError(f"unaligned or empty Training record: {dialogue_id}")
        if not supervision.any():
            raise ValueError(f"all-invalid Training dialogue: {dialogue_id}")
        records[dialogue_id] = record
    total = sum(len(r["labels"]) for r in records.values())
    eligible = sum(int(r["supervision_mask"].sum()) for r in records.values())
    if (total, eligible, total - eligible) != (role["utterance_count"], role["supervision_eligible_count"], role["unknown_masked_count"]):
        raise ValueError("Training artifact counts disagree with approved role")
    return {"schema": "protocol_b_training_role_v1", "role": "training", "dataset": dataset,
            "fold": fold, "dialogue_ids": role["dialogue_ids"], "dialogues": records,
            "approved_manifest_sha256": file_sha256(manifest_path),
            "approved_training_role": role,
            "provenance": {"approved_manifest_path": str(manifest_path.resolve()),
                           "approved_manifest_sha256": file_sha256(manifest_path),
                           "approved_manifest_content_sha256": manifest["content_sha256"],
                           "training_role_sha256": role["sha256"],
                           "source_data_sha256": {str(p.resolve()): file_sha256(p) for p in paths},
                           "extraction_access": "monolithic_deserialization_then_training_only_extraction",
                           "builder_sha256": file_sha256(__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", required=True, choices=["IEMOCAP", "MELD"])
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = build_artifact(args.manifest or args.workspace_root / MANIFEST_RELATIVE,
                             args.data_root or args.workspace_root / "CMERC/data", args.dataset, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata = {k: v for k, v in payload.items() if k != "dialogues"}
    metadata["artifact_sha256"] = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"artifact": str(args.output), "sha256": metadata["artifact_sha256"],
                      "dialogues": len(payload["dialogues"])}))


if __name__ == "__main__":
    main()
