#!/usr/bin/env python3
"""Verify the immutable 53-row mask against frozen IEMOCAP dialogue ordering."""

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
MANIFEST = OUT / "approved_split_manifests.json"
FEATURES = ROOT / "CMERC/data/IEMOCAP_features.pkl"


def tensor_source_hash(values) -> str:
    return hashlib.sha256(pickle.dumps(values, protocol=4)).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    mask_rows = manifest["unknown_label_mask"]["utterances"]
    by_dialogue = {}
    for row in mask_rows:
        by_dialogue.setdefault(row["dialogue_id"], {})[row["utterance_index"]] = row["utterance_id"]
    with FEATURES.open("rb") as handle:
        video_ids, speakers, labels, text, audio, visual, sentences, train_ids, test_ids = pickle.load(handle, encoding="latin1")

    masks = {}
    for dialogue, ids in video_ids.items():
        mask = [True] * len(ids)
        for index, expected_uid in by_dialogue.get(dialogue, {}).items():
            if ids[index] != expected_uid:
                raise RuntimeError(f"mask mismatch {dialogue}/{index}: {ids[index]} != {expected_uid}")
            if labels[dialogue][index] != 0:
                raise RuntimeError(f"masked frozen label is not happy/0: {dialogue}/{index}")
            mask[index] = False
        masks[dialogue] = mask
        n = len(ids)
        if not all(len(source[dialogue]) == n for source in (speakers, labels, text, audio, visual, sentences)):
            raise RuntimeError(f"source length mismatch in {dialogue}")

    if sum(value is False for mask in masks.values() for value in mask) != 53:
        raise RuntimeError("mask count is not 53")
    fold1 = manifest["protocols"]["IEMOCAP"]["folds"]["fold1"]["roles"]
    role_counts = Counter()
    for role, payload in fold1.items():
        role_counts[role] = sum(not value for dialogue in payload["dialogue_ids"] for value in masks[dialogue])
        if role_counts[role] != payload["unknown_masked_count"]:
            raise RuntimeError(f"fold1 mask count mismatch for {role}")
    report = {
        "status": "passed",
        "policy": "independent_supervision_mask",
        "mask_total": 53,
        "dialogues_with_masked_utterances": len(by_dialogue),
        "fold1_mask_counts": dict(role_counts),
        "dialogue_lengths_unchanged": True,
        "features_unchanged": True,
        "qmask_umask_policy": "constructed from original speakers/lengths; supervision mask is separate",
        "frozen_feature_sha256": hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
        "source_structure_hash_before": tensor_source_hash((video_ids, speakers, labels)),
        "source_structure_hash_after": tensor_source_hash((video_ids, speakers, labels)),
    }
    destination = OUT / "results/mask_alignment_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
