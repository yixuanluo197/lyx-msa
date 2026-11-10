#!/usr/bin/env python3
"""Create final hashes and verification metadata inside this audit bundle only."""

import csv
import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data2/yb/paper/lyx-msa")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


required_reports = [
    "README.md", "RAW_TEXT_AUDIT.md", "RAW_AUDIO_AUDIT.md", "RAW_VIDEO_AUDIT.md",
    "TEXT_FEATURE_EXTRACTION_AUDIT.md", "AUDIO_FEATURE_EXTRACTION_AUDIT.md",
    "VISUAL_FEATURE_EXTRACTION_AUDIT.md", "T1_T4_LAYER_MAPPING.md",
    "MULTIMODAL_ID_ALIGNMENT_AUDIT.md", "PERTURBATION_FEASIBILITY_MATRIX.md",
    "FINAL_DATA_READINESS_DECISION.md",
]
required_csv = [
    "file_inventory.csv", "utterance_alignment_audit.csv", "modality_feature_statistics.csv",
    "t1_t4_mapping.csv", "perturbation_feasibility.csv",
]
source_paths = [
    WORKSPACE / "CMERC/experiment_records/04_xp_mainline_validation/code/dataloader.py",
    WORKSPACE / "CMERC/experiment_records/04_xp_mainline_validation/code/model_cm.py",
    WORKSPACE / "CMERC/experiment_records/04_xp_mainline_validation/code/train_cm.py",
    Path("/data2/yb/paper/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction/roberta_feature_extract_iemocap.py"),
    Path("/data2/yb/paper/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction/roberta_feature_extract_meld.py"),
    Path("/data2/yb/paper/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction/roberta_train_iemocap.sh"),
    Path("/data2/yb/paper/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction/roberta_train_meld.sh"),
    WORKSPACE / "CMERC/data/IEMOCAP_features.pkl",
    WORKSPACE / "CMERC/data/iemocap_features_roberta.pkl",
    WORKSPACE / "CMERC/data/MELD_features_raw1.pkl",
    WORKSPACE / "CMERC/data/meld_features_roberta.pkl",
]

hash_entries = []
for path in source_paths + [AUDIT / name for name in required_reports + required_csv] + [Path(__file__).resolve(), AUDIT / "provenance/collect_audit.py"]:
    resolved = path.resolve()
    hash_entries.append({
        "path": str(path), "resolved_path": str(resolved), "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    })
(AUDIT / "provenance/source_hashes.json").write_text(
    json.dumps({"algorithm": "SHA-256", "entries": hash_entries}, indent=2) + "\n",
    encoding="utf-8",
)

with (AUDIT / "utterance_alignment_audit.csv").open(encoding="utf-8") as handle:
    alignment = list(csv.DictReader(handle))
with (AUDIT / "file_inventory.csv").open(encoding="utf-8") as handle:
    inventory = list(csv.DictReader(handle))
checkpoint_rows = [row for row in inventory if row["category"] == "xp04_best_checkpoint"]
checkpoint_sha256_match_count = sum(
    sha256(Path(row["resolved_path"])) == row["sha256"] for row in checkpoint_rows
)

checks = {
    "all_required_reports_exist": all((AUDIT / name).is_file() for name in required_reports),
    "all_required_csv_exist": all((AUDIT / name).is_file() for name in required_csv),
    "alignment_row_count": len(alignment),
    "iemocap_alignment_rows": sum(row["dataset"] == "IEMOCAP" for row in alignment),
    "meld_alignment_rows": sum(row["dataset"] == "MELD" for row in alignment),
    "all_internal_feature_lengths_match": all(row["all_feature_lengths_match"] == "True" for row in alignment),
    "all_internal_raw_roberta_text_match": all(row["raw_roberta_sentence_match"] == "True" for row in alignment),
    "candidate_raw_text_mismatch_count": sum(row["candidate_raw_text_status"] != "MATCH" for row in alignment),
    "iemocap_unknown_label_remap_count": sum(row["candidate_raw_label_status"] == "UNKNOWN_MAPPED_TO_HAP" for row in alignment),
    "checkpoint_inventory_count": len(checkpoint_rows),
    "checkpoint_inventory_all_readable": all(row["readable"] == "True" for row in checkpoint_rows),
    "checkpoint_sha256_recomputed_match_count": checkpoint_sha256_match_count,
    "feature_pickle_inventory_count": sum(row["category"] == "preprocessed_feature" for row in inventory),
    "source_hash_entry_count": len(hash_entries),
}
expected = {
    "alignment_row_count": 21141, "iemocap_alignment_rows": 7433, "meld_alignment_rows": 13708,
    "candidate_raw_text_mismatch_count": 1, "iemocap_unknown_label_remap_count": 53,
    "checkpoint_inventory_count": 20, "checkpoint_sha256_recomputed_match_count": 20,
    "feature_pickle_inventory_count": 4,
}
passed = (
    checks["all_required_reports_exist"] and checks["all_required_csv_exist"]
    and checks["all_internal_feature_lengths_match"] and checks["all_internal_raw_roberta_text_match"]
    and checks["checkpoint_inventory_all_readable"]
    and all(checks[key] == value for key, value in expected.items())
)
verification = {
    "audit_timestamp_utc": "2026-11-10T05:56:04Z",
    "scope": "XP04 read-only data provenance and feature extraction audit",
    "status": "PASS_WITH_DOCUMENTED_PROVENANCE_GAPS" if passed else "FAIL",
    "checks": checks,
    "expected": expected,
    "limitations": [
        "Exact released RoBERTa pickle to producer checkpoint/Fairseq-version linkage is UNVERIFIED.",
        "Audio and visual raw-to-feature extraction code/manifests are MISSING.",
        "Local IEMOCAP and MELD raw collections are candidates, not XP04 runtime dependencies.",
        "Media codec/header checks were sampled; large media were not fully decoded or hashed.",
    ],
}
(AUDIT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
print(verification["status"])
