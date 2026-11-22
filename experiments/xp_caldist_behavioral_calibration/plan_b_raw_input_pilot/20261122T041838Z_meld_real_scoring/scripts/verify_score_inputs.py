#!/usr/bin/env python3
"""Independently audit saved probabilities and declare the frozen CPU scorer input."""
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
PRIVATE = RUN / "local_only"
RESULTS = RUN / "results"
PROVENANCE = RUN / "provenance"
ROLES = {"score_fit", "score_eval"}
VERSIONS = {"paraphrase_a", "paraphrase_b"}


def digest(path):
    result = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require(flag, message):
    if not flag:
        raise ValueError(message)


def unique(rows, fields):
    index = {tuple(row[field] for field in fields): row for row in rows}
    require(len(index) == len(rows), f"Duplicate IDs for {fields}")
    return index


def probabilities(blob):
    values = tuple(float(value) for value in json.loads(blob))
    require(len(values) == 7 and all(math.isfinite(x) and 0 <= x <= 1 for x in values)
            and abs(sum(values) - 1) <= 1e-6, "Invalid full Softmax vector")
    return values


def verify_predictions():
    input_path = RESULTS / "score_inputs.csv"
    transforms = RESULTS / "score_transformations.csv"
    inference = load(RESULTS / "score_prediction_verification.json")
    require(inference["status"] == "COMPLETE" and inference["no_test"]
            and inference["no_model_fit"] and inference["fp32"] and inference["text_modality_only"]
            and inference["target_only_replacement"] and inference["full_dialogue_context"]
            and inference["model_state_before"] == inference["model_state_after"], "Frozen model checks failed")
    for path in (input_path, transforms):
        require(digest(path) == inference["outputs"][path.name], "Inference output changed")
    review_lock_path = PROVENANCE / "ai_review_lock.json"
    review_lock = load(review_lock_path)
    review_path = PRIVATE / "reviewed_text.jsonl"
    require(review_lock["human_confirmation"] is False and digest(review_path) == review_lock["reviewed_text_sha256"],
            "Review artifact changed or misattributed")
    for name, expected in review_lock["source_hashes"].items():
        require(digest(RUN / name) == expected, f"Blind review source changed: {name}")
    with review_path.open(encoding="utf-8") as handle:
        reviews = unique([json.loads(line) for line in handle], ("sample_id", "variant_id"))
    attempts = unique(read_csv(PRIVATE / "perturbation_attempt_manifest.csv"), ("sample_id",))
    require(len(attempts) == 192 and len(reviews) == 384, "Incorrect attempted roster size")
    require(reviews.keys() == {(sid, version) for sid, in attempts for version in VERSIONS}, "Review attempt roster differs")
    manifest = PRIVATE / "SPLIT_UTTERANCES.csv"
    all_split = read_csv(manifest)
    base_dialogues = {row["dialogue_id"] for row in all_split if row["role"] in {"base_fit", "base_inner_dev"}}
    cohort = unique([row for row in all_split if row["role"] in ROLES], ("sample_id",))
    clean = unique(read_csv(input_path), ("sample_id",))
    perturbed = unique(read_csv(transforms), ("sample_id", "variant_id"))
    require(len(clean) == 2976 and clean.keys() == cohort.keys(), "Clean output does not exactly cover both frozen score groups")
    require(perturbed.keys() == {(sid, version) for sid, in clean for version in VERSIONS},
            "All transformation outcomes (including missing) must be recorded")
    roles = defaultdict(Counter)
    missing = defaultdict(Counter)
    errors = []
    for (sid,), row in clean.items():
        who = cohort[(sid,)]
        role = row["role"]
        identity = ("role", "dialogue_id", "utterance_index", "utterance_id")
        require(all(row[k] == who[k] for k in identity) and row["dialogue_id"] not in base_dialogues,
                "Prediction identity or train isolation failed")
        roles[role]["clean"] += 1
        p = probabilities(row["original_probabilities"])
        label = int(row["original_label"])
        pred = max(range(7), key=p.__getitem__)
        correct = int(pred == label)
        roles[role]["clean_correct"] += correct
        require(0 <= label < 7 and pred == int(row["original_prediction"]) and correct == int(row["original_correct"]),
                "Clean class/correctness does not match probabilities")
        if (sid,) in attempts:
            roles[role]["attempted"] += 1
        collected = []
        for version in VERSIONS:
            variant = perturbed[(sid, version)]
            review = reviews.get((sid, version))
            require(all(row[k] == variant[k] for k in identity), "Variant identity changed")
            require(probabilities(variant["original_probabilities"]) == p, "Variant clean control changed")
            if review is not None:
                require(all(variant[k] == review[k] for k in ("review_status", "review_method", "reviewer",
                            "original_text_sha256", "processed_text_sha256")), "Inference review binding mismatch")
            if variant["eligible"] == "1":
                require(review is not None and review["review_status"] == "PASS"
                        and variant["inference_status"] == "PREDICTED", "Unreviewed or invalid version counted")
                q = probabilities(variant["perturbed_probabilities"])
                require(abs(q[pred] - float(variant["perturbed_original_class_probability"])) <= 1e-10,
                        "Perturbed original-class probability mismatch")
                collected.append(q[pred])
                roles[role]["predicted_versions"] += 1
            else:
                require(not variant["perturbed_probabilities"], "Ineligible version has unexpected probability")
                if review is not None:
                    missing[role][variant["reason"]] += 1
        if len(collected) == 2:
            require(row["k2_available"] == "1" and row["D_pred"], "Valid K=2 omitted")
            roles[role]["covered"] += 1
            roles[role]["covered_correct"] += correct
            errors.append(abs(p[pred] - sum(collected) / 2 - float(row["D_pred"])))
        else:
            require(row["k2_available"] == "0" and not row["D_pred"], "Missing K=2 was imputed")
    require(max(errors, default=0) <= 1e-10 and sum(row["covered"] for row in roles.values()) == inference["available_D_pred"],
            "D_pred differs from saved probability vectors")
    for name, total in (("score_fit", 1569), ("score_eval", 1407)):
        require(roles[name]["clean"] == total and roles[name]["attempted"] == 96, "Role scope changed")
    return dict(roles={role: dict(counts) for role, counts in roles.items()},
                attempted_version_exclusions={role: dict(counts) for role, counts in missing.items()},
                max_D_pred_recomputed_absolute_error=max(errors, default=0))


def main():
    protocol = load(RUN / "configs/REAL_SCORING_PROTOCOL.json")
    split_lock = RUN / "SPLIT_LOCK.json"
    require(digest(split_lock) == protocol["split_lock_sha256"], "Frozen assignment changed")
    for stage in ("text", "downstream"):
        require(load(PROVENANCE / f"{stage}_execution.json")["status"] == "COMPLETE", f"Incomplete {stage} training")
    report = verify_predictions()
    sources = dict(input_sha256=RESULTS / "score_inputs.csv", dialogue_manifest_sha256=PRIVATE / "SPLIT_DIALOGUES.csv",
                   utterance_manifest_sha256=PRIVATE / "SPLIT_UTTERANCES.csv", split_lock_sha256=split_lock)
    declaration = dict(
        data_kind="declared_real",
        declarant="Codex Plan B MELD single-Seed scoring run",
        confidence_definition="original_max_softmax_probability",
        outcome_definition="original_correct_1",
        split_frozen=True,
        declaration_note="Fresh full supervised chain on disjoint base Dialogues. Frozen natural Train score groups, "
                         "local AI author and separate blinded AI reviewer, limited randomized text coverage, "
                         "no human confirmation, official Validation and Test excluded.",
        **{key: digest(path) for key, path in sources.items()},
        independent_probability_audit=report,
    )
    with (PROVENANCE / "scoring_input_declaration.json").open("x") as handle:
        json.dump(declaration, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
