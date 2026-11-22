"""Verify reviewed typo candidates change the frozen tokenizer input before model execution."""
from collections import Counter
import importlib.util
import json

from transformers import RobertaTokenizer

from audit_inputs import BASE, RUN, SCORE, digest, read_json
from orthographic_edits import SHARDS, rows


def main():
    audit = read_json(RUN / "provenance/INPUT_AUDIT.json")
    path = SCORE.parent / "20261120T143131Z/scripts/text_encoder.py"
    assert digest(path) == audit["sha256_relative_to_pilot_parent"][str(path.relative_to(BASE))]
    spec = importlib.util.spec_from_file_location("frozen_text_encoder", path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    lock = read_json(RUN / "provenance/CORRUPTION_REVIEW_LOCK.json")
    reviewed = RUN / "local_only/corruption_reviewed.jsonl"
    assert digest(reviewed) == lock["reviewed_sha256"]
    tokenizer = RobertaTokenizer.from_pretrained(
        SCORE / "local_only/text_model/best_classifier", local_files_only=True)
    originals = {row["blind_id"]: row["original_text"] for shard in SHARDS
                 for row in rows(RUN / "local_only/review" / f"source_{shard}.jsonl")}
    paired = {}
    counts = Counter()
    for row in rows(reviewed):
        edit = row["edit"]
        original = originals[row["blind_id"]]
        if edit:
            start, end = edit["word_start"], edit["word_end"]
            adjacent = original[max(start-1, 0):start] + original[end:end+1]
            # Unicode punctuation rejected in this frozen cohort even though the
            # generator already excludes ASCII and curly contractions/hyphens.
            assert not set(adjacent).intersection("\u0092\u02bc\u2018\u201b\uff07\u2010\u2011\u2012\u2013\u2014\u2212")
        if row["review_status"] != "PASS":
            continue
        ids, stats = encoder.tokenize(tokenizer, [original, row["processed_text"]], "MELD")
        key = (row["role"], row["sample_id"])
        counts[row["role"] + "/ai_pass_versions"] += 1
        if ids[0] != ids[1] and not stats[1]["truncated"] and not stats[1]["empty_after_processing"]:
            paired.setdefault(key, []).append(tuple(ids[1]))
            counts[row["role"] + "/token_qc_pass_versions"] += 1
    for (role, _), variants in paired.items():
        counts[role + "/distinct_k2_token_pass"] += len(set(variants)) == 2
    result = dict(status="PASS_TOKEN_PREFLIGHT", model_weights_loaded=False, test_accessed=False,
                  counts=dict(sorted(counts.items())), review_sha256=digest(reviewed),
                  extra_unicode_apostrophe_and_dash_boundary_violations=0)
    with (RUN / "provenance/TOKEN_PREFLIGHT.json").open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
