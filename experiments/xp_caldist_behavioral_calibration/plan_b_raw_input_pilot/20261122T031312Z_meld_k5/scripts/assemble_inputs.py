#!/usr/bin/env python3
"""Bind single-pass AI candidates and independently assigned blind AI verdicts."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PARTS = RUN/"local_only/parts"
NEW = ("paraphrase_c","paraphrase_d","paraphrase_e")
VERDICTS = {"PASS","FAIL","UNCERTAIN"}
AUTHORS = ("codex/aps_local_audit", "codex/converter_protocol_review", "codex/diag_framework")
REVIEWERS = ("codex/audio_upstream", "codex/diag_train_gap", "codex/converter_engine")


def sha_text(text):
    return sha256(text.encode("utf-8")).hexdigest()


def sha_file(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True)+"\n")


def read(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def shard(index):
    template = read(PARTS/f"template_{index}.jsonl")
    generated_rows = read(PARTS/f"generated_{index}.jsonl")
    generated = {r["sample_id"]:r["variants"] for r in generated_rows}
    assert len(template) == len(generated_rows) == len(generated) == 16
    assert set(generated) == {r["sample_id"] for r in template}
    for r in template:
        candidates = generated[r["sample_id"]]
        assert set(candidates) == set(NEW)
        for c in candidates.values():
            assert isinstance(c["text"],str) and c["status"] in VERDICTS and c["notes"].strip()
    return template,generated


def blind(index):
    template, generated = shard(index)
    path = PARTS/f"blind_review_{index}.jsonl"
    write(path, [dict(row,candidates={variant:generated[row["sample_id"]][variant]["text"] for variant in NEW}) for row in template])
    print(path, sha_file(path))


def assemble():
    for path,h in json.loads((RUN/"provenance/cohort_lock.json").read_text()).items():
        assert sha_file(RUN/path) == h, path
    old = read(RUN/"local_only/original_variants.jsonl")
    old_by_sample = {r["sample_id"]:r for r in old if r["variant_id"] == "paraphrase_a"}
    results = list(old)
    review_summary = Counter()
    parts = {}
    for idx in range(3):
        template, generated = shard(idx)
        blinded = read(PARTS/f"blind_review_{idx}.jsonl")
        expected_blind = [dict(row,candidates={v:generated[row["sample_id"]][v]["text"] for v in NEW}) for row in template]
        assert blinded == expected_blind
        review_rows = read(PARTS/f"reviewed_{idx}.jsonl")
        reviewed = {r["sample_id"]:r["variants"] for r in review_rows}
        assert len(reviewed) == len(review_rows) == 16 and set(reviewed) == set(generated)
        for item in template:
            sid = item["sample_id"]
            previous = old_by_sample[sid]
            assert set(reviewed[sid]) == set(NEW)
            for v in NEW:
                author = generated[sid][v]
                reviewer = reviewed[sid][v]
                status = reviewer["status"]
                assert status in VERDICTS and reviewer["notes"].strip()
                assert sha_text(author["text"]) == reviewer["text_sha256"]
                final = "FAIL" if "FAIL" in (author["status"], status) else (
                    "UNCERTAIN" if "UNCERTAIN" in (author["status"],status) else "PASS")
                record = {k:previous[k] for k in (
                    "sample_id","dialogue_id","utterance_index","original_text","original_text_sha256",
                    "original_total_tokens","maximum_neutral_added_tokens","dialogue_context","perturbation_type")}
                record.update(variant_id=v, processed_text=author["text"], review_status=final,
                              reviewer=REVIEWERS[idx], review_method="AI_CONTEXT_SEMANTIC_REVIEW",
                              review_notes=reviewer["notes"], generation_reviewer=AUTHORS[idx],
                              generation_review_status=author["status"], generation_review_notes=author["notes"],
                              second_ai_review={"status":status,"reviewer":REVIEWERS[idx],"notes":reviewer["notes"]},
                              instruction="AI-only input-blind semantic paraphrase; original a/b unchanged")
                results.append(record)
                review_summary[final] += 1
        for stem in ("template","generated","blind_review","reviewed"):
            name=f"local_only/parts/{stem}_{idx}.jsonl"
            parts[name]=sha_file(RUN/name)
    assert len(results) == 240 and len({(r["sample_id"],r["variant_id"]) for r in results}) == 240
    out=RUN/"local_only/matched_text_reviewed.jsonl"
    write(out,results)
    with (RUN/"provenance/input_lock.json").open("x") as stream:
        json.dump(dict(preserved_old_variants=96, new_variants=144,
                       new_ai_status_counts=dict(review_summary), author_and_reviewer_distinct=True,
                       human_review_used=False, source_part_sha256=parts,
                       final_private_inputs_sha256=sha_file(out)),stream,indent=2)
        stream.write("\n")
    print("Frozen 240 inputs; new AI statuses:",dict(review_summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind",type=int,choices=(0,1,2))
    args = parser.parse_args()
    blind(args.blind) if args.blind is not None else assemble()
