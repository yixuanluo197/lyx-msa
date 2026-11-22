#!/usr/bin/env python3
"""Prepare blinded AI review and lock the complete no-replacement roster."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
PARTS = RUN / "local_only/parts"
VARIANTS = {"paraphrase_a", "paraphrase_b"}
STATES = {"PASS": 0, "UNCERTAIN": 1, "FAIL": 2}
SHARDS = range(6)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def text_digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def read_rows(path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            blind_id = row["blind_id"]
            if blind_id in rows:
                raise ValueError(f"Duplicate blind ID in {path.name}")
            rows[blind_id] = row
    if len(rows) != 32:
        raise ValueError(f"Expected all 32 targets in {path.name}")
    return rows


def check_status(entry):
    if entry.get("status") not in STATES or not str(entry.get("notes", "")).strip():
        raise ValueError("Every candidate needs an explicit verdict and reason")


def check_roster():
    roster_path = RUN / "provenance/ai_attempt_roster_lock.json"
    frozen = json.loads(roster_path.read_text())
    for name, expected in frozen["hashes"].items():
        if digest(RUN / name) != expected:
            raise ValueError(f"Changed roster source: {name}")
    return roster_path


def author_rows(index):
    source = read_rows(PARTS / f"source_{index}.jsonl")
    generated = read_rows(PARTS / f"generated_{index}.jsonl")
    if source.keys() != generated.keys():
        raise ValueError(f"Generated target set differs for shard {index}")
    for blind_id, row in generated.items():
        variants = row["variants"]
        if variants.keys() != VARIANTS:
            raise ValueError("Wrong candidate family or count")
        for entry in variants.values():
            check_status(entry)
            if not isinstance(entry["text"], str):
                raise ValueError("Candidate must be text")
            if entry["status"] == "PASS" and not entry["text"].strip():
                raise ValueError(f"Empty candidate marked PASS for {blind_id}")
    return source, generated


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def blind(index):
    check_roster()
    source, generated = author_rows(index)
    rows = []
    for blind_id, row in source.items():
        candidates = {key: entry["text"] for key, entry in generated[blind_id]["variants"].items()}
        rows.append({**row, "candidates": candidates})
    write_jsonl(PARTS / f"blind_review_{index}.jsonl", rows)
    print(f"Ready for blind review: shard {index}, {len(rows)} targets")


def assemble():
    roster_path = check_roster()
    assignments_path = PARTS / "review_assignments.json"
    assignments = json.loads(assignments_path.read_text())
    if set(assignments) != set(map(str, SHARDS)):
        raise ValueError("Incomplete review attribution")
    with (RUN / "local_only/perturbation_attempt_manifest.csv").open() as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["blind_id"]: row for row in manifest_rows}
    if (len(manifest) != len(manifest_rows) or len(manifest) != 192
            or Counter(row["role"] for row in manifest.values()) != {"score_fit": 96, "score_eval": 96}):
        raise ValueError("Unexpected original attempt roster")
    hashes = {str(p.relative_to(RUN)): digest(p) for p in (roster_path, assignments_path, Path(__file__))}
    counts = Counter()
    complete = Counter()
    seen = set()
    output = []
    for index in SHARDS:
        source, generated = author_rows(index)
        blind_rows = read_rows(PARTS / f"blind_review_{index}.jsonl")
        reviewed = read_rows(PARTS / f"reviewed_{index}.jsonl")
        author, reviewer = (assignments[str(index)][key] for key in ("author", "reviewer"))
        if not author or not reviewer or author == reviewer:
            raise ValueError(f"Author and reviewer must differ for shard {index}")
        if not (source.keys() == blind_rows.keys() == reviewed.keys()) or seen.intersection(source):
            raise ValueError(f"Review roster mismatch for shard {index}")
        seen.update(source)
        for blind_id, row in source.items():
            ids = {k: v for k, v in manifest[blind_id].items() if k != "blind_id"}
            curated = generated[blind_id]["variants"]
            verdicts = reviewed[blind_id]["variants"]
            if verdicts.keys() != VARIANTS or blind_rows[blind_id] != {
                **row, "candidates": {k: v["text"] for k, v in curated.items()}
            }:
                raise ValueError("Candidate or review context changed")
            final = []
            for key in sorted(VARIANTS):
                candidate, verdict = curated[key], verdicts[key]
                check_status(verdict)
                if verdict["text_sha256"] != text_digest(candidate["text"]):
                    raise ValueError("Reviewed candidate hash differs")
                status = max(candidate["status"], verdict["status"], key=STATES.get)
                if status == "PASS" and candidate["text"].strip() == row["original_text"].strip():
                    raise ValueError("Unchanged input marked PASS")
                output.append({
                    **ids,
                    "variant_id": key,
                    "original_text_sha256": text_digest(row["original_text"]),
                    "processed_text": candidate["text"],
                    "review_status": status,
                    "review_method": "AI_CONTEXT_SEMANTIC_REVIEW",
                    "reviewer": reviewer,
                    "review_notes": verdict["notes"],
                    "blind_id": blind_id,
                    "author": author,
                    "author_status": candidate["status"],
                    "author_notes": candidate["notes"],
                    "blind_review_status": verdict["status"],
                    "processed_text_sha256": verdict["text_sha256"],
                })
                final.append(status)
                counts[(ids["role"], status)] += 1
            if final == ["PASS", "PASS"]:
                complete[ids["role"]] += 1
        for stem in ("source", "generated", "blind_review", "reviewed"):
            path = PARTS / f"{stem}_{index}.jsonl"
            hashes[str(path.relative_to(RUN))] = digest(path)
    if seen != manifest.keys() or len(output) != 384:
        raise ValueError("All attempts, including failed versions, must be retained")
    reviewed_path = RUN / "local_only/reviewed_text.jsonl"
    write_jsonl(reviewed_path, output)
    lock = {
        "review_kind": "Two-stage AI-only context review, different reviewer from author",
        "human_confirmation": False,
        "n_targets": len(seen),
        "n_variants": len(output),
        "complete_ai_pass_targets": dict(complete),
        "role_variant_verdicts": {f"{role}/{status}": count for (role, status), count in sorted(counts.items())},
        "reviewed_text_sha256": digest(reviewed_path),
        "source_hashes": hashes,
    }
    with (RUN / "provenance/ai_review_lock.json").open("x") as handle:
        json.dump(lock, handle, indent=2)
    print(json.dumps({key: value for key, value in lock.items() if key != "source_hashes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--blind", type=int, choices=SHARDS)
    actions.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.assemble:
        assemble()
    else:
        blind(args.blind)
