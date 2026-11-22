"""Generate locked one-swap controls and assemble AI-only context judgments."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import re

from audit_inputs import BASE, RUN, SCORE, digest, read_csv, read_json


PRIVATE = RUN / "local_only"
SHARDS = range(4)
WORD = re.compile(r"(?<![\w'’\-])[a-z]{5,}(?![\w'’\-])")
STATES = {"PASS", "FAIL", "UNCERTAIN"}


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, entries):
    with path.open("x", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=True) + "\n")


def lock_inputs():
    path = RUN / "provenance/INPUT_AUDIT.json"
    audit = read_json(path)
    assert audit["status"] == "INPUTS_VERIFIED" and audit["n_fixed_targets"] == 192
    config = RUN / "EXPERIMENT_PROTOCOL.json"
    assert audit["sha256_relative_to_pilot_parent"][str(config.relative_to(BASE))] == digest(config)
    options = read_json(config)["text"]["corruption"]
    return path, options


def text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def candidates(original, blind_id, config):
    allowed = set(config["allowlist"])
    selected = {}
    for match in WORD.finditer(original):
        word = match.group()
        if word not in allowed:
            continue
        for index in range(1, len(word) - 2):
            position = match.start() + index
            before = original[position:position + 2]
            after = before[::-1]
            if before == after:
                continue
            changed = original[:position] + after + original[position + 2:]
            edit = dict(position=position, before=before, after=after, original_word=word,
                        edited_word=changed[match.start():match.end()], word_start=match.start(), word_end=match.end())
            selected.setdefault(changed, edit)
    rank = lambda item: (text_hash(f"{config['seed']}|{blind_id}|{item[1]['position']}|{item[0]}"), item[0])
    return sorted(selected.items(), key=rank)[:2]


def generate():
    verified, options = lock_inputs()
    source = []
    for shard in range(6):
        path = SCORE / "local_only/parts" / f"source_{shard}.jsonl"
        key = str(path.relative_to(BASE))
        assert digest(path) == read_json(verified)["sha256_relative_to_pilot_parent"][key]
        source.extend(rows(path))
    assert len(source) == len({row["blind_id"] for row in source}) == 192
    output = []
    counts = Counter()
    sources = {}
    for row in source:
        original = row["original_text"]
        offered = candidates(original, row["blind_id"], options)
        planned = {}
        for i, variant in enumerate(options["variants"]):
            if i < len(offered):
                changed, edit = offered[i]
                planned[variant] = dict(text=changed, edit=edit, generation_status="GENERATED")
            else:
                reason = "NO_ELIGIBLE_EDIT" if not offered else "NO_SECOND_DISTINCT_EDIT"
                planned[variant] = dict(text=original, edit=None, generation_status=reason)
            counts[planned[variant]["generation_status"]] += 1
        entry = {**row, "candidates": planned}
        output.append(entry)
    folder = PRIVATE / "review"
    folder.mkdir(exist_ok=False)
    for shard in SHARDS:
        path = folder / f"source_{shard}.jsonl"
        write(path, output[shard::len(SHARDS)])
        sources[str(path.relative_to(RUN))] = digest(path)
    summary = dict(n_targets=len(output), n_planned_versions=384, generation_counts=dict(counts),
                   generation_sha256=digest(RUN / "scripts/orthographic_edits.py"),
                   input_audit_sha256=digest(verified), output_hashes=sources)
    with (RUN / "provenance/GENERATION_LOCK.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["generation_counts"]))


def assemble():
    audit_path, options = lock_inputs()
    generation_path = RUN / "provenance/GENERATION_LOCK.json"
    lock = read_json(generation_path)
    assert digest(audit_path) == lock["input_audit_sha256"]
    folder = PRIVATE / "review"
    assignments_path = folder / "review_assignments.json"
    assigned = read_json(assignments_path)
    assert set(assigned) == set(map(str, SHARDS)) and all(assigned.values())
    original = SCORE / "local_only/perturbation_attempt_manifest.csv"
    assert digest(original) == read_json(audit_path)["sha256_relative_to_pilot_parent"][str(original.relative_to(BASE))]
    mapping = {row["blind_id"]: row for row in read_csv(original)}
    counts = Counter()
    complete = Counter()
    reviewed = []
    seen = set()
    files = {str(assignments_path.relative_to(RUN)): digest(assignments_path)}
    for shard in SHARDS:
        path = folder / f"source_{shard}.jsonl"
        assert digest(path) == lock["output_hashes"][str(path.relative_to(RUN))]
        source = rows(path)
        verdict_path = folder / f"reviewed_{shard}.jsonl"
        decisions = rows(verdict_path)
        verdicts = {entry["blind_id"]: entry["variants"] for entry in decisions}
        assert len(verdicts) == len(decisions) == len(source) == 48
        assert set(verdicts) == {entry["blind_id"] for entry in source}
        files[str(verdict_path.relative_to(RUN))] = digest(verdict_path)
        for row in source:
            blind_id = row["blind_id"]
            assert blind_id not in seen
            seen.add(blind_id)
            original_text = row["original_text"]
            target = mapping[blind_id]
            offered = candidates(original_text, blind_id, options)
            assert set(verdicts[blind_id]) == set(options["variants"])
            passes = 0
            for index, variant in enumerate(options["variants"]):
                candidate = row["candidates"][variant]
                decision = verdicts[blind_id][variant]
                status = decision["status"]
                generation = candidate["generation_status"]
                assert status in STATES and decision["notes"].strip()
                processed = candidate["text"]
                assert decision["text_sha256"] == text_hash(processed)
                if generation == "GENERATED":
                    assert offered[index] == (processed, candidate["edit"])
                    method = "AI_CONTEXT_ORTHOGRAPHIC_REVIEW"
                else:
                    assert index >= len(offered) and status == "FAIL"
                    assert candidate["edit"] is None and processed == original_text
                    method = "NO_CANDIDATE"
                counts[(target["role"], status, generation)] += 1
                passes += status == "PASS"
                reviewed.append({
                    **target,
                    "variant_id": variant,
                    "original_text_sha256": text_hash(original_text),
                    "processed_text_sha256": text_hash(processed),
                    "processed_text": processed,
                    "edit": candidate["edit"],
                    "generation_status": generation,
                    "review_status": status,
                    "review_method": method,
                    "review_notes": decision["notes"],
                    "reviewer": assigned[str(shard)],
                })
            complete[target["role"]] += passes == 2
    assert seen == mapping.keys() and len(reviewed) == 384
    reviewed_path = PRIVATE / "corruption_reviewed.jsonl"
    write(reviewed_path, reviewed)
    result = dict(status="AI_REVIEW_LOCKED", n_targets=len(seen), n_versions=len(reviewed),
                  human_review=False, blind_ai_review=True, input_generation_sha256=digest(generation_path),
                  counts={"/".join(k): v for k, v in sorted(counts.items())}, complete_ai_pass=dict(complete),
                  reviewed_sha256=digest(reviewed_path), review_input_files=files)
    with (RUN / "provenance/CORRUPTION_REVIEW_LOCK.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ("counts", "complete_ai_pass")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "assemble"))
    {"generate": generate, "assemble": assemble}[parser.parse_args().action]()
