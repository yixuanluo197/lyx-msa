#!/usr/bin/env python3
"""Freeze disjoint future scoring roles from MELD Train metadata (CPU only)."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / "20261120T143131Z"
SOURCE = PARENT / "local_only/data/MELD/manifest.csv"
BASE_SEED = 67137
ROLE_ORDER = ("base_fit", "base_inner_dev", "score_fit", "score_eval")
CLASSES = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def order(dialogue):
    return sha256(f"MELD|train|score-holdout-v1|{BASE_SEED}|{dialogue}".encode()).hexdigest(), dialogue


def assignments(dialogues):
    """Fixed hash order based on IDs only. Separate 15% fit and 15% eval groups."""
    ordered = sorted(dialogues, key=order)
    score_count = len(ordered)*15//100
    base = ordered[2*score_count:]
    dev_count = len(base)*10//100
    roles = {role: set(ids) for role, ids in zip(ROLE_ORDER, (
        base[dev_count:], base[:dev_count], ordered[:score_count], ordered[score_count:2*score_count]
    ))}
    assert all(roles.values()) and sum(map(len, roles.values())) == len(ordered) == len(set.union(*roles.values()))
    return roles


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    roles_path = RUN / "local_only/SPLIT_DIALOGUES.csv"
    roster_path = RUN / "local_only/SPLIT_UTTERANCES.csv"
    lock_path = RUN / "SPLIT_LOCK.json"
    for path in (roles_path, roster_path, lock_path):
        if path.exists():
            raise FileExistsError(path)
    before = digest(SOURCE)
    dialogues = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "MELD" or row["split"] != "train":
                continue
            did, iid = row["dialogue_id"], row["utterance_id"]
            if not re.fullmatch(r"[0-9]+", did) or not re.fullmatch(r"[0-9]+", iid):
                raise ValueError("Unexpected MELD Train ID")
            dialogues[did].append((iid, row["utterance_index"], row["label"], row["inner_subset"]))
    utterance_count = sum(map(len, dialogues.values()))
    assert utterance_count == 9989
    assert all(len({r[3] for r in items}) == 1 for items in dialogues.values()), "Original Dialogue role inconsistency"
    roles = assignments(dialogues)
    assert roles == assignments(reversed(list(dialogues))), "Assignment depends on input order"
    roster, groups, counts = [], [], {}
    for role in ROLE_ORDER:
        label_counts, prior_roles = Counter(), Counter()
        for did in sorted(roles[role], key=int):
            items = sorted(dialogues[did], key=lambda row: int(row[1]))
            assert [int(r[1]) for r in items] == list(range(len(items)))
            groups.append(dict(dialogue_id=did, role=role, utterance_count=len(items)))
            for iid, index, label, old in items:
                roster.append(dict(sample_id=f"MELD:train:{did}:{iid}", dialogue_id=did,
                                   utterance_id=iid, utterance_index=index, role=role))
                label_counts[CLASSES[int(label)]] += 1
                prior_roles[old] += 1
        counts[role] = dict(dialogues=len(roles[role]), utterances=sum(label_counts.values()),
                            historical_train_role_utterances=dict(prior_roles), emotion_labels=dict(label_counts))
    assert len({row["sample_id"] for row in roster}) == utterance_count
    assert digest(SOURCE) == before
    # Labels are observed only after the hash assignment, for descriptive support checks.
    # No rebalancing or reroll happens if a class has a small count.
    common = [k for k in CLASSES if all(counts[role]["emotion_labels"].get(k, 0) > 0 for role in ROLE_ORDER)]
    write_csv(roles_path, groups)
    write_csv(roster_path, roster)
    head = subprocess.check_output(["git","-C",str(RUN.parents[3]/"CMERC"),"rev-parse","HEAD"],text=True).strip()
    lock = dict(status="FROZEN_PRE_MODEL_OUTPUT", frozen_at=datetime.now(timezone.utc).isoformat(),
                dataset="MELD", source_split="Train", branch_seed=BASE_SEED, git_head=head,
                policy="SHA256-sorted Dialogue IDs. First floor(15%) score_fit, next floor(15%) score_eval, remaining base with first floor(10%) inner_dev.",
                train_dialogues=len(dialogues), train_utterances=utterance_count, counts=counts, labels_present_in_all_roles=common,
                no_test_data=True, official_validation_rows_used=0, raw_texts_read=False,
                possible_roles_of_existing_model="Existing baseline used Train fit or inner_dev categories; a new independent training chain is required for these heldouts.",
                source_manifest_sha256=before, hashes={p.name:digest(p) for p in (roles_path,roster_path,Path(__file__))})
    with lock_path.open("x") as stream:
        json.dump(lock,stream,indent=2)
        stream.write("\n")
    print(json.dumps(lock,indent=2))

if __name__ == "__main__":
    main()
