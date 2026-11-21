#!/usr/bin/env python3
"""Audit source content, event and target identities, and aggregate accounting."""
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import rel_entr

RUN = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((RUN / "PROTOCOL.json").read_text())
PARENT = RUN.parent / PROTOCOL["parent"]
METRICS = (
    "mu", "delta", "JS", "flip_rate", "D_pred", "D_margin", "D_C", "V_C",
    "mean_abs_delta_C", "mean_abs_delta_P", "max_abs_delta_C", "std_original_class_probability",
)
TARGET_KEY = ("sample_id", "perturbation_type")
EVENT_KEY = TARGET_KEY + ("variant_id",)


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def unique(rows, fields):
    indexed = {tuple(r[f] for f in fields): r for r in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"Duplicate key for {fields}")
    return indexed


def count_tuple_mapping(mapping):
    result = {v: set() for values in mapping.values() for v in values}
    for meaning, labels in mapping.items():
        assert len(labels) == 1
        result[next(iter(labels))].add(meaning)
    assert all(len(meanings) == 1 for meanings in result.values())


def num(value, expected, why):
    if not np.isclose(float(value), expected, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{why}: {value} != {expected}")


def change_bin(value):
    if value < -.20:
        return 0
    if value < -.10:
        return 1
    if value < -.02:
        return 2
    if value <= .02:
        return 3
    if value <= .10:
        return 4
    if value <= .20:
        return 5
    return 6


def verify_event_summaries(events, originals):
    cross = read(RUN / "FLIP_CONFIDENCE_CROSS_TAB.csv")
    distribution = read(RUN / "CONFIDENCE_CHANGE_DISTRIBUTION.csv")
    checks = 0
    for family in PROTOCOL["families"]:
        valid = [(key, r) for key, r in events.items() if key[1] == family and r["metric_available"] == "1"]
        expected_states = defaultdict(Counter)
        for key, row in valid:
            counts = expected_states[row["confidence_response_state"]]
            old = originals[key]
            correct = old["original_correct"] == "1"
            pred = int(np.argmax(json.loads(old["perturbed_probabilities"])))
            pred_correct = pred == int(old["true_label"])
            counts["state_count"] += 1
            counts["original_correct_count"] += int(correct)
            counts["original_wrong_count"] += int(not correct)
            counts["reference_correct_to_wrong_count"] += int(correct and not pred_correct)
            counts["reference_wrong_to_correct_count"] += int(not correct and pred_correct)
        common = lambda row: (
            row["perturbation_type"] == family and row["analysis_scope"] == "INPUT_QC_AVAILABLE"
            and row["original_correctness_group"] == "ALL" and row["aggregation"] == "EVENT_WEIGHTED"
        )
        all_cross = [r for r in cross if common(r) and r["confidence_definition"] == "THRESHOLD_0.02"
                     and r["confidence_response_state"] != "NA_MISSING_PROBABILITY"]
        assert len(all_cross) == 6
        for row in all_cross:
            expected = expected_states[row["confidence_response_state"]]
            assert all(int(row[k]) == v for k, v in expected.items())
            num(row["fraction_metric_available"], expected["state_count"]/len(valid), "Cross-table denominator")
            checks += 6
        for metric in ("delta_C", "delta_P"):
            values = [float(r[metric]) for _, r in valid]
            observed = [r for r in distribution if common(r) and r["flip_group"] == "ALL" and r["metric"] == metric]
            assert len(observed) == 1
            observed = observed[0]
            wanted = Counter(change_bin(v)+1 for v in values)
            assert [int(observed[f"bin_{i}_count"]) for i in range(1, 8)] == [wanted[i] for i in range(1, 8)]
            num(observed["mean"], np.mean(values), "Distribution mean")
            num(observed["q50"], np.median(values), "Distribution median")
            num(observed["std_population"], np.std(values), "Distribution standard deviation")
            checks += 10
    return checks


def main():
    expected_files = [
        "PREDICTION_RESPONSE_DECOMPOSITION.md", "FLIP_CONFIDENCE_CROSS_TAB.csv",
        "CONFIDENCE_CHANGE_DISTRIBUTION.csv", "ORIGINAL_CLASS_SUPPORT_ANALYSIS.csv",
        "SAMPLE_RESPONSE_PATTERN.csv", "METRIC_RELATIONSHIP_ANALYSIS.csv",
        "CORRECT_WRONG_RESPONSE_COMPARISON.csv", "UNIFIED_BEHAVIOR_FEATURES.csv",
        "METRIC_SELECTION_RECOMMENDATION.md", "EVENT_RESPONSES.csv",
    ]
    assert all((RUN / name).is_file() and (RUN / name).stat().st_size for name in expected_files)
    for f in (RUN / "scripts").glob("*.py"):
        ast.parse(f.read_text())
    assert all(digest(RUN.parent / k) == v for k, v in PROTOCOL["source_sha256"].items())

    original = unique(read(PARENT / "results/perturbed_predictions.csv"), EVENT_KEY)
    events = unique(read(RUN / "EVENT_RESPONSES.csv"), EVENT_KEY)
    assert set(original) == set(events) and len(events) == 1624
    count_failed = 0
    group_stats = defaultdict(Counter)
    target_events = defaultdict(list)
    states, c_bins, p_bins = defaultdict(set), defaultdict(set), defaultdict(set)
    for key, row in events.items():
        before = original[key]
        assert row["status"] == before["status"]
        np_clean = np.asarray(json.loads(before["original_probabilities"]), dtype=float)
        clean_top = int(np_clean.argmax())
        np_pert = before["perturbed_probabilities"]
        if not np_pert:
            assert row["metric_available"] == "0"
            assert all(row[k] in ("", "NA") for k in ["delta_C", "delta_P", "JS", "flip"])
            count_failed += 1
            continue
        assert row["metric_available"] == "1" and row["status"] == "SUCCESS"
        q = np.asarray(json.loads(np_pert), dtype=float)
        top = int(q.argmax())
        flip = int(top != clean_top)
        non_original = np.arange(len(q)) != clean_top
        cm = np_clean[clean_top] - np_clean[non_original].max()
        pm = q[clean_top] - q[non_original].max()
        midpoint = (q + np_clean)/2
        js = (rel_entr(np_clean, midpoint).sum() + rel_entr(q, midpoint).sum())/2
        dc, dp = float(q[top]-np_clean[clean_top]), float(q[clean_top]-np_clean[clean_top])
        checks = dict(C_clean=np_clean[clean_top], C_perturbed=q[top], P_clean=np_clean[clean_top],
                      P_perturbed=q[clean_top], flip=flip, delta_C=dc, delta_P=dp, JS=js,
                      clean_margin=cm, perturbed_margin=pm)
        for field, expected in checks.items():
            num(row[field], expected, f"{key}, {field}")
        assert flip or dc == dp
        c_bins[-1 if dc < -.02 else 1 if dc > .02 else 0].add(row["delta_C_bin"])
        p_bins[-1 if dp < -.02 else 1 if dp > .02 else 0].add(row["delta_P_bin"])
        states[flip, "down" if dc < -.02 else "up" if dc > .02 else "unchanged"].add(row["confidence_response_state"])
        counters = group_stats[key[1]]
        counters["valid_events"] += 1
        counters["flip_events"] += flip
        counters["nonflip_events"] += (not flip)
        counters["nonflip_delta_C_negative"] += (not flip and dc < 0)
        counters["nonflip_delta_P_negative"] += (not flip and dp < 0)
        counters["flip_delta_C_positive"] += (flip and dc > 0)
        counters["flip_delta_C_above_0.02"] += (flip and dc > .02)
        target_events[key[:2]].append(checks)
    assert count_failed == 80
    assert len(c_bins) == len(p_bins) == 3 and len(states) == 6
    for mapping in (states, c_bins, p_bins):
        count_tuple_mapping(mapping)
    # Exercise bin boundary inclusivity without relying on observed values alone.
    assert [change_bin(t) for t in (-.21, -.20, -.10, -.02, .02, .10, .20, .21)] == [0, 1, 2, 3, 3, 4, 5, 6]
    summary_checks = verify_event_summaries(events, original)

    targets = unique(read(RUN / "UNIFIED_BEHAVIOR_FEATURES.csv"), TARGET_KEY)
    patterns = unique(read(RUN / "SAMPLE_RESPONSE_PATTERN.csv"), TARGET_KEY)
    expected_targets = {key[:2] for key in original}
    assert len(targets) == len(patterns) == 812 and set(targets) == set(patterns) == expected_targets
    complete_k2 = Counter()
    for key, row in targets.items():
        sample = target_events[key]
        if len(sample) != 2:
            assert all(not row[field] for field in METRICS)
            continue
        complete_k2[key[1]] += 1
        c = np.array([event["C_perturbed"] for event in sample])
        dc = np.array([event["delta_C"] for event in sample])
        dp = np.array([event["delta_P"] for event in sample])
        f = np.array([event["flip"] for event in sample])
        expected = dict(mu=np.mean(f*c), delta=abs(np.mean(dc)), JS=np.mean([r["JS"] for r in sample]),
                        flip_rate=f.mean(), D_pred=-dp.mean(), D_margin=sample[0]["clean_margin"]-np.mean(
                            [r["perturbed_margin"] for r in sample]),
                        D_C=dc.mean(), V_C=c.std(ddof=1), mean_abs_delta_C=np.abs(dc).mean(),
                        mean_abs_delta_P=np.abs(dp).mean(), max_abs_delta_C=np.abs(dc).max(),
                        std_original_class_probability=np.array([r["P_perturbed"] for r in sample]).std(ddof=1))
        for field, value in expected.items():
            num(row[field], value, f"{key}, {field}")
        prob_std = float(row["std_original_class_probability"])
        confidence_dispersion = float(row["V_C"]) / np.sqrt(2)
        num(row["max_abs_delta_C"], expected["delta"] + confidence_dispersion, "K2 max abs delta-C identity")
        num(row["mean_abs_delta_C"], max(expected["delta"], confidence_dispersion), "K2 mean abs delta-C identity")
        num(row["mean_abs_delta_P"], max(abs(expected["D_pred"]), prob_std / np.sqrt(2)), "K2 mean abs delta-P identity")
    assert dict(complete_k2) == {"neutral_append": 353, "semantic_paraphrase": 389}

    compare = read(RUN / "CORRECT_WRONG_RESPONSE_COMPARISON.csv")
    assert len(compare) == 576 and all(int(r["bootstrap_requested"]) == 2000 for r in compare)
    human = [r for r in compare if r["analysis_scope"] == "HUMAN_CONFIRMED"]
    assert len(human) == 192 and all(
        r["status"] == "NA_PENDING_ACTUAL_DUAL_REVIEW" and not r["error_minus_correct"]
        and not r["ci95_low"] and not r["ci95_high"] and not int(r["bootstrap_valid"]) for r in human
    )
    comparison_log = json.loads((RUN / "logs/comparison.json").read_text())
    assert comparison_log["old_scope_support_checks"] == 18 and comparison_log["old_effect_and_interval_checks"] == 96
    git_workdir = RUN.parents[3] / "CMERC"
    git_head = subprocess.check_output(["git", "-C", str(git_workdir), "rev-parse", "HEAD"], text=True).strip()
    assert git_head == PROTOCOL["git_head"]
    assert not (RUN / "verification.json").exists()
    result = {
        "status": "PASS_OFFLINE_EXPLORATORY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "MELD matched Validation, Seed 67137",
        "events": len(events), "failed_with_missing_perturbed_probability": count_failed,
        "full_probability_event_field_checks": 11*(len(events)-count_failed),
        "cross_table_and_distribution_summary_checks":summary_checks,
        "target_family_rows": len(targets), "complete_k2_counts": dict(complete_k2),
        "independent_k2_metric_checks": len(METRICS)*sum(complete_k2.values()),
        "verified_k2_exact_redundancy_checks": 3*sum(complete_k2.values()),
        "family_event_counts": {k: dict(v) for k, v in group_stats.items()},
        "comparison_rows": len(compare), "human_pending_rows": len(human),
        "legacy_matched_points_and_ci_replayed": 96, "legacy_scope_support_replayed": 18,
        "frozen_bootstrap_repetitions": 2000, "frozen_bootstrap_dialogues": 80,
        "original_sources_unchanged": True, "source_sha256": PROTOCOL["source_sha256"],
        "git_head": git_head, "no_models_loaded": True, "no_training": True, "no_new_inference": True,
        "no_test": True, "new_text_generated": False, "human_table_modified": False,
        "files_sha256": {str(f.relative_to(RUN)): digest(f) for f in sorted(RUN.rglob("*"))
                         if f.is_file() and "__pycache__" not in f.parts},
    }
    with (RUN / "verification.json").open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("files_sha256", "source_sha256")}, indent=2))

if __name__ == "__main__":
    main()
