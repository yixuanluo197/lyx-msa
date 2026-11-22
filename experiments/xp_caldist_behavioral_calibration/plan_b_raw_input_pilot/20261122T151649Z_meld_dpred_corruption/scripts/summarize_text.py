"""Recompute signed and absolute text response from saved variant probabilities."""
from collections import Counter, defaultdict
import csv
import json
from math import isclose
from statistics import mean, median

from audit_inputs import RUN, SCORE, read_csv, read_json


def vector(field):
    values = json.loads(field)
    assert len(values) == 7 and all(0 <= value <= 1 for value in values)
    assert abs(sum(values) - 1) < 1e-6
    return values


def outputs(rows, original, planned):
    if len(rows) != 2 or not all(row["eligible"] in {"1", "True"} for row in rows):
        assert not planned
        return ("", "") * 2
    label = max(range(7), key=original.__getitem__)
    values = [vector(row["perturbed_probabilities"]) for row in rows]
    changes = [original[label] - p[label] for p in values]
    cost = mean(changes)
    assert isclose(cost, float(planned), abs_tol=1e-12, rel_tol=0)
    return (cost, mean(map(abs, changes)), mean(max(range(7), key=p.__getitem__) != label for p in values),
            max(abs(a - b) for p in values for a, b in zip(original, p)))


def main():
    folder = RUN / "results/text_comparison"
    score_rows = read_csv(folder / "score_inputs.csv")
    variant_rows = read_csv(folder / "corruption_variants.csv")
    semantic = read_csv(SCORE / "results/score_transformations.csv")
    assert len(score_rows) == 2976 and len(variant_rows) == 384 and len(semantic) == 5952
    joins = defaultdict(list)
    joint = []
    causes = Counter()
    categories = {"semantic": semantic, "corruption": variant_rows}
    for family, variants in categories.items():
        for row in variants:
            joins[(family, row["sample_id"])].append(row)
            if family == "corruption":
                causes[(row["role"], row["reason"])] += 1
    attempted = {row["sample_id"] for row in variant_rows}
    assert len(attempted) == 192
    for row in score_rows:
        sid = row["sample_id"]
        if sid not in attempted:
            continue
        clean = vector(row["original_probabilities"])
        s = outputs(joins[("semantic", sid)], clean, row["D_pred"])
        c = outputs(joins[("corruption", sid)], clean, row["corruption_d_pred"])
        value = {key: row[key] for key in ("sample_id", "dialogue_id", "role", "original_correct", "original_confidence")}
        value["common"] = s[0] != "" and c[0] != ""
        for name, result in (("semantic", s), ("corruption", c)):
            value.update({f"{name}_{k}": v for k, v in zip(("D_pred", "D_abs", "flip_rate", "max_vector_delta"), result)})
        joint.append(value)
    assert len(joint) == 192
    verification = read_json(folder / "inference_verification.json")
    assert sum(row["corruption_D_pred"] != "" for row in joint) == verification["available_corruption_d_pred"]
    with (folder / "paired_behavior.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joint[0]))
        writer.writeheader()
        writer.writerows(joint)
    groups = []
    for scope in ("family_available", "common"):
        for role in ("score_fit", "score_eval"):
            for family in categories:
                for outcome in ("all", "1", "0"):
                    chosen = [row for row in joint if row["role"] == role and row[family + "_D_pred"] != ""
                              and (scope != "common" or row["common"])
                              and (outcome == "all" or row["original_correct"] == outcome)]
                    record = dict(scope=scope, role=role, family=family, original_correct=outcome, n=len(chosen))
                    for metric in ("D_pred", "D_abs", "flip_rate", "max_vector_delta"):
                        vals = [row[family + "_" + metric] for row in chosen]
                        record["mean_" + metric] = mean(vals) if vals else None
                        record["median_" + metric] = median(vals) if vals else None
                    groups.append(record)
    with (folder / "family_response_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    summary = dict(
        total_planned_versions=len(variant_rows),
        response=[
            row for row in groups if row["original_correct"] == "all"],
        variant_reasons={"/".join(k): v for k, v in sorted(causes.items())},
        no_candidate_is_separate_from_context_failure=True,
    )
    with (folder / "text_response_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Verified D_pred for both conditions on all {len(joint)} targets.")


if __name__ == "__main__":
    main()
