"""Check the completed comparison from its saved probabilities and hash the delivery."""
from collections import Counter
import csv
import json
import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from audit_inputs import BASE, RUN, digest, read_csv, read_json


def calculate_bootstrap(rows, seed, draws):
    covered = [row for row in rows if row["common_eval_eligible"] == "True"]
    universe = sorted({row["dialogue_id"] for row in rows})
    index = {key: i for i, key in enumerate(universe)}
    locations = [index[row["dialogue_id"]] for row in covered]
    y = np.asarray([int(row["original_correct"]) for row in covered])
    p1 = np.clip([float(row["E1_common_fit_R"]) for row in covered], 1e-15, 1 - 1e-15)
    p2 = np.clip([float(row["E2_common_fit_R"]) for row in covered], 1e-15, 1 - 1e-15)
    difference = np.where(y == 1, np.log(p2 / p1), np.log((1 - p2) / (1 - p1)))
    values = []
    for draw in np.random.RandomState(seed).randint(0, len(universe), size=(draws, len(universe))):
        weights = np.bincount(draw, minlength=len(universe))[locations]
        if weights.sum():
            values.append(np.average(difference, weights=weights))
    if len(difference) == 0 or len(values) == 0:
        return None
    return float(difference.mean()), np.quantile(values, [.025, .975]).tolist(), len(values)


def metric_values(rows, field):
    y = np.asarray([int(row["original_correct"]) for row in rows])
    p = np.asarray([float(row[field]) for row in rows])
    nll = log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1], eps=1e-15)
    rankable = len(set(y)) == 2
    return dict(binary_nll=nll, brier=brier_score_loss(y, p),
                auroc_error=roc_auc_score(1-y, 1-p) if rankable else None,
                auprc_error=average_precision_score(1-y, 1-p) if rankable else None)


def main():
    audit = read_json(RUN / "provenance/INPUT_AUDIT.json")
    for path, expected in audit["sha256_relative_to_pilot_parent"].items():
        assert digest(BASE / path) == expected, path
    fit = RUN / "results/scoring"
    scorer = read_json(fit / "scoring_report.json")
    evaluation = read_csv(fit / "evaluation_predictions.csv")
    assert len(evaluation) == len({row["sample_id"] for row in evaluation}) == 1407
    assert all(row["role"] == "score_eval" and row["sample_id"].startswith("MELD:train:") for row in evaluation)
    inference = read_json(RUN / "results/text_comparison/inference_verification.json")
    for filename, expected in inference["outputs"].items():
        assert digest(RUN / "results/text_comparison" / filename) == expected
    checks = {}
    for arm in ["E1", "E2", "E3"]:
        missing = [row for row in evaluation if row[arm + "_fallback_to_E0_full"] == "True"]
        assert all(row[arm + "_policy_R"] == row["E0_full_fit_R"] for row in missing)
        assert len(missing) == scorer["coverage"][arm]["deployment_fallback"]
        checks[arm + "_exact_fallback_rows"] = len(missing)
    mutually_eligible = [row for row in evaluation if row["common_eval_eligible"] == "True"]
    primary = next(row for row in scorer["comparisons"]
                   if row["scope"] == "PRIMARY_COMMON_FIT_COMMON_EVAL" and row["comparison"] == "E1_to_E2")
    assert primary["evaluation_support"]["n"] == len(mutually_eligible)
    if primary["status"] == "COMPUTED":
        for column, name in (("E1_common_fit_R", "baseline"), ("E2_common_fit_R", "augmented")):
            for metric, result in metric_values(mutually_eligible, column).items():
                actual = primary[name][metric]
                assert (actual is None and result is None) or math.isclose(actual, result, rel_tol=0, abs_tol=1e-12)
        settings = read_json(RUN / "EXPERIMENT_PROTOCOL.json")["scoring"]["bootstrap"]
        estimate, interval, n_draws = calculate_bootstrap(evaluation, settings["seed"], settings["replicates"])
        b = primary["paired_dialogue_improvement"]["binary_nll"]
        assert math.isclose(b["improvement"], estimate, rel_tol=0, abs_tol=1e-12)
        assert np.allclose([b["improvement_ci_low"], b["improvement_ci_high"]], interval, atol=1e-12, rtol=0)
        assert b["valid_draws"] == n_draws
        checks["primary_binary_nll_independent_replay"] = dict(estimate=estimate, interval=interval, n_draws=n_draws)
    else:
        checks["primary_binary_nll_independent_replay"] = primary["status"]
    for stage in ("dpred", "audio"):
        assert (RUN / f"results/{stage}/verification.json").is_file()
    assert read_json(RUN / "provenance/PILOT_EXECUTION.json")["status"] == "COMPLETE"
    checks["original_inputs_unchanged"] = len(audit["sha256_relative_to_pilot_parent"])
    checks["summary_files"] = sorted(p.name for p in RUN.glob("*.md"))
    files = [
        p for p in RUN.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name not in {"SHA256SUMS.csv", "verification.json"}
    ]
    files.extend(RUN / "results" / stage / "verification.json" for stage in ("audio", "dpred"))
    assert len(set(files)) == len(files)
    paths = sorted(files)
    manifest = RUN / "SHA256SUMS.csv"
    with manifest.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for path in paths:
            writer.writerow([path.relative_to(RUN), digest(path), path.stat().st_size])
    out = dict(status="COMPLETE_EXPLORATORY_PILOT", new_independent_evaluation=False,
               test_accessed=False, official_validation_new_inference=False,
               evidence=checks, sha256_manifest=digest(manifest),
               files_by_top_level=dict(Counter(path.relative_to(RUN).parts[0] for path in paths)))
    with (RUN / "verification.json").open("x") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
