#!/usr/bin/env python3
"""Compare matched error/correct responses using only frozen Dialogue draws."""
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((RUN / "PROTOCOL.json").read_text())
PARENT = RUN.parent / PROTOCOL["parent"]
sys.path.insert(0, str(PARENT / "scripts"))
import analyze_matched as frozen

SCOPES = {
    "AUTOMATIC_INPUT_QC_DIAGNOSTIC": "automatic_qc_eligible",
    "AI_REVIEWED_EXPLORATORY": "ai_reviewed_exploratory_eligible",
    "HUMAN_CONFIRMED": "human_confirmed_eligible",
}
METRICS = (
    "mu", "delta", "JS", "flip_rate", "D_pred", "D_margin", "D_C", "V_C",
    "mean_abs_delta_C", "mean_abs_delta_P", "max_abs_delta_C",
    "std_original_class_probability",
)
SORTERS = {
    "all": lambda row: True,
    "gap_le_0.02": lambda row: float(row["confidence_abs_gap"]) <= .02,
    "gap_le_0.01": lambda row: float(row["confidence_abs_gap"]) <= .01,
    "both_mu_zero": lambda row: row["both_mu_zero"],
}
LEGACY = {"mu": "mu", "delta": "delta", "JS": "js_mean", "flip_rate": "flip_rate"}


def require(test, why):
    if not test:
        raise ValueError(why)


def read(name):
    return frozen.read_csv(PARENT / "results" / name)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def source_checks():
    for rel, expected in PROTOCOL["source_sha256"].items():
        require(digest(RUN.parent / rel) == expected, f"Frozen source changed: {rel}")
    # Human confirmation is explicitly unavailable in this run.
    forms = RUN.parent / PROTOCOL["prior_experiment"] / "local_only/human_review"
    require(not list(forms.glob("reviewer_*_completed.csv")) and not list(forms.rglob("human_consensus.csv")),
            "Human files arrived after freeze; validate provenance before processing human results")


def main():
    source_checks()
    targets = frozen.read_csv(RUN / "UNIFIED_BEHAVIOR_FEATURES.csv")
    pairs = read("matched_pair_manifest.csv")
    frozen.validate_cohort(pairs, read("matched_sample_manifest.csv"))
    orig_targets = {(r["sample_id"], r["perturbation_type"]): r for r in read("matched_analysis/target_behavior.csv")}
    by_key = {(r["sample_id"], r["perturbation_type"]): r for r in targets}
    require(len(by_key) == len(targets) == 812 and set(by_key) == set(orig_targets), "Target-family coverage")
    require(len(pairs) == 203, "Frozen cohort must contain 203 pairs")
    for key, r in by_key.items():
        prior = orig_targets[key]
        require(r["original_correct"] == prior["original_correct"], "Correctness mismatch")
        require(all(frozen.truth(r[g]) == frozen.truth(prior[g]) for g in tuple(SCOPES.values())[:2]),
                "Historical QC or AI gate mismatch")
        require(not r["human_confirmed_eligible"], "A pending human flag must be NA")

    old_pairs = {(r["pair_id"], r["perturbation_type"], r["analysis_scope"]): r
                 for r in read("matched_analysis/pair_behavior_and_eligibility.csv")}
    old_support = {(r["perturbation_type"], r["analysis_scope"], r["confidence_caliper"]): r
                   for r in read("matched_analysis/condition_support.csv")
                   if r["residual_confidence_group"] == "all"}
    old_contrasts = {(r["perturbation_type"], r["analysis_scope"], r["confidence_caliper"], r["metric"], r["estimand"]): r
                     for r in read("matched_analysis/paired_metric_contrasts.csv")
                     if r["residual_confidence_group"] == "all"}
    with np.load(PARENT / "results/matched_analysis/bootstrap_draw_indices.npz", allow_pickle=False) as saved:
        dialogues, indices = saved["dialogue_ids"].tolist(), saved["indices"]
    require(indices.shape == (2000, 80) and set(dialogues) == {p["dialogue_id"] for p in pairs}, "Bootstrap shape or IDs")
    require(np.array_equal(indices, frozen.bootstrap_draws(dialogues)), "Frozen Bootstrap draws do not replay")
    registry = defaultdict(list)
    for pair in pairs:
        for family in PROTOCOL["families"]:
            correct = by_key[pair["correct_sample_id"], family]
            error = by_key[pair["error_sample_id"], family]
            for scope, gate in SCOPES.items():
                eligible = frozen.truth(correct[gate]) and frozen.truth(error[gate])
                prior = old_pairs[pair["pair_id"], family, scope]
                require(eligible == frozen.truth(prior["eligible"]), "Pair eligibility mismatch")
                if not eligible:
                    continue
                row = dict(pair, both_mu_zero=float(correct["mu"]) == float(error["mu"]) == 0,
                           confidence_rank=frozen.concordance(float(pair["correct_confidence"]) - float(pair["error_confidence"])))
                for metric in METRICS:
                    c, e = float(correct[metric]), float(error[metric])
                    diff = e - c
                    rank = frozen.concordance(-diff if metric == "D_C" else diff)
                    row.update({f"{metric}_correct": c, f"{metric}_error": e, f"{metric}_diff": diff,
                                f"{metric}_rank": rank, f"{metric}_rank_gain": rank - row["confidence_rank"]})
                registry[family, scope].append(row)

    rows = []
    validated_support = validated_contrasts = 0
    for family in PROTOCOL["families"]:
        for scope in SCOPES:
            for subset, select in SORTERS.items():
                use = [r for r in registry[family, scope] if select(r)]
                if subset != "both_mu_zero":
                    candidate_count = sum(select(r) for r in pairs)
                else:
                    # Show conditional count on verified scope, in addition to frozen cohort coverage.
                    candidate_count = len(registry[family, scope])
                n_dialogues = len({r["dialogue_id"] for r in use})
                if subset != "both_mu_zero":
                    expected = old_support[family, scope, subset]
                    require((len(use), n_dialogues) ==
                            (int(expected["eligible_pairs"]), int(expected["eligible_dialogues"])), "Scope support mismatch")
                    validated_support += 1
                info = dict(perturbation_type=family, analysis_scope=scope, subset=subset,
                            candidate_pairs=candidate_count, pairs=len(use), dialogues=n_dialogues,
                            candidate_pair_coverage=len(use)/candidate_count if candidate_count else None,
                            fraction_of_frozen_pairs=len(use)/len(pairs),
                            minimum_descriptive_support=len(use)>=30 and n_dialogues>=20,
                            status=("NA_PENDING_ACTUAL_DUAL_REVIEW" if scope == "HUMAN_CONFIRMED" else
                                    "EXPLORATORY" if use else "NA_NO_ELIGIBLE_PAIRS"))
                for metric in METRICS:
                    fields = {stat: frozen.summarize_values(use, f"{metric}_{stat}", dialogues, indices)
                              for stat in ("correct", "error", "diff", "rank", "rank_gain")}
                    baseline = frozen.summarize_values(use, "confidence_rank", dialogues, indices)
                    for estimand in ("equal_dialogue", "equal_pair"):
                        estimates = {stat: vals[estimand][0] for stat, vals in fields.items()}
                        diff_samples = fields["diff"][estimand][1]
                        rank_samples = fields["rank_gain"][estimand][1]
                        low, high = frozen.interval(diff_samples)
                        gain_low, gain_high = frozen.interval(rank_samples)
                        standard_dev = lambda stat: float(np.std([r[f"{metric}_{stat}"] for r in use], ddof=1)) if len(use)>1 else None
                        row = dict(info, metric=metric, estimand=estimand, original_correct_mean=estimates["correct"],
                                   original_wrong_mean=estimates["error"], correct_sd_unweighted=standard_dev("correct"),
                                   wrong_sd_unweighted=standard_dev("error"), error_minus_correct=estimates["diff"],
                                   ci95_low=low, ci95_high=high, rank=estimates["rank"],
                                   confidence_pair_rank=baseline[estimand][0], rank_gain_vs_confidence=estimates["rank_gain"],
                                   gain_ci95_low=gain_low, gain_ci95_high=gain_high, bootstrap_requested=len(indices),
                                   bootstrap_valid=int(np.isfinite(diff_samples).sum()))
                        rows.append(row)
                        if subset != "both_mu_zero" and metric in LEGACY and use:
                            expected = old_contrasts[family, scope, subset, LEGACY[metric], estimand]
                            pairs_to_check = [(estimates["diff"], "error_minus_correct"), (low, "ci95_low"), (high, "ci95_high")]
                            require(all(np.isclose(v, float(expected[k]), atol=1e-12, rtol=1e-12)
                                        for v, k in pairs_to_check), "Legacy point/CI mismatch")
                            validated_contrasts += 1
    require(validated_support == 18 and validated_contrasts == 96, "Replay completeness")
    require(len(rows) == len(PROTOCOL["families"])*len(SCOPES)*len(SORTERS)*len(METRICS)*2,
            "All metric/scope/subset combinations must be accounted for")
    require(not (RUN / "CORRECT_WRONG_RESPONSE_COMPARISON.csv").exists(), "Use a new run for comparisons")
    source_checks()
    frozen.write_csv(RUN / "CORRECT_WRONG_RESPONSE_COMPARISON.csv", rows)
    audit = dict(status="PASS", row_count=len(rows), nonempty_rows=sum(r["bootstrap_valid"]>0 for r in rows),
                 old_scope_support_checks=validated_support, old_effect_and_interval_checks=validated_contrasts,
                 source_locks=len(PROTOCOL["source_sha256"]), bootstrap_identical=True,
                 human_data_available=False, no_model_or_fitting=True)
    frozen.save_json(RUN / "logs/comparison.json", audit)
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
