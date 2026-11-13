"""Read-only Seed 1475 true-Validation confidence diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
PILOT = OUT.parents[1] / "pilot" / "20261113T025053Z"
KEY = ["dataset", "dialogue_id", "utterance_index"]
CLASSES = {
    "IEMOCAP": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "MELD": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
}
MIN_CLASS_N = 20
COLORS = {"IEMOCAP": "#0072B2", "MELD": "#D55E00"}


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.12g")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "NA"


def stats(values: pd.Series) -> dict:
    values = values.astype(float)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else np.nan,
        "median": float(values.median()) if len(values) else np.nan,
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "q25": float(values.quantile(.25)) if len(values) else np.nan,
        "q75": float(values.quantile(.75)) if len(values) else np.nan,
        "q90": float(values.quantile(.90)) if len(values) else np.nan,
    }


def ece15(conf: np.ndarray, correct: np.ndarray) -> float:
    edges = np.linspace(0, 1, 16)
    ix = np.minimum(np.searchsorted(edges, conf, side="right") - 1, 14)
    return sum(
        np.sum(ix == b) / len(conf) * abs(np.mean(correct[ix == b]) - np.mean(conf[ix == b]))
        for b in range(15) if np.any(ix == b)
    )


def correctness_auc(confidence: pd.Series, correct: pd.Series) -> float:
    """Mann–Whitney AUC with average ranks for ties; no extra dependency."""
    positive = correct.eq(1)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    ranks = confidence.rank(method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_gap(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    """Dialogue-cluster percentile interval for accuracy minus mean confidence."""
    groups = [g for _, g in frame.groupby("dialogue_id", sort=True)]
    rng = np.random.default_rng(1475)
    sums = np.array([g.correct.sum() - g[column].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(np.quantile(estimates, [.025, .975]).tolist())


def main() -> None:
    original_path = PILOT / "original_predictions.csv"
    oof_path = PILOT / "oof_calibration_predictions.csv"
    metrics_path = PILOT / "calibration_metrics.csv"
    original = pd.read_csv(original_path, dtype={"dialogue_id": str})
    oof = pd.read_csv(oof_path, dtype={"dialogue_id": str})
    pilot_metrics = pd.read_csv(metrics_path)
    assert original[KEY].notna().all().all() and not original.duplicated(KEY).any()
    assert set(original.seed) == {1475}
    assert set(original.dataset) == set(CLASSES)
    e1 = oof.query("method == 'E1' and corruption == 'none' and modality == 'none'").copy()
    e0 = oof.query("method == 'E0' and corruption == 'none' and modality == 'none'").copy()
    assert not e0.duplicated(KEY).any() and not e1.duplicated(KEY).any()
    joined = original.merge(
        e1[KEY + ["label", "prediction", "correct", "confidence"]].rename(columns={
            "label": "e1_label", "prediction": "e1_prediction", "correct": "e1_correct",
            "confidence": "e1_confidence"}),
        on=KEY, how="inner", validate="one_to_one",
    ).merge(e0[KEY + ["confidence"]].rename(columns={"confidence": "e0_oof_confidence"}),
            on=KEY, how="inner", validate="one_to_one")
    assert len(joined) == len(original) == len(e0) == len(e1)
    assert (joined.label == joined.e1_label).all()
    assert (joined.prediction == joined.e1_prediction).all()
    joined["correct"] = (joined.prediction == joined.label).astype(int)
    assert (joined.correct == joined.e1_correct).all()
    assert np.allclose(joined.confidence, joined.e0_oof_confidence, atol=1e-12, rtol=0)
    assert joined[["confidence", "e1_confidence"]].notna().all().all()
    assert joined[["confidence", "e1_confidence"]].ge(0).all().all()
    assert joined[["confidence", "e1_confidence"]].le(1).all().all()
    joined["shift"] = joined.e1_confidence - joined.confidence
    joined["predicted_emotion"] = [CLASSES[d][int(i)] for d, i in zip(joined.dataset, joined.prediction)]
    joined["true_emotion"] = [CLASSES[d][int(i)] for d, i in zip(joined.dataset, joined.label)]
    expected = {"IEMOCAP": (527, 12), "MELD": (1109, 114)}
    for dataset, frame in joined.groupby("dataset"):
        assert (len(frame), frame.dialogue_id.nunique()) == expected[dataset]
        for method, col in [("E0", "confidence"), ("E1", "e1_confidence")]:
            p = pilot_metrics.query("dataset == @dataset and method == @method and corruption == 'none'").iloc[0]
            assert abs(ece15(frame[col].to_numpy(), frame.correct.to_numpy()) - p.ece15) < 1e-10
            assert abs(np.mean((frame[col] - frame.correct) ** 2) - p.binary_brier) < 1e-10
            clipped = np.clip(frame[col].to_numpy(), 1e-12, 1 - 1e-12)
            r = frame.correct.to_numpy()
            nll = -np.mean(r * np.log(clipped) + (1 - r) * np.log1p(-clipped))
            assert abs(nll - p.binary_nll) < 1e-10
    save_csv(joined[KEY + ["label", "true_emotion", "prediction", "predicted_emotion", "correct",
                           "confidence", "e1_confidence", "shift"]], "e0_e1_confidence_shift.csv")

    dist_rows, bin_rows, class_rows, comparison_rows = [], [], [], []
    for dataset, frame in joined.groupby("dataset", sort=True):
        for correct, group in frame.groupby("correct"):
            dist_rows.append({"dataset": dataset, "group": "correct" if correct else "wrong",
                              **stats(group.confidence), "fraction_below_0_5": float((group.confidence < .5).mean())})
            delta = group["shift"]
            comparison_rows.append({"dataset": dataset, "group": "correct" if correct else "wrong",
                                    "n": len(group), "mean_e0": group.confidence.mean(),
                                    "mean_e1": group.e1_confidence.mean(), "mean_shift": delta.mean(),
                                    "median_shift": delta.median(), "raised_fraction": (delta > 1e-12).mean(),
                                    "lowered_fraction": (delta < -1e-12).mean(),
                                    "unchanged_fraction": (delta.abs() <= 1e-12).mean()})
        for method, col in [("E0", "confidence"), ("E1", "e1_confidence")]:
            for i in range(10):
                lo, hi = i / 10, (i + 1) / 10
                b = frame[(frame[col] >= lo) & ((frame[col] < hi) if i < 9 else (frame[col] <= hi))]
                bin_rows.append({"dataset": dataset, "method": method, "bin": i,
                                 "interval": f"[{lo:.1f}, {hi:.1f}{']' if i == 9 else ')'}",
                                 "n": len(b), "mean_confidence": b[col].mean(),
                                 "accuracy": b.correct.mean(),
                                 "gap_accuracy_minus_confidence": b.correct.mean() - b[col].mean()})
        for condition, column in [("true", "true_emotion"), ("predicted", "predicted_emotion")]:
            for label, group in frame.groupby(column, sort=False):
                right = group[group.correct.eq(1)]
                wrong = group[group.correct.eq(0)]
                class_rows.append({"dataset": dataset, "condition": condition, "emotion": label,
                                   "n": len(group), "correct_n": len(right), "wrong_n": len(wrong),
                                   "mean_confidence": group.confidence.mean(),
                                   "correct_mean_confidence": right.confidence.mean(),
                                   "wrong_mean_confidence": wrong.confidence.mean(),
                                   "correct_minus_wrong_mean_confidence": right.confidence.mean() - wrong.confidence.mean(),
                                   "conditional_accuracy": group.correct.mean(),
                                   "gap_accuracy_minus_confidence": group.correct.mean() - group.confidence.mean(),
                                   "small_n_lt_20": len(group) < MIN_CLASS_N})
    dist = pd.DataFrame(dist_rows)
    bins = pd.DataFrame(bin_rows)
    classes = pd.DataFrame(class_rows)
    comparison = pd.DataFrame(comparison_rows)
    save_csv(dist, "confidence_distribution.csv")
    save_csv(bins, "confidence_bins.csv")
    save_csv(classes, "classwise_confidence.csv")
    save_csv(comparison, "correct_wrong_comparison.csv")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (dataset, frame) in zip(axes, joined.groupby("dataset", sort=True)):
        ax.hist(frame.loc[frame.correct.eq(1), "confidence"], bins=np.linspace(0, 1, 21),
                density=True, alpha=.55, color="#0072B2", label="Correct")
        ax.hist(frame.loc[frame.correct.eq(0), "confidence"], bins=np.linspace(0, 1, 21),
                density=True, alpha=.55, color="#D55E00", label="Wrong")
        ax.set(title=dataset, xlabel="Original top-label confidence", xlim=(0, 1))
        ax.legend()
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(OUT / "figures/correct_wrong_distributions.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, dataset in zip(axes, sorted(CLASSES)):
        ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="Perfect calibration")
        for method, color in [("E0", "#D55E00"), ("E1", "#0072B2")]:
            part = bins.query("dataset == @dataset and method == @method and n > 0")
            ax.plot(part.mean_confidence, part.accuracy, marker="o", color=color, label=method)
            ax.scatter(part.mean_confidence, part.accuracy, s=part.n * .35, color=color, alpha=.3)
            for point in part.itertuples():
                if point.n < 40:
                    ax.annotate(f"n={point.n}", (point.mean_confidence, point.accuracy),
                                xytext=(4, 5 if method == "E0" else -12),
                                textcoords="offset points", fontsize=7, color=color)
        ax.set(title=dataset, xlabel="Mean confidence", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Observed accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "figures/reliability_diagram.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, dataset in enumerate(sorted(CLASSES)):
        for col, condition in enumerate(["true", "predicted"]):
            ax = axes[row, col]
            part = classes.query("dataset == @dataset and condition == @condition").set_index("emotion").reindex(CLASSES[dataset])
            x = np.arange(len(part))
            ax.bar(x - .18, part.mean_confidence, width=.36, color="#0072B2", label="Mean confidence")
            ax.bar(x + .18, part.conditional_accuracy, width=.36, color="#D55E00", label="Conditional accuracy")
            for i, (_, item) in enumerate(part.iterrows()):
                ax.text(i, min(1, max(item.mean_confidence, item.conditional_accuracy) + .025),
                        f"n={int(item.n)}{'*' if item.small_n_lt_20 else ''}", ha="center", fontsize=8)
            ax.set(title=f"{dataset} — {condition} label", ylim=(0, 1.12), xticks=x,
                   xticklabels=part.index, ylabel="Rate / confidence")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures/classwise_mean_confidence.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (dataset, frame) in zip(axes, joined.groupby("dataset", sort=True)):
        positions = np.arange(2)
        for j, correct in enumerate([0, 1]):
            part = frame[frame.correct.eq(correct)]
            ax.scatter(np.full(len(part), j) + np.random.default_rng(1475 + j).uniform(-.12, .12, len(part)),
                       part["shift"], s=5, alpha=.12, color="#0072B2" if correct else "#D55E00")
            ax.plot([j - .2, j + .2], [part["shift"].mean()] * 2, color="black", lw=2)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set(title=dataset, xticks=positions, xticklabels=["Wrong", "Correct"], xlabel="Original prediction")
    axes[0].set_ylabel("E1 confidence − E0 confidence")
    fig.tight_layout()
    fig.savefig(OUT / "figures/e0_e1_confidence_shift.pdf")
    plt.close(fig)

    lines = ["# Confidence distribution diagnostic", "", "Seed 1475, true Validation only; no model or perturbation rerun.",
             "Top-label confidence and binary correctness are analyzed. Sample standard deviations use ddof=1.",
             "Dialogue-cluster percentile intervals use 2,000 resamples with fixed RNG seed 1475; they describe uncertainty across these Validation dialogues, not five-seed stability.", ""]
    class_lines = ["# Classwise confidence diagnostic", "", "Class IDs follow the frozen XP04 evaluation code. `true` condition accuracy is recall; `predicted` condition accuracy is precision. A true-class gap is descriptive and cannot be used at inference. Classes with n<20 are flagged; classwise gaps are exploratory, without multiplicity-adjusted inference.", ""]
    shift_lines = ["# E0 versus E1 confidence shift", "", "E1 is the existing dialogue-level OOF logistic confidence mapping. Its held-out sample label was not used in that fold's fit.", ""]
    final_lines = ["# E3 design recommendation", "", "This is a single-seed Validation diagnostic, not an E3 training result.", ""]
    verification = {"status": "completed", "source_files": {}, "datasets": {}, "plots": []}
    for path in [original_path, oof_path, metrics_path]:
        verification["source_files"][str(path.relative_to(OUT.parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for dataset, frame in joined.groupby("dataset", sort=True):
        accuracy = frame.correct.mean()
        mean_c = frame.confidence.mean()
        mean_e1 = frame.e1_confidence.mean()
        gap = accuracy - mean_c
        ci0 = bootstrap_gap(frame, "confidence")
        ci1 = bootstrap_gap(frame, "e1_confidence")
        d = dist.query("dataset == @dataset")
        right = d.query("group == 'correct'").iloc[0]
        wrong = d.query("group == 'wrong'").iloc[0]
        auc = correctness_auc(frame.confidence, frame.correct)
        bb = bins.query("dataset == @dataset and method == 'E0' and n > 0")
        under = bb.query("gap_accuracy_minus_confidence > 0")
        over = bb.query("gap_accuracy_minus_confidence < 0")
        e0ece = ece15(frame.confidence.to_numpy(), frame.correct.to_numpy())
        e1ece = ece15(frame.e1_confidence.to_numpy(), frame.correct.to_numpy())
        e1auc = correctness_auc(frame.e1_confidence, frame.correct)
        lines += [f"## {dataset}", "", f"n={len(frame)}, dialogues={frame.dialogue_id.nunique()}, accuracy={fmt(accuracy)}, mean E0 confidence={fmt(mean_c)}, gap (accuracy−confidence)={fmt(gap)}; dialogue bootstrap 95% CI [{fmt(ci0[0])}, {fmt(ci0[1])}].",
                  f"Correct: n={int(right.n)}, mean={fmt(right['mean'])}, median={fmt(right['median'])}, SD={fmt(right.std_sample)}, Q25/Q75/Q90={fmt(right.q25)}/{fmt(right.q75)}/{fmt(right.q90)}, C<0.5={fmt(right.fraction_below_0_5)}.",
                  f"Wrong: n={int(wrong.n)}, mean={fmt(wrong['mean'])}, median={fmt(wrong['median'])}, SD={fmt(wrong.std_sample)}, Q25/Q75/Q90={fmt(wrong.q25)}/{fmt(wrong.q75)}/{fmt(wrong.q90)}, C<0.5={fmt(wrong.fraction_below_0_5)}.",
                  f"Confidence separates correctness to some extent: mean difference={fmt(right['mean']-wrong['mean'])}; ROC AUC={fmt(auc)}.",
                  f"E0 occupied bins: {len(bb)}; underconfident bins: {len(under)} ({', '.join(under.interval)}); overconfident bins: {len(over)} ({', '.join(over.interval)}). Sparse bins should not be overinterpreted.", "",
                  "| E0 interval | n | Mean C | Accuracy | Gap |", "|---|---:|---:|---:|---:|",
                  *[f"| {r.interval} | {int(r.n)} | {fmt(r.mean_confidence)} | {fmt(r.accuracy)} | {fmt(r.gap_accuracy_minus_confidence)} |" for r in bb.itertuples()],
                  "", "See `figures/correct_wrong_distributions.pdf` and `figures/reliability_diagram.pdf`.", ""]
        cc = classes.query("dataset == @dataset and condition == 'predicted'").copy()
        cc = cc.sort_values("gap_accuracy_minus_confidence", ascending=False)
        class_lines += [f"## {dataset}: predicted-class calibration gaps", "", "Positive gap means observed precision exceeds mean top-label confidence within this predicted class. Separation is mean C(correct)−mean C(wrong).", "", "| Class | n | Correct | Mean C | Precision | Gap | Correct C | Wrong C | Separation | Small n |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|", *[
            f"| {r.emotion} | {int(r.n)} | {int(r.correct_n)} | {fmt(r.mean_confidence)} | {fmt(r.conditional_accuracy)} | {fmt(r.gap_accuracy_minus_confidence)} | {fmt(r.correct_mean_confidence)} | {fmt(r.wrong_mean_confidence)} | {fmt(r.correct_minus_wrong_mean_confidence)} | {'yes' if r.small_n_lt_20 else 'no'} |"
            for r in cc.itertuples()], ""]
        true_cc = classes.query("dataset == @dataset and condition == 'true'").set_index("emotion").reindex(CLASSES[dataset])
        class_lines += [f"### {dataset}: true-class descriptive breakdown", "", "Here conditional accuracy is recall; the gap is not a deployable top-label calibration target.", "", "| Class | n | Correct | Mean C | Recall | Gap | Correct C | Wrong C | Small n |", "|---|---:|---:|---:|---:|---:|---:|---:|---|", *[
            f"| {r.emotion} | {int(r.n)} | {int(r.correct_n)} | {fmt(r.mean_confidence)} | {fmt(r.conditional_accuracy)} | {fmt(r.gap_accuracy_minus_confidence)} | {fmt(r.correct_mean_confidence)} | {fmt(r.wrong_mean_confidence)} | {'yes' if r.small_n_lt_20 else 'no'} |"
            for r in true_cc.reset_index().itertuples()], "", "See `classwise_confidence.csv` and `figures/classwise_mean_confidence.pdf`.", ""]
        cmp = comparison.query("dataset == @dataset")
        rc = cmp.query("group == 'correct'").iloc[0]
        wc = cmp.query("group == 'wrong'").iloc[0]
        shift_lines += [f"## {dataset}", "", f"E0 accuracy/mean C={fmt(accuracy)}/{fmt(mean_c)}; E1 mean C={fmt(mean_e1)}; E1 global gap={fmt(accuracy-mean_e1)} (dialogue bootstrap 95% CI [{fmt(ci1[0])}, {fmt(ci1[1])}]). ECE15 {fmt(e0ece,5)} → {fmt(e1ece,5)}. Correctness AUC {fmt(auc)} → {fmt(e1auc)}; OOF fold-specific mappings can change cross-fold ranking.",
                        f"Correct predictions: mean C {fmt(rc.mean_e0)} → {fmt(rc.mean_e1)}, mean shift {fmt(rc.mean_shift)}, raised {fmt(rc.raised_fraction)}, lowered {fmt(rc.lowered_fraction)}.",
                        f"Wrong predictions: mean C {fmt(wc.mean_e0)} → {fmt(wc.mean_e1)}, mean shift {fmt(wc.mean_shift)}, raised {fmt(wc.raised_fraction)}, lowered {fmt(wc.lowered_fraction)}.",
                        "This is an aggregate OOF calibration result: a higher confidence for a wrong prediction is not individually beneficial, even when aggregate Brier/NLL improve. Bins have different membership after calibration.", "",
                        "| Method | Interval | n | Mean C | Accuracy | Gap |", "|---|---|---:|---:|---:|---:|",
                        *[f"| {r.method} | {r.interval} | {int(r.n)} | {fmt(r.mean_confidence)} | {fmt(r.accuracy)} | {fmt(r.gap_accuracy_minus_confidence)} |"
                          for r in bins.query("dataset == @dataset and n > 0").itertuples()],
                        "", "See `figures/e0_e1_confidence_shift.pdf` and `figures/reliability_diagram.pdf`.", ""]
        if dataset == "MELD":
            shift_lines += ["MELD E1 still overpredicts accuracy in the occupied 0.3–0.6 confidence bins by about 0.05–0.07, while the 0.7–0.9 bins remain mildly underconfident (gap about +0.04). The 0.9–1.0 bin has only six items. Aggregate improvement does not imply uniform bin calibration.", ""]
        else:
            shift_lines += ["IEMOCAP E1 has a 0.8–0.9 bin with 31 items and accuracy 0.419 against mean confidence 0.830; this local mismatch and the lower OOF correctness AUC warrant caution despite aggregate ECE improvement.", ""]
        verification["datasets"][dataset] = {"utterances": len(frame), "dialogues": frame.dialogue_id.nunique(),
                                             "accuracy": accuracy, "mean_e0_confidence": mean_c,
                                             "mean_e1_confidence": mean_e1, "e0_gap": gap,
                                             "e0_gap_dialogue_bootstrap_ci95": ci0,
                                             "e1_gap_dialogue_bootstrap_ci95": ci1,
                                             "roc_auc_correctness_from_e0_confidence": auc,
                                             "roc_auc_correctness_from_e1_confidence": e1auc,
                                             "e0_ece15_verified": e0ece, "e1_ece15_verified": e1ece}
    meld = joined[joined.dataset.eq("MELD")]
    iem = joined[joined.dataset.eq("IEMOCAP")]
    meld_pred = classes.query("dataset == 'MELD' and condition == 'predicted' and small_n_lt_20 == False")
    negative_classes = meld_pred.query("gap_accuracy_minus_confidence < 0").emotion.tolist()
    positive_classes = meld_pred.query("gap_accuracy_minus_confidence > 0").emotion.tolist()
    final_lines += ["## Findings", "",
        f"1. IEMOCAP is overconfident in aggregate: accuracy−mean C={fmt(iem.correct.mean()-iem.confidence.mean())}. This is not a claim that every bin or class is overconfident.",
        f"2. MELD is underconfident in aggregate: accuracy−mean C={fmt(meld.correct.mean()-meld.confidence.mean())}. Its correct predictions have mean C={fmt(meld.loc[meld.correct.eq(1),'confidence'].mean())}; {fmt((meld.loc[meld.correct.eq(1),'confidence'] < .5).mean())} are below 0.5. Wrong predictions have mean C={fmt(meld.loc[meld.correct.eq(0),'confidence'].mean())}.",
        f"3. Among MELD predicted classes with n≥{MIN_CLASS_N}, positive gaps: {', '.join(positive_classes) or 'none'}; negative gaps: {', '.join(negative_classes) or 'none'}. Class gaps are descriptive and can reflect prediction-mix effects; inspect sample counts before treating them as a correction rule.",
        f"4. MELD E1 raises confidence in {fmt((meld['shift'] > 1e-12).mean())} of all samples, including {fmt((meld.loc[meld.correct.eq(0),'shift'] > 1e-12).mean())} of wrong predictions. It improves aggregate ECE/Brier/NLL in the Pilot, but the increase is not targeted only at correct predictions.",
        "5. A future E3 should allow both upward and downward adjustment around C because the datasets have opposite aggregate bias and class/bin biases may differ. Start with a low-capacity, regularized correctness model using C plus pre-registered behavior features, evaluated dialogue-level OOF; do not alter predicted labels.",
        "6. Class-conditional information is plausible but not yet established as useful. Test predicted-class features as a separate pre-registered ablation with shrinkage and minimum support; never use true class at inference. Compare against C-only E1 and behavior-only variants on the same dialogue folds, checking ECE15, binary Brier, and binary NLL together.",
        "", "No E3 was trained. This diagnostic alone does not authorize the five-seed or raw-data stage; review the Pilot decision first.", ""]
    for name, content in [("CONFIDENCE_DISTRIBUTION_REPORT.md", lines), ("CLASSWISE_CONFIDENCE_REPORT.md", class_lines),
                          ("E0_E1_CONFIDENCE_SHIFT.md", shift_lines), ("FINAL_E3_DESIGN_RECOMMENDATION.md", final_lines)]:
        (OUT / name).write_text("\n".join(content), encoding="utf-8")
    verification["plots"] = sorted(str(p.relative_to(OUT)) for p in (OUT / "figures").glob("*.pdf"))
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verification["datasets"], indent=2))


if __name__ == "__main__":
    main()
