#!/usr/bin/env python
"""Produce preregistered cross-seed diagnostic tables from saved CSV results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TARGET_LABEL = {"T-All": "text", "A": "audio", "V": "visual"}


def save(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"refusing to save empty table: {path}")
    frame.to_csv(path, index=False)


def direction(metric: str, difference: float) -> int:
    """Return +1 when the candidate improves correctness discrimination."""
    return int(np.sign(difference if metric in {"auroc_success", "auprc_success"} else -difference))


def correctness_cross_seed(root: Path) -> pd.DataFrame:
    data = pd.read_csv(root / "results/paired_bootstrap_results.csv")
    rows = []
    keys = ["dataset", "model", "comparator", "metric"]
    for key, group in data.groupby(keys, sort=True):
        values = group["difference"].to_numpy(float)
        directions = np.asarray([direction(key[-1], value) for value in values])
        rows.append(dict(zip(keys, key), n_model_seeds=len(group), seed_mean_difference=values.mean(),
                         seed_sample_sd_difference=values.std(ddof=1), improved_seeds=(directions > 0).sum(),
                         worsened_seeds=(directions < 0).sum(), tied_seeds=(directions == 0).sum(),
                         all_five_improve=bool(np.all(directions > 0))))
    return pd.DataFrame(rows)


def correctness_contrasts(root: Path) -> pd.DataFrame:
    data = pd.read_csv(root / "results/per_utterance_sensitivity.csv", low_memory=False)
    data = data[(data.reference == "main") & np.isclose(data.strength, .5) & data.target.isin(TARGET_LABEL)]
    data["original_correct"] = data.original_prediction == data.label
    rows = []
    metrics = ["flip", "absolute_confidence_change", "js_divergence"]
    for key, group in data.groupby(["dataset", "seed", "target"], sort=True):
        correct, wrong = group[group.original_correct], group[~group.original_correct]
        for metric in metrics:
            cm, wm = correct[metric].mean(), wrong[metric].mean()
            rows.append({"dataset": key[0], "seed": key[1], "target": key[2], "metric": metric,
                         "n_correct": len(correct), "n_wrong": len(wrong), "correct_mean": cm,
                         "wrong_mean": wm, "wrong_minus_correct": wm-cm})
    return pd.DataFrame(rows)


def classwise(root: Path) -> pd.DataFrame:
    data = pd.read_csv(root / "results/per_utterance_sensitivity.csv", low_memory=False)
    data = data[(data.reference == "main") & data.target.isin(TARGET_LABEL)]
    data["original_correct"] = data.original_prediction == data.label
    rows = []
    for key, group in data.groupby(["dataset", "seed", "target", "strength", "label"], sort=True):
        rows.append({"dataset": key[0], "seed": key[1], "target": key[2], "strength": key[3],
                     "true_class": key[4], "n": len(group), "original_accuracy": group.original_correct.mean(),
                     "flip_rate": group.flip.mean(), "mean_true_probability_change": group.true_probability_change.mean(),
                     "mean_confidence_change": group.confidence_change.mean(), "mean_js_divergence": group.js_divergence.mean(),
                     "correct_to_wrong_rate": group.correct_to_wrong.mean(), "wrong_to_correct_rate": group.wrong_to_correct.mean()})
    return pd.DataFrame(rows)


def historical_joint(root: Path, config: dict) -> pd.DataFrame:
    mild = pd.read_csv(Path(config["historical_pilot"]) / "results/utterance_behavior_metrics.csv")
    medium = pd.read_csv(root / "results/per_utterance_sensitivity.csv", low_memory=False)
    medium = medium[(medium.reference == "main") & np.isclose(medium.strength, .5) & medium.target.isin(TARGET_LABEL)]
    medium["modality"] = medium.target.map(TARGET_LABEL)
    medium.loc[medium.modality == "text", "modality"] = "text_representation"
    medium = medium.rename(columns={"flip": "medium_flip", "absolute_confidence_change": "medium_abs_confidence_change",
                                    "js_divergence": "medium_js_divergence"})
    ids = ["dataset", "seed", "modality", "dialogue_id", "utterance_index"]
    medium.dialogue_id = medium.dialogue_id.astype(str)
    mild.dialogue_id = mild.dialogue_id.astype(str)
    joined = mild.merge(medium[ids + ["medium_flip", "medium_abs_confidence_change", "medium_js_divergence"]],
                        on=ids, how="inner", validate="many_to_one")
    rows = []
    pairs = [("flip_rate", "medium_flip"), ("confidence_mad", "medium_abs_confidence_change"),
             ("js_divergence", "medium_js_divergence")]
    for key, group in joined.groupby(["dataset", "seed", "modality", "perturbation_type"], sort=True):
        common = {"dataset": key[0], "seed": key[1], "modality": key[2], "light_perturbation_type": key[3],
                  "n_utterances": len(group), "anchor_only": True,
                  "light_stable_n": int((group.flip_rate == 0).sum()),
                  "light_stable_medium_flip_n": int(((group.flip_rate == 0) & (group.medium_flip == 1)).sum()),
                  "light_stable_medium_flip_rate": group.loc[group.flip_rate == 0, "medium_flip"].mean(),
                  "high_confidence_stable_wrong_n": int(((group.original_confidence >= .8) & (~group.correct.astype(bool)) & (group.flip_rate == 0)).sum())}
        for left, right in pairs:
            coefficient, pvalue = spearmanr(group[left], group[right])
            rows.append({**common, "light_metric": left, "medium_metric": right,
                         "spearman_rho": coefficient, "spearman_pvalue_utterance_naive": pvalue,
                         "inference_warning": "descriptive anchor-only; p-value ignores dialogue clustering"})
    return pd.DataFrame(rows)


def notable_cases(root: Path) -> pd.DataFrame:
    data = pd.read_csv(root / "results/per_utterance_sensitivity.csv", low_memory=False)
    data = data[(data.reference == "main") & np.isclose(data.strength, .5) & data.target.isin(TARGET_LABEL)].copy()
    data["original_correct"] = data.original_prediction == data.label
    wrong = data[(~data.original_correct) & (data.original_confidence >= .8) & (data.flip == 0)].copy()
    wrong["case_type"] = "high_confidence_stable_wrong"
    sensitive = data[data.original_correct & (data.flip == 1)].copy()
    sensitive["case_type"] = "correct_but_sensitive"
    columns = ["case_type", "dataset", "seed", "target", "dialogue_id", "utterance_index", "label",
               "original_prediction", "original_confidence", "flip", "absolute_confidence_change", "js_divergence"]
    return pd.concat([wrong, sensitive], ignore_index=True)[columns]


def main(root: Path) -> None:
    config = json.loads((root / "configs/preregistered_config.json").read_text())
    output = root / "results"
    tables = {
        "correctness_cross_seed_summary.csv": correctness_cross_seed(root),
        "correctness_sensitivity_contrasts.csv": correctness_contrasts(root),
        "classwise_sensitivity.csv": classwise(root),
        "historical_joint_diagnostics.csv": historical_joint(root, config),
        "notable_cases.csv": notable_cases(root),
    }
    for name, frame in tables.items():
        save(frame, output / name)
    print(json.dumps({name: len(frame) for name, frame in tables.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root.resolve())
