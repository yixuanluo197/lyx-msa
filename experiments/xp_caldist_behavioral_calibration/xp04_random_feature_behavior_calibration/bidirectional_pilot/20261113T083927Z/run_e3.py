"""CPU-only dialogue OOF E3 calibration from frozen XP04 Pilot CSVs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


OUT = Path(__file__).resolve().parent
PILOT = OUT.parents[1] / "pilot" / "20261113T025053Z"
KEY = ["dataset", "seed", "dialogue_id", "utterance_index"]
ROW_KEY = ["dataset", "dialogue_id", "utterance_index"]
MODALITIES = ["joint", "text", "audio", "visual"]
MEASURES = ["mu", "delta", "js"]
BASE = ["logit_C"]
JOINT = [f"joint_{x}" for x in MEASURES]
MODAL = [f"{m}_{x}" for m in ["text", "audio", "visual"] for x in MEASURES]
ABLATIONS = {
    "A_mu": BASE + ["joint_mu"],
    "A_delta": BASE + ["joint_delta"],
    "B_js": BASE + ["joint_js"],
    "C_text": BASE + [f"text_{x}" for x in MEASURES],
    "C_audio": BASE + [f"audio_{x}" for x in MEASURES],
    "C_visual": BASE + [f"visual_{x}" for x in MEASURES],
    "C_no_text": BASE + JOINT + [f"{m}_{x}" for m in ["audio", "visual"] for x in MEASURES],
    "C_no_audio": BASE + JOINT + [f"{m}_{x}" for m in ["text", "visual"] for x in MEASURES],
    "C_no_visual": BASE + JOINT + [f"{m}_{x}" for m in ["text", "audio"] for x in MEASURES],
}
MAIN = {
    "L0": BASE,
    "E3-A": BASE + JOINT[:2],
    "E3-B": BASE + JOINT,
    "E3-C": BASE + JOINT + MODAL,
    "E3-D": BASE + JOINT + MODAL + ["predicted_class_onehot"],
}
ALL = {**MAIN, **ABLATIONS}


def scoring(conf: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    c = np.clip(np.asarray(conf, float), 1e-15, 1 - 1e-15)
    y = np.asarray(correct, int)
    b = np.minimum(np.floor(c * 15).astype(int), 14)
    ece = sum((b == i).mean() * abs(c[b == i].mean() - y[b == i].mean())
              for i in range(15) if np.any(b == i))
    return {"ece15": float(ece), "binary_brier": float(np.mean((c - y) ** 2)),
            "binary_nll": float(-np.mean(y * np.log(c) + (1 - y) * np.log1p(-c)))}


def auc(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(roc_auc_score(correct, conf)) if len(np.unique(correct)) == 2 else float("nan")


def aggregate_features(original: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    assert behavior[KEY + ["modality", "corruption"]].notna().all().all()
    assert not behavior.duplicated(KEY + ["modality", "corruption"]).any()
    assert set(behavior.seed) == {1475}
    assert set(behavior.modality) == set(MODALITIES)
    assert set(behavior.corruption) == {"gaussian", "dropout"}
    assert np.allclose(behavior.strength, .05)
    assert len(behavior) == 8 * len(original)
    assert behavior[MEASURES].notna().all().all()
    complete = behavior.groupby(KEY, sort=False).size()
    assert complete.eq(8).all() and len(complete) == len(original)
    aligned = original[KEY + ["confidence", "prediction", "label"]].copy()
    idx = pd.MultiIndex.from_frame(aligned[KEY])
    grouped = behavior.groupby(KEY + ["modality"], sort=False)[MEASURES].mean()
    for modality in MODALITIES:
        one = grouped.xs(modality, level="modality").reindex(idx)
        assert one.notna().all().all(), modality
        for measure in MEASURES:
            aligned[f"{modality}_{measure}"] = one[measure].to_numpy(float)
    clipped = np.clip(aligned.confidence.to_numpy(float), 1e-6, 1 - 1e-6)
    aligned["logit_C"] = np.log(clipped) - np.log1p(-clipped)
    aligned["correct"] = aligned.label.eq(aligned.prediction).astype(int)
    assert not aligned.duplicated(KEY).any()
    return aligned


def bootstrap_delta(groups: np.ndarray, correct: np.ndarray, reference: np.ndarray,
                    candidate: np.ndarray, repeats: int = 2000) -> dict:
    """Paired cluster bootstrap; delta=candidate−reference, lower is better."""
    unique = sorted(set(groups))
    group_indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(1475)
    res = {k: np.empty(repeats) for k in ["ece15", "binary_brier", "binary_nll"]}
    for repeat in range(repeats):
        chosen = rng.integers(0, len(unique), len(unique))
        take = np.concatenate([group_indices[i] for i in chosen])
        left, right = scoring(reference[take], correct[take]), scoring(candidate[take], correct[take])
        for key in res:
            res[key][repeat] = right[key] - left[key]
    return {key: {"estimate": scoring(candidate, correct)[key] - scoring(reference, correct)[key],
                  "ci_low": float(np.quantile(values, .025)),
                  "ci_high": float(np.quantile(values, .975)),
                  "bootstrap_fraction_better": float(np.mean(values < 0))}
            for key, values in res.items()}


def run_dataset(name: str, original: pd.DataFrame, behavior: pd.DataFrame, saved_e1: pd.DataFrame,
                pilot_metrics: pd.DataFrame):
    orig = original[original.dataset.eq(name)].copy().reset_index(drop=True)
    feat = aggregate_features(orig, behavior[behavior.dataset.eq(name)])
    n, n_dialogues = (527, 12) if name == "IEMOCAP" else (1109, 114)
    assert len(orig) == n and orig.dialogue_id.nunique() == n_dialogues
    groups = orig.dialogue_id.to_numpy(str)
    y = feat.correct.to_numpy(int)
    c = orig.confidence.to_numpy(float)
    labels = orig.label.to_numpy(int)
    predictions = orig.prediction.to_numpy(int)
    num_classes = 6 if name == "IEMOCAP" else 7
    assert predictions.min() >= 0 and predictions.max() < num_classes
    for i in range(7):
        feat[f"predclass_{i}"] = (predictions == i).astype(int)
    splits = list((LeaveOneGroupOut() if name == "IEMOCAP" else GroupKFold(n_splits=5))
                  .split(orig, groups=groups))
    assert len(splits) == (12 if name == "IEMOCAP" else 5)
    fold_id = np.full(n, -1, int)
    for i, (train, held) in enumerate(splits):
        assert not set(groups[train]) & set(groups[held])
        fold_id[held] = i
    assert np.all(fold_id >= 0)
    e1 = np.full(n, np.nan)
    e1_params = []
    for fold, (train, held) in enumerate(splits):
        model = LogisticRegression(C=1, max_iter=1000, solver="lbfgs")
        model.fit(c[train, None], y[train])
        e1[held] = model.predict_proba(c[held, None])[:, 1]
        e1_params.append({"fold": fold, "train_dialogues": sorted(set(groups[train])),
                          "held_dialogues": sorted(set(groups[held])),
                          "intercept": float(model.intercept_[0]),
                          "coefficient": float(model.coef_[0, 0])})
    e1_saved = orig[ROW_KEY].merge(saved_e1[saved_e1.dataset.eq(name)][ROW_KEY + ["confidence"]],
                                   on=ROW_KEY, validate="one_to_one", how="left")
    assert len(e1_saved) == n and e1_saved.confidence.notna().all()
    max_e1_error = float(np.max(np.abs(e1 - e1_saved.confidence.to_numpy(float))))
    assert max_e1_error < 1e-12, max_e1_error
    output = {"E0": c, "E1": e1}
    parameters = {"E1": e1_params}
    models_converged = True
    for method, columns in ALL.items():
        expanded = []
        for column in columns:
            expanded.extend([f"predclass_{i}" for i in range(num_classes)]
                            if column == "predicted_class_onehot" else [column])
        x = feat[expanded].to_numpy(float)
        result = np.full(n, np.nan)
        folds = []
        for fold, (train, held) in enumerate(splits):
            scaler = StandardScaler().fit(x[train])
            train_x, held_x = scaler.transform(x[train]), scaler.transform(x[held])
            assert np.isfinite(train_x).all() and np.isfinite(held_x).all()
            model = LogisticRegression(C=1, max_iter=1000, solver="lbfgs")
            model.fit(train_x, y[train])
            result[held] = model.predict_proba(held_x)[:, 1]
            converged = int(model.n_iter_[0]) < model.max_iter
            models_converged &= converged
            folds.append({"fold": fold, "features": expanded, "train_dialogues": sorted(set(groups[train])),
                          "held_dialogues": sorted(set(groups[held])), "train_n": len(train), "held_n": len(held),
                          "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                          "intercept": float(model.intercept_[0]), "coefficients": model.coef_[0].tolist(),
                          "iterations": int(model.n_iter_[0]), "converged": converged})
        assert np.isfinite(result).all() and np.all((result > 0) & (result < 1))
        output[method] = result
        parameters[method] = folds
    assert models_converged
    metric_rows, pred_rows, shift_rows, compare_rows, dialogue_rows = [], [], [], [], []
    for method, scores in output.items():
        base = scoring(scores, y)
        reference = pilot_metrics.query("dataset == @name and method == @method and corruption == 'none'")
        if method in ("E0", "E1"):
            assert len(reference) == 1
            for key in base:
                assert abs(base[key] - float(reference.iloc[0][key])) < 1e-10
        extra = {"accuracy": float(accuracy_score(labels, predictions)),
                 "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
                 "macro_f1": float(f1_score(labels, predictions, average="macro")),
                 "correctness_roc_auc": auc(scores, y),
                 "mean_confidence": float(np.mean(scores)),
                 "correct_mean_confidence": float(np.mean(scores[y == 1])),
                 "wrong_mean_confidence": float(np.mean(scores[y == 0])),
                 "fraction_raised_vs_e0": float(np.mean(scores > c + 1e-12)),
                 "fraction_lowered_vs_e0": float(np.mean(scores < c - 1e-12))}
        metric_rows.append({"dataset": name, "method": method, "utterances": n,
                            "dialogues": n_dialogues, **base, **extra})
        for i in range(n):
            pred_rows.append({"dataset": name, "seed": 1475, "dialogue_id": groups[i],
                              "utterance_index": int(orig.utterance_index.iloc[i]), "fold": int(fold_id[i]),
                              "label": int(labels[i]), "prediction": int(predictions[i]), "correct": int(y[i]),
                              "method": method, "original_confidence": float(c[i]),
                              "oof_confidence": float(scores[i])})
        if method in ["E1", *MAIN]:
            for i in range(n):
                shift_rows.append({"dataset": name, "dialogue_id": groups[i],
                                   "utterance_index": int(orig.utterance_index.iloc[i]), "fold": int(fold_id[i]),
                                   "correct": int(y[i]), "method": method,
                                   "original_confidence": float(c[i]), "oof_confidence": float(scores[i]),
                                   "shift": float(scores[i] - c[i])})
            for correct in (0, 1):
                mask = y == correct
                delta = scores[mask] - c[mask]
                compare_rows.append({"dataset": name, "method": method, "correct": correct,
                                     "n": int(mask.sum()), "mean_e0": float(np.mean(c[mask])),
                                     "mean_calibrated": float(np.mean(scores[mask])),
                                     "mean_shift": float(np.mean(delta)),
                                     "fraction_raised": float(np.mean(delta > 1e-12)),
                                     "fraction_lowered": float(np.mean(delta < -1e-12))})
            for dialogue in sorted(set(groups)):
                mask = groups == dialogue
                dialogue_rows.append({"dataset": name, "dialogue_id": dialogue, "method": method,
                                      "n": int(mask.sum()), "accuracy": float(np.mean(y[mask])),
                                      **scoring(scores[mask], y[mask]),
                                      "mean_confidence": float(np.mean(scores[mask]))})
    bootstrap_rows = []
    for method in ["L0", "E3-A", "E3-B", "E3-C", "E3-D"]:
        for metric, values in bootstrap_delta(groups, y, e1, output[method]).items():
            bootstrap_rows.append({"dataset": name, "method": method, "reference": "E1",
                                   "metric": metric, **values})
    # Added after the main OOF readout as a diagnostic against the logit(C) link control.
    # Every fixed feature configuration is included, avoiding selective uncertainty reports.
    for method in ALL:
        if method == "L0":
            continue
        for metric, values in bootstrap_delta(groups, y, output["L0"], output[method]).items():
            bootstrap_rows.append({"dataset": name, "method": method, "reference": "L0",
                                   "metric": metric, **values})
    verification = {"dataset": name, "utterances": n, "dialogues": n_dialogues,
                    "behavior_rows": len(behavior[behavior.dataset.eq(name)]),
                    "behavior_conditions_per_utterance": 8, "folds": len(splits),
                    "e1_max_abs_error_vs_saved": max_e1_error,
                    "models_converged": bool(models_converged),
                    "one_training_record_per_utterance": True,
                    "train_held_dialogues_disjoint": True}
    return (pd.DataFrame(pred_rows), pd.DataFrame(metric_rows), pd.DataFrame(shift_rows),
            pd.DataFrame(compare_rows), pd.DataFrame(dialogue_rows), pd.DataFrame(bootstrap_rows),
            feat, parameters, verification)


def main() -> None:
    source_names = ["original_predictions.csv", "behavior_features.csv", "reliability_scores.csv",
                    "oof_calibration_predictions.csv", "calibration_metrics.csv"]
    source_hashes = {name: hashlib.sha256((PILOT / name).read_bytes()).hexdigest() for name in source_names}
    original = pd.read_csv(PILOT / source_names[0], dtype={"dialogue_id": str})
    behavior = pd.read_csv(PILOT / source_names[1], dtype={"dialogue_id": str})
    reliable = pd.read_csv(PILOT / source_names[2], dtype={"dialogue_id": str})
    oof = pd.read_csv(PILOT / source_names[3], dtype={"dialogue_id": str})
    pilot_metrics = pd.read_csv(PILOT / source_names[4])
    assert not original.duplicated(KEY).any() and set(original.seed) == {1475}
    assert len(reliable) == len(behavior)
    assert not reliable.duplicated(ROW_KEY + ["corruption", "modality"]).any()
    assert not behavior.duplicated(KEY + ["corruption", "modality"]).any()
    matched = reliable.merge(behavior, on=ROW_KEY + ["corruption", "modality"],
                             validate="one_to_one", suffixes=("_reliability", "_behavior"))
    assert len(matched) == len(behavior)
    assert np.allclose(matched.mu_reliability, matched.mu_behavior)
    assert np.allclose(matched.delta_reliability, matched.delta_behavior)
    saved_e1 = oof.query("method == 'E1' and corruption == 'none' and modality == 'none'")
    assert len(saved_e1) == len(original) and not saved_e1.duplicated(ROW_KEY).any()
    all_outputs = [[], [], [], [], [], []]
    features = []
    params = {}
    verification = {"status": "completed", "seed": 1475, "source_sha256": source_hashes,
                    "no_cm_erc_training": True, "no_gpu_inference": True, "no_new_perturbations": True,
                    "no_test_labels": True, "no_temperature_scaling": True, "datasets": {}}
    for name in ("IEMOCAP", "MELD"):
        result = run_dataset(name, original, behavior, saved_e1, pilot_metrics)
        for i in range(6):
            all_outputs[i].append(result[i])
        features.append(result[6])
        params[name] = result[7]
        verification["datasets"][name] = result[8]
    names = ["e3_oof_predictions.csv", "e3_metrics.csv", "confidence_shift.csv",
             "correct_wrong_adjustment.csv", "per_dialogue_results.csv", "bootstrap_results.csv"]
    for name, pieces in zip(names, all_outputs):
        pd.concat(pieces, ignore_index=True).to_csv(OUT / name, index=False)
    metrics = pd.read_csv(OUT / "e3_metrics.csv")
    metrics[metrics.method.isin(ABLATIONS)].to_csv(OUT / "e3_feature_ablation.csv", index=False)
    pd.concat(features, ignore_index=True).to_csv(OUT / "e3_feature_matrix.csv", index=False)
    (OUT / "fold_parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(metrics[metrics.method.isin(["E0", "E1", *MAIN])][
        ["dataset", "method", "ece15", "binary_brier", "binary_nll", "correctness_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
