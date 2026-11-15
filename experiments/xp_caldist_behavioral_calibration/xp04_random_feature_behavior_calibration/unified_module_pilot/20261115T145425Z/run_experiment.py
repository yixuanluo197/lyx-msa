"""Run the preregistered unified-module CPU Pilot on saved XP04 CSVs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from calibration_core import auxiliary, base_logits, fit_base, fit_nested_fold, metrics, predict


OUT = Path(__file__).resolve().parent
PILOT = OUT.parents[1] / "pilot" / "20261113T025053Z"
KEY = ["dataset", "seed", "dialogue_id", "utterance_index"]
ROWKEY = ["dataset", "dialogue_id", "utterance_index"]
FEATURES = {"DA": ["gaussian_D", "gaussian_A", "dropout_D", "dropout_A"],
            "MUDELTA": ["gaussian_mu", "gaussian_delta", "dropout_mu", "dropout_delta"]}
MAIN = "E1_DA"
METHODS = ["XP04", "E1", "L0", "E1_MUDELTA", "E1_DA", "L0_MUDELTA", "L0_DA"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def build_features():
    original = pd.read_csv(PILOT / "original_predictions.csv", dtype={"dialogue_id": str})
    behavior = pd.read_csv(PILOT / "behavior_features.csv", dtype={"dialogue_id": str})
    perturbations = pd.read_csv(PILOT / "perturbed_predictions.csv", dtype={"dialogue_id": str})
    assert len(original) == 1636 and not original.duplicated(KEY).any()
    assert set(original.seed) == {1475}
    assert len(perturbations) == 130880
    assert not perturbations.duplicated(KEY + ["modality", "corruption", "repeat_id"]).any()
    assert len(behavior) == 8 * len(original)
    assert not behavior.duplicated(KEY + ["modality", "corruption"]).any()
    for row in original.itertuples():
        p = np.asarray(json.loads(row.probabilities))
        lp = np.asarray(json.loads(row.log_probabilities))
        assert abs(p.sum() - 1) < 1e-5 and np.isfinite(lp).all()
        assert np.allclose(np.exp(lp), p, atol=1e-6, rtol=1e-5)
        assert int(p.argmax()) == row.prediction and abs(p.max() - row.confidence) < 1e-12
    joint = perturbations[perturbations.modality.eq("joint")].copy()
    assert len(joint) == 20 * len(original)
    assert np.allclose(joint.strength, .05)
    joint = joint.merge(original[KEY + ["prediction", "confidence"]].rename(columns={
        "prediction": "original_prediction", "confidence": "original_confidence"}), on=KEY, validate="many_to_one")
    probabilities = [np.asarray(json.loads(p), float) for p in joint.probabilities]
    assert all(np.isfinite(p).all() and abs(p.sum() - 1) < 1e-5 for p in probabilities)
    assert np.allclose([p.max() for p in probabilities], joint.confidence, atol=1e-12, rtol=0)
    assert np.array_equal([p.argmax() for p in probabilities], joint.prediction)
    joint["q_original_class"] = [p[int(k)] for p, k in zip(probabilities, joint.original_prediction)]
    feature_records = []
    for key, g in joint.groupby(KEY + ["corruption"], sort=False):
        g = g.sort_values("repeat_id")
        assert g.repeat_id.tolist() == list(range(10))
        c = float(g.original_confidence.iloc[0])
        q = g.q_original_class.to_numpy(float)
        changed = g.prediction.ne(g.original_prediction).to_numpy(int)
        assert g.visual_missing_skipped.nunique() == 1
        feature_records.append({**dict(zip(KEY + ["corruption"], key)),
                                "D": float(c - q.mean()), "A": float(np.sqrt(np.mean((q - c) ** 2))),
                                "mu": float(np.mean(changed * g.confidence.to_numpy(float))),
                                "delta": float(abs(c - g.confidence.mean())),
                                "visual_missing": bool(g.visual_missing_skipped.iloc[0]),
                                "original_confidence": c})
    long = pd.DataFrame(feature_records)
    reference = behavior[behavior.modality.eq("joint")]
    check = long.merge(reference, on=KEY + ["corruption"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(check) == 2 * len(original)
    assert np.allclose(check.mu_new, check.mu_old, atol=1e-12, rtol=0)
    assert np.allclose(check.delta_new, check.delta_old, atol=1e-12, rtol=0)
    feature = original.copy()
    for corruption in ["gaussian", "dropout"]:
        part = long[long.corruption.eq(corruption)].drop(columns=["corruption", "original_confidence"])
        part = part.rename(columns={k: f"{corruption}_{k}" for k in ["D", "A", "mu", "delta", "visual_missing"]})
        feature = feature.merge(part, on=KEY, validate="one_to_one", how="left", sort=False)
    assert feature[sum(FEATURES.values(), [])].notna().all().all()
    assert feature.gaussian_visual_missing.equals(feature.dropout_visual_missing)
    feature["visual_missing"] = feature.gaussian_visual_missing
    feature["correct"] = feature.prediction.eq(feature.label).astype(int)
    feature.drop(columns=["probabilities", "log_probabilities"]).to_csv(OUT / "feature_matrix.csv", index=False)
    long.to_csv(OUT / "behavior_features_recomputed.csv", index=False)
    return feature


def bootstrap_comparisons(frame, config):
    """Cluster weights avoid repeated row concatenation; same draws for every comparison."""
    groups = np.unique(frame.dialogue_id.to_numpy(str))
    index = np.searchsorted(groups, frame.dialogue_id.to_numpy(str))
    y = frame.correct.to_numpy(int)
    counts = np.bincount(index, minlength=len(groups))
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    draw = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    totals = draw @ counts
    scores = {}
    full = {}
    for name in METHODS:
        p = np.clip(frame[name].to_numpy(float), 1e-15, 1 - 1e-15)
        errors = (p - y) ** 2
        logs = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        bin_id = np.minimum(np.floor(p * 15).astype(int), 14)
        residual_sums = np.zeros((len(groups), 15))
        np.add.at(residual_sums, (index, bin_id), p - y)
        scores[name] = {"binary_brier": (draw @ np.bincount(index, weights=errors, minlength=len(groups))) / totals,
                        "binary_nll": (draw @ np.bincount(index, weights=logs, minlength=len(groups))) / totals,
                        "ece15": np.abs(draw @ residual_sums).sum(axis=1) / totals}
        full[name] = metrics(p, y)
    result = []
    for candidate, reference in config["bootstrap_comparisons"]:
        for metric in ["ece15", "binary_brier", "binary_nll"]:
            d = scores[candidate][metric] - scores[reference][metric]
            result.append({"candidate": candidate, "reference": reference, "metric": metric,
                           "difference": full[candidate][metric] - full[reference][metric],
                           "ci_low": float(np.quantile(d, .025)), "ci_high": float(np.quantile(d, .975)),
                           "bootstrap_fraction_improving": float(np.mean(d < 0)), "resamples": 2000})
    return result


def run_dataset(name, feature, config):
    f = feature[feature.dataset.eq(name)].reset_index(drop=True)
    c, y, groups = f.confidence.to_numpy(float), f.correct.to_numpy(int), f.dialogue_id.to_numpy(str)
    assert len(f) == config["datasets"][name]["utterances"]
    assert len(set(groups)) == config["datasets"][name]["dialogues"]
    assert int(f.visual_missing.sum()) == (1 if name == "IEMOCAP" else 95)
    splits = list((LeaveOneGroupOut() if name == "IEMOCAP" else GroupKFold(n_splits=5)).split(f, groups=groups))
    scores = {method: np.full(len(f), np.nan) for method in METHODS}
    scores["XP04"] = c.copy()
    foldid = np.full(len(f), -1, int)
    parameters, choices = [], []
    for fold, (train, held) in enumerate(splits):
        train_groups, held_groups = sorted(set(groups[train])), sorted(set(groups[held]))
        assert set(train_groups).isdisjoint(held_groups)
        foldid[held] = fold
        shared = {"dataset": name, "fold": fold, "train_dialogues": train_groups, "held_dialogues": held_groups}
        for base in ["E1", "L0"]:
            params = fit_base(c[train], y[train], base)
            scores[base][held] = expit(base_logits(c[held], params))
            parameters.append({**shared, "method": base, "base": params})
            for variant in ["MUDELTA", "DA"]:
                method = f"{base}_{variant}"
                b = f[FEATURES[variant]].to_numpy(float)
                fitted = fit_nested_fold(c[train], b[train], y[train], groups[train], base)
                assert fitted["base"] == params
                scores[method][held] = predict(c[held], b[held], fitted)
                parameters.append({**shared, "method": method, "features": FEATURES[variant], **fitted})
                for item in fitted["candidates"]:
                    choices.append({"dataset": name, "method": method, "outer_fold": fold,
                                    "penalty": item["penalty"], "selected": item["penalty"] == fitted["selected_penalty"], **metrics_fields(item)})
        print(json.dumps({"dataset": name, "outer_fold_done": fold + 1, "outer_folds": len(splits)}), flush=True)
    assert np.all(foldid >= 0)
    for p in scores.values():
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    saved = pd.read_csv(PILOT / "oof_calibration_predictions.csv", dtype={"dialogue_id": str})
    original_e1 = saved[(saved.dataset.eq(name)) & saved.method.eq("E1")]
    ref = f[ROWKEY].merge(original_e1[ROWKEY + ["confidence"]], on=ROWKEY, validate="one_to_one", how="left")
    assert len(ref) == len(f) and ref.confidence.notna().all()
    parity = float(np.max(np.abs(scores["E1"] - ref.confidence.to_numpy(float))))
    assert parity < 1e-12
    prior_l0 = pd.read_csv(OUT.parents[1] / "bidirectional_pilot/20261113T083927Z/e3_oof_predictions.csv", dtype={"dialogue_id": str})
    prior_l0 = prior_l0[prior_l0.dataset.eq(name) & prior_l0.method.eq("L0")]
    ref_l0 = f[ROWKEY].merge(prior_l0[ROWKEY + ["oof_confidence"]], on=ROWKEY, validate="one_to_one", how="left")
    l0_parity = float(np.max(np.abs(scores["L0"] - ref_l0.oof_confidence.to_numpy(float))))
    assert l0_parity < 1e-12
    pilot_metrics = pd.read_csv(PILOT / "calibration_metrics.csv")
    prediction_rows, summary, shift, per_dialogue = [], [], [], []
    for method in METHODS:
        p = scores[method]
        for i in range(len(f)):
            prediction_rows.append({**{k: f.iloc[i][k] for k in KEY}, "fold": int(foldid[i]),
                                    "label": int(f.label.iloc[i]), "prediction": int(f.prediction.iloc[i]),
                                    "correct": int(y[i]), "method": method, "original_confidence": c[i],
                                    "oof_confidence": p[i], "shift": p[i] - c[i], "visual_missing": bool(f.visual_missing.iloc[i])})
        values = metrics(p, y)
        if method in ("XP04", "E1"):
            old = pilot_metrics[pilot_metrics.dataset.eq(name) & pilot_metrics.method.eq("E0" if method == "XP04" else method)].iloc[0]
            assert all(abs(values[k] - old[k]) < 1e-10 for k in values)
        summary.append({"dataset": name, "seed": 1475, "method": method, "utterances": len(f),
                        "dialogues": len(set(groups)), **values,
                        **auxiliary(p, y, c, f.label, f.prediction)})
        for correct in [0, 1]:
            mask = y == correct
            shift.append({"dataset": name, "method": method, "correct": correct, "n": int(mask.sum()),
                          "mean_original": float(c[mask].mean()), "mean_calibrated": float(p[mask].mean()),
                          "mean_shift": float((p[mask] - c[mask]).mean()),
                          "raised_fraction": float(np.mean(p[mask] > c[mask] + 1e-12)),
                          "lowered_fraction": float(np.mean(p[mask] < c[mask] - 1e-12))})
        for dialogue in sorted(set(groups)):
            mask = groups == dialogue
            per_dialogue.append({"dataset": name, "dialogue_id": dialogue, "method": method,
                                 "n": int(mask.sum()), "accuracy": float(y[mask].mean()), **metrics(p[mask], y[mask])})
    wide = f[ROWKEY + ["correct"]].copy()
    for name_, p in scores.items():
        wide[name_] = p
    boots = [{"dataset": name, **row} for row in bootstrap_comparisons(wide, config)]
    verified = {"utterances": len(f), "dialogues": len(set(groups)), "outer_folds": len(splits),
                "visual_missing_retained": int(f.visual_missing.sum()), "e1_parity_max_error": parity,
                "l0_parity_max_error": l0_parity, "inner_group_disjoint": True,
                "one_record_per_utterance": True}
    return prediction_rows, summary, shift, per_dialogue, parameters, choices, boots, verified


def metrics_fields(item):
    return {k: item[k] for k in ["ece15", "binary_brier", "binary_nll"]}


def main():
    start = time.time()
    config = json.loads((OUT / "preregistration.json").read_text())
    assert config["methods"] == METHODS and config["primary_method"] == MAIN
    sources = [PILOT / n for n in ["original_predictions.csv", "behavior_features.csv", "perturbed_predictions.csv",
                                  "oof_calibration_predictions.csv", "calibration_metrics.csv", "verification.json"]]
    sources += [OUT.parents[1] / "bidirectional_pilot/20261113T083927Z/e3_oof_predictions.csv"]
    hashes = {str(p): sha(p) for p in sources}
    code = [OUT / x for x in ["calibration_core.py", "run_experiment.py", "preregistration.json", "EXPERIMENT_PROTOCOL.md"]]
    code_before = {str(p.relative_to(OUT)): sha(p) for p in code}
    write_json("provenance/source_hashes.json", hashes)
    write_json("provenance/run_snapshot.json", {"start_unix": start, "pid": os.getpid(), "host": platform.node(),
        "device": "CPU", "python": sys.executable, "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "argv": sys.argv,
        "git_head": subprocess.check_output(["git", "-C", str(PILOT.parents[4] / "CMERC"), "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": code_before})
    feature = build_features()
    outputs = [[] for _ in range(7)]
    verification = {"status": "running", "primary_method": MAIN, "datasets": {},
                    "cpu_only": True, "new_gpu_inference": False, "new_perturbations": False,
                    "cmerc_weights_changed": False, "temperature_scaling": False, "test_labels_used": False}
    for dataset in ["IEMOCAP", "MELD"]:
        result = run_dataset(dataset, feature, config)
        for i in range(7):
            outputs[i].extend(result[i])
        verification["datasets"][dataset] = result[7]
    filenames = ["oof_predictions.csv", "metrics.csv", "correct_wrong_adjustment.csv", "per_dialogue_results.csv",
                 "fold_parameters.json", "inner_cv_results.csv", "bootstrap_results.csv"]
    for name, data in zip(filenames, outputs):
        if name.endswith(".json"):
            write_json(name, data)
        else:
            pd.DataFrame(data).to_csv(OUT / name, index=False)
    assert all(sha(p) == h for p, h in hashes.items()), "source changed during run"
    assert all(sha(OUT / p) == h for p, h in code_before.items()), "running implementation changed"
    verification.update(status="completed", seconds=time.time() - start,
                        source_hashes_unchanged=True, running_code_hashes_unchanged=True)
    write_json("verification.json", verification)
    print(pd.DataFrame(outputs[1])[["dataset", "method", "ece15", "binary_brier", "binary_nll", "correctness_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
