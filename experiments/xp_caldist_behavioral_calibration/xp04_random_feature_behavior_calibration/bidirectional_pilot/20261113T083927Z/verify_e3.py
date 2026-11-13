"""Independent verification of stored E3 fold parameters and OOF artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


OUT = Path(__file__).resolve().parent
PILOT = OUT.parents[1] / "pilot" / "20261113T025053Z"
ROW_KEY = ["dataset", "dialogue_id", "utterance_index"]


def scoring(confidence, correct):
    c = np.clip(np.asarray(confidence, float), 1e-15, 1 - 1e-15)
    y = np.asarray(correct, int)
    b = np.minimum(np.floor(c * 15).astype(int), 14)
    ece = sum(np.mean(b == i) * abs(np.mean(c[b == i]) - np.mean(y[b == i]))
              for i in range(15) if np.any(b == i))
    return ece, np.mean((c - y) ** 2), -np.mean(y * np.log(c) + (1 - y) * np.log1p(-c))


def main():
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification["status"] == "completed"
    for filename, expected in verification["source_sha256"].items():
        assert hashlib.sha256((PILOT / filename).read_bytes()).hexdigest() == expected
    predictions = pd.read_csv(OUT / "e3_oof_predictions.csv", dtype={"dialogue_id": str})
    features = pd.read_csv(OUT / "e3_feature_matrix.csv", dtype={"dialogue_id": str})
    metrics = pd.read_csv(OUT / "e3_metrics.csv")
    params = json.loads((OUT / "fold_parameters.json").read_text())
    assert not predictions.duplicated(ROW_KEY + ["method"]).any()
    checked = 0
    for dataset in ("IEMOCAP", "MELD"):
        f = features[features.dataset == dataset].copy()
        assert not f.duplicated(ROW_KEY).any()
        for method, folds in params[dataset].items():
            p = predictions[(predictions.dataset == dataset) & (predictions.method == method)]
            assert len(p) == len(f)
            merged = p.merge(f, on=ROW_KEY, validate="one_to_one", suffixes=("_pred", "_feature"))
            assert len(merged) == len(f)
            assert (merged.correct_pred == merged.correct_feature).all()
            assert (merged.prediction_pred == merged.prediction_feature).all()
            assert (merged.label_pred == merged.label_feature).all()
            held_seen = set()
            for fold in folds:
                train_dialogues = set(fold["train_dialogues"])
                held_dialogues = set(fold.get("held_dialogues", fold.get("held_out_dialogues", [])))
                assert train_dialogues.isdisjoint(held_dialogues)
                assert held_seen.isdisjoint(held_dialogues)
                held_seen |= held_dialogues
                part = merged[merged.dialogue_id.isin(held_dialogues)]
                assert len(part) == fold.get("held_n", len(part))
                assert part.fold.eq(fold["fold"]).all()
                if method == "E1":
                    logits = fold["intercept"] + fold["coefficient"] * part.original_confidence.to_numpy(float)
                else:
                    train = f[f.dialogue_id.isin(train_dialogues)]
                    names = fold["features"]
                    x = part[names].to_numpy(float)
                    mean = np.asarray(fold["scaler_mean"], float)
                    scale = np.asarray(fold["scaler_scale"], float)
                    assert np.allclose(train[names].to_numpy(float).mean(axis=0), mean, atol=1e-10)
                    expected_scale = train[names].to_numpy(float).std(axis=0)
                    expected_scale[expected_scale == 0] = 1.0
                    assert np.allclose(expected_scale, scale, atol=1e-10)
                    logits = fold["intercept"] + ((x - mean) / scale) @ np.asarray(fold["coefficients"])
                assert np.max(np.abs(expit(logits) - part.oof_confidence.to_numpy(float))) < 1e-10
                checked += len(part)
            assert held_seen == set(f.dialogue_id)
        for method, p in predictions[predictions.dataset == dataset].groupby("method"):
            row = metrics[(metrics.dataset == dataset) & (metrics.method == method)]
            assert len(row) == 1
            row = row.iloc[0]
            ece, brier, nll = scoring(p.oof_confidence, p.correct)
            assert abs(ece - row.ece15) < 1e-10
            assert abs(brier - row.binary_brier) < 1e-10
            assert abs(nll - row.binary_nll) < 1e-10
            assert abs(accuracy_score(p.label, p.prediction) - row.accuracy) < 1e-10
            assert abs(f1_score(p.label, p.prediction, average="weighted") - row.weighted_f1) < 1e-10
            assert abs(f1_score(p.label, p.prediction, average="macro") - row.macro_f1) < 1e-10
            assert abs(roc_auc_score(p.correct, p.oof_confidence) - row.correctness_roc_auc) < 1e-10
        verification["datasets"][dataset]["independent_metric_rows_checked"] = len(metrics[metrics.dataset == dataset])
        verification["datasets"][dataset]["folds_parameter_verified"] = len(params[dataset]["E1"])
    expected_outputs = ["E3_METHOD.md", "E3_OOF_RESULTS.md", "E1_VS_E3_COMPARISON.md",
                        "BEHAVIOR_FEATURE_ABLATION.md", "CONFIDENCE_ADJUSTMENT_ANALYSIS.md",
                        "FINAL_E3_DECISION.md", "e3_oof_predictions.csv", "e3_metrics.csv",
                        "e3_feature_ablation.csv", "confidence_shift.csv",
                        "correct_wrong_adjustment.csv", "per_dialogue_results.csv",
                        "bootstrap_results.csv", "e3_feature_matrix.csv", "fold_parameters.json"]
    for filename in expected_outputs:
        assert (OUT / filename).is_file() and (OUT / filename).stat().st_size > 0
    verification["independent_verification"] = {
        "status": "passed", "fold_model_predictions_checked": checked,
        "all_source_hashes_match": True, "all_required_outputs_present": True,
        "all_oof_metric_rows_recomputed": True,
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification["independent_verification"], indent=2))


if __name__ == "__main__":
    main()
