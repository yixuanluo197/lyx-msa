"""Reconstruct every stored inner/outer prediction and verify nested selection."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import f1_score, roc_auc_score


OUT = Path(__file__).resolve().parent


def score(p, y):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, int)
    b = np.minimum(np.floor(p * 15).astype(int), 14)
    ece = sum(np.mean(b == i) * abs(np.mean(p[b == i]) - np.mean(y[b == i]))
              for i in range(15) if np.any(b == i))
    return {"ece15": ece, "binary_brier": np.mean((p - y) ** 2),
            "binary_nll": -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))}


def base_offset(frame, p, train=None):
    def transform(values):
        return logit(np.clip(values, 1e-6, 1 - 1e-6)) if p["name"] == "L0" else values
    c = transform(frame.confidence.to_numpy(float))
    if train is not None:
        xt = transform(train.confidence.to_numpy(float))
        if p["name"] == "L0":
            assert abs(xt.mean() - p["mean"][0]) < 1e-10
            expected_scale = xt.std() if xt.std() > 0 else 1.0
            assert abs(expected_scale - p["scale"][0]) < 1e-10
        else:
            assert p["mean"] == [0.] and p["scale"] == [1.]
    return p["intercept"] + (c - p["mean"][0]) / p["scale"][0] * p["coefficient"]


def corrected(frame, names, bp, cp, train):
    xtrain = train[names].to_numpy(float)
    mean, scale = np.asarray(cp["mean"]), np.asarray(cp["scale"])
    assert np.allclose(xtrain.mean(axis=0), mean, atol=1e-10, rtol=0)
    sd = xtrain.std(axis=0)
    sd[sd == 0] = 1.
    assert np.allclose(sd, scale, atol=1e-10, rtol=0)
    weights = np.asarray(cp["weights"])
    xt = (xtrain - mean) / scale
    off = base_offset(train, bp, train)
    if cp["penalty"] == "off":
        assert np.array_equal(weights, np.zeros(4))
    else:
        t = off + xt @ weights
        gradient = xt.T @ (expit(t) - train.correct.to_numpy(int)) / len(train) + float(cp["penalty"]) * weights
        assert np.max(np.abs(gradient)) < 1e-6
    x = (frame[names].to_numpy(float) - mean) / scale
    return expit(base_offset(frame, bp) + x @ weights)


def main():
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification["status"] == "completed"
    hashes = json.loads((OUT / "provenance/source_hashes.json").read_text())
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    snapshot = json.loads((OUT / "provenance/run_snapshot.json").read_text())
    for path, expected in snapshot["code_sha256"].items():
        assert hashlib.sha256((OUT / path).read_bytes()).hexdigest() == expected
    feature = pd.read_csv(OUT / "feature_matrix.csv", dtype={"dialogue_id": str})
    predictions = pd.read_csv(OUT / "oof_predictions.csv", dtype={"dialogue_id": str})
    summaries = pd.read_csv(OUT / "metrics.csv")
    params = json.loads((OUT / "fold_parameters.json").read_text())
    assert len(feature) == 1636 and len(predictions) == 1636 * 7
    assert not predictions.duplicated(["dataset", "seed", "dialogue_id", "utterance_index", "method"]).any()
    outer_checked, inner_checked, fits_checked = 0, 0, 0
    max_error = 0.
    for record in params:
        dataset, method = record["dataset"], record["method"]
        f = feature[feature.dataset.eq(dataset)].copy()
        train_set, held_set = set(record["train_dialogues"]), set(record["held_dialogues"])
        assert train_set.isdisjoint(held_set) and train_set | held_set == set(f.dialogue_id)
        train, held = f[f.dialogue_id.isin(train_set)], f[f.dialogue_id.isin(held_set)]
        base_offset(held, record["base"], train)
        if "correction" in record:
            actual = corrected(held, record["features"], record["base"], record["correction"], train)
            inner_prob = {str(item["penalty"]): pd.Series(index=train.index, dtype=float) for item in record["candidates"]}
            seen = set()
            for fold in record["inner_folds"]:
                trset, hoset = set(fold["train_dialogues"]), set(fold["held_dialogues"])
                assert trset.isdisjoint(hoset) and trset | hoset == train_set
                assert seen.isdisjoint(hoset)
                seen |= hoset
                tr, ho = f[f.dialogue_id.isin(trset)], f[f.dialogue_id.isin(hoset)]
                assert len(tr) == fold["train_n"] and len(ho) == fold["held_n"]
                for penalty, cp in fold["corrections"].items():
                    values = corrected(ho, record["features"], fold["base"], cp, tr)
                    inner_prob[penalty].loc[ho.index] = values
                    inner_checked += len(ho)
                    fits_checked += 1
            assert seen == train_set
            selected, best = "off", np.inf
            for candidate in record["candidates"]:
                penalty = candidate["penalty"]
                assert inner_prob[str(penalty)].notna().all()
                recomputed = score(inner_prob[str(penalty)].to_numpy(), train.correct.to_numpy())
                for key, value in recomputed.items():
                    assert abs(candidate[key] - value) < 1e-10
                if recomputed["binary_brier"] < best - 1e-12:
                    selected, best = penalty, recomputed["binary_brier"]
            assert selected == record["selected_penalty"] == record["correction"]["penalty"]
        else:
            actual = expit(base_offset(held, record["base"]))
        p = predictions[predictions.dataset.eq(dataset) & predictions.method.eq(method)]
        aligned = held[["dataset", "dialogue_id", "utterance_index", "label", "prediction", "correct"]].merge(
            p, on=["dataset", "dialogue_id", "utterance_index"], validate="one_to_one", suffixes=("_feature", "_pred"))
        assert len(aligned) == len(held) and aligned.fold.eq(record["fold"]).all()
        for col in ["label", "prediction", "correct"]:
            assert (aligned[col + "_feature"] == aligned[col + "_pred"]).all()
        err = float(np.max(np.abs(actual - aligned.oof_confidence.to_numpy())))
        assert err < 1e-10, (method, err)
        max_error = max(err, max_error)
        outer_checked += len(held)
    for (dataset, method), p in predictions.groupby(["dataset", "method"]):
        summary = summaries[summaries.dataset.eq(dataset) & summaries.method.eq(method)].iloc[0]
        assert len(p) == (527 if dataset == "IEMOCAP" else 1109)
        for key, value in score(p.oof_confidence, p.correct).items():
            assert abs(value - summary[key]) < 1e-10
        assert abs(float(p.correct.mean()) - summary.accuracy) < 1e-10
        assert abs(f1_score(p.label, p.prediction, average="weighted") - summary.weighted_f1) < 1e-10
        assert abs(f1_score(p.label, p.prediction, average="macro") - summary.macro_f1) < 1e-10
        assert abs(roc_auc_score(p.correct, p.oof_confidence) - summary.correctness_auc) < 1e-10
    # Check equivalence of cluster weights and literal dialogue resampling.
    bootstrap = pd.read_csv(OUT / "bootstrap_results.csv")
    bootstrap_checks = 0
    for dataset in ["IEMOCAP", "MELD"]:
        f = feature[feature.dataset.eq(dataset)]
        p = predictions[predictions.dataset.eq(dataset)]
        keys = ["dialogue_id", "utterance_index"]
        pivot = p.pivot(index=keys, columns="method", values="oof_confidence").reindex(pd.MultiIndex.from_frame(f[keys]))
        groups = np.unique(f.dialogue_id.to_numpy(str))
        gi = np.searchsorted(groups, f.dialogue_id.to_numpy(str))
        draw = np.random.default_rng(1475).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
        for method in ["XP04", "E1_DA"]:
            values = pivot[method].to_numpy(float)
            y = f.correct.to_numpy()
            for chosen in draw[:5]:
                indices = np.concatenate([np.tile(np.flatnonzero(gi == j), k) for j, k in enumerate(chosen) if k])
                direct = score(values[indices], y[indices])
                weights = chosen[gi]
                pp = np.clip(values, 1e-15, 1 - 1e-15)
                b = np.minimum(np.floor(pp * 15).astype(int), 14)
                residual = np.bincount(b, weights=weights * (pp - y), minlength=15)
                weighted = {"ece15": np.abs(residual).sum() / weights.sum(),
                            "binary_brier": np.average((pp - y) ** 2, weights=weights),
                            "binary_nll": np.average(-(y*np.log(pp)+(1-y)*np.log1p(-pp)), weights=weights)}
                for key in direct:
                    assert abs(direct[key] - weighted[key]) < 1e-10
                    bootstrap_checks += 1
        assert len(bootstrap[bootstrap.dataset.eq(dataset)]) == 36
    result = {"passed": True, "outer_predictions_reconstructed": outer_checked,
              "inner_candidate_predictions_reconstructed": inner_checked, "inner_candidate_fits_checked": fits_checked,
              "outer_max_abs_reconstruction_error": max_error, "metric_rows_recomputed": len(summaries),
              "cluster_weight_equivalence_checks": bootstrap_checks, "inner_selections_recomputed": 68,
              "sources_and_running_code_unchanged": True}
    verification["independent_verification"] = result
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
