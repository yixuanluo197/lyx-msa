"""Independent audit of the preregistered confidence-dependent mu/delta interaction score.

This verifier deliberately does not import the fitting implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
KEY = ["dataset", "seed", "dialogue_id", "utterance_index"]
METRICS = ["ece15", "binary_brier", "binary_nll"]
SEEDS = [1475, 67137, 2024, 3407, 42]
COUNTS = {"IEMOCAP": (527, 12), "MELD": (1109, 114)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            h.update(block)
    return h.hexdigest()


def csv(path):
    return pd.read_csv(path, dtype={"dialogue_id": str})


def close(a, b, tolerance=1e-10):
    aa, bb = np.asarray(a), np.asarray(b)
    assert aa.shape == bb.shape, (aa.shape, bb.shape)
    assert np.all(np.isfinite(aa)) and np.all(np.isfinite(bb))
    assert np.allclose(aa, bb, rtol=0, atol=tolerance), float(np.max(np.abs(aa - bb)))


def metrics(probability, correctness):
    q = np.clip(np.asarray(probability, float), 1e-15, 1 - 1e-15)
    y = np.asarray(correctness, int)
    bins = np.minimum(np.floor(q * 15).astype(int), 14)
    residual = np.bincount(bins, weights=q-y, minlength=15)
    return {
        "ece15": float(np.abs(residual).sum() / len(y)),
        "binary_brier": float(np.mean((q-y)**2)),
        "binary_nll": float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),
    }


def check_inputs(config):
    path = ROOT/"inputs/features.csv"
    joint = csv(path)
    historical_path = Path(config["source_mainline"])/"inputs/joint_features.csv"
    historical = csv(historical_path)
    previous = csv(Path(config["source_five_seed"])/"behavior/feature_matrix.csv")
    columns = config["behavior_features"]
    assert len(joint) == config["expected"]["input_rows"] == 8180
    assert not joint.duplicated(KEY).any()
    assert set(joint.seed) == set(SEEDS) and set(joint.modality) == {"joint"}
    assert joint.columns.tolist() == historical.columns.tolist(), "New primitive inputs introduced"
    assert sha(path) == sha(historical_path), "Inputs must be exact historical copy"
    for (ds, seed), frame in joint.groupby(["dataset", "seed"]):
        n, groups = COUNTS[ds]
        assert len(frame) == n and frame.dialogue_id.nunique() == groups
        assert frame.visual_missing.sum() == (1 if ds == "IEMOCAP" else 95)
        assert np.array_equal(frame.correct, (frame.prediction == frame.label).astype(int))
        assert not frame.perturbation_skipped.any()
        assert np.isfinite(frame[["confidence"]+columns]).all().all()
        assert frame.confidence.between(0, 1).all()
    paired = joint.merge(previous, on=KEY, validate="one_to_one", suffixes=("", "_old"))
    assert len(paired) == len(joint)
    for column in columns+["confidence", "label", "prediction", "correct"]:
        close(paired[column], paired[column+"_old"])
    return {"joint_rows": len(joint), "one_original_record_per_utterance_and_seed": True,
            "exact_historical_input_sha256": sha(path), "no_new_primitive_inputs": True,
            "historical_behavior_values_matched": len(paired)*len(columns),
            "visual_missing_rows_preserved": True}


def population_scaler(values):
    """Independent population scaler, including sklearn's constant bound."""
    values = np.asarray(values, float)
    n = len(values)
    mean, variance = values.mean(axis=0), values.var(axis=0)
    constant = np.ptp(values, axis=0) == 0
    mean[constant] = values[0, constant]
    upper = n*np.finfo(float).eps*variance+(n*mean*np.finfo(float).eps)**2
    scale = np.sqrt(variance)
    scale[variance <= upper] = 1
    return mean, scale


def objective(theta, design, target, penalties):
    score = design @ theta
    return float(np.mean(np.logaddexp(0, score)-target*score)
                 + np.dot(penalties, theta**2)/2)


def newton_refine(theta, design, target, penalties):
    """Independent convex refinement for optimizer accuracy sensitivity only."""
    theta = np.asarray(theta, float).copy()
    initial = objective(theta, design, target, penalties)
    for _ in range(60):
        q = expit(design @ theta)
        gradient = design.T @ (q-target)/len(target) + penalties*theta
        if np.max(np.abs(gradient)) < 1e-12:
            break
        hessian = (design.T*(q*(1-q))) @ design/len(target) + np.diag(penalties)
        step = np.linalg.solve(hessian, gradient)
        before = objective(theta, design, target, penalties)
        factor = 1.0
        while factor > 2**-30:
            candidate = theta-factor*step
            if objective(candidate, design, target, penalties) <= before+1e-15:
                theta = candidate
                break
            factor /= 2
        else:
            break
    return theta, initial-objective(theta, design, target, penalties)


def bootstrap_values(frame, groups, draws):
    """Recompute cluster bootstrap, including nonlinear fixed-bin ECE."""
    gi = np.searchsorted(groups, frame.dialogue_id.to_numpy(str))
    assert np.array_equal(groups[gi], frame.dialogue_id.to_numpy(str))
    multiplicities = np.stack([np.bincount(row, minlength=len(groups)) for row in draws])
    counts = np.bincount(gi, minlength=len(groups))
    denominator = multiplicities @ counts
    q = np.clip(frame.oof_confidence.to_numpy(float), 1e-15, 1-1e-15)
    y = frame.correct.to_numpy(int)
    bins = np.minimum((q*15).astype(int), 14)
    residual = np.zeros((len(groups), 15))
    np.add.at(residual, (gi, bins), q-y)
    loss = -(y*np.log(q)+(1-y)*np.log1p(-q))
    answer = {
        "ece15": np.abs(multiplicities @ residual).sum(axis=1)/denominator,
        "binary_brier": multiplicities @ np.bincount(gi, weights=(q-y)**2,
                                                       minlength=len(groups))/denominator,
        "binary_nll": multiplicities @ np.bincount(gi, weights=loss,
                                                     minlength=len(groups))/denominator,
    }
    literal_checks = 0
    for draw in range(3):
        indices = np.concatenate([np.flatnonzero(gi == g) for g in draws[draw]])
        direct = metrics(q[indices], y[indices])
        for metric in METRICS:
            close(answer[metric][draw], direct[metric])
            literal_checks += 1
    return answer, literal_checks


class FitAudit:
    def __init__(self, frame):
        self.frame = frame
        self.cache = {}
        self.max_gradient = 0.0
        self.max_objective_gap = 0.0
        self.max_refined_probability_gap = 0.0
        self.over_1e6 = 0
        self.fits = 0

    def check(self, fitted, expected_indices):
        ix = np.asarray(expected_indices, int)
        assert fitted["train_indices"] == ix.tolist()
        training = self.frame.iloc[ix]
        assert len(ix) == len(set(ix)) == fitted["n_train"]
        assert fitted["train_dialogues"] == sorted(training.dialogue_id.unique().tolist())
        features = fitted["features"]
        interaction = fitted["interaction"]
        key = (tuple(ix), tuple(features), str(fitted["penalty"]), interaction)
        if key in self.cache:
            old, theta, refined, penalties = self.cache[key]
            assert old == fitted, "Same exact fitting task produced inconsistent saved fits"
            return theta, refined
        assert fitted["optimizer"] == {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-8}
        n = len(training)
        close(fitted["c0"], training.confidence.mean())
        if fitted["penalty"] == "off":
            assert features == [] and not interaction
            assert fitted["scaler_mean"] == fitted["scaler_scale"] == []
            assert fitted["interaction_scaler_mean"] == fitted["interaction_scaler_scale"] == []
            assert fitted["effective_behavior_coefficients"] == fitted["effective_interaction_coefficients"] == []
            assert fitted["design_divisor"] == 1.0
            penalty = None
        else:
            assert features == ["gaussian_mu", "gaussian_delta", "dropout_mu", "dropout_delta"]
            penalty = float(fitted["penalty"])
            assert penalty in [1., .1, .01]
            values = training[features].to_numpy(float)
            mean, scale = population_scaler(values)
            close(mean, fitted["scaler_mean"])
            close(scale, fitted["scaler_scale"])
            z = (values-mean)/scale
            h = (training.confidence.to_numpy()-training.confidence.mean())[:, None]*z
            hmean, hscale = population_scaler(h)
            close(hmean, fitted["interaction_scaler_mean"])
            close(hscale, fitted["interaction_scaler_scale"])
            constant = np.ptp(values, axis=0) == 0
            if constant.any():
                close(z[:, constant], np.zeros((n, int(constant.sum()))), 0)
                t = (h-hmean)/hscale
                close(t[:, constant], np.zeros((n, int(constant.sum()))), 0)
            if not interaction:
                assert fitted["effective_interaction_coefficients"] == []
            close(fitted["design_divisor"], np.sqrt(n*penalty))
        coeff = np.asarray(fitted["raw_coefficients"], float)
        width = len(features)*(2 if interaction else 1)
        assert len(coeff) == 1+width
        close(coeff[0], fitted["confidence_coefficient"])
        effective = fitted["effective_behavior_coefficients"]+fitted["effective_interaction_coefficients"]
        close(coeff[1:]/fitted["design_divisor"], effective)
        theta = np.r_[fitted["intercept"], fitted["confidence_coefficient"], effective]
        design = self.design(training, fitted)
        target = training.correct.to_numpy(int)
        penalties = np.r_[0., 1/n, np.repeat(penalty, width)] if features else np.array([0., 1/n])
        gradient = design.T@(expit(design@theta)-target)/n+penalties*theta
        gradient_max = float(np.max(np.abs(gradient)))
        self.max_gradient = max(self.max_gradient, gradient_max)
        self.over_1e6 += int(gradient_max > 1e-6)
        # Check equivalence to sklearn's unscaled sum-loss plus L2 objective.
        natural = objective(theta, design, target, penalties)
        raw_equivalent = (np.mean(np.logaddexp(0, design@theta)-target*(design@theta))
                          + np.dot(coeff, coeff)/(2*n))
        close(natural, raw_equivalent)
        refined, gap = newton_refine(theta, design, target, penalties)
        assert gap >= -1e-12, "Convex refinement increased objective"
        self.max_objective_gap = max(self.max_objective_gap, gap)
        self.max_refined_probability_gap = max(self.max_refined_probability_gap,
            float(np.max(np.abs(expit(design@theta)-expit(design@refined)))))
        self.cache[key] = (fitted, theta, refined, penalties)
        self.fits += 1
        return theta, refined

    @staticmethod
    def design(frame, fitted):
        columns = [np.ones(len(frame)), frame.confidence.to_numpy(float)]
        if fitted["features"]:
            z = (frame[fitted["features"]].to_numpy(float)-fitted["scaler_mean"])/fitted["scaler_scale"]
            columns.extend(z.T)
            if fitted["interaction"]:
                h = (frame.confidence.to_numpy(float)-fitted["c0"])[:, None]*z
                t = (h-fitted["interaction_scaler_mean"])/fitted["interaction_scaler_scale"]
                columns.extend(t.T)
        return np.column_stack(columns)

    def predict(self, frame, fitted, expected_indices):
        theta, refined = self.check(fitted, expected_indices)
        x = self.design(frame, fitted)
        base = theta[0]+theta[1]*frame.confidence.to_numpy(float)
        k = len(fitted["features"])
        additive = x[:, 2:2+k]@theta[2:2+k]
        interaction = x[:, 2+k:]@theta[2+k:]
        behavior = additive+interaction
        score = x@theta
        close(score, base+behavior)
        return {"base_score": base, "additive_score": additive, "interaction_score": interaction,
                "behavior_score": behavior, "reliability_score": score, "oof_confidence": expit(score),
                "refined_oof_confidence": expit(x@refined)}


def confidence_bin_statistics(predicted, bins):
    """Description by ORIGINAL C and correctness, with all empty cells retained."""
    assert bins == [0., .4, .6, .8, 1.]
    records = []
    for identity, frame in predicted.groupby(["dataset", "seed", "modality", "method"], sort=False):
        original = frame.confidence.to_numpy(float)
        membership = np.minimum(np.searchsorted(bins, original, side="right")-1, len(bins)-2)
        assert np.all((membership >= 0) & (membership < 4))
        for band in range(4):
            for correct in [0, 1]:
                cell = frame[(membership == band)&frame.correct.eq(correct)]
                record = dict(zip(["dataset", "seed", "modality", "method"], identity))
                record.update(bin_index=band, bin_low=bins[band], bin_high=bins[band+1],
                              correct=correct, n=len(cell))
                q, c = cell.oof_confidence.to_numpy(float), cell.confidence.to_numpy(float)
                record.update(original_mean_confidence=float(c.mean()) if len(cell) else np.nan,
                              calibrated_mean_confidence=float(q.mean()) if len(cell) else np.nan,
                              mean_change=float((q-c).mean()) if len(cell) else np.nan,
                              raised_fraction=float(np.mean(q > c+1e-12)) if len(cell) else np.nan,
                              lowered_fraction=float(np.mean(q < c-1e-12)) if len(cell) else np.nan)
                records.append(record)
    return pd.DataFrame(records)


def verify_confidence_diagnostics(predicted, config):
    saved = csv(ROOT/"results/confidence_bins.csv")
    calculated = confidence_bin_statistics(predicted, config["confidence_diagnostic_bins"])
    key = ["dataset", "seed", "modality", "method", "bin_index", "correct"]
    assert len(saved) == len(calculated) == 320 and not saved.duplicated(key).any()
    merged = saved.merge(calculated, on=key, suffixes=("", "_independent"), validate="one_to_one")
    assert len(merged) == 320
    for column in ["bin_low", "bin_high", "n", "original_mean_confidence", "calibrated_mean_confidence",
                   "mean_change", "raised_fraction", "lowered_fraction"]:
        a, b = merged[column], merged[column+"_independent"]
        assert np.array_equal(a.isna(), b.isna()), column
        close(a[a.notna()], b[b.notna()])
    assert saved.n.sum() == len(predicted)
    return len(merged)


def primary_signal(frame):
    answer = []
    for ds, rows in frame.groupby("dataset"):
        delta = {}
        for seed in SEEDS:
            a = rows[rows.seed.eq(seed)&rows.method.eq("M_interaction")]
            b = rows[rows.seed.eq(seed)&rows.method.eq("E1")]
            ma, mb = metrics(a.oof_confidence, a.correct), metrics(b.oof_confidence, b.correct)
            delta[seed] = {m: ma[m]-mb[m] for m in METRICS}
        means = {m: float(np.mean([delta[s][m] for s in SEEDS])) for m in METRICS}
        improving = {m: int(sum(delta[s][m] < 0 for s in SEEDS)) for m in METRICS}
        proper = ["binary_brier", "binary_nll"]
        signal = means["ece15"] < 0 and all(means[m] <= 0 for m in proper) and any(
            means[m] < 0 and improving[m] >= 3 for m in proper)
        answer.append({"dataset": ds, "candidate": "M_interaction", "reference": "E1",
                       "mean_differences": means, "improving_seed_counts": improving,
                       "preliminary_signal": bool(signal),
                       "ece_lower_but_full_signal_absent": bool(means["ece15"] < 0 and not signal),
                       "exploratory_validation_only": True})
    return answer


def self_test():
    rng = np.random.default_rng(1475)
    n = 100
    c = rng.uniform(.15, .95, n)
    behavior = rng.normal(size=(n, 4))
    behavior[:, 3] = .1
    mean, scale = population_scaler(behavior)
    z = (behavior-mean)/scale
    c0 = c.mean()
    h = (c-c0)[:, None]*z
    hmean, hscale = population_scaler(h)
    t = (h-hmean)/hscale
    close(z[:, 3], np.zeros(n)); close(t[:, 3], np.zeros(n))
    close(z.mean(axis=0), np.zeros(4)); close(t.mean(axis=0), np.zeros(4))
    fitted = {"features": ["gaussian_mu", "gaussian_delta", "dropout_mu", "dropout_delta"],
              "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(), "c0": c0,
              "interaction": True, "interaction_scaler_mean": hmean.tolist(),
              "interaction_scaler_scale": hscale.tolist()}
    frame = pd.DataFrame(behavior, columns=fitted["features"])
    frame["confidence"] = c
    design = FitAudit.design(frame, fitted)
    close(design, np.column_stack([np.ones(n), c, z, t]))
    held = frame.iloc[:3].copy()
    held.loc[:, fitted["features"]] += 15
    held["confidence"] += .1
    expected_z = (held[fitted["features"]].to_numpy()-mean)/scale
    expected_t = ((held.confidence.to_numpy()-c0)[:, None]*expected_z-hmean)/hscale
    close(FitAudit.design(held, fitted), np.column_stack([np.ones(3), held.confidence, expected_z, expected_t]))
    target = rng.binomial(1, expit(-.7+1.9*c+.3*z[:, 0]+.4*t[:, 1]))
    penalties = np.r_[0, 1/n, np.repeat(.1, 8)]
    theta, gap = newton_refine(np.zeros(10), design, target, penalties)
    gradient = design.T@(expit(design@theta)-target)/n+penalties*theta
    assert np.max(np.abs(gradient)) < 1e-9 and gap > 0
    raw_coeff = theta[2:]*np.sqrt(n*.1)
    raw_design = np.column_stack([c, np.c_[z, t]/np.sqrt(n*.1)])
    close(design@theta, theta[0]+raw_design@np.r_[theta[1], raw_coeff])
    close(objective(theta, design, target, penalties),
          np.mean(np.logaddexp(0, design@theta)-target*(design@theta))
          +(theta[1]**2+np.dot(raw_coeff, raw_coeff))/(2*n))
    # Behavioral score allows both upward and downward corrections.
    effect = z[:, 0]+t[:, 1]
    assert effect.min() < 0 < effect.max()
    assert metrics([.2, .8], [0, 1])["ece15"] > .19
    groups = np.array(["a", "b", "c"])
    sample = pd.DataFrame({"dialogue_id": ["a", "a", "b", "c"],
                           "oof_confidence": [.1, .6, .7, .9], "correct": [0, 1, 0, 1]})
    _, literal_checks = bootstrap_values(sample, groups, np.array([[0, 0, 2], [2, 1, 0], [1, 1, 1]]))
    assert literal_checks == 9
    diagnostic = sample.assign(dataset="synthetic", seed=1475, modality="joint", method="E1",
                               confidence=[0., .4, .8, 1.])
    bins = confidence_bin_statistics(diagnostic, [0., .4, .6, .8, 1.])
    assert bins.n.sum() == 4 and len(bins) == 8
    assert bins[bins.bin_index.eq(0)].n.sum() == 1
    assert bins[bins.bin_index.eq(1)].n.sum() == 1
    assert bins[bins.bin_index.eq(2)].n.sum() == 0
    assert bins[bins.bin_index.eq(3)].n.sum() == 2
    print(json.dumps({"engineering_self_test_passed": True,
                      "formal_results_not_claimed": True,
                      "train_only_transform_constant_column_and_regularization_checked": True,
                      "convex_refinement_objective_gap": gap}))


def verify_batch():
    config = json.loads((ROOT/"preregistration.json").read_text())
    out = ROOT/"results"
    completion = json.loads((out/"verification.json").read_text())
    assert completion["status"] == "completed", "Batch not complete"
    inputs = check_inputs(config)
    for manifest in [ROOT/"inputs/source_hashes.json", out/"provenance/source_hashes.json"]:
        for path, expected in json.loads(manifest.read_text()).items():
            assert sha(path) == expected, path
    before = json.loads((out/"provenance/source_hashes_before.json").read_text())
    after = json.loads((out/"provenance/source_hashes_after.json").read_text())
    assert before == after == json.loads((out/"provenance/source_hashes.json").read_text())
    assert str(Path(__file__).resolve()) in before, "Independent verifier absent from frozen snapshot"
    for local, original in [("fold_definitions.json", "baseline/fold_parameters.json"),
                            ("IEMOCAP_dialogue_draws.npz", "baseline/bootstrap/IEMOCAP_dialogue_draws.npz"),
                            ("MELD_dialogue_draws.npz", "baseline/bootstrap/MELD_dialogue_draws.npz")]:
        assert sha(ROOT/"inputs"/local) == sha(Path(config["source_five_seed"])/original)
    frame = csv(ROOT/"inputs/features.csv")
    predicted = csv(out/"oof_predictions.csv")
    per_seed = csv(out/"per_seed_metrics.csv")
    inner_saved = csv(out/"inner_cv_results.csv")
    aggregate = csv(out/"aggregate_metrics.csv")
    bootstrap = csv(out/"bootstrap_results.csv")
    key = KEY+["modality", "method"]
    methods = config["methods"]
    assert len(predicted) == 32720
    assert len(per_seed) == 40
    assert len(bootstrap) == 24
    assert len(inner_saved) == 680
    assert not predicted.duplicated(key).any()
    assert set(predicted.method) == set(methods)
    assert not per_seed.duplicated(["dataset", "seed", "modality", "method"]).any()
    old_folds = json.loads((ROOT/"inputs/fold_definitions.json").read_text())
    outer_count = inner_count = selections = fit_count = gradient_large = 0
    max_gradient = max_objective_gap = max_fit_refinement_gap = max_error = 0.0
    refined_rows = []
    selection_sensitivity = 0
    from sklearn.model_selection import GroupKFold
    for (ds, seed, modality), group in frame.groupby(["dataset", "seed", "modality"], sort=False):
        group = group.reset_index(drop=True)
        audit = FitAudit(group)
        records = json.loads((out/f"fold_parameters/{ds}_seed{seed}_{modality}.json").read_text())
        definitions = {r["fold"]: r for r in old_folds if r["dataset"] == ds and r["seed"] == seed}
        assert len(records) == len(definitions) == (12 if ds == "IEMOCAP" else 5)
        p = predicted[predicted.dataset.eq(ds)&predicted.seed.eq(seed)&predicted.modality.eq(modality)]
        seen = np.zeros(len(group), int)
        for record in records:
            fold = record["fold"]
            definition = definitions[fold]
            train, held = np.array(record["train_indices"]), np.array(record["held_indices"])
            assert sorted(record["train_dialogues"]) == sorted(definition["train_dialogues"])
            assert sorted(record["held_dialogues"]) == sorted(definition["held_dialogues"])
            assert set(record["train_dialogues"]).isdisjoint(record["held_dialogues"])
            assert set(train).isdisjoint(held) and sorted(np.r_[train, held].tolist()) == list(range(len(group)))
            assert np.array_equal(train, np.flatnonzero(group.dialogue_id.isin(record["train_dialogues"])))
            assert np.array_equal(held, np.flatnonzero(group.dialogue_id.isin(record["held_dialogues"])))
            seen[held] += 1
            check = group.iloc[held]
            e1 = audit.predict(check, record["E1"], train)
            actual = {"E1": e1}
            splits = list(GroupKFold(5).split(train, groups=group.iloc[train].dialogue_id))
            assert len(record["inner_folds"]) == 5
            for inner, (tr, va) in zip(record["inner_folds"], splits):
                assert inner["train_indices"] == train[tr].tolist()
                assert inner["held_indices"] == train[va].tolist()
                assert set(inner["train_dialogues"]).isdisjoint(inner["held_dialogues"])
                assert set(inner["train_dialogues"]) | set(inner["held_dialogues"]) == set(record["train_dialogues"])
                audit.check(inner["E1"], train[tr])
            assert set(record["methods"]) == set(methods)-{"E0", "E1"}
            for method, item in record["methods"].items():
                assert item["features"] == config["behavior_features"]
                best = refined_best = "off"
                best_loss = refined_best_loss = np.inf
                for penalty in ["off", 1., .1, .01]:
                    candidates = item["inner_candidate_fits"][str(penalty)]
                    assert len(candidates) == 5
                    sq, refined_sq = [], []
                    for candidate, inner in zip(candidates, record["inner_folds"]):
                        assert candidate["held_indices"] == inner["held_indices"]
                        assert candidate["held_dialogues"] == inner["held_dialogues"]
                        fitted = candidate["fit"]
                        assert fitted["penalty"] == penalty
                        assert fitted["features"] == ([] if penalty == "off" else item["features"])
                        if penalty != "off":
                            assert fitted["interaction"] == (method == "M_interaction")
                        if penalty == "off":
                            assert fitted == inner["E1"], "off must reuse E1 exactly"
                        inner_frame = group.iloc[inner["held_indices"]]
                        result = audit.predict(inner_frame, fitted, inner["train_indices"])
                        error = (result["oof_confidence"]-inner_frame.correct.to_numpy())**2
                        refined_error = (result["refined_oof_confidence"]-inner_frame.correct.to_numpy())**2
                        close(candidate["held_brier"], error.mean())
                        sq.extend(error.tolist()); refined_sq.extend(refined_error.tolist())
                        inner_count += len(inner_frame)
                    loss = float(np.mean(sq)); refined_loss = float(np.mean(refined_sq))
                    close(loss, item["candidate_brier"][str(penalty)])
                    saved = inner_saved[inner_saved.dataset.eq(ds)&inner_saved.seed.eq(seed)&inner_saved.modality.eq(modality)
                                        &inner_saved.fold.eq(fold)&inner_saved.method.eq(method)&inner_saved.penalty.eq(str(penalty))]
                    assert len(saved) == 1
                    close(saved.iloc[0].pooled_inner_brier, loss)
                    if loss < best_loss-1e-12:
                        best, best_loss = penalty, loss
                    if refined_loss < refined_best_loss-1e-12:
                        refined_best, refined_best_loss = penalty, refined_loss
                assert best == item["selected_penalty"] == item["outer_fit"]["penalty"]
                selection_sensitivity += int(best != refined_best)
                if best == "off":
                    assert item["outer_fit"] == record["E1"]
                if best != "off":
                    assert item["outer_fit"]["interaction"] == (method == "M_interaction")
                actual[method] = audit.predict(check, item["outer_fit"], train)
                selections += 1
            for method in methods:
                saved = p[p.method.eq(method)&p.fold.eq(fold)].set_index(KEY)
                expected = check.set_index(KEY)
                assert set(saved.index) == set(expected.index)
                saved = saved.loc[expected.index].reset_index()
                for column in ["confidence", "label", "prediction", "correct"]:
                    close(saved[column], check[column].to_numpy())
                if method == "E0":
                    close(saved.oof_confidence, check.confidence.to_numpy())
                    assert saved.base_score.isna().all() and saved.reliability_score.isna().all()
                    for score_column in ["behavior_score", "additive_score", "interaction_score"]:
                        close(saved[score_column], np.zeros(len(check)))
                    refined_q = check.confidence.to_numpy()
                else:
                    for column in ["base_score", "additive_score", "interaction_score", "behavior_score", "reliability_score", "oof_confidence"]:
                        close(saved[column], actual[method][column])
                    max_error = max(max_error, float(np.max(np.abs(saved.oof_confidence-actual[method]["oof_confidence"]))))
                    refined_q = actual[method]["refined_oof_confidence"]
                refined = saved[KEY+["modality", "method", "correct"]].copy()
                refined["oof_confidence"] = refined_q
                refined_rows.append(refined)
                outer_count += len(check)
        assert (seen == 1).all()
        fit_count += audit.fits; gradient_large += audit.over_1e6
        max_gradient = max(max_gradient, audit.max_gradient)
        max_objective_gap = max(max_objective_gap, audit.max_objective_gap)
        max_fit_refinement_gap = max(max_fit_refinement_gap, audit.max_refined_probability_gap)
    assert outer_count == len(predicted) and selections == 170
    for row in per_seed.to_dict("records"):
        p = predicted[predicted.dataset.eq(row["dataset"])&predicted.seed.eq(row["seed"])
                      &predicted.modality.eq(row["modality"])&predicted.method.eq(row["method"])]
        q, y = p.oof_confidence.to_numpy(), p.correct.to_numpy()
        for metric, value in metrics(q, y).items():
            close(row[metric], value)
        extras = {"accuracy": y.mean(), "weighted_f1": f1_score(p.label, p.prediction, average="weighted"),
                  "macro_f1": f1_score(p.label, p.prediction, average="macro"), "correctness_auc": roc_auc_score(y, q),
                  "mean_confidence": q.mean(), "correct_mean_confidence": q[y == 1].mean(),
                  "wrong_mean_confidence": q[y == 0].mean(),
                  "raised_fraction": np.mean(q > p.confidence.to_numpy()+1e-12),
                  "lowered_fraction": np.mean(q < p.confidence.to_numpy()-1e-12)}
        for metric, value in extras.items():
            close(row[metric], value)
        assert len(p) == row["utterances"] and p.dialogue_id.nunique() == row["dialogues"]
    for _, group in per_seed.groupby(["dataset", "seed", "modality"]):
        for metric in ["accuracy", "weighted_f1", "macro_f1"]:
            assert group[metric].nunique() == 1
    for row in aggregate.to_dict("records"):
        g = per_seed[per_seed.dataset.eq(row["dataset"])&per_seed.modality.eq(row["modality"])&per_seed.method.eq(row["method"])]
        for column, value in row.items():
            if column.endswith("_mean"):
                close(value, g[column[:-5]].mean())
            elif column.endswith("_sample_sd"):
                close(value, g[column[:-10]].std(ddof=1) if len(g) > 1 else 0.)
    bootstrap_checks = literal_checks = 0
    pairs = config["comparisons"]
    seeds = SEEDS
    for (ds, modality), rows in bootstrap.groupby(["dataset", "modality"]):
        assert len(rows) == 3*len(pairs)
        assert set(zip(rows.candidate, rows.reference)) == {tuple(v) for v in pairs}
        source = np.load(ROOT/f"inputs/{ds}_dialogue_draws.npz")
        groups, draws = source["dialogue_ids"].astype(str), source["draws"]
        assert draws.shape == (2000, COUNTS[ds][1])
        values, points = {}, {}
        for seed in seeds:
            for method in methods:
                p = predicted[predicted.dataset.eq(ds)&predicted.seed.eq(seed)&predicted.modality.eq(modality)&predicted.method.eq(method)]
                values[seed, method], count = bootstrap_values(p, groups, draws)
                points[seed, method] = metrics(p.oof_confidence, p.correct)
                literal_checks += count
        for row in rows.to_dict("records"):
            metric, candidate, reference = row["metric"], row["candidate"], row["reference"]
            assert metric in METRICS
            delta = np.array([points[s, candidate][metric]-points[s, reference][metric] for s in seeds])
            distribution = np.mean([values[s, candidate][metric]-values[s, reference][metric] for s in seeds], axis=0)
            close(row["mean_difference"], delta.mean())
            close(row["seed_difference_sample_sd"], delta.std(ddof=1) if len(seeds) > 1 else 0.)
            close(row["ci_low"], np.quantile(distribution, .025))
            close(row["ci_high"], np.quantile(distribution, .975))
            close(row["bootstrap_fraction_improving"], np.mean(distribution < 0))
            assert row["improved_seeds"] == int(np.sum(delta < 0)) and row["seeds"] == len(seeds)
            assert row["resamples"] == 2000 and not row["refit"]
            bootstrap_checks += 1
    refined = pd.concat(refined_rows, ignore_index=True)
    precision = []
    for (ds, modality), data in predicted.groupby(["dataset", "modality"]):
        ref = refined[refined.dataset.eq(ds)&refined.modality.eq(modality)]
        for metric in METRICS:
            before, after = [], []
            for seed in seeds:
                point = {}
                refined_point = {}
                for method in ["M_interaction", "M_joint", "E1"]:
                    rows = data[data.seed.eq(seed)&data.method.eq(method)]
                    r = ref[ref.seed.eq(seed)&ref.method.eq(method)]
                    point[method] = metrics(rows.oof_confidence, rows.correct)[metric]
                    refined_point[method] = metrics(r.oof_confidence, r.correct)[metric]
                before.append(point["M_interaction"]-point["E1"])
                after.append(refined_point["M_interaction"]-refined_point["E1"])
            precision.append({"dataset": ds, "modality": modality, "metric": metric,
                              "candidate": "M_interaction", "reference": "E1",
                              "reported_mean_difference": float(np.mean(before)),
                              "reported_improving_seeds": int(np.sum(np.array(before) < 0)),
                              "refined_improving_seeds": int(np.sum(np.array(after) < 0)),
                              "refined_same_selected_penalty_mean_difference": float(np.mean(after)),
                              "difference_sign_unchanged": bool(np.sign(np.mean(before)) == np.sign(np.mean(after)))})
    historical = csv(ROOT/"inputs/reference_oof.csv")
    paired = predicted[predicted.method.isin(["E0", "E1", "M_joint"])].merge(
        historical, on=KEY+["method"], validate="one_to_one", suffixes=("", "_old"))
    assert len(paired) == 24540 and np.array_equal(paired.fold, paired.fold_old)
    parity = {}
    for method, p in paired.groupby("method"):
        for column in ["oof_confidence", "label", "prediction", "correct"]:
            close(p[column], p[column+"_old"], 1e-12)
        parity[method] = {"rows": len(p), "max_absolute_confidence_difference":
                         float(np.max(np.abs(p.oof_confidence-p.oof_confidence_old))),
                         "folds_identical": True}
    diagnostic_count = verify_confidence_diagnostics(predicted, config)
    signal = primary_signal(predicted)
    refined_signal = primary_signal(refined)
    result = {"all_passed": True, "batch": "interaction", "input_precheck": inputs,
              "oof_predictions_reconstructed": outer_count, "inner_candidate_predictions_reconstructed": inner_count,
              "unique_fits_objective_and_train_scaler_checked": fit_count,
              "nested_selections_recomputed": selections, "metric_rows_recomputed": len(per_seed),
              "bootstrap_rows_recomputed": bootstrap_checks, "literal_cluster_resampling_checks": literal_checks,
              "max_oof_reconstruction_error": max_error, "C_input_raw_and_unstandardized": True,
              "only_registered_mu_delta_scoring_features": True, "off_exact_E1_reuse_verified": True,
              "train_only_standardization_and_dialogue_isolation_verified": True,
              "accuracy_and_F1_unchanged": True, "source_and_implementation_hashes_verified": True,
              "parity": parity, "primary_signal_independently_recomputed": signal,
              "refined_primary_signal_same_selected_penalty": refined_signal,
              "fixed_raw_C_diagnostic_rows_recomputed": diagnostic_count,
              "outer_folds_checked": selections//2, "interaction_train_only_scalers_verified": True,
              "optimizer_accuracy": {"max_mean_objective_gradient": max_gradient,
                 "fits_gradient_above_1e_6_diagnostic_only": gradient_large,
                 "max_independent_refinement_mean_objective_gap": max_objective_gap,
                 "max_training_probability_refinement_gap": max_fit_refinement_gap,
                 "inner_selection_changes_under_refinement": selection_sensitivity,
                 "primary_direction_sensitivity_same_selected_penalty": precision,
                 "original_results_not_replaced": True,
                 "refinement_does_not_establish_new_calibration_method": True}}
    (out/"independent_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        self_test()
        verify_batch()
