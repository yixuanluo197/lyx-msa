"""Independent modality audit, preserving the immutable joint verifier.

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


def _assert_missing(value):
    assert pd.isna(value), "Single-seed sample SD must be missing, not zero"


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
    joint = csv(ROOT/"inputs/joint_features.csv")
    modality = csv(ROOT/"inputs/modality_features.csv")
    previous = csv(Path(config["source_five_seed"])/"behavior/feature_matrix.csv")
    pilot = csv(Path(config["source_pilot"])/"behavior_features.csv")
    columns = config["features"]["M_joint"]
    assert len(joint) == 8180 and len(modality) == 6544
    assert not joint.duplicated(KEY).any() and not modality.duplicated(KEY+["modality"]).any()
    for seed, data in [(None, joint), (1475, modality)]:
        for (ds, actual_seed, mod), frame in data.groupby(["dataset", "seed", "modality"]):
            n, groups = COUNTS[ds]
            assert len(frame) == n and frame.dialogue_id.nunique() == groups
            assert frame.visual_missing.sum() == (1 if ds == "IEMOCAP" else 95)
            assert np.array_equal(frame.correct, (frame.prediction == frame.label).astype(int))
            if seed is not None:
                assert actual_seed == seed
            close(frame.perturbation_skipped.astype(int),
                  (frame.visual_missing & (mod == "visual")).astype(int))
            assert np.isfinite(frame[["confidence"]+columns]).all().all()
    paired = joint.merge(previous, on=KEY, validate="one_to_one", suffixes=("", "_old"))
    assert len(paired) == len(joint)
    for column in columns+["confidence", "label", "prediction", "correct"]:
        close(paired[column], paired[column+"_old"])
    matched = modality[modality.modality.eq("joint")].merge(
        joint[joint.seed.eq(1475)], on=KEY+["modality"], validate="one_to_one", suffixes=("", "_joint"))
    assert len(matched) == 1636
    for column in columns+["confidence", "label", "prediction", "correct"]:
        close(matched[column], matched[column+"_joint"])
    checked = 0
    for corruption in ["gaussian", "dropout"]:
        paired = modality.merge(pilot[pilot.corruption.eq(corruption)], on=KEY+["modality"],
                                validate="one_to_one", suffixes=("", "_old"))
        assert len(paired) == 6544
        for metric in ["mu", "delta"]:
            close(paired[corruption+"_"+metric], paired[metric])
            checked += len(paired)
    return {"joint_rows": len(joint), "modality_rows": len(modality),
            "historical_modality_mu_delta_values_matched": checked,
            "visual_missing_rows_preserved": True,
            "joint_modality_seed1475_inputs_identical": True}


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
        key = (tuple(ix), tuple(features), str(fitted["penalty"]))
        if key in self.cache:
            old, theta, refined, penalties = self.cache[key]
            assert old == fitted, "Same exact fitting task produced inconsistent saved fits"
            return theta, refined
        assert fitted["optimizer"] == {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-8}
        n = len(training)
        if fitted["penalty"] == "off":
            assert features == [] and fitted["scaler_mean"] == fitted["scaler_scale"] == []
            assert fitted["effective_behavior_coefficients"] == []
            assert fitted["design_divisor"] == 1.0
            penalty = None
        else:
            assert features and set(features) <= {
                "gaussian_mu", "gaussian_delta", "dropout_mu", "dropout_delta"}
            penalty = float(fitted["penalty"])
            assert penalty in [1., .1, .01]
            values = training[features].to_numpy(float)
            mean = values.mean(axis=0)
            variance = values.var(axis=0)
            # Match population standard deviation and sklearn's numerical constant bound.
            upper = n*np.finfo(float).eps*variance+(n*mean*np.finfo(float).eps)**2
            scale = np.sqrt(variance)
            scale[variance <= upper] = 1
            close(mean, fitted["scaler_mean"])
            close(scale, fitted["scaler_scale"])
            close(fitted["design_divisor"], np.sqrt(n*penalty))
        coeff = np.asarray(fitted["raw_coefficients"], float)
        assert len(coeff) == 1+len(features)
        close(coeff[0], fitted["confidence_coefficient"])
        close(coeff[1:]/fitted["design_divisor"], fitted["effective_behavior_coefficients"])
        theta = np.r_[fitted["intercept"], fitted["confidence_coefficient"],
                      fitted["effective_behavior_coefficients"]]
        design = self.design(training, fitted)
        target = training.correct.to_numpy(int)
        penalties = np.r_[0., 1/n, np.repeat(penalty, len(features))] if features else np.array([0., 1/n])
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
        return np.column_stack(columns)

    def predict(self, frame, fitted, expected_indices):
        theta, refined = self.check(fitted, expected_indices)
        x = self.design(frame, fitted)
        base = theta[0]+theta[1]*frame.confidence.to_numpy(float)
        behavior = x[:, 2:]@theta[2:]
        score = x@theta
        close(score, base+behavior)
        return {"base_score": base, "behavior_score": behavior,
                "reliability_score": score, "oof_confidence": expit(score),
                "refined_oof_confidence": expit(x@refined)}


def self_test():
    rng = np.random.default_rng(1475)
    n = 100
    c = rng.uniform(.15, .95, n)
    behavior = rng.normal(size=(n, 4))
    z = (behavior-behavior.mean(axis=0))/behavior.std(axis=0)
    design = np.column_stack([np.ones(n), c, z])
    target = rng.binomial(1, expit(-.7+1.9*c+.3*z[:, 0]))
    penalties = np.array([0, 1/n, .1, .1, .1, .1])
    theta, gap = newton_refine(np.zeros(6), design, target, penalties)
    gradient = design.T@(expit(design@theta)-target)/n+penalties*theta
    assert np.max(np.abs(gradient)) < 1e-9 and gap > 0
    raw_coeff = theta[2:]*np.sqrt(n*.1)
    raw_design = np.column_stack([c, z/np.sqrt(n*.1)])
    close(design@theta, theta[0]+raw_design@np.r_[theta[1], raw_coeff])
    close(objective(theta, design, target, penalties),
          np.mean(np.logaddexp(0, design@theta)-target*(design@theta))
          +(theta[1]**2+np.dot(raw_coeff, raw_coeff))/(2*n))
    assert metrics([.2, .8], [0, 1])["ece15"] > .19
    print(json.dumps({"engineering_self_test_passed": True,
                      "formal_results_not_claimed": True,
                      "convex_refinement_objective_gap": gap}))


def verify_batch(batch):
    config = json.loads((ROOT/"preregistration.json").read_text())
    out = ROOT/batch
    completion = json.loads((out/"verification.json").read_text())
    assert completion["status"] == "completed", "Batch not complete"
    inputs = check_inputs(config)
    for path, expected in json.loads((out/"provenance/source_hashes.json").read_text()).items():
        assert sha(path) == expected, path
    frame = csv(ROOT/f"inputs/{'joint' if batch == 'joint' else 'modality'}_features.csv")
    predicted = csv(out/"oof_predictions.csv")
    per_seed = csv(out/"per_seed_metrics.csv")
    inner_saved = csv(out/"inner_cv_results.csv")
    aggregate = csv(out/"aggregate_metrics.csv")
    bootstrap = csv(out/"bootstrap_results.csv")
    key = KEY+["modality", "method"]
    methods = config["methods"] if batch == "joint" else ["E0", "E1", "M_joint"]
    assert len(predicted) == (57260 if batch == "joint" else 19632)
    assert len(per_seed) == (70 if batch == "joint" else 24)
    assert len(bootstrap) == (54 if batch == "joint" else 24)
    assert not predicted.duplicated(key).any()
    assert set(predicted.method) == set(methods)
    assert not per_seed.duplicated(["dataset", "seed", "modality", "method"]).any()
    old_folds = json.loads((Path(config["source_five_seed"])/"baseline/fold_parameters.json").read_text())
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
                assert item["features"] == config["features"][method]
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
                    close(saved.behavior_score, np.zeros(len(check)))
                    refined_q = check.confidence.to_numpy()
                else:
                    for column in ["base_score", "behavior_score", "reliability_score", "oof_confidence"]:
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
    assert outer_count == len(predicted)
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
                if len(g) > 1:
                    close(value, g[column[:-10]].std(ddof=1))
                else:
                    _assert_missing(value)
    bootstrap_checks = literal_checks = 0
    pairs = config["joint_comparisons"] if batch == "joint" else [["M_joint", "E1"]]
    seeds = SEEDS if batch == "joint" else [1475]
    for (ds, modality), rows in bootstrap.groupby(["dataset", "modality"]):
        assert len(rows) == 3*len(pairs)
        assert set(zip(rows.candidate, rows.reference)) == {tuple(v) for v in pairs}
        source = np.load(Path(config["source_five_seed"])/f"baseline/bootstrap/{ds}_dialogue_draws.npz")
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
            if len(seeds) > 1:
                close(row["seed_difference_sample_sd"], delta.std(ddof=1))
            else:
                _assert_missing(row["seed_difference_sample_sd"])
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
                for method in ["M_joint", "E1"]:
                    rows = data[data.seed.eq(seed)&data.method.eq(method)]
                    r = ref[ref.seed.eq(seed)&ref.method.eq(method)]
                    point[method] = metrics(rows.oof_confidence, rows.correct)[metric]
                    refined_point[method] = metrics(r.oof_confidence, r.correct)[metric]
                before.append(point["M_joint"]-point["E1"])
                after.append(refined_point["M_joint"]-refined_point["E1"])
            precision.append({"dataset": ds, "modality": modality, "metric": metric,
                              "reported_mean_difference": float(np.mean(before)),
                              "refined_same_selected_penalty_mean_difference": float(np.mean(after)),
                              "difference_sign_unchanged": bool(np.sign(np.mean(before)) == np.sign(np.mean(after)))})
    parity = None
    if batch == "joint":
        legacy = csv(Path(config["source_five_seed"])/"baseline/oof_predictions.csv")
        paired = predicted[predicted.method.eq("E1")].merge(legacy[legacy.method.eq("E1")], on=KEY,
                                                          validate="one_to_one", suffixes=("", "_old"))
        assert len(paired) == 8180 and np.array_equal(paired.fold, paired.fold_old)
        saved = json.loads((out/"legacy_e1_precision_comparison.json").read_text())
        for row in saved["comparisons"]:
            g = paired[paired.dataset.eq(row["dataset"])&paired.seed.eq(row["seed"])]
            close(row["max_absolute_confidence_difference"], np.max(np.abs(g.oof_confidence-g.oof_confidence_old)))
        parity = {"legacy_e1_max_absolute_difference": float(np.max(np.abs(paired.oof_confidence-paired.oof_confidence_old))),
                  "legacy_e1_folds_identical": True, "precision_change_not_behavior_gain": True}
    else:
        joint = csv(ROOT/"joint/oof_predictions.csv")
        paired = predicted[predicted.modality.eq("joint")].merge(joint[joint.seed.eq(1475)&joint.method.isin(methods)],
                          on=key, validate="one_to_one", suffixes=("", "_joint"))
        assert len(paired) == 4908
        for column in ["oof_confidence", "behavior_score"]:
            close(paired[column], paired[column+"_joint"], 1e-12)
        assert np.array_equal(paired.fold, paired.fold_joint)
        parity = {"seed1475_joint_oof_rows_identical_to_main_batch": len(paired)}
    result = {"all_passed": True, "batch": batch, "input_precheck": inputs,
              "oof_predictions_reconstructed": outer_count, "inner_candidate_predictions_reconstructed": inner_count,
              "unique_fits_objective_and_train_scaler_checked": fit_count,
              "nested_selections_recomputed": selections, "metric_rows_recomputed": len(per_seed),
              "bootstrap_rows_recomputed": bootstrap_checks, "literal_cluster_resampling_checks": literal_checks,
              "max_oof_reconstruction_error": max_error, "C_input_raw_and_unstandardized": True,
              "only_registered_mu_delta_scoring_features": True, "off_exact_E1_reuse_verified": True,
              "train_only_standardization_and_dialogue_isolation_verified": True,
              "accuracy_and_F1_unchanged": True, "source_and_implementation_hashes_verified": True,
              "parity": parity,
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
    parser.add_argument("--batch", choices=["joint", "modality"], default="modality")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        verify_batch(args.batch)
