"""Independent numerical audit; never imports calibration implementation functions.

All reconstruction below uses serialized coefficients and independently computed
design matrices, likelihoods, scaler statistics, CV selection and cluster draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calibration"
PLAN = json.loads((ROOT / "preregistration.json").read_text())
SOURCE = Path(PLAN["source_run"])
KEY = ["dataset", "seed", "dialogue_id", "utterance_index"]
METHODS = PLAN["methods"]
SEEDS = PLAN["seeds"]
METRICS = ["ece15", "binary_brier", "binary_nll"]
OPTIMALITY_CACHE = {}
PRIMARY_REFINEMENT = []


def model_cache_key(fit):
    return json.dumps([fit["columns"],fit["C"],fit["base"]],sort_keys=True)


def read_csv(path):
    return pd.read_csv(path, dtype={"dialogue_id": str})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, tolerance=1e-10, context=""):
    assert np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=True), context


def scores(probability, correct):
    p = np.clip(np.asarray(probability, float), 1e-15, 1 - 1e-15)
    r = np.asarray(correct, int)
    bins = np.minimum(np.floor(15 * p).astype(int), 14)
    residual = np.bincount(bins, weights=p-r, minlength=15)
    return {"ece15": float(np.abs(residual).sum()/len(p)),
            "binary_brier": float(np.mean((p-r)**2)),
            "binary_nll": float(np.mean(np.logaddexp(0, np.log(p)-np.log1p(-p)) - r*(np.log(p)-np.log1p(-p))))}


def scaler_statistics(x):
    x = np.asarray(x, float)
    mean, variance = x.mean(axis=0), x.var(axis=0)
    eps = np.finfo(np.float64).eps
    constant = variance <= len(x)*eps*variance + (len(x)*mean*eps)**2
    scale = np.sqrt(variance)
    scale[constant] = 1
    return mean, scale


def independent_random_features(dataset, dialogue, utterance):
    key = [dataset, str(dialogue), int(utterance), "capacity_control_seed1475"]
    encoded = json.dumps(key, separators=(",", ":"), ensure_ascii=True).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(4)


def expected_candidates(method, dataset):
    static = {"T": ["logit_C"], "S": ["logit_C", "margin", "normalized_entropy"] +
              [f"pred_class_{k}" for k in range(1, PLAN["datasets"][dataset]["classes"])]}
    grid = PLAN["C_grid_tie_order"]
    def joint(columns, mode):
        return [dict(kind="joint", columns=columns, C=c, mode=mode) for c in grid]
    if method in static:
        return joint(static[method], "base")
    if method == "S_DA_OFFSET":
        return [dict(kind="offset", columns=static["S"], C=c,
                     mode="base" if penalty == "off" else "augmented", penalty=penalty)
                for penalty in PLAN["offset_lambda_tie_order"] for c in grid]
    name = "T" if method == "T_DA" else "S"
    extra = [f"rand_{k}" for k in range(4)] if method == "S_RAND" else PLAN["DA_features"]
    return joint(static[name], "base") + joint(static[name] + extra, "augmented")


def candidate_equal(saved, expected):
    for key, value in expected.items():
        assert saved[key] == value, (key, saved.get(key), value)


def reconstructed(frame, train, fit, audit):
    """Rebuild predictions and verify fitted objective from serialized parameters."""
    cols = fit["columns"]
    x_train, x = train[cols].to_numpy(float), frame[cols].to_numpy(float)
    p = fit["base"]
    mean, scale = scaler_statistics(x_train)
    close(mean, p["mean"], context="training-only scaler mean")
    close(scale, p["scale"], context="training-only scaler scale")
    w, intercept = np.asarray(p["weights"]), float(p["intercept"])
    assert p["train_n"] == len(train) and p["parameter_count"] == len(cols)+1
    assert len(w) == len(cols) and np.isfinite(w).all() and p["iterations"] < 2000
    constant = x_train.var(axis=0) == 0
    if np.any(constant):
        close(w[constant], np.zeros(constant.sum()))
        audit["unseen_held_constant_values_checked"] += int(np.sum(x[:,constant] != mean[constant]))
    train_x = (x_train-mean)/scale
    y = train.correct.to_numpy(int)
    train_z = train_x@w + intercept
    residual = expit(train_z)-y
    # sklearn's sum-loss + ||w||^2/(2C), divided here by n.
    gradient = np.r_[train_x.T@residual/len(y)+w/(float(fit["C"])*len(y)), residual.mean()]
    error = float(np.max(abs(gradient)))
    audit["max_lr_mean_gradient"] = max(audit["max_lr_mean_gradient"], error)
    if error > 2e-5:
        audit["lr_fit_calls_gradient_above_2e5"] += 1
    # L-BFGS may stop on relative objective change before its gradient tolerance.
    # Assess its actual objective gap using a separate exact-Hessian Newton solve;
    # do not impose a gradient stopping guarantee the registered optimizer lacks.
    cache_key = model_cache_key(fit)
    if cache_key not in OPTIMALITY_CACHE:
        design = np.column_stack([train_x,np.ones(len(y))])
        penalty = np.r_[np.full(len(w),1/(float(fit["C"])*len(y))),0.]
        theta = np.r_[w,intercept]
        def objective(v):
            t = design@v
            return np.mean(np.logaddexp(0,t)-y*t)+np.dot(penalty*v,v)/2
        initial = objective(theta)
        for _ in range(30):
            probability = expit(design@theta)
            grad = design.T@(probability-y)/len(y)+penalty*theta
            if np.max(abs(grad)) < 1e-10:
                break
            hessian = (design.T*(probability*(1-probability)))@design/len(y)+np.diag(penalty)
            step = np.linalg.solve(hessian,grad)
            old = objective(theta)
            rate = 1.
            while rate > 2**-30 and objective(theta-rate*step) > old+1e-15:
                rate /= 2
            theta -= rate*step
        gap = max(0.,float(initial-objective(theta)))
        assert gap < 1e-6, ("LR objective optimality gap",gap,fit["C"],cols)
        OPTIMALITY_CACHE[cache_key] = dict(gap=gap,theta=theta.tolist(),gradient=error)
    refined = OPTIMALITY_CACHE[cache_key]
    audit["max_lr_objective_gap"] = max(audit["max_lr_objective_gap"],refined["gap"])
    theta = np.asarray(refined["theta"])
    train_delta = np.max(abs(expit(np.column_stack([train_x,np.ones(len(y))])@theta)-expit(train_z)))
    audit["max_newton_training_base_probability_change"] = max(audit["max_newton_training_base_probability_change"],float(train_delta))
    z = (x-mean)/scale@w+intercept
    held_refined = expit(np.column_stack([(x-mean)/scale,np.ones(len(x))])@theta)
    audit["max_newton_held_base_probability_change"] = max(audit["max_newton_held_base_probability_change"],float(np.max(abs(held_refined-expit(z)))))
    if fit["kind"] == "offset":
        correction = fit["correction"]
        cb = train[PLAN["DA_features"]].to_numpy(float)
        cm, cs = scaler_statistics(cb)
        close(cm, correction["mean"]); close(cs, correction["scale"])
        cw = np.asarray(correction["weights"])
        assert cw.shape == (4,) and correction["penalty"] == fit["penalty"]
        if fit["penalty"] == "off":
            assert np.array_equal(cw, np.zeros(4))
            assert correction["objective"] is None
            audit["offset_off_exact"] += 1
        else:
            cx = (cb-cm)/cs
            logits = train_z+cx@cw
            lam = float(fit["penalty"])
            objective = np.mean(np.logaddexp(0, logits)-y*logits)+lam*np.dot(cw,cw)/2
            grad = cx.T@(expit(logits)-y)/len(y)+lam*cw
            close(objective, correction["objective"])
            close(max(abs(grad)), correction["gradient_max_abs"], tolerance=1e-9)
            assert max(abs(grad)) < 1e-7
            audit["offset_stationary"] += 1
        z += (frame[PLAN["DA_features"]].to_numpy(float)-cm)/cs@cw
    audit["serialized_fits_checked"] += 1
    audit["constant_columns_checked"] += int(np.sum(x_train.var(axis=0) == 0))
    return expit(z)


def verify_features(features):
    assert len(features) == 8180 and not features.duplicated(KEY).any()
    original = read_csv(SOURCE/"baseline/original_predictions.csv")
    prior = read_csv(SOURCE/"behavior/feature_matrix.csv")
    for ds, config in PLAN["datasets"].items():
        for seed in SEEDS:
            f = features[features.dataset.eq(ds)&features.seed.eq(seed)].reset_index(drop=True)
            source = original[original.dataset.eq(ds)&original.seed.eq(seed)].reset_index(drop=True)
            b = prior[prior.dataset.eq(ds)&prior.seed.eq(seed)].reset_index(drop=True)
            assert len(f) == config["n"] and f.dialogue_id.nunique() == config["dialogues"]
            assert f[KEY].equals(source[KEY]) and f[KEY].equals(b[KEY])
            for col in ["label", "prediction", "confidence"]:
                close(f[col], source[col])
            assert np.array_equal(f.correct, (source.prediction==source.label).astype(int))
            probs = np.stack(source.probabilities.map(json.loads))
            confidence = source.confidence.to_numpy(float)
            clipped = np.clip(confidence, 1e-6, 1-1e-6)
            close(f.logit_C, np.log(clipped)-np.log1p(-clipped))
            sorted_p = np.sort(probs, axis=1)
            close(f.margin, sorted_p[:,-1]-sorted_p[:,-2])
            close(f.normalized_entropy, -(probs*np.log(np.maximum(probs, 1e-300))).sum(axis=1)/np.log(config["classes"]))
            for category in range(1, config["classes"]):
                assert np.array_equal(f[f"pred_class_{category}"], (source.prediction==category).astype(int))
            for col in PLAN["DA_features"]:
                close(f[col], b[col])
            expected_rand = np.array([independent_random_features(ds, r.dialogue_id, r.utterance_index) for r in f.itertuples()])
            close(f[[f"rand_{k}" for k in range(4)]], expected_rand)
        # Random controls share the same vectors across model seeds by identity.
        subsets = [features[features.dataset.eq(ds)&features.seed.eq(s)].set_index(["dialogue_id","utterance_index"]).sort_index() for s in SEEDS]
        for frame in subsets[1:]:
            close(frame[[f"rand_{k}" for k in range(4)]], subsets[0][[f"rand_{k}" for k in range(4)]])


def verify_folds(features, predictions, audit):
    old = read_csv(SOURCE/"baseline/oof_predictions.csv")
    for ds, config in PLAN["datasets"].items():
        for seed in SEEDS:
            frame = features[features.dataset.eq(ds)&features.seed.eq(seed)].reset_index(drop=True)
            saved = predictions[predictions.dataset.eq(ds)&predictions.seed.eq(seed)]
            legacy = old[old.dataset.eq(ds)&old.seed.eq(seed)]
            for method in ["XP04", "E1", "L0"]:
                a = saved[saved.method.eq(method)].set_index(KEY).sort_index()
                b = legacy[legacy.method.eq(method)].set_index(KEY).sort_index()
                assert len(a) == len(b) == config["n"] and a.index.equals(b.index)
                close(a.oof_confidence, b.oof_confidence, tolerance=1e-12)
            fold_reference = legacy[legacy.method.eq("XP04")].set_index(KEY).fold
            for method in METHODS:
                p = saved[saved.method.eq(method)]
                aligned = frame[KEY+["label","prediction","correct","confidence"]].merge(p, on=KEY, validate="one_to_one", suffixes=("", "_saved"))
                assert len(aligned) == len(frame)
                for col in ["label","prediction","correct"]:
                    assert np.array_equal(aligned[col], aligned[col+"_saved"])
                close(aligned.confidence,aligned.original_confidence)
                close(aligned["shift"],aligned.oof_confidence-aligned.original_confidence)
                assert np.array_equal(p.set_index(KEY).fold.sort_index(), fold_reference.sort_index())
            groups = frame.dialogue_id.to_numpy(str)
            splitter = LeaveOneGroupOut() if ds == "IEMOCAP" else GroupKFold(5)
            outer = list(splitter.split(frame, groups=groups))
            for fold, (tr, ho) in enumerate(outer):
                record = json.loads((OUT/f"parameters/{ds}_seed{seed}_fold{fold}.json").read_text())
                assert record["dataset"] == ds and record["seed"] == seed and record["fold"] == fold
                assert np.array_equal(tr, record["train_indices"]) and np.array_equal(ho, record["held_indices"])
                train, held = frame.iloc[tr], frame.iloc[ho]
                assert set(train.dialogue_id).isdisjoint(held.dialogue_id)
                assert record["train_dialogues"] == sorted(set(train.dialogue_id))
                assert record["held_dialogues"] == sorted(set(held.dialogue_id))
                for method, baseline in record["historical_baselines"].items():
                    assert method in ["E1", "L0"]
                    train_x = train[["confidence"]].to_numpy(float) if method == "E1" else train[["logit_C"]].to_numpy(float)
                    held_x = held[["confidence"]].to_numpy(float) if method == "E1" else held[["logit_C"]].to_numpy(float)
                    if method == "E1":
                        close(baseline["mean"],[0.]); close(baseline["scale"],[1.])
                    else:
                        mean, scale = scaler_statistics(train_x)
                        close(baseline["mean"],mean); close(baseline["scale"],scale)
                    q = expit(((held_x-baseline["mean"])/baseline["scale"])[:,0]*baseline["coefficient"]+baseline["intercept"])
                    wanted = held[KEY].merge(saved[saved.method.eq(method)],on=KEY,validate="one_to_one")
                    close(q,wanted.oof_confidence)
                    audit["historical_models_reconstructed"] += 1
                inner_splits = list(GroupKFold(5).split(train, groups=train.dialogue_id))
                methods = [m for m in METHODS if m not in ["XP04","E1","L0"]]
                pool = {m: np.full((len(expected_candidates(m, ds)), len(train)), np.nan) for m in methods}
                assert len(record["inner_folds"]) == 5
                for inner, (itr, iho) in zip(record["inner_folds"], inner_splits):
                    assert np.array_equal(itr, inner["train_indices"]) and np.array_equal(iho, inner["held_indices"])
                    ti, hi = train.iloc[itr], train.iloc[iho]
                    assert set(ti.dialogue_id).isdisjoint(hi.dialogue_id)
                    assert inner["train_dialogues"] == sorted(set(ti.dialogue_id)) and inner["held_dialogues"] == sorted(set(hi.dialogue_id))
                    for method in methods:
                        candidates = expected_candidates(method, ds)
                        fits = inner["models"][method]
                        assert len(fits) == len(candidates)
                        for index, (fit, candidate) in enumerate(zip(fits, candidates)):
                            candidate_equal(fit, candidate)
                            pool[method][index, iho] = reconstructed(hi, ti, fit, audit)
                    audit["inner_folds_checked"] += 1
                for method in methods:
                    model = record["methods"][method]
                    actual_candidates = expected_candidates(method, ds)
                    assert len(model["candidates"]) == len(actual_candidates)
                    assert np.isfinite(pool[method]).all()
                    values = np.mean((pool[method]-train.correct.to_numpy(int))**2, axis=1)
                    choice, best = 0, np.inf
                    for index, (candidate, declared, value) in enumerate(zip(actual_candidates, model["candidates"], values)):
                        candidate_equal(declared, candidate)
                        close(value, declared["binary_brier"])
                        if value < best-1e-12:
                            choice, best = index, value
                    assert choice == model["selected_index"]
                    candidate_equal(model["fit"], actual_candidates[choice])
                    if method in ["T_DA","S_DA","S_RAND","S_DA_OFFSET"] and model["fit"]["mode"] == "base":
                        parent = record["methods"]["T" if method == "T_DA" else "S"]
                        assert choice == parent["selected_index"]
                        assert model["fit"]["base"] == parent["fit"]["base"]
                        audit["controlled_base_fallbacks"] += 1
                    for metric, value in scores(pool[method][choice], train.correct).items():
                        close(value, model["inner_metrics"][metric])
                    q = reconstructed(held, train, model["fit"], audit)
                    if method in ["S","S_DA"]:
                        fit = model["fit"]
                        theta = np.asarray(OPTIMALITY_CACHE[model_cache_key(fit)]["theta"])
                        x = (held[fit["columns"]].to_numpy(float)-fit["base"]["mean"])/fit["base"]["scale"]
                        refined = expit(np.column_stack([x,np.ones(len(x))])@theta)
                        PRIMARY_REFINEMENT.append(pd.DataFrame({"dataset":ds,"seed":seed,"method":method,
                            "index":ho,"correct":held.correct.to_numpy(),"original":q,"refined":refined}))
                    held_saved = held[KEY].merge(saved[saved.method.eq(method)], on=KEY, validate="one_to_one")
                    assert held_saved.selected_mode.eq(model["fit"]["mode"]).all()
                    close(held_saved.selected_C,float(model["fit"]["C"]))
                    error = float(np.max(abs(q-held_saved.oof_confidence)))
                    audit["max_outer_reconstruction_error"] = max(audit["max_outer_reconstruction_error"], error)
                    assert error < 1e-10
                    train_q = reconstructed(train, train, model["fit"], audit)
                    for metric, value in scores(train_q, train.correct).items():
                        close(value, model["train_metrics"][metric])
                    audit["outer_models_checked"] += 1
                audit["outer_folds_checked"] += 1


def verify_metrics(predictions, metrics, aggregate):
    assert len(predictions) == 73620 and len(metrics) == 90 and len(aggregate) == 90
    assert not predictions.duplicated(KEY+["method"]).any()
    assert not metrics.duplicated(["dataset","seed","method"]).any()
    assert not aggregate.duplicated(["dataset","method","metric"]).any()
    for row in metrics.to_dict("records"):
        f = predictions[predictions.dataset.eq(row["dataset"])&predictions.seed.eq(row["seed"])&predictions.method.eq(row["method"])]
        y, p = f.correct.to_numpy(int), f.oof_confidence.to_numpy(float)
        c = f.original_confidence.to_numpy(float)
        expected = {**scores(p,y), "accuracy": y.mean(),
                    "weighted_f1": f1_score(f.label,f.prediction,average="weighted"),
                    "macro_f1": f1_score(f.label,f.prediction,average="macro"),
                    "correctness_auc": roc_auc_score(y,p), "mean_confidence": p.mean(),
                    "correct_mean_confidence": p[y==1].mean(), "wrong_mean_confidence": p[y==0].mean(),
                    "raised_fraction": np.mean(p>c+1e-12), "lowered_fraction": np.mean(p<c-1e-12)}
        for key, value in expected.items(): close(value,row[key],context=key)
    for row in aggregate.to_dict("records"):
        f = metrics[metrics.dataset.eq(row["dataset"])&metrics.method.eq(row["method"])]
        assert len(f) == 5
        metric = row["metric"]
        assert metric in METRICS+["correctness_auc","mean_confidence"]
        close(f[metric].mean(),row["mean"])
        close(f[metric].std(ddof=1),row["sample_sd"])


def verify_supplemental(predictions):
    shift = read_csv(OUT/"confidence_shift.csv")
    correct_wrong = read_csv(OUT/"correct_wrong_adjustment.csv")
    assert len(shift) == 90 and len(correct_wrong) == 180
    for row in shift.to_dict("records"):
        p = predictions[predictions.dataset.eq(row["dataset"])&predictions.seed.eq(row["seed"])&predictions.method.eq(row["method"])]
        d = p.oof_confidence.to_numpy()-p.original_confidence.to_numpy()
        close(row["mean_shift"],d.mean())
        close(row["raised_fraction"],np.mean(d>1e-12));close(row["lowered_fraction"],np.mean(d< -1e-12))
    for row in correct_wrong.to_dict("records"):
        p = predictions[predictions.dataset.eq(row["dataset"])&predictions.seed.eq(row["seed"])&predictions.method.eq(row["method"])&predictions.correct.eq(row["correct"])]
        d = p.oof_confidence.to_numpy()-p.original_confidence.to_numpy()
        assert row["n"] == len(p)
        close(row["original_mean"],p.original_confidence.mean());close(row["calibrated_mean"],p.oof_confidence.mean())
        close(row["mean_shift"],d.mean());close(row["raised_fraction"],np.mean(d>1e-12));close(row["lowered_fraction"],np.mean(d< -1e-12))


def verify_bootstrap(predictions, bootstrap, audit):
    assert len(bootstrap) == 60
    expected_pairs = {tuple(pair) for pair in PLAN["comparisons"]}
    for ds in PLAN["datasets"]:
        records = bootstrap[bootstrap.dataset.eq(ds)]
        assert set(zip(records.method,records.reference)) == expected_pairs
        assert not records.duplicated(["method","reference","metric"]).any()
        source = SOURCE/f"baseline/bootstrap/{ds}_dialogue_draws.npz"
        with np.load(source) as z:
            dialogues, draws = z["dialogue_ids"], z["draws"]
        assert draws.shape == (2000,len(dialogues))
        multiplicities = np.stack([np.bincount(d,minlength=len(dialogues)) for d in draws])
        distribution, full = {}, {}
        for seed in SEEDS:
            for method in METHODS:
                f = predictions[predictions.dataset.eq(ds)&predictions.seed.eq(seed)&predictions.method.eq(method)]
                g = np.searchsorted(dialogues,f.dialogue_id.to_numpy(str))
                assert np.array_equal(dialogues[g],f.dialogue_id)
                n = np.bincount(g,minlength=len(dialogues))
                denom = multiplicities@n
                y, p = f.correct.to_numpy(int),np.clip(f.oof_confidence.to_numpy(float),1e-15,1-1e-15)
                bins = np.minimum((15*p).astype(int),14)
                residual = np.zeros((len(dialogues),15));np.add.at(residual,(g,bins),p-y)
                logloss = -(y*np.log(p)+(1-y)*np.log1p(-p))
                distribution[seed,method] = {
                    "ece15": np.abs(multiplicities@residual).sum(axis=1)/denom,
                    "binary_brier": multiplicities@np.bincount(g,weights=(p-y)**2,minlength=len(dialogues))/denom,
                    "binary_nll": multiplicities@np.bincount(g,weights=logloss,minlength=len(dialogues))/denom}
                full[seed,method] = scores(p,y)
                if method in ["XP04","S","S_DA"]:
                    for which in [0,1,1999]:
                        indices = np.concatenate([np.flatnonzero(g==dialogue) for dialogue in draws[which]])
                        for metric,value in scores(p[indices],y[indices]).items():
                            close(value,distribution[seed,method][metric][which])
                            audit["literal_bootstrap_checks"] += 1
        for record in records.to_dict("records"):
            candidate,reference,metric = record["method"],record["reference"],record["metric"]
            delta = np.mean([distribution[s,candidate][metric]-distribution[s,reference][metric] for s in SEEDS],axis=0)
            observed = np.array([full[s,candidate][metric]-full[s,reference][metric] for s in SEEDS])
            expected = {"mean_difference": observed.mean(),
                        "ci_low": np.quantile(delta,.025), "ci_high": np.quantile(delta,.975),
                        "bootstrap_probability_improved": np.mean(delta<0)}
            for key,value in expected.items():close(value,record[key])
            assert record["improved_seeds"] == np.sum(observed<0) and record["repeats"] == 2000
            assert record["draw_source_sha256"] == sha(source) and not record["refit"]
            with np.load(OUT/f"bootstrap/{ds}_metric_differences.npz") as saved:
                close(delta,saved[f"{candidate}__vs__{reference}__{metric}"])
            audit["bootstrap_rows_checked"] += 1


def main():
    audit = {"status": "independently_verified", "all_passed": True,
             "implementation_functions_imported": False,
             "serialized_fits_checked": 0, "constant_columns_checked": 0,
             "unseen_held_constant_values_checked": 0,
             "offset_off_exact": 0, "offset_stationary": 0,
             "outer_folds_checked": 0, "inner_folds_checked": 0,
             "outer_models_checked": 0, "historical_models_reconstructed": 0,
             "controlled_base_fallbacks": 0,
             "max_outer_reconstruction_error": 0.,
             "max_lr_mean_gradient": 0., "bootstrap_rows_checked": 0,
             "max_lr_objective_gap": 0.,
             "lr_fit_calls_gradient_above_2e5": 0,
             "max_newton_training_base_probability_change": 0.,
             "max_newton_held_base_probability_change": 0.,
             "literal_bootstrap_checks": 0}
    features = read_csv(ROOT/"diagnostic/feature_matrix.csv")
    for path,digest in json.loads((OUT/"source_hashes.json").read_text()).items():
        assert sha(path) == digest, ("source changed",path)
    for path,digest in json.loads((OUT/"artifact_hashes.json").read_text()).items():
        assert sha(OUT/path) == digest, ("artifact changed",path)
    context = json.loads((ROOT/"provenance/source_context.json").read_text())
    assert sha(ROOT/"provenance/frozen_correction_core.py") == context["frozen_core_sha256"]
    predictions = read_csv(OUT/"oof_predictions.csv")
    verify_features(features)
    verify_folds(features,predictions,audit)
    verify_metrics(predictions,read_csv(OUT/"per_seed_metrics.csv"),read_csv(OUT/"aggregate_metrics.csv"))
    verify_supplemental(predictions)
    verify_bootstrap(predictions,read_csv(OUT/"bootstrap_results.csv"),audit)
    audit["lr_unique_models_refined"] = len(OPTIMALITY_CACHE)
    audit["lr_unique_models_gradient_above_2e5"] = sum(v["gradient"]>2e-5 for v in OPTIMALITY_CACHE.values())
    sensitivity = pd.concat(PRIMARY_REFINEMENT,ignore_index=True)
    rows = []
    for ds in PLAN["datasets"]:
        for metric in METRICS:
            old, new = [], []
            for seed in SEEDS:
                values = {}
                for method in ["S","S_DA"]:
                    f = sensitivity[sensitivity.dataset.eq(ds)&sensitivity.seed.eq(seed)&sensitivity.method.eq(method)]
                    assert len(f) == PLAN["datasets"][ds]["n"] and not f["index"].duplicated().any()
                    values[method] = [scores(f.original,f.correct)[metric],scores(f.refined,f.correct)[metric]]
                old.append(values["S_DA"][0]-values["S"][0]); new.append(values["S_DA"][1]-values["S"][1])
            rows.append({"dataset":ds,"metric":metric,"original_mean_S_DA_minus_S":float(np.mean(old)),
                         "newton_sensitivity_mean_S_DA_minus_S":float(np.mean(new)),
                         "max_seed_difference_change":float(np.max(abs(np.array(old)-new))),
                         "original_improved_seeds":int(np.sum(np.array(old)<0)),
                         "sensitivity_improved_seeds":int(np.sum(np.array(new)<0)),
                         "mean_direction_unchanged":bool((np.mean(old)<0)==(np.mean(new)<0))})
    audit["numerical_sensitivity_only_not_replacement_model"] = True
    audit["primary_comparison_numeric_sensitivity"] = rows
    audit.update(oof_rows=len(predictions), metric_rows=90, input_utterances=len(features),
                 source_validation_only=True, test_labels_used=False,
                 random_control_label_independent=True,
                 verifier_sha256=sha(Path(__file__)))
    (OUT/"independent_verification.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


def basic_self_test():
    # Fixed points and non-linear ECE cancellation exercise independent scoring.
    assert scores([.2, .8], [0, 1])["binary_brier"] == .04 or abs(scores([.2, .8], [0, 1])["binary_brier"]-.04)<1e-15
    close(scores([.5, .5], [0, 1])["ece15"], 0)
    x = np.array([[1., 0., 2.], [2., 0., 2.], [3., 0., 2.]])
    mean, scale = scaler_statistics(x)
    close(mean, [2, 0, 2]); close(scale, [np.sqrt(2/3), 1, 1])
    assert np.array_equal(independent_random_features("MELD", "d", 1), independent_random_features("MELD", "d", 1))
    print(json.dumps({"independent_formula_self_test": "passed", "formal_results_verified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        basic_self_test()
    else:
        main()
