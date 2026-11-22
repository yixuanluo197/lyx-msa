"""Replay MELD K2 behavior and fit six CPU-only, same-cohort correctness models."""
import ast
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


RUN = Path(__file__).resolve().parents[1]
BASE = RUN.parent
PROJECT = RUN.parents[3]
SOURCE = BASE / "20261122T041838Z_meld_real_scoring"
SCORER = BASE / "20261121T175233Z_reliability_readiness/scripts/score_heldout.py"
REPO = PROJECT / "CMERC"
MAINLINE = (REPO / "experiments/xp_caldist_behavioral_calibration/xp04_random_feature_behavior_calibration"
            / "mainline_mu_delta/20261115T165025Z/scripts/build_inputs.py")
PLAN_B = BASE / "20261120T143131Z/scripts/pilot_analysis.py"
PATH_PROTOCOL = RUN / "EXPERIMENT_PROTOCOL.json"
OUT = RUN / "results"
ROLE_COUNTS = {"score_fit": 72, "score_eval": 69}
VARIANTS = ("paraphrase_a", "paraphrase_b")
ARMS = ("E0", "E1", "E2", "E3", "E4", "E5")
METRICS = ("binary_nll", "brier", "auroc_error", "auprc_error", "ece15")
SMALLER_IS_BETTER = {"binary_nll", "brier", "ece15"}
ECE_EDGES = np.linspace(0.0, 1.0, 16)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_response():
    tree = ast.parse(MAINLINE.read_text(encoding="utf-8"))
    response = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "response")
    env = {"np": np}
    exec(compile(ast.Module(body=[response], type_ignores=[]), str(MAINLINE), "exec"), env)
    return env["response"]


def bind_sources(protocol):
    assets={}

    def bind(path, expected):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Historical source mismatch: {path}")
        assets[str(path)] = actual

    bind(SCORER, protocol["scoring"]["old_scorer_sha256"])
    bind(MAINLINE, protocol["formula_source_sha256"]["mainline_build_inputs.py"])
    bind(PLAN_B, protocol["formula_source_sha256"]["plan_b_pilot_analysis.py"])
    prior_check = SOURCE / "verification.json"
    rel = prior_check.relative_to(PROJECT)
    original = subprocess.check_output(["git", "-C", str(REPO), "show",
                                       f"{protocol['source_publication_commit']}:{rel}"])
    bind(prior_check, sha256(original).hexdigest())
    check = load(prior_check)
    assert check["status"] == "COMPLETE_VERIFIED" and not check["test_used"] and not check["official_validation_used"]
    for key in ("results/score_inputs.csv", "results/scoring/evaluation_predictions.csv",
                "results/scoring/fitted_parameters.json"):
        bind(SOURCE / key, check["artifacts"][key])
    prediction_path = SOURCE / "results/score_prediction_verification.json"
    inference = load(prediction_path)
    assert inference["status"] == "COMPLETE" and inference["no_test"] and inference["no_model_fit"]
    for name, expected in inference["outputs"].items():
        bind(SOURCE / "results" / name, expected)
    split_path = SOURCE / "SPLIT_LOCK.json"
    bind(split_path, inference["inputs"][str(split_path)])
    split = load(split_path)
    assert split["source_split"] == "Train" and split["no_test_data"] and split["official_validation_rows_used"] == 0
    for name in ("SPLIT_DIALOGUES.csv", "SPLIT_UTTERANCES.csv"):
        bind(SOURCE / "local_only" / name, split["hashes"][name])
    for p in (PATH_PROTOCOL, Path(__file__), prediction_path):
        assets[str(p)] = digest(p)
    shared = import_file("prior_fixed_scorer",SCORER)
    assert protocol["scoring"]["logistic"] == shared.LOGISTIC
    assert protocol["scoring"]["bootstrap"]["draws"] == shared.BOOTSTRAPS
    assert protocol["scoring"]["bootstrap"]["seed"] == shared.SEED
    assert protocol["eligibility"]["variants"] == list(VARIANTS)
    return assets, shared


def probability(value):
    row = np.asarray(json.loads(value),float)
    if row.shape != (7,) or not np.isfinite(row).all() or (row < 0).any() or (row>1).any() or abs(row.sum()-1)>1e-6:
        raise ValueError("Invalid saved seven-class probability vector")
    return row


def construct_features(shared):
    clean=shared.read_csv(SOURCE/"results/score_inputs.csv")
    events=shared.read_csv(SOURCE/"results/score_transformations.csv")
    parsed,scalar_validation=shared.validate_rows(clean,
        shared.read_csv(SOURCE/"local_only/SPLIT_DIALOGUES.csv"),
        shared.read_csv(SOURCE/"local_only/SPLIT_UTTERANCES.csv"))
    assert len(clean)==len(parsed)==2976 and len(events)==5952
    by_id=defaultdict(dict)
    for event in events:
        sid,variant=event["sample_id"],event["variant_id"]
        if variant not in VARIANTS or variant in by_id[sid]:
            raise ValueError("Repeated, wrong or extra K2 variant")
        by_id[sid][variant]=event
    assert set(by_id)=={r["sample_id"] for r in parsed}
    calculate=historical_response()
    features=[];replay=[];max_err={"mu":0.0,"delta":0.0,"D_pred_alignment":0.0}
    exclusions=Counter()
    for row,raw in zip(parsed,clean):
        sid=row["sample_id"]
        assert sid==raw["sample_id"] and sid.startswith("MELD:train:")
        original=probability(raw["original_probabilities"])
        original_class=int(original.argmax())
        clean_conf=float(original[original_class])
        assert raw["original_correct"]==str(int(original_class==int(raw["original_label"])))
        assert int(raw["original_prediction"])==original_class
        pair=by_id[sid]
        assert set(pair)==set(VARIANTS)
        selected=[]
        for name in VARIANTS:
            event=pair[name]
            assert all(event[k]==raw[k] for k in ("role","dialogue_id","utterance_index","utterance_id","original_prediction"))
            assert np.array_equal(original,probability(event["original_probabilities"]))
            if event["eligible"]=="1":
                assert event["review_status"]=="PASS" and event["review_method"]=="AI_CONTEXT_SEMANTIC_REVIEW"
                assert event["inference_status"]=="PREDICTED" and event["encoding_status"]=="ENCODED"
                q=probability(event["perturbed_probabilities"])
                label=int(q.argmax())
                assert label==int(event["perturbed_prediction"])
                assert abs(q[original_class]-float(event["perturbed_original_class_probability"]))<=1e-12
                selected.append((name,q,label))
            else:
                assert event["eligible"]=="0" and not event["perturbed_probabilities"]
        included=len(selected)==2
        assert included==(raw["k2_available"]=="1")==bool(raw["D_pred"])
        if not included:
            exclusions[row["role"],len(selected)] += 1
            continue
        maxima=np.array([q.max() for _,q,_ in selected])
        labels=np.array([label for _,_,label in selected])
        flips=labels!=original_class
        mu=float((maxima*flips).mean())
        delta=float(abs(clean_conf-maxima.mean()))
        original_response=calculate(clean_conf,labels,maxima,original_class)
        stored_d=clean_conf-float(np.mean([q[original_class] for _,q,_ in selected]))
        diff={"mu":abs(mu-original_response["mu"]),"delta":abs(delta-original_response["delta"]),
              "D_pred_alignment":abs(stored_d-float(raw["D_pred"]))}
        for name,error in diff.items():
            max_err[name]=max(max_err[name],error)
            if error>1e-12:
                raise ValueError(f"Behavior replay mismatch for {name}")
        base={**row,"mu":mu,"delta":delta,"flip_count":int(flips.sum()),
              "mean_max_confidence_perturbed":float(maxima.mean()),"signed_mean_confidence_change":float(maxima.mean()-clean_conf)}
        features.append(base)
        for name,q,label in selected:
            replay.append(dict(sample_id=sid,dialogue_id=row["dialogue_id"],role=row["role"],variant_id=name,
                original_correct=row["original_correct"],clean_predicted_class=original_class,
                clean_max_confidence=clean_conf,perturbed_predicted_class=label,perturbed_max_confidence=float(q.max()),
                flip=int(label!=original_class),mu_term=float(q.max()) if label!=original_class else 0.0,
                signed_confidence_change=float(q.max()-clean_conf)))
    assert Counter(r["role"] for r in features)==ROLE_COUNTS
    assert len(replay)==282
    verification = dict(checked_source_clean=len(clean),checked_source_versions=len(events),valid_K2=len(features),
        eligible_variants=len(replay),formula_max_absolute_errors=max_err,
        excluded_counts_by_role_and_valid_version_count={"/".join(map(str,k)):v for k,v in sorted(exclusions.items())},
        no_missing_value_imputed=True,scalar_validation=scalar_validation,source_eval_previously_viewed=True)
    return features,replay,verification


def ece_bins(p):
    return np.clip(np.searchsorted(ECE_EDGES,p,side="right")-1,0,14)


def ece_metric(y,p,weights):
    bin_error=(np.eye(15)[ece_bins(p)])*(y-p)[:,None]
    totals=weights.sum(axis=1)
    return np.divide(np.abs(weights@bin_error).sum(axis=1),totals,
                     out=np.full(totals.shape,np.nan),where=totals>0)


def interval(samples):
    a=np.asarray(samples)
    a=a[np.isfinite(a)]
    return [float(x) for x in np.quantile(a,[.025,.975])] if len(a) else [None,None]


def distribute(rows):
    result=[]
    for role in ROLE_COUNTS:
        sub=[r for r in rows if r["role"]==role]
        for group,y in (("all",None),("correct",1),("error",0)):
            ss=[r for r in sub if y is None or r["original_correct"]==y]
            for feature in ("mu","delta"):
                x=np.asarray([r[feature] for r in ss])
                result.append(dict(role=role,group=group,feature=feature,n=len(x),
                    mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),
                    q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),min=float(x.min()),
                    max=float(x.max()),zero=int((x==0).sum()),nonzero=int((x!=0).sum())))
    return result


def score(shared,protocol,rows):
    fit=[r for r in rows if r["role"]=="score_fit"]
    evaluation=[r for r in rows if r["role"]=="score_eval"]
    assert {r["dialogue_id"] for r in fit}.isdisjoint(r["dialogue_id"] for r in evaluation)
    frozen=load(SOURCE/"results/scoring/fitted_parameters.json")
    models={};coefs=[];pfit={};peval={}
    y=np.array([r["original_correct"] for r in evaluation])
    shared_scalers={};baseline_difference={}
    for arm in ARMS:
        trained=shared.fit_logistic(fit,protocol["arms"][arm])
        model=shared.model_record(trained)
        models[arm]=model
        assert model["fit_sample_ids"]==[r["sample_id"] for r in fit]
        pfit[arm]=shared.predict(trained,fit)
        peval[arm]=shared.predict(trained,evaluation)
        for i,field in enumerate(model["features"]):
            stats=tuple(model[k][i] for k in ("scaler_mean","scaler_scale","scaler_variance"))
            previous=shared_scalers.setdefault(field,stats)
            assert np.allclose(previous,stats,atol=1e-12,rtol=1e-12)
            raw=model["coefficient"][0][i]
            coefs.append(dict(arm=arm,feature=field,standardized_coefficient=raw,
                raw_coefficient=raw/model["scaler_scale"][i],scale=model["scaler_scale"][i],
                fit_mean=model["scaler_mean"][i],intercept=model["intercept"][0]))
        if arm in {"E0","E1"}:
            key="weak_baseline_covered_fit" if arm=="E0" else "strong_baseline_covered_fit"
            old=frozen[key]
            assert model["fit_sample_ids"]==old["fit_sample_ids"] and model["features"]==old["features"]
            max_diff=max(float(np.max(np.abs(np.array(model[k])-np.array(old[k]))))
                         for k in ("coefficient","intercept","scaler_mean","scaler_scale"))
            assert max_diff<=1e-12
            baseline_difference[arm]=max_diff
    peval={"RAW_C":np.array([r["original_confidence"] for r in evaluation]),**peval}
    weights,bootstrap=shared.bootstrap_weights([r["dialogue_id"] for r in evaluation])
    all_weights=np.vstack([np.ones(len(evaluation)),weights])
    metrics={};draw_scores={};bins=[]
    for arm,p in peval.items():
        arrays=shared.metric_batch(y,p,all_weights)
        arrays["ece15"]=ece_metric(y,p,all_weights)
        for k,a in arrays.items():
            assert np.isfinite(a[0])
            bounds=interval(a[1:])
            metrics[arm,k]=dict(arm=arm,metric=k,value=float(a[0]),ci_low=bounds[0],ci_high=bounds[1],
                               valid_draws=int(np.isfinite(a[1:]).sum()))
            draw_scores[arm,k]=a[1:]
        total_binned=0.0
        for j in range(15):
            sel=ece_bins(p)==j
            count=int(sel.sum())
            error=float(np.abs((y[sel]-p[sel]).sum())/len(y)) if count else 0.0
            total_binned+=error
            bins.append(dict(arm=arm,bin=j,left=ECE_EDGES[j],right=ECE_EDGES[j+1],n=count,
                             mean_probability=float(p[sel].mean()) if count else None,
                             correct_fraction=float(y[sel].mean()) if count else None,
                             ece_contribution=error))
        assert abs(total_binned-metrics[arm,"ece15"]["value"])<=1e-12
    contrasts=[]
    for a,b in protocol["scoring"]["comparisons"]:
        for metric in METRICS:
            mult=1 if metric in SMALLER_IS_BETTER else -1
            x,y1=metrics[a,metric]["value"],metrics[b,metric]["value"]
            sims=mult*(draw_scores[a,metric]-draw_scores[b,metric])
            bounds=interval(sims)
            contrasts.append(dict(reference=a,model=b,metric=metric,reference_value=x,model_value=y1,
                 improvement=mult*(x-y1),ci_low=bounds[0],ci_high=bounds[1],
                 valid_draws=int(np.isfinite(sims).sum()),requested_draws=len(weights)))
    predictions=[]
    for cohort,p in ((fit,pfit),(evaluation,peval)):
        for i,r in enumerate(cohort):
            pred={k:r[k] for k in ("sample_id","dialogue_id","role","original_correct","original_confidence",
                                 "clean_margin","clean_entropy","mu","delta","flip_count")}
            predictions.append({**pred,"in_sample":r["role"]=="score_fit",
                                **{f"{arm}_R":float(p[arm][i]) for arm in ARMS}})
    write_csv(OUT/"scorer_metrics.csv",list(metrics.values()))
    write_csv(OUT/"paired_contrasts.csv",contrasts)
    write_csv(OUT/"probability_predictions.csv",predictions)
    write_csv(OUT/"coefficients.csv",coefs)
    write_csv(OUT/"ece15_bins.csv",bins)
    dump(OUT/"fitted_parameters.json",models)
    np.savez_compressed(OUT/"bootstrap_draws.npz",dialogue_row_weights=weights,
                        **{f"{a}__{k}":v for (a,k),v in draw_scores.items()})
    summary=dict(status="COMPLETE_OFFLINE_EXPLORATORY_SCORING",
        roles={str(group[0]["role"]):dict(n=len(group),dialogues=len({r["dialogue_id"] for r in group}),
                correct=sum(r["original_correct"] for r in group),
                errors=sum(1-r["original_correct"] for r in group)) for group in (fit,evaluation)},
        baseline_parameter_replay_max_absolute_errors=baseline_difference,bootstrap=bootstrap,
        positive_improvement_is_better=True,pointwise_unadjusted=True,confirmatory=False)
    dump(OUT/"scoring_summary.json",summary)
    return summary


def main():
    if any(OUT.iterdir()):
        raise FileExistsError("Refusing to overwrite experiment outputs")
    protocol=load(PATH_PROTOCOL)
    assets,shared=bind_sources(protocol)
    rows,version_rows,checks=construct_features(shared)
    dump(OUT/"formula_verification.json",checks)
    write_csv(OUT/"k2_features.csv",rows)
    write_csv(OUT/"version_replay.csv",version_rows)
    write_csv(OUT/"correct_wrong_distributions.csv",distribute(rows))
    summary=score(shared,protocol,rows)
    if any(digest(Path(p))!=v for p,v in assets.items()):
        raise ValueError("A source, code or protocol file changed during scoring")
    source=RUN/"provenance/SOURCE_LOCK.json"
    dump(source,dict(source_read_only=True,sources=assets,no_model_or_test_access=True))
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    main()
