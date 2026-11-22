"""Independently verify K2 reductions, fitted score replay and paired statistics."""
from collections import defaultdict
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path
import sys

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from score_mu_delta import OUT, RUN, digest, dump, load, write_csv
import csv


def read(path):
    with path.open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def equal(x,y):
    assert isclose(float(x),float(y),abs_tol=1e-12,rel_tol=0)


def check_responses(features):
    replay=defaultdict(list)
    for row in read(OUT/"version_replay.csv"):
        replay[row["sample_id"]].append(row)
    assert len(features)==len(replay)==141
    for row in features:
        variants=replay[row["sample_id"]]
        assert {r["variant_id"] for r in variants}=={"paraphrase_a","paraphrase_b"}
        delta=abs(float(row["original_confidence"])-fsum(float(r["perturbed_max_confidence"]) for r in variants)/2)
        mu=fsum(float(r["perturbed_max_confidence"]) for r in variants if
                r["perturbed_predicted_class"]!=r["clean_predicted_class"])/2
        equal(row["delta"],delta)
        equal(row["mu"],mu)


def check_scores():
    features=read(OUT/"k2_features.csv")
    check_responses(features)
    rows=read(OUT/"probability_predictions.csv")
    assert len(rows)==len({r["sample_id"] for r in rows})==141
    lookup={r["sample_id"]:r for r in rows}
    models=load(OUT/"fitted_parameters.json")
    fit_ids=[r["sample_id"] for r in features if r["role"]=="score_fit"]
    significance=load(OUT/"scoring_summary.json")
    for arm,model in models.items():
        assert model["fit_sample_ids"]==fit_ids
        for row in features:
            v=lookup[row["sample_id"]]
            assert v["role"]==row["role"] and v["original_correct"]==row["original_correct"]
            standardized=(np.array([float(row[field]) for field in model["features"]])
                          -np.array(model["scaler_mean"]))/np.array(model["scaler_scale"])
            p=expit(standardized@np.array(model["coefficient"][0])+model["intercept"][0])
            equal(p,v[arm+"_R"])
    eval_rows=[r for r in rows if r["role"]=="score_eval"]
    assert len(eval_rows)==69 and sum(r["original_correct"]=="0" for r in eval_rows)==25
    y=np.array([int(r["original_correct"]) for r in eval_rows])
    draws=np.load(OUT/"bootstrap_draws.npz",allow_pickle=False)
    settings=load(RUN/"EXPERIMENT_PROTOCOL.json")["scoring"]["bootstrap"]
    dids=sorted({r["dialogue_id"] for r in eval_rows})
    draws_index=np.random.RandomState(settings["seed"]).randint(0,len(dids),(settings["draws"],len(dids)))
    assert sha256(draws_index.astype("<i8").tobytes()).hexdigest()==significance["bootstrap"]["draw_indices_sha256"]
    groups=[dids.index(row["dialogue_id"]) for row in eval_rows]
    weights=np.array([np.bincount(row,minlength=len(dids))[groups] for row in draws_index])
    assert np.array_equal(weights,draws["dialogue_row_weights"])
    points={(r["arm"],r["metric"]):float(r["value"]) for r in read(OUT/"scorer_metrics.csv")}
    for arm in ("RAW_C",*models):
        field="original_confidence" if arm=="RAW_C" else arm+"_R"
        p=np.array([float(r[field]) for r in eval_rows])
        edges=[i/15 for i in range(16)]
        groups=np.array([sum(s>=left for left in edges[1:-1]) for s in p])
        ece=lambda w:np.abs(np.bincount(groups,weights=w*(y-p),minlength=15)).sum()/w.sum()
        equal(points[arm,"ece15"],ece(np.ones(len(y))))
        equal(points[arm,"binary_nll"],log_loss(y,p,labels=[0,1],eps=1e-15))
        equal(points[arm,"brier"],brier_score_loss(y,p))
        equal(points[arm,"auroc_error"],roc_auc_score(1-y,1-p))
        equal(points[arm,"auprc_error"],average_precision_score(1-y,1-p))
        np.testing.assert_allclose(draws[arm+"__ece15"],[ece(w) for w in weights],atol=1e-12,rtol=0)
        nll=-(y*np.log(np.clip(p,1e-15,1-1e-15))+(1-y)*np.log1p(-np.clip(p,1e-15,1-1e-15)))
        mean=lambda x:np.array([np.average(x,weights=w) for w in weights])
        np.testing.assert_allclose(draws[arm+"__binary_nll"],mean(nll),atol=1e-12,rtol=0)
        np.testing.assert_allclose(draws[arm+"__brier"],mean((p-y)**2),atol=1e-12,rtol=0)
    for row in read(OUT/"paired_contrasts.csv"):
        a,b,metric=(row[k] for k in ("reference","model","metric"))
        sign=-1 if metric in {"auroc_error","auprc_error"} else 1
        equal(row["improvement"],sign*(points[a,metric]-points[b,metric]))
        diff=sign*(draws[a+"__"+metric]-draws[b+"__"+metric])
        valid=diff[np.isfinite(diff)]
        assert len(valid)==int(row["valid_draws"])
        lo,hi=np.quantile(valid,[.025,.975])
        equal(row["ci_low"],lo)
        equal(row["ci_high"],hi)
    return dict(formula_K2_rows=141,per_row_score_replay=len(rows)*len(models),verified_metric_points=len(points),
                verified_paired_intervals=len(read(OUT/"paired_contrasts.csv")),
                independently_verified_bootstrap_draws=len(weights))


def main():
    checks=check_scores()
    for path, expected in load(RUN/"provenance/SOURCE_LOCK.json")["sources"].items():
        assert digest(Path(path))==expected
    assert (RUN/"FINAL_DECISION.md").is_file()
    artifacts=sorted(p for p in RUN.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.name not in {"verification.json","SHA256SUMS.csv"})
    manifest=RUN/"SHA256SUMS.csv"
    write_csv(manifest,[dict(path=str(p.relative_to(RUN)),sha256=digest(p),bytes=p.stat().st_size) for p in artifacts])
    result=dict(status="COMPLETE_VERIFIED_EXPLORATORY_OFFLINE_SCORING",checks=checks,no_model_inference=True,
                no_test_access=True,confirmatory=False,files=len(artifacts),sha256_manifest=digest(manifest),
                versions=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__))
    dump(RUN/"verification.json",result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
