"""Independently recompute both responses and hash the completed pilot delivery."""
from collections import Counter, defaultdict
import json
from math import isclose
from statistics import mean

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from freeze_pilot import PRIVATE, PROV, RUN, VARIANTS, csv_read, csv_write, digest, dump, load


def probability(value):
    values = json.loads(value)
    assert len(values)==7 and all(0<=p<=1 for p in values) and abs(sum(values)-1)<1e-6
    return values


def check_inference():
    folder = RUN / "results/inference"
    evidence = load(folder / "inference_verification.json")
    assert evidence["status"] == "COMPLETE" and evidence["unexpected_failures"] == 0
    expected = load(PROV/"SOURCE_LOCK.json")
    for filename, value in expected["exports"].items():
        assert digest(RUN/filename)==value
    for name, value in evidence["outputs"].items():
        assert digest(folder/name)==value
    source=csv_read(PRIVATE/"source_clean.csv")
    scores=csv_read(folder/"score_inputs.csv")
    assert len(source)==len(scores)==1412
    assert all(a=={k:b[k] for k in a} for a,b in zip(source,scores))
    paired=defaultdict(dict)
    events=csv_read(folder/"corruption_variants.csv")
    assert len(events)==270
    for row in events:
        sid,version=row["sample_id"],row["variant_id"]
        assert version in VARIANTS and version not in paired[sid]
        paired[sid][version]=row
    assert len(paired)==135 and all(set(rows)==set(VARIANTS) for rows in paired.values())
    assert set(paired)=={r["sample_id"] for r in csv_read(PRIVATE/"target_manifest.csv")}
    n=Counter(); opposite=Counter(); max_error=0
    for row in scores:
        sid=row["sample_id"];role=row["role"]
        clean=probability(row["original_probabilities"])
        c=max(range(7),key=clean.__getitem__)
        assert int(row["original_prediction"])==c and abs(float(row["original_confidence"])-clean[c])<=1e-12
        rows=paired.get(sid,{}).values()
        deltas=[]
        for event in rows:
            assert str(event["dialogue_id"])==row["dialogue_id"] and event["role"]==role
            if event["eligible"] in {"True","1"}:
                assert event["review_status"]=="PASS" and event["inference_status"]=="PREDICTED"
                p=probability(event["perturbed_probabilities"])
                deltas.append(clean[c]-p[c])
        if len(deltas)==2:
            directed=mean(deltas); absolute=mean(map(abs,deltas))
            max_error=max(max_error,abs(directed-float(row["D_pred"])),abs(absolute-float(row["D_abs"])))
            assert absolute+1e-12>=abs(directed)
            n[role] += 1
            opposite[role] += deltas[0]*deltas[1]<0
        else:
            assert not row["D_pred"] and not row["D_abs"], "Missing K2 produced a behavior value"
    assert max_error<=1e-12
    return dict(all_clean_rows=len(scores),planned_variants=len(events),complete_K2=dict(n),
                opposite_sign_K2=dict(opposite),absolute_formula_max_error=max_error)


def check_scoring():
    scoring=RUN/"results/scoring"
    report=load(scoring/"scoring_report.json")
    rows=csv_read(scoring/"evaluation_predictions.csv")
    assert len(rows)==len({r["sample_id"] for r in rows})==671 and {r["role"] for r in rows}=={"pilot_eval"}
    covered=[r for r in rows if r["common_eval_eligible"]=="True"]
    fallback={}
    for method in ("E1","E2","E3"):
        missed=[r for r in rows if r[f"{method}_fallback_to_E0_full"]=="True"]
        assert all(r[f"{method}_policy_R"]==r["E0_full_fit_R"] for r in missed)
        assert len(missed)==len(rows)-len(covered)
        fallback[method]=len(missed)
    primary=next(x for x in report["comparisons"] if x["comparison"]=="E1_to_E2" and
                 x["scope"]=="PRIMARY_COMMON_FIT_COMMON_EVAL")
    if primary["status"]!="COMPUTED":
        return dict(primary_status=primary["status"],shared_eligible_n=len(covered),exact_fallback=fallback)
    correct=np.array([int(r["original_correct"]) for r in covered])
    scores={}
    for arm,key in (("E1","baseline"),("E2","augmented")):
        p=np.array([float(r[f"{arm}_common_fit_R"]) for r in covered])
        scores[arm]=np.clip(p,1e-15,1-1e-15)
        values={"binary_nll":log_loss(correct,scores[arm],labels=[0,1],eps=1e-15),
                "brier":brier_score_loss(correct,p)}
        if len(set(correct))==2:
            values.update(auroc_error=roc_auc_score(1-correct,1-p),auprc_error=average_precision_score(1-correct,1-p))
        assert all(isclose(v,primary[key][m],rel_tol=0,abs_tol=1e-12) for m,v in values.items())
    gain=np.where(correct==1,np.log(scores["E2"]/scores["E1"]),
                  np.log((1-scores["E2"])/(1-scores["E1"])))
    cfg=load(RUN/"EXPERIMENT_PROTOCOL.json")["scoring"]
    universe=sorted({r["dialogue_id"] for r in rows})
    assert len(universe)==60
    pos={did:i for i,did in enumerate(universe)}
    eligible_indices=[pos[r["dialogue_id"]] for r in covered]
    estimates=[]
    draws=np.random.RandomState(cfg["bootstrap_seed"]).randint(0,len(universe),
            (cfg["bootstrap_dialogue_replicates"],len(universe)))
    for d in draws:
        r=np.bincount(d,minlength=len(universe))[eligible_indices]
        if r.sum():
            estimates.append(np.average(gain,weights=r))
    ci=np.quantile(estimates,[.025,.975])
    saved=primary["paired_dialogue_improvement"]["binary_nll"]
    assert isclose(gain.mean(),saved["improvement"],abs_tol=1e-12,rel_tol=0)
    assert np.allclose(ci,[saved["improvement_ci_low"],saved["improvement_ci_high"]],rtol=0,atol=1e-12)
    assert len(estimates)==saved["valid_draws"]
    return dict(primary_status="INDEPENDENT_NUMERIC_REPLAY",shared_eligible_n=len(covered),exact_fallback=fallback,
                primary_nll_improvement=float(gain.mean()),interval=ci.tolist(),valid_draws=len(estimates))


def main():
    assert load(PROV/"PILOT_EXECUTION.json")["status"]=="COMPLETE"
    assert load(PROV/"INDEPENDENCE_GATE.json")["status"]=="CONFIRMATION_PAUSED"
    checks=check_inference()
    scoring=check_scoring()
    for report in ("INDEPENDENCE_AUDIT.md","D_PRED_D_ABS_COMPARISON.md","FINAL_DECISION.md"):
        assert (RUN/report).is_file()
    files=sorted(p for p in RUN.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p.name not in {"SHA256SUMS.csv","verification.json"})
    # Component verifications with this same basename remain part of the delivery.
    files.extend(sorted((RUN/"results").rglob("verification.json")))
    records=[dict(path=str(p.relative_to(RUN)),sha256=digest(p),bytes=p.stat().st_size) for p in files]
    manifest=RUN/"SHA256SUMS.csv"
    csv_write(manifest,records)
    output=dict(status="COMPLETE_TRAIN_ONLY_EXPLORATORY_PILOT",new_independent_evaluation=False,
                confirmatory_scoring_paused=True,test_accessed=False,official_validation_accessed=False,
                inference_checks=checks,scoring_checks=scoring,
                delivery_files=len(records),sha256_manifest=digest(manifest))
    dump(RUN/"verification.json",output)
    print(json.dumps(output))


if __name__=="__main__":
    main()
