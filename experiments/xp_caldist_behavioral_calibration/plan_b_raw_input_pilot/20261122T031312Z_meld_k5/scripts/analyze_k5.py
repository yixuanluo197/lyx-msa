#!/usr/bin/env python3
"""Compute fixed-cohort MELD paraphrase repeatability from frozen inference CSVs."""
import csv
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent/"20261120T170835Z"
VARIANTS = tuple(f"paraphrase_{c}" for c in "abcde")
SCOPES = ("AI_COMPLETE","INPUT_QC_COMPLETE")
METRICS = ("D_pred","flip_rate","JS","D_C","mu","delta","V_C","D_margin")
BOOTSTRAP = 2000
SEED = 20261122


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        raise ValueError(f"Missing rows for {path}")
    with path.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def check(test, message):
    if not test:
        raise ValueError(message)


def truth(value):
    return str(value).lower() in {"true","1"}


def probability(payload):
    p=np.asarray(json.loads(payload),dtype=np.float64)
    check(p.shape==(7,) and np.isfinite(p).all() and (p>=0).all() and
          np.isclose(p.sum(),1,atol=1e-5,rtol=0),"Invalid seven-class probability")
    return p


def js(p,q):
    half=(p+q)/2
    a,b=p>0,q>0
    return .5*float(np.sum(p[a]*np.log(p[a]/half[a]))+np.sum(q[b]*np.log(q[b]/half[b])))


def metrics(p, q):
    c=int(p.argmax())
    other=np.arange(len(p))!=c
    conf=q.max(1)
    flipped=q.argmax(1)!=c
    output=dict(D_pred=p[c]-q[:,c].mean(), flip_rate=flipped.mean(),
                JS=np.mean([js(p,v) for v in q]), D_C=conf.mean()-p[c],
                mu=np.mean(flipped*conf), delta=abs(conf.mean()-p[c]), V_C=conf.std(ddof=1),
                D_margin=(p[c]-p[other].max())-np.mean(q[:,c]-q[:,other].max(1)))
    check(np.isclose(output["delta"],abs(output["D_C"]),atol=1e-14),"delta identity")
    if not flipped.any():
        check(np.isclose(output["D_pred"],-output["D_C"],atol=1e-14),"No-flip identity")
    return {k:float(v) for k,v in output.items()}


def available(row, scope):
    if row["status"] != "SUCCESS" or not truth(row["automatic_qc_pass"]):
        return False
    return scope == SCOPES[1] or truth(row["label_preserving_main_eligible"])


def analysis_rows(samples, variants):
    target, subgroups, subset_values = [], defaultdict(dict), []
    for sid, sample in samples.items():
        p=probability(sample["original_probabilities"])
        event=variants[sid]
        for scope in SCOPES:
            usable={v:probability(event[v]["perturbed_probabilities"]) for v in VARIANTS if available(event[v],scope)}
            info = dict(sample_id=sid, dialogue_id=sample["dialogue_id"], pair_id=sample["pair_id"],
                        original_correct=int(sample["original_correct"]), scope=scope)
            solutions = {}
            for version_set, wanted in (("K2_ab",VARIANTS[:2]), ("K2_cd",VARIANTS[2:4]), ("K5_all",VARIANTS)):
                data=metrics(p,np.asarray([usable[v] for v in wanted])) if all(v in usable for v in wanted) else None
                solutions[version_set]=data
                target.append(dict(info, version_set=version_set, planned_k=len(wanted),
                    valid_k=sum(v in usable for v in wanted), eligible=data is not None,
                    exclusion="|".join(v for v in wanted if v not in usable), **(data or dict.fromkeys(METRICS))))
            subgroups[scope][sid]=dict(info, values=solutions)
            if solutions["K5_all"] is None:
                continue
            for k in (2,3,4):
                # These combinations overlap; only their conditional spread is interpreted.
                for ix,combo in enumerate(combinations(VARIANTS,k)):
                    behavior=metrics(p,np.asarray([usable[v] for v in combo]))
                    subset_values.append(dict(info,k=k,subset_id=ix,variants="|".join(combo),
                                              D_pred=behavior["D_pred"]))
    return target, subgroups, subset_values


def cluster_vector(rows, field, universe, draws):
    groups=defaultdict(list)
    for r in rows:
        if r.get(field) is not None:
            groups[r["dialogue_id"]].append(float(r[field]))
    if not groups:
        return (None,None,None,0)
    per=np.array([np.mean(groups[d]) if d in groups else np.nan for d in universe])
    points=per[draws]
    num=np.nansum(points,axis=1)
    den=np.isfinite(points).sum(axis=1)
    boot=np.divide(num,den,out=np.full(len(den),np.nan),where=den>0)
    finite=boot[np.isfinite(boot)]
    return (float(np.nanmean(per)),*(float(x) for x in np.quantile(finite,[.025,.975])),len(finite))


def correlation(x,y,rank=False):
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:
        return None
    a,b=(rankdata(x),rankdata(y)) if rank else (x,y)
    return float(np.corrcoef(a,b)[0,1])


def pairwise(pairs, subgroups, subset_values, universe, draws):
    condition_rows=[]; pair_records=[]; agreement_rows=[]; dispersion_rows=[]
    for scope, entries in subgroups.items():
        pool={"K2_ab_all_eligible_pairs":[],"K2_ab_common_K5_pairs":[],"K5_all_common_pairs":[]}
        common_targets=set()
        for pair in pairs:
            endpoints=[entries[pair[f"{side}_sample_id"]] for side in ("correct","error")]
            statuses={k:all(x["values"][k] is not None for x in endpoints) for k in ("K2_ab","K2_cd","K5_all")}
            if statuses["K5_all"]:
                check(statuses["K2_ab"] and statuses["K2_cd"],"Complete K5 must have both fixed K2")
                common_targets|={x["sample_id"] for x in endpoints}
            for key,source,gate in (("K2_ab_all_eligible_pairs","K2_ab","K2_ab"),
                                    ("K2_ab_common_K5_pairs","K2_ab","K5_all"),
                                    ("K5_all_common_pairs","K5_all","K5_all")):
                if statuses[gate]:
                    c,e=(x["values"][source] for x in endpoints)
                    info=dict(scope=scope, subset=key, pair_id=pair["pair_id"], dialogue_id=pair["dialogue_id"])
                    diffs={m:e[m]-c[m] for m in METRICS}
                    for m in METRICS:
                        risk=-diffs[m] if m=="D_C" else diffs[m]
                        info.update({f"{m}_correct":c[m],f"{m}_error":e[m],f"{m}_diff":diffs[m],
                                     f"{m}_rank":float(risk>0)+.5*float(risk==0)})
                    conf_diff=float(pair["correct_confidence"])-float(pair["error_confidence"])
                    info["confidence_rank"]=float(conf_diff>0)+.5*float(conf_diff==0)
                    pool[key].append(info)
            pair_records.append(dict(scope=scope,pair_id=pair["pair_id"],dialogue_id=pair["dialogue_id"],
                                     K2_ab_eligible=statuses["K2_ab"],K2_cd_eligible=statuses["K2_cd"],
                                     K5_complete=statuses["K5_all"]))
        for name, group in pool.items():
            for metric in METRICS:
                mean,lo,hi,n=cluster_vector(group,f"{metric}_diff",universe,draws)
                rank=cluster_vector(group,f"{metric}_rank",universe,draws)[0]
                clean=cluster_vector(group,"confidence_rank",universe,draws)[0]
                condition_rows.append(dict(scope=scope,subset=name,metric=metric,pairs=len(group),
                    dialogues=len({r["dialogue_id"] for r in group}),error_minus_correct=mean,
                    ci95_low=lo,ci95_high=hi,bootstrap_valid=n,
                    metric_pair_rank=rank,clean_confidence_pair_rank=clean,
                    risk_direction="LOWER_D_C" if metric=="D_C" else "HIGHER_METRIC"))
        old={r["pair_id"]:r for r in pool["K2_ab_common_K5_pairs"]}
        changes=[{**r,**{m:r[f"{m}_diff"]-old[r["pair_id"]][f"{m}_diff"] for m in METRICS}}
                 for r in pool["K5_all_common_pairs"]]
        for m in METRICS:
            mean,lo,hi,n=cluster_vector(changes,m,universe,draws)
            condition_rows.append(dict(scope=scope,subset="K5_minus_K2_same_pair_difference",metric=m,
                pairs=len(changes),dialogues=len({r["dialogue_id"] for r in changes}),error_minus_correct=mean,
                ci95_low=lo,ci95_high=hi,bootstrap_valid=n))
        for ref, alternative, label in (("K2_ab","K2_cd","DISJOINT_K2"),
                                        ("K2_ab","K5_all","NESTED_K2_K5")):
            ids=[sid for sid in sorted(entries) if sid in common_targets]
            for metric in METRICS:
                a=np.asarray([entries[sid]["values"][ref][metric] for sid in ids])
                b=np.asarray([entries[sid]["values"][alternative][metric] for sid in ids])
                both=[{"dialogue_id":entries[sid]["dialogue_id"],"abs_diff":abs(float(x-y))}
                       for sid,x,y in zip(ids,a,b)]
                mean,lo,hi,n=cluster_vector(both,"abs_diff",universe,draws)
                agreement_rows.append(dict(scope=scope,comparison=label,metric=metric,targets=len(ids),
                    dialogues=len({r["dialogue_id"] for r in both}),MAE=mean,MAE_ci95_low=lo,MAE_ci95_high=hi,
                    bootstrap_valid=n,pearson=correlation(a,b),spearman=correlation(a,b,rank=True),
                    overlap_versions=0 if label=="DISJOINT_K2" else 2))
        for k in (2,3,4):
            groups=defaultdict(list)
            for r in subset_values:
                if r["scope"]==scope and r["k"]==k and r["sample_id"] in common_targets:
                    groups[r["sample_id"]].append(r["D_pred"])
            stats=[{"dialogue_id":entries[sid]["dialogue_id"],"spread":float(np.std(vals,ddof=0))}
                   for sid,vals in groups.items()]
            mean,lo,hi,n=cluster_vector(stats,"spread",universe,draws)
            dispersion_rows.append(dict(scope=scope,k=k,targets=len(groups),
                                        combinations_per_target={2:10,3:10,4:5}[k],
                mean_within_Utterance_conditional_SD=mean,ci95_low=lo,ci95_high=hi,bootstrap_valid=n,
                interpretation="Subsets share five fixed versions; not independent replicates or validation"))
    return pair_records, condition_rows, agreement_rows, dispersion_rows


def main():
    protocol=json.loads((RUN/"configs/EXPLORATORY_PROTOCOL.json").read_text())
    def check_sources():
        for path, expected in protocol["frozen_input_sha256"].items():
            check(sha(RUN.parents[3]/path)==expected,f"Frozen prior asset changed: {path}")
        for rel,h in json.loads((RUN/"provenance/cohort_lock.json").read_text()).items():
            check(sha(RUN/rel)==h,f"Matched cohort changed: {rel}")
    check_sources()
    records=read(RUN/"results/perturbed_predictions.csv")
    controls=read(RUN/"results/clean_controls.csv")
    pairs=read(RUN/"results/matched_pair_manifest.csv")
    samples={r["sample_id"]:r for r in read(RUN/"results/matched_sample_manifest.csv")}
    check(len(samples)==48 and len(controls)==48 and len(pairs)==24 and len(records)==240,"Frozen cardinality")
    check(all(all(truth(r[k]) for k in ("clean_feature_allclose","clean_probability_allclose",
              "original_probability_allclose","clean_prediction_unchanged")) for r in controls),"Clean controls not all passed")
    lookup=defaultdict(dict)
    for r in records:
        sid, v=r["sample_id"],r["variant_id"]
        check(r["perturbation_type"]=="semantic_paraphrase" and v in VARIANTS and v not in lookup[sid],"Unexpected variant")
        sample=samples[sid]
        check(all(r[k]==sample[k] for k in ("dialogue_id","original_probabilities","pair_id","original_correct")),"Sample fields changed")
        if available(r,SCOPES[0]):
            check(r["label_review_status"]=="PASS", "AI gate inconsistent")
        lookup[sid][v]=r
    check(set(lookup)==set(samples) and all(set(v)==set(VARIANTS) for v in lookup.values()), "Lost variants")
    # Replayed a/b must use the original private text hashes and closely match the old probabilities.
    old={(r["sample_id"],r["variant_id"]):r for r in read(PARENT/"results/perturbed_predictions.csv")
         if r["sample_id"] in samples and r["perturbation_type"]=="semantic_paraphrase"}
    dists=[]
    max_conf_effect=0
    for sid, versions in lookup.items():
        for v in VARIANTS[:2]:
            before, after=old[sid,v], versions[v]
            check(before["status"]==after["status"]=="SUCCESS" and available(after,SCOPES[0]),"Inherited a/b lost QC")
            check(before["processed_input_sha256"]==after["processed_input_sha256"],"Inherited a/b text changed")
            p,q=[probability(r["perturbed_probabilities"]) for r in (before,after)]
            check(np.allclose(p,q,atol=1e-4,rtol=1e-4) and p.argmax()==q.argmax(),"Inherited a/b probability replay failed")
            dists.append(float(abs(q-p).max()))
            max_conf_effect=max(max_conf_effect,float(abs(q.max()-p.max())))
    controls_hash=sha(RUN/"results/clean_controls.csv")
    pred_hash=sha(RUN/"results/perturbed_predictions.csv")
    universe=sorted({p["dialogue_id"] for p in pairs})
    check(len(universe)==24,"Expected one frozen pair per dialogue")
    draws=np.random.RandomState(SEED).randint(0,len(universe), (BOOTSTRAP,len(universe)))
    targets, subgroups, values=analysis_rows(samples,lookup)
    pair_rows, contrasts, agreement, dispersion=pairwise(pairs,subgroups,values,universe,draws)
    for name,rows in (("target_metrics.csv",targets),("subset_D_pred.csv",values),("pair_eligibility.csv",pair_rows),
        ("paired_comparison.csv",contrasts),("repeatability.csv",agreement),("subset_dispersion.csv",dispersion)):
        if rows:
            write(RUN/"results"/name,rows)
    check_sources()
    check(sha(RUN/"results/clean_controls.csv")==controls_hash and sha(RUN/"results/perturbed_predictions.csv")==pred_hash,
          "Inputs changed while analyzing")
    verification=dict(status="PASS_AI_ONLY_EXPLORATORY",planned_conditions=len(records),targets=len(samples),pairs=len(pairs),
        inference_status=dict(Counter(r["status"] for r in records)),ai_status=dict(Counter(r["label_review_status"] for r in records)),
        new_ai_status=dict(Counter(r["label_review_status"] for r in records if r["variant_id"] in VARIANTS[2:])),
        clean_controls_passed=len(controls),original_version_replays=len(dists),parent_ab_max_probability_abs_difference=max(dists),
        parent_ab_max_top_probability_abs_difference=max_conf_effect,
        full_AI_K5_pairs=sum(r["scope"]==SCOPES[0] and r["K5_complete"] for r in pair_rows),
        full_QC_K5_pairs=sum(r["scope"]==SCOPES[1] and r["K5_complete"] for r in pair_rows),
        common_study_scope=protocol["analysis"]["scope"],bootstrap_dialogues=universe,bootstrap_seed=SEED,
        bootstrap_repetitions=BOOTSTRAP,bootstrap_indices_sha256=sha256(draws.astype("<i8").tobytes()).hexdigest(),
        human_review_used=False,new_score_fit=False,no_training=True,no_Test=True,
        outputs_sha256={f.name:sha(f) for f in sorted((RUN/"results").glob("*.csv"))})
    with (RUN/"verification.json").open("x") as f:
        json.dump(verification,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in verification.items() if k not in {"outputs_sha256","bootstrap_dialogues"}},indent=2))


if __name__=="__main__":
    main()
