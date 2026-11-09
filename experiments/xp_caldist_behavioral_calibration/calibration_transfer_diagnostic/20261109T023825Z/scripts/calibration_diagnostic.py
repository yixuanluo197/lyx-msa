#!/usr/bin/env python
"""Validation-only CMERC calibration-transfer diagnostic."""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

DEFAULT_ROOT = Path(__file__).resolve().parent.parent

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def stable_log_softmax(x):
    x=np.asarray(x,np.float64); z=x-x.max(1,keepdims=True)
    return z-np.log(np.exp(z).sum(1,keepdims=True))

def calibrated(logp, temperature):
    lp=stable_log_softmax(np.asarray(logp,np.float64)/float(temperature)); return lp,np.exp(lp)

def nll_log(logp,y): return float(-logp[np.arange(len(y)),y].mean())

def fit_temperature(logp,y,cfg,lamb=0.0):
    lo,hi=np.log(cfg["bounds"])
    def objective(theta): return nll_log(stable_log_softmax(logp/np.exp(theta)),y)+lamb*theta*theta
    res=minimize_scalar(objective,bounds=(lo,hi),method="bounded",options={"xatol":cfg["xatol"],"maxiter":cfg["maxiter"]})
    candidates=[(float(res.x),float(objective(res.x))),(lo,float(objective(lo))),(hi,float(objective(hi))),(0.0,float(objective(0.0)))]
    candidates.sort(key=lambda z:(z[1],abs(z[0]),z[0])); theta,value=candidates[0]
    return {"temperature":float(np.exp(theta)),"log_temperature":theta,"penalized_objective":value,
            "unpenalized_nll":nll_log(stable_log_softmax(logp/np.exp(theta)),y),"lambda":float(lamb),
            "success":bool(res.success),"boundary":bool(abs(theta-lo)<1e-8 or abs(theta-hi)<1e-8),"nfev":int(res.nfev)}

def bins_metrics(p,y,n_bins=15):
    p=np.asarray(p,np.float64); y=np.asarray(y,int); pred=p.argmax(1); conf=p.max(1); correct=pred==y
    idx=np.minimum((conf*n_bins).astype(int),n_bins-1); gaps=[]; weights=[]
    for b in range(n_bins):
        mask=idx==b
        if mask.any(): gaps.append(abs(conf[mask].mean()-correct[mask].mean())); weights.append(mask.mean())
    one=np.eye(p.shape[1])[y]
    return {"n":len(y),"accuracy":float(correct.mean()),"mean_confidence":float(conf.mean()),
            "signed_gap":float(conf.mean()-correct.mean()),"ece15":float(np.dot(gaps,weights)),
            "nll":float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-300)).mean()),
            "brier":float(np.square(p-one).sum(1).mean())}

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load_payload(path,expected):
    with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
    required={"labels","dialogue_ids","utterance_indices","log_probabilities","probabilities","predictions"}
    if not required.issubset(d): raise ValueError(f"missing fields: {path}")
    if len(d["labels"])!=expected["utterances"] or len(set(map(str,d["dialogue_ids"])))!=expected["dialogues"]: raise ValueError("validation size mismatch")
    if not np.array_equal(d["predictions"],d["probabilities"].argmax(1)): raise ValueError("prediction mismatch")
    if not np.allclose(np.exp(d["log_probabilities"]),d["probabilities"],atol=1e-6,rtol=1e-5): raise ValueError("probability mismatch")
    return d

def deterministic_seed(base,*parts): return int.from_bytes(hashlib.sha256("|".join(map(str,(base,*parts))).encode()).digest()[:8],"big")

def inner_group_folds(dialogues,seed,nfold=5):
    values=np.array(sorted(set(map(str,dialogues))))
    rng=np.random.default_rng(seed); rng.shuffle(values)
    return [list(x) for x in np.array_split(values,min(nfold,len(values)))]

def analyze(root):
    cfg=json.loads((root/"configs/diagnostic_config.json").read_text()); results=root/"results"; results.mkdir(parents=True,exist_ok=True)
    all_dialogue=[]; all_lodo_temp=[]; all_oof=[]; all_class=[]; all_size=[]; all_reg_folds=[]; all_reg_compare=[]; all_boot=[]
    audit={"status":"passed","validation_only":True,"test_files_read":False,"inputs":{},"historical_global_ts_root":cfg["historical_global_ts_root"]}
    for dataset in cfg["datasets"]:
      class_names=cfg["classes"][dataset]
      for seed in cfg["seeds"]:
        path=Path(cfg["validation_prediction_template"].format(dataset=dataset,seed=seed)); d=load_payload(path,cfg["expected_validation"][dataset])
        checkpoint=Path(cfg["checkpoint_template"].format(dataset=dataset,seed=seed))
        audit["inputs"][f"{dataset}/seed{seed}"]={"validation_path":str(path),"validation_sha256":file_hash(path),"checkpoint_path":str(checkpoint),"checkpoint_sha256":file_hash(checkpoint)}
        y=np.asarray(d["labels"],int); logp=np.asarray(d["log_probabilities"],np.float64); p=np.asarray(d["probabilities"],np.float64); dialogs=np.asarray(d["dialogue_ids"],str)
        unique=np.array(sorted(set(dialogs))); full=fit_temperature(logp,y,cfg["temperature"],0.0); _,pfull=calibrated(logp,full["temperature"])
        poof=np.empty_like(p); preg=np.empty_like(p); fold_t={}; fold_reg_t={}; fold_lambda={}
        # Outer LODO and nested regularization.
        for outer in unique:
            hold=dialogs==outer; train=~hold
            ordinary=fit_temperature(logp[train],y[train],cfg["temperature"],0.0); fold_t[outer]=ordinary["temperature"]
            _,poof[hold]=calibrated(logp[hold],ordinary["temperature"])
            cv_scores={}
            inner_dialogs=dialogs[train]
            folds=inner_group_folds(inner_dialogs,deterministic_seed(cfg["sample_size"]["base_seed"],dataset,seed,outer,"inner"))
            train_indices=np.flatnonzero(train)
            for lamb in cfg["regularization"]["lambdas"]:
                loss_sum=0.0; count=0
                for val_dialogues in folds:
                    inner_val=np.isin(dialogs,val_dialogues)&train; inner_train=train&~inner_val
                    fit=fit_temperature(logp[inner_train],y[inner_train],cfg["temperature"],lamb)
                    lp,_=calibrated(logp[inner_val],fit["temperature"])
                    loss_sum += -lp[np.arange(inner_val.sum()),y[inner_val]].sum(); count += int(inner_val.sum())
                cv_scores[float(lamb)]=float(loss_sum/count)
            best_score=min(cv_scores.values()); eligible=[l for l,v in cv_scores.items() if v<=best_score+1e-12]; selected=max(eligible)
            regfit=fit_temperature(logp[train],y[train],cfg["temperature"],selected); fold_reg_t[outer]=regfit["temperature"]; fold_lambda[outer]=selected
            _,preg[hold]=calibrated(logp[hold],regfit["temperature"])
            all_lodo_temp.append({"dataset":dataset,"seed":seed,"held_out_dialogue":outer,"held_out_utterances":int(hold.sum()),
                "training_dialogues":len(unique)-1,"training_utterances":int(train.sum()),"full_validation_temperature":full["temperature"],
                "lodo_temperature":ordinary["temperature"],"temperature_influence":ordinary["temperature"]-full["temperature"],
                "heldout_nll_vanilla":bins_metrics(p[hold],y[hold])["nll"],"heldout_nll_lodo":bins_metrics(poof[hold],y[hold])["nll"],
                "heldout_brier_vanilla":bins_metrics(p[hold],y[hold])["brier"],"heldout_brier_lodo":bins_metrics(poof[hold],y[hold])["brier"],
                "heldout_ece15_vanilla":bins_metrics(p[hold],y[hold])["ece15"],"heldout_ece15_lodo":bins_metrics(poof[hold],y[hold])["ece15"]})
            all_reg_folds.append({"dataset":dataset,"seed":seed,"held_out_dialogue":outer,"selected_lambda":selected,
                "regularized_temperature":regfit["temperature"],"ordinary_temperature":ordinary["temperature"],
                **{f"inner_oof_nll_lambda_{str(l).replace('.','p')}":cv_scores[float(l)] for l in cfg["regularization"]["lambdas"]}})
        if not np.array_equal(poof.argmax(1),p.argmax(1)) or not np.array_equal(preg.argmax(1),p.argmax(1)): raise RuntimeError("positive temperature changed argmax")

        # OOF rows and method summaries.
        for i in range(len(y)):
            all_oof.append({"dataset":dataset,"seed":seed,"dialogue_id":dialogs[i],"utterance_index":int(d["utterance_indices"][i]),"ground_truth":int(y[i]),
                "prediction":int(p[i].argmax()),"lodo_temperature":fold_t[dialogs[i]],"regularized_temperature":fold_reg_t[dialogs[i]],"selected_lambda":fold_lambda[dialogs[i]],
                "original_probabilities":json.dumps(p[i].tolist(),separators=(",",":")),"lodo_probabilities":json.dumps(poof[i].tolist(),separators=(",",":")),
                "regularized_probabilities":json.dumps(preg[i].tolist(),separators=(",",":"))})
        methods={"vanilla":p,"full_global_ts_in_sample":pfull,"lodo_global_ts":poof,"regularized_lodo_ts":preg}
        lambda_counts=Counter(fold_lambda.values())
        for name,probs in methods.items():
            m=bins_metrics(probs,y)
            all_reg_compare.append({"dataset":dataset,"seed":seed,"method":name,**m,"full_temperature":full["temperature"],
                "lodo_temperature_mean":float(np.mean(list(fold_t.values()))),"lodo_temperature_std":float(np.std(list(fold_t.values()),ddof=1)),
                "regularized_temperature_mean":float(np.mean(list(fold_reg_t.values()))),"regularized_temperature_std":float(np.std(list(fold_reg_t.values()),ddof=1)),
                "selected_lambda_counts":json.dumps({str(k):lambda_counts[k] for k in sorted(lambda_counts)})})

        # Per-dialogue composition and calibration.
        for dialog in unique:
            mask=dialogs==dialog; pred=p.argmax(1); errors=mask&(pred!=y); true_counts=np.bincount(y[mask],minlength=len(class_names)); pred_counts=np.bincount(pred[mask],minlength=len(class_names))
            composition=json.dumps({class_names[j]:int(true_counts[j]) for j in range(len(class_names))},separators=(",",":")); predcomp=json.dumps({class_names[j]:int(pred_counts[j]) for j in range(len(class_names))},separators=(",",":"))
            errconf=p[errors].max(1) if errors.any() else np.array([])
            for method,probs in methods.items():
                m=bins_metrics(probs[mask],y[mask])
                all_dialogue.append({"dataset":dataset,"seed":seed,"dialogue_id":dialog,"method":method,**m,
                    "true_class_distribution":composition,"predicted_class_distribution":predcomp,"observed_true_classes":int((true_counts>0).sum()),
                    "error_count":int(errors.sum()),"error_confidence_mean":float(errconf.mean()) if len(errconf) else np.nan,
                    "error_confidence_median":float(np.median(errconf)) if len(errconf) else np.nan,"error_confidence_p90":float(np.quantile(errconf,.9)) if len(errconf) else np.nan,
                    "full_validation_temperature":full["temperature"],"lodo_temperature":fold_t[dialog],"regularized_temperature":fold_reg_t[dialog],"selected_lambda":fold_lambda[dialog],
                    "temperature_influence":fold_t[dialog]-full["temperature"]})

        # True-label and predicted-label class calibration, before/after global and OOF scaling.
        pred=p.argmax(1)
        for grouping,groups in (("true_class",y),("predicted_class",pred)):
          for cls,name in enumerate(class_names):
            mask=groups==cls
            for method,probs in methods.items():
                conf=probs[mask].max(1); correct=probs[mask].argmax(1)==y[mask]; one=np.eye(len(class_names))[y[mask]]
                all_class.append({"dataset":dataset,"seed":seed,"grouping":grouping,"class_index":cls,"class_name":name,"method":method,
                    "support":int(mask.sum()),"small_support":bool(mask.sum()<cfg["small_class_support_threshold"]),"accuracy":float(correct.mean()) if mask.any() else np.nan,
                    "mean_confidence":float(conf.mean()) if mask.any() else np.nan,"signed_gap":float(conf.mean()-correct.mean()) if mask.any() else np.nan,
                    "mean_nll":float(-np.log(np.maximum(probs[mask,np.asarray(y[mask],int)],1e-300)).mean()) if mask.any() else np.nan,
                    "mean_brier_contribution":float(np.square(probs[mask]-one).sum(1).mean()) if mask.any() else np.nan})

        # Dialogue-subset sample-size sensitivity.
        for proportion in cfg["sample_size"]["dialogue_proportions"]:
            key=str(proportion); repeats=cfg["sample_size"]["repeats"][key]; count=max(2,int(math.floor(proportion*len(unique)+.5)))
            for repeat in range(repeats):
                if proportion==1.0: chosen=unique.copy()
                else:
                    rng=np.random.default_rng(deterministic_seed(cfg["sample_size"]["base_seed"],dataset,seed,proportion,repeat)); chosen=np.sort(rng.choice(unique,size=count,replace=False))
                mask=np.isin(dialogs,chosen); fit=fit_temperature(logp[mask],y[mask],cfg["temperature"],0.0)
                all_size.append({"dataset":dataset,"seed":seed,"dialogue_proportion":proportion,"repeat":repeat,"dialogue_count":len(chosen),
                    "utterance_count":int(mask.sum()),"temperature":fit["temperature"],"log_temperature":fit["log_temperature"],"boundary":fit["boundary"],
                    "selected_dialogues_sha256":canonical_hash(chosen.tolist())})

        # Fixed-OOF paired dialogue bootstrap, distinct from refit uncertainty.
        rng=np.random.default_rng(deterministic_seed(cfg["bootstrap"]["base_seed"],dataset,seed,"oof-bootstrap")); distributions={(method,metric):[] for method in ("lodo_global_ts","regularized_lodo_ts") for metric in ("nll","brier","ece15")}
        for _ in range(cfg["bootstrap"]["repeats"]):
            sampled=rng.choice(unique,size=len(unique),replace=True); idx=np.concatenate([np.flatnonzero(dialogs==x) for x in sampled]); base=bins_metrics(p[idx],y[idx])
            for method,probs in (("lodo_global_ts",poof),("regularized_lodo_ts",preg)):
                mm=bins_metrics(probs[idx],y[idx])
                for metric in ("nll","brier","ece15"): distributions[(method,metric)].append(mm[metric]-base[metric])
        base=bins_metrics(p,y)
        for (method,metric),values in distributions.items():
            probs=poof if method=="lodo_global_ts" else preg; point=bins_metrics(probs,y)[metric]-base[metric]; values=np.asarray(values)
            all_boot.append({"dataset":dataset,"seed":seed,"method":method,"metric":metric,"difference_definition":"method_minus_vanilla",
                "point_difference":point,"bootstrap_mean_difference":float(values.mean()),"ci95_lower":float(np.quantile(values,.025)),"ci95_upper":float(np.quantile(values,.975)),
                "repeats":len(values),"dialogues":len(unique),"fixed_oof_predictions":True,"temperature_refit_in_bootstrap":False})

    write_csv(results/"per_dialogue_metrics.csv",all_dialogue); write_csv(results/"lodo_temperatures.csv",all_lodo_temp); write_csv(results/"lodo_oof_predictions.csv",all_oof)
    write_csv(results/"classwise_calibration.csv",all_class); write_csv(results/"temperature_sample_size.csv",all_size); write_csv(results/"regularization_outer_folds.csv",all_reg_folds)
    write_csv(results/"regularization_oof_comparison.csv",all_reg_compare); write_csv(results/"paired_bootstrap_oof.csv",all_boot)
    audit["completed_at_utc"]=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"); audit["row_counts"]={p.name:sum(1 for _ in p.open())-1 for p in results.glob("*.csv")}
    (root/"provenance").mkdir(exist_ok=True); (root/"provenance/input_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":"complete","row_counts":audit["row_counts"]},indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=DEFAULT_ROOT); args=ap.parse_args(); analyze(args.output.resolve())
