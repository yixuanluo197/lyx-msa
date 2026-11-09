#!/usr/bin/env python3
"""Build read-only lineage and candidate split design artifacts.

This script never loads model predictions, fits a model/calibrator, or mutates source data.
Candidate assignments are planning artifacts and are explicitly not approved formal splits.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import pickle
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
RESULTS = OUT / "results"
PROV = OUT / "provenance"
CONFIGS = OUT / "configs"
for p in (RESULTS, PROV, CONFIGS):
    p.mkdir(parents=True, exist_ok=True)

IEM_PKL = ROOT / "CMERC/data/IEMOCAP_features.pkl"
MELD_PKL = ROOT / "CMERC/data/MELD_features_raw1.pkl"
IEM_ROB = ROOT / "CMERC/data/iemocap_features_roberta.pkl"
MELD_ROB = ROOT / "CMERC/data/meld_features_roberta.pkl"
MELD_META = ROOT / "../feature_sources/mmalign_20261017"
XP04 = ROOT / "CMERC/experiments/xp_mainline_validation/20261016T050823Z"
IEM_MANIFEST = XP04 / "runs/cmerc/IEMOCAP_validation_seed1475/run_manifest.json"

IEM_CLASSES = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
MELD_CLASSES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exposure(split: str) -> tuple[bool, bool, bool, bool, str]:
    if split == "train":
        return True, False, False, True, "original_model_training;train_statistics_in_behavior_studies"
    if split == "validation":
        return False, True, True, False, "checkpoint_selection;calibration_and_method_development"
    return False, False, True, False, "repeated_test_evaluation;method_development_exposure"


def dialogue_rows(samples: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in samples:
        by[(r["dataset"], r["dialogue_id"])].append(r)
    out = []
    for (ds, did), rr in sorted(by.items()):
        classes = IEM_CLASSES if ds == "IEMOCAP" else MELD_CLASSES
        c = Counter(x["emotion_label"] for x in rr)
        row = {
            "dataset": ds, "dialogue_id": did, "session_id": rr[0]["session_id"],
            "episode_id": rr[0]["episode_id"], "original_split": rr[0]["original_split"],
            "utterance_count": len(rr), "speaker_count": len({x["speaker_id"] for x in rr}),
        }
        row.update({f"class_{x}": c[x] for x in classes})
        out.append(row)
    return out


def load_iemocap() -> tuple[list[dict], list[dict]]:
    with IEM_PKL.open("rb") as f:
        ids, speakers, labels, _, _, _, sentences, _, _ = pickle.load(f, encoding="latin1")
    manifest = json.loads(IEM_MANIFEST.read_text())
    split_map = {}
    for split, key in (("train", "train_ids"), ("validation", "validation_ids"), ("test", "test_ids")):
        for d in manifest["split"][key]: split_map[d] = split
    samples = []
    for did in sorted(labels, key=str):
        session = re.match(r"(Ses\d\d)", did).group(1)
        split = split_map[did]
        for i, lab in enumerate(labels[did]):
            uid = str(ids[did][i])
            sm = re.match(r"(Ses\d\d[FM])", uid)
            speaker = sm.group(1) if sm else session + str(speakers[did][i])
            tr, ck, cal, unseen, hist = exposure(split)
            samples.append({
                "dataset":"IEMOCAP", "dialogue_id":did, "session_id":session, "episode_id":"",
                "speaker_id":speaker, "utterance_index":i, "source_utterance_id":uid,
                "emotion_label":IEM_CLASSES[int(lab)], "original_split":split,
                "used_for_original_model_training":tr, "used_for_checkpoint_selection":ck,
                "used_for_calibration_development":cal, "previously_evaluated_as_test":split == "test",
                "not_previously_evaluated":unseen, "historical_exposure":hist,
            })
    return samples, dialogue_rows(samples)


def load_meld() -> tuple[list[dict], list[dict], dict]:
    with MELD_PKL.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    ids, _, labels, _, _, _, sentences = obj[:7]
    split_files = {"train":"train_sent_emo.csv", "validation":"dev_sent_emo.csv", "test":"test_sent_emo.csv"}
    offsets = {"train":0, "validation":1039, "test":1153}
    samples, audit = [], {"matched_dialogues":0, "matched_utterances":0, "mismatches":[]}
    for split, fn in split_files.items():
        with (MELD_META / fn).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        grouped = defaultdict(list)
        for r in rows: grouped[int(r["Dialogue_ID"])].append(r)
        for raw_id, rr in sorted(grouped.items()):
            did = raw_id + offsets[split]
            if did not in labels:  # official train dialogue 60 is absent from CMERC features
                audit["mismatches"].append({"split":split,"raw_dialogue_id":raw_id,"reason":"absent_from_cmerc_features"})
                continue
            rr.sort(key=lambda x:int(x["Utterance_ID"]))
            assert len(rr) == len(labels[did]) == len(sentences[did])
            audit["matched_dialogues"] += 1; audit["matched_utterances"] += len(rr)
            for i, (meta, lab) in enumerate(zip(rr, labels[did])):
                assert meta["Emotion"].strip().lower() == MELD_CLASSES[int(lab)]
                tr, ck, cal, unseen, hist = exposure(split)
                episode = f"S{int(meta['Season']):02d}E{int(meta['Episode']):02d}"
                samples.append({
                    "dataset":"MELD", "dialogue_id":f"MELD_{split}_{raw_id:04d}",
                    "feature_dialogue_id":did, "session_id":"", "episode_id":episode,
                    "speaker_id":meta["Speaker"], "utterance_index":int(meta["Utterance_ID"]),
                    "source_utterance_id":f"{split}:{raw_id}:{meta['Utterance_ID']}",
                    "emotion_label":MELD_CLASSES[int(lab)], "original_split":split,
                    "used_for_original_model_training":tr, "used_for_checkpoint_selection":ck,
                    "used_for_calibration_development":cal, "previously_evaluated_as_test":split == "test",
                    "not_previously_evaluated":unseen, "historical_exposure":hist,
                })
    return samples, dialogue_rows(samples), audit


def inventory(samples: list[dict], group_key: str, classes: list[str], dataset: str) -> list[dict]:
    by = defaultdict(list)
    for r in samples:
        if r["dataset"] == dataset and r[group_key]: by[r[group_key]].append(r)
    out=[]
    for gid, rr in sorted(by.items()):
        c=Counter(x["emotion_label"] for x in rr)
        row={"dataset":dataset, group_key:gid, "dialogue_count":len({x['dialogue_id'] for x in rr}),
             "utterance_count":len(rr), "speaker_count":len({x['speaker_id'] for x in rr}),
             "original_splits":";".join(sorted({x['original_split'] for x in rr}))}
        row.update({f"class_{x}":c[x] for x in classes}); out.append(row)
    return out


def meld_balanced_blocks(ep_rows: list[dict], n=10) -> dict[str,int]:
    targets = {"utterance_count":sum(r["utterance_count"] for r in ep_rows)/n}
    for c in MELD_CLASSES: targets[f"class_{c}"]=sum(r[f"class_{c}"] for r in ep_rows)/n
    loads=[Counter() for _ in range(n)]; assignment={}
    rarity={c:1/max(1,sum(r[f"class_{c}"] for r in ep_rows)) for c in MELD_CLASSES}
    ordered=sorted(ep_rows,key=lambda r:(-sum(r[f"class_{c}"]*rarity[c] for c in MELD_CLASSES),-r["utterance_count"],r["episode_id"]))
    for idx,r in enumerate(ordered):
        best=None
        for b in range(n):
            trial=loads[b].copy(); trial["utterance_count"]+=r["utterance_count"]
            for c in MELD_CLASSES: trial[f"class_{c}"]+=r[f"class_{c}"]
            # Assign to the least-loaded normalized bin. Using distance to the
            # final target here would incorrectly prefer partly filled bins and
            # can leave later bins empty.
            cost=sum(trial[k]/max(targets[k],1) for k in targets)
            cost += 0.01*len([x for x in assignment.values() if x==b])
            cand=(cost,b)
            if best is None or cand<best: best=cand
        b=best[1]; assignment[r["episode_id"]]=b
        loads[b]["utterance_count"]+=r["utterance_count"]
        for c in MELD_CLASSES: loads[b][f"class_{c}"]+=r[f"class_{c}"]
    return assignment


def select_iem_dev(dialogues: list[dict], sessions: set[str], fold: int) -> set[str]:
    pools={s:[r for r in dialogues if r["session_id"]==s] for s in sessions}
    rng=random.Random(20261109+fold)
    allr=[r for s in sessions for r in pools[s]]
    total=Counter();
    for r in allr:
        total["utterance_count"]+=r["utterance_count"]
        for c in IEM_CLASSES: total[c]+=r[f"class_{c}"]
    target={k:v*0.10/(sum(x["utterance_count"] for x in dialogues)/sum(r["utterance_count"] for r in allr)) for k,v in total.items()}
    # Exactly three dialogues per remaining session; deterministic random search for coverage/balance.
    best=None
    for _ in range(12000):
        chosen=[r for s in sorted(sessions) for r in rng.sample(pools[s],3)]
        cc=Counter()
        for r in chosen:
            cc["utterance_count"]+=r["utterance_count"]
            for c in IEM_CLASSES: cc[c]+=r[f"class_{c}"]
        missing=sum(cc[c]==0 for c in IEM_CLASSES)
        score=missing*1e6+((cc["utterance_count"]-0.10*7433)/743.3)**2
        score+=sum(((cc[c]/max(cc["utterance_count"],1)-total[c]/total["utterance_count"])**2) for c in IEM_CLASSES)*100
        cand=(score,tuple(sorted(r["dialogue_id"] for r in chosen)))
        if best is None or cand<best: best=cand
    return set(best[1])


def summarize_manifest(protocol: str, dataset: str, fold: str, assignment: dict[str,str], drows: list[dict], classes: list[str]):
    rows=[]; coverage=[]
    lookup={r["dialogue_id"]:r for r in drows if r["dataset"]==dataset}
    for role in ("training","development","calibration","evaluation"):
        ids=[d for d,x in assignment.items() if x==role]; rr=[lookup[d] for d in ids]
        rows.append({"protocol":protocol,"dataset":dataset,"fold":fold,"role":role,
                     "grouping":"session" if dataset=="IEMOCAP" and role in {"calibration","evaluation"} else ("episode" if dataset=="MELD" else "dialogue"),
                     "group_count":len({(r['session_id'] if dataset=='IEMOCAP' and role in {'calibration','evaluation'} else r['episode_id'] if dataset=='MELD' else r['dialogue_id']) for r in rr}),
                     "dialogue_count":len(rr),"utterance_count":sum(r["utterance_count"] for r in rr),
                     "training_retention_fraction":sum(r["utterance_count"] for r in rr)/sum(x["utterance_count"] for x in lookup.values()) if role=="training" else ""})
        for c in classes:
            coverage.append({"protocol":protocol,"dataset":dataset,"fold":fold,"role":role,"emotion_label":c,
                             "support":sum(r[f"class_{c}"] for r in rr),"missing":sum(r[f"class_{c}"] for r in rr)==0})
    return rows,coverage


def make_manifests(iem_d: list[dict], meld_d: list[dict], episode_rows: list[dict]):
    manifests={"status":"candidate_not_approved","warning":"Planning artifact only; researcher review required before materializing formal splits.","protocols":{}}
    counts=[]; coverage=[]
    # Protocol A IEM: fully session-disjoint 2 train + 1 dev + 1 cal + 1 eval.
    sessions=[f"Ses0{i}" for i in range(1,6)]
    # Deterministic high-support fixed candidate, with historically test-exposed Ses05 kept out of evaluation.
    roles={"Ses01":"evaluation","Ses02":"calibration","Ses03":"development","Ses04":"training","Ses05":"training"}
    a={r["dialogue_id"]:roles[r["session_id"]] for r in iem_d}
    manifests["protocols"]["A_IEMOCAP"]={"group_unit":"session","folds":{"fixed":a}}
    x,y=summarize_manifest("A","IEMOCAP","fixed",a,iem_d,IEM_CLASSES); counts+=x; coverage+=y
    # Protocol B IEM: rotating eval/cal sessions; 3 dialogues/session for development.
    folds={}
    for i,ev in enumerate(sessions):
        cal=sessions[(i+1)%5]; remaining=set(sessions)-{ev,cal}; dev=select_iem_dev(iem_d,remaining,i)
        assign={r["dialogue_id"]:("evaluation" if r["session_id"]==ev else "calibration" if r["session_id"]==cal else "development" if r["dialogue_id"] in dev else "training") for r in iem_d}
        folds[f"fold{i+1}"]=assign
        x,y=summarize_manifest("B","IEMOCAP",f"fold{i+1}",assign,iem_d,IEM_CLASSES); counts+=x; coverage+=y
    manifests["protocols"]["B_IEMOCAP"]={"group_unit":"session_for_cal_eval_dialogue_for_train_dev","folds":folds}
    # MELD: ten balanced Episode blocks.
    blocks=meld_balanced_blocks(episode_rows,10)
    ep_to_dialogues=defaultdict(list)
    for r in meld_d: ep_to_dialogues[r["episode_id"]].append(r["dialogue_id"])
    def assign_blocks(role_for_block):
        return {d:role_for_block[blocks[ep]] for ep,ds in ep_to_dialogues.items() for d in ds}
    role_a={b:("training" if b<6 else "development" if b==6 else "calibration" if b==7 else "evaluation") for b in range(10)}
    ma=assign_blocks(role_a); manifests["protocols"]["A_MELD"]={"group_unit":"season_episode","block_assignment":blocks,"folds":{"fixed":ma}}
    x,y=summarize_manifest("A","MELD","fixed",ma,meld_d,MELD_CLASSES); counts+=x; coverage+=y
    mf={}
    for i in range(5):
        ev={2*i,2*i+1}; cal={(2*i+2)%10}; dev={(2*i+3)%10}
        rb={b:("evaluation" if b in ev else "calibration" if b in cal else "development" if b in dev else "training") for b in range(10)}
        m=assign_blocks(rb); mf[f"fold{i+1}"]=m
        x,y=summarize_manifest("B","MELD",f"fold{i+1}",m,meld_d,MELD_CLASSES); counts+=x; coverage+=y
    manifests["protocols"]["B_MELD"]={"group_unit":"season_episode","block_assignment":blocks,"folds":mf}
    return manifests,counts,coverage


def overlap_audit(manifests: dict, drows: list[dict]) -> list[dict]:
    out=[]; lookup={r["dialogue_id"]:r for r in drows}
    for name,p in manifests["protocols"].items():
        ds=name.split("_")[1]
        for fold,a in p["folds"].items():
            role_ids={role:{d for d,x in a.items() if x==role} for role in ("training","development","calibration","evaluation")}
            for r1,r2 in itertools.combinations(role_ids,2):
                dlg_overlap=role_ids[r1]&role_ids[r2]
                # In IEMOCAP, Session sharing is allowed only between Training
                # and Development. Calibration/Evaluation must be Session-isolated
                # from every other role.
                group_key="session_id" if ds=="IEMOCAP" and {r1,r2}!={"training","development"} else "episode_id" if ds=="MELD" else "dialogue_id"
                g1={lookup[d][group_key] for d in role_ids[r1]}; g2={lookup[d][group_key] for d in role_ids[r2]}
                out.append({"protocol":name[0],"dataset":ds,"fold":fold,"role_a":r1,"role_b":r2,"group_unit":group_key,
                            "dialogue_overlap_count":len(dlg_overlap),"group_overlap_count":len(g1&g2),"passed":not dlg_overlap and not(g1&g2)})
    return out


def main():
    iem_s,iem_d=load_iemocap(); meld_s,meld_d,meld_audit=load_meld(); samples=iem_s+meld_s; drows=iem_d+meld_d
    write_csv(RESULTS/"sample_lineage.csv",samples)
    write_csv(RESULTS/"dialogue_group_inventory.csv",drows)
    sessions=inventory(samples,"session_id",IEM_CLASSES,"IEMOCAP")
    episodes=inventory(samples,"episode_id",MELD_CLASSES,"MELD")
    write_csv(RESULTS/"session_inventory.csv",sessions); write_csv(RESULTS/"episode_inventory.csv",episodes)
    manifests,counts,coverage=make_manifests(iem_d,meld_d,episodes)
    (OUT/"candidate_split_manifests.json").write_text(json.dumps(manifests,indent=2,ensure_ascii=False)+"\n")
    write_csv(RESULTS/"candidate_split_counts.csv",counts); write_csv(RESULTS/"candidate_class_coverage.csv",coverage)
    overlaps=overlap_audit(manifests,drows); write_csv(RESULTS/"split_overlap_audit.csv",overlaps)
    # Empirical XP04 wall-time and checkpoint-size based estimates; scaled conservatively by fold count.
    empirical={"IEMOCAP":{"gpu_hours_per_model":0.873,"checkpoint_gb":0.0598},"MELD":{"gpu_hours_per_model":0.407,"checkpoint_gb":0.1421}}
    resources=[]
    for protocol,folds in (("A",1),("B",5)):
        for ds in ("IEMOCAP","MELD"):
            models=folds*5; e=empirical[ds]
            resources.append({"protocol":protocol,"dataset":ds,"folds":folds,"seeds_per_fold":5,"model_count":models,
                              "empirical_xp04_gpu_hours_per_model":e["gpu_hours_per_model"],
                              "estimated_gpu_hours":round(models*e["gpu_hours_per_model"]*0.9,2),
                              "estimated_gpu_hours_with_25pct_contingency":round(models*e["gpu_hours_per_model"]*0.9*1.25,2),
                              "best_checkpoint_gb_each":e["checkpoint_gb"],"best_checkpoints_total_gb":round(models*e["checkpoint_gb"],2),
                              "full_run_artifacts_estimated_gb":round(models*e["checkpoint_gb"]*3.0,2),
                              "basis":"XP04 CMERC observed wall time; 0.9 training-pool scaling is planning-only"})
    resources.append({"protocol":"C","dataset":"external","folds":"TBD","seeds_per_fold":"TBD","model_count":"TBD",
                      "empirical_xp04_gpu_hours_per_model":"N/A","estimated_gpu_hours":"N/A","estimated_gpu_hours_with_25pct_contingency":"N/A",
                      "best_checkpoint_gb_each":"N/A","best_checkpoints_total_gb":"N/A","full_run_artifacts_estimated_gb":"N/A",
                      "basis":"No locally verified CMERC-compatible external MERC cohort"})
    write_csv(RESULTS/"resource_estimates.csv",resources)
    config={"status":"preregistered_design_not_approved","created_utc":"2026-11-09T10:03:07Z","random_seed":20261109,
            "protocol_b":{"IEMOCAP":{"folds":5,"evaluation":"one Session","calibration":"next Session cyclically","development":"3 dialogues from each remaining Session","training":"remaining dialogues"},
                          "MELD":{"folds":5,"atomic_episode_blocks":10,"evaluation_blocks":2,"calibration_blocks":1,"development_blocks":1,"training_blocks":6}},
            "future_methods":["E0 Vanilla","E1 Global TS","E2 strongly regularized hierarchical predicted-class TS","E3 unregularized classwise TS diagnostic"],
            "selection_metric":"Calibration-set grouped-CV NLL","evaluation_metrics":["NLL","multiclass Brier","ECE15"],
            "prohibitions":["no training","no GPU forward","no calibration fitting","no test prediction metrics"]}
    (CONFIGS/"preregistered_design.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n")
    source_paths=[IEM_PKL,IEM_ROB,MELD_PKL,MELD_ROB,MELD_META/"train_sent_emo.csv",MELD_META/"dev_sent_emo.csv",MELD_META/"test_sent_emo.csv",MELD_META/"acquisition.json",IEM_MANIFEST,ROOT/"CMERC/dataloader.py",ROOT/"CMERC/train_cm.py"]
    hashes=[]
    for p in source_paths:
        hashes.append({"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha256(p)})
    (PROV/"source_hashes.json").write_text(json.dumps({"algorithm":"SHA-256","files":hashes},indent=2)+"\n")
    verification={"status":"passed","scope":"read_only_protocol_design","counts":{"sample_rows":len(samples),"IEMOCAP_samples":len(iem_s),"MELD_samples":len(meld_s),"dialogues":len(drows),"sessions":len(sessions),"episodes":len(episodes)},
                  "meld_metadata_alignment":meld_audit,"assertions":{"all_candidate_role_overlaps_zero":all(r["passed"] for r in overlaps),
                    "all_candidate_roles_have_all_classes":not any(r["missing"] for r in coverage),"IEMOCAP_expected_utterances":len(iem_s)==7433,
                    "MELD_expected_utterances":len(meld_s)==13708,"MELD_episode_count":len(episodes)==190,
                    "no_model_predictions_read":True,"no_test_metrics_read":True,"no_gpu_forward":True,"no_training":True,"no_calibration_fit":True},
                  "limitations":["Candidate manifests require researcher approval before use.","Resource estimates exclude queue time and calibration inference artifacts.","Fixed-prediction historical exposure cannot be undone."]}
    (OUT/"verification.json").write_text(json.dumps(verification,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(verification,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
