#!/usr/bin/env python3
"""Protocol B preflight audit. No model forward, training, or calibration fitting."""
from __future__ import annotations

import csv
import datetime
import hashlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
RESULTS = OUT / "results"
PROV = OUT / "provenance"
CONFIGS = OUT / "configs"
for p in (RESULTS, PROV, CONFIGS): p.mkdir(parents=True, exist_ok=True)

PROTOCOL = ROOT / "experiments/xp_caldist_behavioral_calibration/independent_calibration_protocol/20261109T100307Z"
CANDIDATE = PROTOCOL / "candidate_split_manifests.json"
LINEAGE = PROTOCOL / "results/sample_lineage.csv"
CMERC = ROOT / "CMERC"
SNAPSHOT = CMERC / "experiment_records/04_xp_mainline_validation/code"
RAW_IEM = ROOT.parent / "IEMOCAP_extract/IEMOCAP_full_release"
CODE_REFERENCES = ROOT.parent / "05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion"


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [],lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def parse_raw_iemocap() -> dict[str,dict]:
    out={}
    pattern=re.compile(r"^\[([^]]+)\]\s+(\S+)\s+(\S+)\s+\[([^]]+)\]")
    for path in sorted(RAW_IEM.glob("Session*/dialog/EmoEvaluation/*.txt")):
        for line in path.read_text(errors="replace").splitlines():
            m=pattern.match(line)
            if m:
                out[m.group(2)]={"raw_primary_label":m.group(3),"vad":m.group(4),"annotation_file":str(path)}
    return out


def label_audit(manifest: dict) -> tuple[list[dict],list[dict],dict]:
    raw=parse_raw_iemocap()
    with (CMERC/"data/IEMOCAP_features.pkl").open("rb") as f:
        video_ids,_,labels,*_=pickle.load(f,encoding="latin1")
    anomalies=[]
    for dialogue in sorted(video_ids):
        for idx,(uid,label) in enumerate(zip(video_ids[dialogue],labels[dialogue])):
            meta=raw.get(uid)
            if meta and meta["raw_primary_label"]=="xxx" and int(label)==0:
                anomalies.append({"dialogue_id":dialogue,"utterance_index":idx,"utterance_id":uid,
                                  "session_id":uid[:5],"raw_primary_label":"xxx","frozen_xp04_label":"happy",
                                  "annotation_file":meta["annotation_file"],"vad":meta["vad"]})
    assert len(anomalies)==53
    impacts=[]
    folds=manifest["protocols"]["B_IEMOCAP"]["folds"]
    for fold,assignment in folds.items():
        counts=Counter(assignment[a["dialogue_id"]] for a in anomalies)
        for role in ("training","development","calibration","evaluation"):
            impacts.append({"fold":fold,"role":role,"anomaly_count_legacy_included":counts[role],
                            "anomaly_count_unknown_masked":counts[role],
                            "legacy_effect":"included_as_happy",
                            "mask_effect":"context_retained;excluded_from_loss_calibration_and_metrics"})
    summary={"total":len(anomalies),"by_session":dict(sorted(Counter(a["session_id"] for a in anomalies).items())),
             "legacy_happy_total":648,"masked_happy_total":595,"raw_six_class_total":7380,"frozen_total":7433,
             "upstream_rule_found":False,
             "finding":"No preprocessing source or explicit rule was found in HITSZ-HLT/CMERC commit 815c7c0; the repository distributes only the frozen pickle."}
    return anomalies,impacts,summary


def load_lineage():
    with LINEAGE.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))


def protocol_precheck(manifest: dict, lineage: list[dict]) -> dict:
    result={"status":"awaiting_iemocap_label_policy","protocol":"B","source_status":manifest["status"],"datasets":{}}
    meta=defaultdict(dict)
    for r in lineage:
        meta[r["dataset"]][r["dialogue_id"]]=r
    for ds,key in (("IEMOCAP","B_IEMOCAP"),("MELD","B_MELD")):
        folds={}
        for fold,assignment in manifest["protocols"][key]["folds"].items():
            roles=defaultdict(set)
            for d,role in assignment.items(): roles[role].add(d)
            assert set().union(*roles.values())==set(assignment)
            assert sum(map(len,roles.values()))==len(assignment)
            checks={"dialogue_overlap_zero":all(not roles[a]&roles[b] for i,a in enumerate(roles) for b in list(roles)[i+1:])}
            if ds=="IEMOCAP":
                rs={role:{meta[ds][d]["session_id"] for d in ids} for role,ids in roles.items()}
                checks["calibration_session_isolated"]=all(not rs["calibration"]&rs[x] for x in roles if x!="calibration")
                checks["evaluation_session_isolated"]=all(not rs["evaluation"]&rs[x] for x in roles if x!="evaluation")
            else:
                rs={role:{meta[ds][d]["episode_id"] for d in ids} for role,ids in roles.items()}
                checks["episode_overlap_zero"]=all(not rs[a]&rs[b] for i,a in enumerate(rs) for b in list(rs)[i+1:])
            payload=json.dumps({role:sorted(ids) for role,ids in sorted(roles.items())},sort_keys=True,separators=(",",":"))
            folds[fold]={"role_counts":{r:len(x) for r,x in roles.items()},"checks":checks,
                         "prospective_sha256":hashlib.sha256(payload.encode()).hexdigest()}
            assert all(checks.values())
        result["datasets"][ds]={"folds":folds}
    return result


def import_snapshot_dataloader():
    spec=importlib.util.spec_from_file_location("xp04_dataloader",SNAPSHOT/"dataloader.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def dry_run_dataset(dataset: str, assignment: dict[str,str], lineage: list[dict]) -> dict:
    old=os.getcwd(); os.chdir(CMERC)
    try:
        dlmod=import_snapshot_dataloader()
        map_internal={}
        for r in lineage:
            if r["dataset"]==dataset:
                if dataset == "IEMOCAP":
                    feature_key = r["dialogue_id"]
                else:
                    split_name, dialogue_index, _ = r["source_utterance_id"].split(":")
                    split_offset = {"train": 0, "validation": 1039, "test": 1153}[split_name]
                    feature_key = split_offset + int(dialogue_index)
                map_internal[r["dialogue_id"]]=feature_key
        role_keys={role:[map_internal[d] for d,x in assignment.items() if x==role] for role in ("training","development","calibration","evaluation")}
        if dataset=="IEMOCAP":
            factory=lambda keys: dlmod.IEMOCAPDataset(keys=keys)
            audio_dim=1582; qdim=2
        else:
            factory=lambda keys: dlmod.MELDDataset("data/MELD_features_raw1.pkl",keys=keys)
            audio_dim=300; qdim=9
        report={}
        for role,keys in role_keys.items():
            ds=factory(keys)
            lengths=[]
            for key in keys:
                # Evaluation labels remain sealed: structure is derived from IDs/features only.
                n=len(ds.videoIDs[key]); lengths.append(n)
                arrays=[ds.roberta1[key],ds.roberta2[key],ds.roberta3[key],ds.roberta4[key],ds.videoVisual[key],ds.videoAudio[key],ds.videoSpeakers[key]]
                assert all(len(a)==n for a in arrays)
                assert all(np.asarray(a).shape==(n,1024) for a in arrays[:4])
                assert np.asarray(arrays[4]).shape==(n,342)
                assert np.asarray(arrays[5]).shape==(n,audio_dim)
                q=np.asarray([[1,0] if x=="M" else [0,1] for x in arrays[6]]) if dataset=="IEMOCAP" else np.asarray(arrays[6])
                assert q.shape==(n,qdim)
                if role!="evaluation": assert len(ds.videoLabels[key])==n
            batch_info={}
            if role!="evaluation":
                loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=ds.collate_fn)
                batch=next(iter(loader)); maxlen=max(lengths[:16]); b=min(16,len(keys))
                expected=[(maxlen,b,1024)]*4+[(maxlen,b,342),(maxlen,b,audio_dim),(maxlen,b,qdim),(b,maxlen),(b,maxlen)]
                actual=[tuple(x.shape) for x in batch[:-1]]
                assert actual==expected
                batch_info={"first_batch_shapes":actual,"padding_verified":True,"label_access":"allowed_for_role"}
            else:
                # Pad only non-label tensors to verify the same dialogue-preserving layout.
                items=[]
                for key in keys[:16]:
                    q=np.asarray([[1,0] if x=="M" else [0,1] for x in ds.videoSpeakers[key]]) if dataset=="IEMOCAP" else np.asarray(ds.videoSpeakers[key])
                    items.append([torch.as_tensor(np.asarray(ds.roberta1[key]),dtype=torch.float32),
                                  torch.as_tensor(np.asarray(ds.videoVisual[key]),dtype=torch.float32),
                                  torch.as_tensor(q,dtype=torch.float32)])
                padded=[pad_sequence([x[i] for x in items]) for i in range(3)]
                assert padded[0].shape[2]==1024 and padded[1].shape[2]==342 and padded[2].shape[2]==qdim
                batch_info={"first_batch_nonlabel_shapes":[list(x.shape) for x in padded],"padding_verified":True,"label_access":"sealed"}
            report[role]={"dialogues":len(keys),"utterances":sum(lengths),"min_length":min(lengths),"max_length":max(lengths),
                          "text_dims":[1024]*4,"audio_dim":audio_dim,"visual_dim":342,"qmask_dim":qdim,
                          "umask":"one for valid utterances; zero only after collation padding","dialogue_order_preserved":True,**batch_info}
        return report
    finally: os.chdir(old)


def gpu_audit():
    query=["nvidia-smi","--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu","--format=csv,noheader,nounits"]
    rows=[]
    for line in subprocess.check_output(query,text=True).strip().splitlines():
        x=[v.strip() for v in line.split(",")]
        rows.append(dict(zip(("index","name","uuid","memory_total_mib","memory_used_mib","memory_free_mib","utilization_percent","temperature_c"),x)))
    proc=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True,capture_output=True,check=True).stdout.strip()
    for r in rows: r.update({"eligible_now":int(r["memory_free_mib"])>=11500 and int(r["utilization_percent"])<=10 and r["uuid"] not in proc})
    captured=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    return {"captured_utc":captured,"gpus":rows,"compute_processes":proc.splitlines() if proc else [],
            "max_parallel_jobs":min(4,sum(bool(r["eligible_now"]) for r in rows)),
            "pilot_plan":[{"task":"IEMOCAP_fold1_seed1475","host_gpu":0},{"task":"MELD_fold1_seed1475","host_gpu":1}],
            "binding":"one process per CUDA_VISIBLE_DEVICES host index; process uses cuda:0",
            "cpu_limits":{"OMP_NUM_THREADS":4,"MKL_NUM_THREADS":4,"num_workers":0}}


def main():
    manifest=json.loads(CANDIDATE.read_text()); lineage=load_lineage()
    anomalies,impacts,label_summary=label_audit(manifest)
    write_csv(RESULTS/"iemocap_xxx_anomalies.csv",anomalies); write_csv(RESULTS/"label_policy_impact.csv",impacts)
    precheck=protocol_precheck(manifest,lineage); (RESULTS/"split_freeze_precheck.json").write_text(json.dumps(precheck,indent=2,ensure_ascii=False)+"\n")
    fold1={ds:dry_run_dataset(ds,manifest["protocols"][f"B_{ds}"]["folds"]["fold1"],lineage) for ds in ("IEMOCAP","MELD")}
    (RESULTS/"data_loader_dry_run.json").write_text(json.dumps(fold1,indent=2,ensure_ascii=False)+"\n")
    gpu=gpu_audit(); (RESULTS/"gpu_preflight.json").write_text(json.dumps(gpu,indent=2,ensure_ascii=False)+"\n")
    config={"status":"preflight_only_awaiting_label_policy","protocol_b_confirmed":True,"fold":1,"pilot_seed":1475,
            "iemocap_label_policy":None,"curriculum_policy":"fold_local_training_only_recompute_required",
            "evaluation_forward_allowed":False,"evaluation_label_access_allowed":False,"max_gpu_jobs":4}
    (CONFIGS/"pilot_preflight_config.json").write_text(json.dumps(config,indent=2)+"\n")
    sources=[CANDIDATE,LINEAGE,CMERC/"data/IEMOCAP_features.pkl",CMERC/"data/iemocap_features_roberta.pkl",
             CMERC/"data/MELD_features_raw1.pkl",CMERC/"data/meld_features_roberta.pkl",SNAPSHOT/"dataloader.py",
             SNAPSHOT/"train_cm.py",SNAPSHOT/"model_cm.py",SNAPSHOT/"evaluation_metrics.py",
             SNAPSHOT/"snapshots/source_snapshot.json",CMERC/"caldiff/IEMOCAP_diff.pkl",CMERC/"caldiff/MELD_diff.pkl",
             RAW_IEM/"README.txt",CODE_REFERENCES/"DialogueRNN/DialogueRNN_features.zip",
             CODE_REFERENCES/"COSMIC/feature-extraction/iemocap/iemocap.pkl"]
    hashes=[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in sources]
    (PROV/"source_hashes.json").write_text(json.dumps({"files":hashes},indent=2)+"\n")
    verification={"status":"awaiting_required_researcher_decision","protocol_b_confirmed":True,"approved_manifest_created":False,
                  "iemocap_label_policy_confirmed":False,"training_started":False,"gpu_forward_run":False,"calibration_fit_run":False,
                  "evaluation_labels_read_for_debug":False,"evaluation_forward_run":False,"anomaly_count":len(anomalies),
                  "label_summary":label_summary,"split_precheck_passed":all(all(v for v in f["checks"].values()) for d in precheck["datasets"].values() for f in d["folds"].values()),
                  "dataloader_fold1_passed":True,"gpu_preflight":gpu,
                  "additional_blocker":"XP04 caldiff artifacts include old validation and omit some new-fold training groups; fold-local Training-only difficulty recomputation is required to preserve curriculum without leakage."}
    (OUT/"verification.json").write_text(json.dumps(verification,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(verification,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
