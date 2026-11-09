#!/usr/bin/env python3
"""Independent structural verification of the protocol-design delivery."""
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
RES=OUT/"results"

def read_csv(name):
    with (RES/name).open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

required=["README.md","DATA_LINEAGE_AUDIT.md","IEMOCAP_GROUP_SPLIT_ANALYSIS.md",
          "MELD_GROUP_SPLIT_ANALYSIS.md","SPLIT_PROTOCOL_COMPARISON.md",
          "CALIBRATION_EVALUATION_PREREGISTRATION.md","TRAINING_RESOURCE_ESTIMATE.md",
          "HISTORICAL_EXPOSURE_AUDIT.md","FINAL_PROTOCOL_RECOMMENDATION.md",
          "candidate_split_manifests.json","verification.json","provenance/source_hashes.json"]
required_results=["sample_lineage.csv","dialogue_group_inventory.csv","session_inventory.csv",
                  "episode_inventory.csv","candidate_split_counts.csv","candidate_class_coverage.csv",
                  "split_overlap_audit.csv","resource_estimates.csv"]
assert all((OUT/x).is_file() for x in required)
assert all((RES/x).is_file() for x in required_results)

samples=read_csv("sample_lineage.csv")
assert len(samples)==21141
assert Counter(r["dataset"] for r in samples)=={"MELD":13708,"IEMOCAP":7433}
assert len({(r["dataset"],r["source_utterance_id"]) for r in samples})==len(samples)
assert len({r["dialogue_id"] for r in samples if r["dataset"]=="IEMOCAP"})==151
assert len({r["dialogue_id"] for r in samples if r["dataset"]=="MELD"})==1432
assert len({r["session_id"] for r in samples if r["dataset"]=="IEMOCAP"})==5
assert len({r["episode_id"] for r in samples if r["dataset"]=="MELD"})==190

manifest=json.loads((OUT/"candidate_split_manifests.json").read_text())
assert manifest["status"]=="candidate_not_approved"
dialogue_meta={r["dialogue_id"]:r for r in read_csv("dialogue_group_inventory.csv")}
for name,proto in manifest["protocols"].items():
    dataset=name.split("_",1)[1]
    expected={d for d,r in dialogue_meta.items() if r["dataset"]==dataset}
    for fold,assignment in proto["folds"].items():
        assert set(assignment)==expected
        assert set(assignment.values())=={"training","development","calibration","evaluation"}
        roles=defaultdict(set)
        for d,role in assignment.items(): roles[role].add(d)
        for a in roles:
            for b in roles:
                if a<b: assert not roles[a]&roles[b]
        if dataset=="MELD":
            ep_roles=defaultdict(set)
            for d,role in assignment.items(): ep_roles[dialogue_meta[d]["episode_id"]].add(role)
            assert max(map(len,ep_roles.values()))==1
        else:
            session_roles=defaultdict(set)
            for d,role in assignment.items(): session_roles[dialogue_meta[d]["session_id"]].add(role)
            assert all(not ({"calibration","evaluation"}&rr) or len(rr)==1 for rr in session_roles.values())

coverage=read_csv("candidate_class_coverage.csv")
assert all(int(r["support"])>0 and r["missing"]=="False" for r in coverage)
overlaps=read_csv("split_overlap_audit.csv")
assert all(r["passed"]=="True" and int(r["dialogue_overlap_count"])==0 and int(r["group_overlap_count"])==0 for r in overlaps)

hash_manifest=json.loads((OUT/"provenance/source_hashes.json").read_text())
for item in hash_manifest["files"]:
    p=Path(item["path"]); assert p.is_file(); assert p.stat().st_size==item["size_bytes"]; assert digest(p)==item["sha256"]

v=json.loads((OUT/"verification.json").read_text())
assert v["status"]=="passed" and all(v["assertions"].values())
independent={"status":"passed","required_files":len(required)+len(required_results),
             "sample_rows":len(samples),"candidate_protocols":len(manifest["protocols"]),
             "candidate_folds":sum(len(x["folds"]) for x in manifest["protocols"].values()),
             "source_hashes_recomputed":len(hash_manifest["files"]),
             "role_and_group_overlap_checks":"passed","class_coverage_checks":"passed"}
v["independent_verification"]=independent
(OUT/"verification.json").write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(independent,indent=2))
