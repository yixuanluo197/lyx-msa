#!/usr/bin/env python3
"""Select attempt coverage before any scores and create input-only private shards."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import torch

RUN=Path(__file__).resolve().parents[1]


def h(data):
    return sha256(data).hexdigest()


def rank(domain, value):
    return h(f"20261122|real_score|{domain}|{value}".encode())


def main():
    path=RUN/"local_only/score_data.pt"
    lock=json.loads((RUN/"provenance/data_lock.json").read_text())
    assert h(path.read_bytes())==lock["output_hashes"]["local_only/score_data.pt"]
    data=torch.load(path,map_location="cpu",weights_only=True)
    roster=[]; blind=[]
    for role in ("score_fit","score_eval"):
        index={str(r["dialogue_id"]):r for r in data["splits"][role]}
        assert len(index)==155
        dids=sorted(index,key=lambda d:(rank(role,d),d))[:96]
        for did in dids:
            row=index[did]
            uids=list(map(str,row["utterance_ids"]))
            idx=min(range(len(uids)),key=lambda i:(rank("target",f"{did}:{uids[i]}"),i))
            uid=uids[idx]
            sid=f"MELD:train:{did}:{uid}"
            blindid=rank("blind",sid)[:24]
            roster.append(dict(blind_id=blindid,sample_id=sid,role=role,dialogue_id=did,utterance_index=idx,utterance_id=uid))
            speaker=row["speaker"].argmax(-1).tolist()
            blind.append(dict(blind_id=blindid,original_text=row["texts"][idx],target_index=idx,
                context=[dict(turn=i,speaker=int(speaker[i]),text=t) for i,t in enumerate(row["texts"])]))
    assert len(roster)==len(blind)==192 and len({r["dialogue_id"] for r in roster})==192
    with (RUN/"local_only/perturbation_attempt_manifest.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(roster[0]));w.writeheader();w.writerows(roster)
    blind.sort(key=lambda r:(rank("shard",r["blind_id"]),r["blind_id"]))
    sources={}
    for i in range(6):
        p=RUN/f"local_only/parts/source_{i}.jsonl"
        with p.open("x",encoding="utf-8") as f:
            for entry in blind[i*32:(i+1)*32]:
                f.write(json.dumps(entry,ensure_ascii=True)+"\n")
        sources[str(p.relative_to(RUN))]=h(p.read_bytes())
    p=RUN/"local_only/perturbation_attempt_manifest.csv"
    with (RUN/"provenance/ai_attempt_roster_lock.json").open("x") as f:
        json.dump(dict(score_fit_attempts=96,score_eval_attempts=96,versions_per_target=2,
                  no_model_or_labels_consumed=True,no_replacement=True,
                  hashes={str(p.relative_to(RUN)):h(p.read_bytes()),**sources}),f,indent=2);f.write("\n")
    print("Frozen 192 of 2976 targets, one per selected Dialogue. Six blinded shards of 32.")
if __name__=="__main__":
    main()
