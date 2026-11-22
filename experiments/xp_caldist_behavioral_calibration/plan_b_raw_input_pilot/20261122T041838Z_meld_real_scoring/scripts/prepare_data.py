#!/usr/bin/env python3
"""Bind frozen train-only Dialogue roles and physically separate training/scoring data."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

import torch

RUN=Path(__file__).resolve().parents[1]
PARENT=RUN.parent/"20261120T143131Z"
DESIGN=RUN.parent/"20261121T175233Z_reliability_readiness"


def dg(path):
    h=sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    audio_src=PARENT/"local_only/features/MELD/audio_features.pt"
    dataset_src=PARENT/"local_only/data/MELD/dataset.pt"
    design_lock=DESIGN/"SPLIT_LOCK.json"
    lock=json.loads(design_lock.read_text())
    parent=json.loads((PARENT/"configs/BASELINE_LOCK_MELD.json").read_text())
    old_src={dataset_src:parent["dataset_sha256"],audio_src:parent["audio_features_sha256"]}
    assert all(dg(p)==h for p,h in old_src.items())
    paths={}
    for n in ("SPLIT_DIALOGUES.csv","SPLIT_UTTERANCES.csv"):
        f=DESIGN/"local_only"/n
        assert dg(f)==lock["hashes"][n]
        target=RUN/"local_only"/n
        if not target.exists():
            with target.open("xb") as out, f.open("rb") as s:
                shutil.copyfileobj(s,out)
        assert dg(target)==dg(f)
        paths[n]=target
    dlock=RUN/"SPLIT_LOCK.json"
    if not dlock.exists():
        with dlock.open("xb") as f:
            f.write(design_lock.read_bytes())
    assert dg(dlock)==dg(design_lock)
    data=torch.load(dataset_src,map_location="cpu",weights_only=True)
    # The hash-verified historical frozen feature artifact contains TorchVersion metadata.
    audio=torch.load(audio_src,map_location="cpu",weights_only=False)
    assert data["dataset"]=="MELD" and data["seed"]==67137
    assert set(data["splits"])=={"train","validation"}
    assert set(audio)=={"train","validation","audio_missing_mask","metadata","utterance_metadata"}
    did={str(x["dialogue_id"]):x for x in data["splits"]["train"]}
    locked=rows(paths["SPLIT_DIALOGUES.csv"])
    assert len(did)==len(locked)==1038
    role={r["dialogue_id"]:r["role"] for r in locked}
    assert set(did)==set(role)
    groups={k:sorted(d for d,v in role.items() if v==k) for k in lock["counts"]}
    for k,g in groups.items():
        assert len(g)==lock["counts"][k]["dialogues"]
        assert sum(len(did[id0]["texts"]) for id0 in g)==lock["counts"][k]["utterances"]
    for item in rows(paths["SPLIT_UTTERANCES.csv"]):
        d=item["dialogue_id"]
        i=int(item["utterance_index"])
        assert item["role"]==role[d] and str(did[d]["utterance_ids"][i])==item["utterance_id"]
        assert item["sample_id"]==f'MELD:train:{d}:{item["utterance_id"]}'
    assert all(audio["train"][d].shape[0]==len(did[d]["texts"]) and torch.isfinite(audio["train"][d]).all() for d in did)
    base_ids=sorted(groups["base_fit"]+groups["base_inner_dev"])
    common={k:data[k] for k in ("dataset","seed","class_names")}
    base=dict(common,splits={"train":[did[d] for d in base_ids],"validation":[]},
              fit_dialogue_ids=groups["base_fit"],inner_dev_dialogue_ids=groups["base_inner_dev"])
    scoring=dict(common,splits={k:[did[d] for d in groups[k]] for k in ("score_fit","score_eval")})
    exports={"local_only/base_data/dataset.pt":base,
             "local_only/base_data/features/audio_features.pt":{"train":{d:audio["train"][d] for d in base_ids},"validation":{}},
             "local_only/score_data.pt":scoring,
             "local_only/score_audio_features.pt":{k:{d:audio["train"][d] for d in groups[k]} for k in ("score_fit","score_eval")}}
    for rel,value in exports.items():
        with (RUN/rel).open("xb") as f:torch.save(value,f)
    assert all(dg(p)==h for p,h in old_src.items())
    manifest=dict(status="PASS_ISOLATED_TRAIN_ONLY_SPLITS",fixed_seed=67137,
       official_val_used=False,test_used=False,base_bundle_score_dialogues=0,
       source_hashes={str(x):v for x,v in old_src.items()},split_lock_sha256=dg(dlock),
       output_hashes={k:dg(RUN/k) for k in exports},counts=lock["counts"],
       shared_components="Fixed frozen public Wav2Vec2 and original fixed visual/audio missing masks. No inherited supervised text, M3Net or CMERC weights.")
    with (RUN/"provenance/data_lock.json").open("x") as f:
        json.dump(manifest,f,indent=2);f.write("\n")
    print("PASS: base fit 6313/dev 700; score fit 1569/eval 1407; all groups physically isolated.")

if __name__=="__main__":
    main()
