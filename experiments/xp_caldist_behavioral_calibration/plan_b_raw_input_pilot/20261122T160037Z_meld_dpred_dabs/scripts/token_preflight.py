"""Audit the frozen tokenizer on the new reviewed roster without weight loading."""
from collections import Counter, defaultdict
import importlib.util
import json

import torch
from transformers import RobertaTokenizer

from freeze_pilot import BASE, PARENT, PRIVATE, PROV, RUN, digest, dump, jsonl_read, load


def main():
    source=load(PROV/"SOURCE_LOCK.json")
    lock=load(PROV/"REVIEW_LOCK.json")
    for name in ("source_cache.pt","source_data.pt"):
        assert digest(PRIVATE/name)==source["exports"]["local_only/"+name]
    reviewed=PRIVATE/"reviewed.jsonl"
    assert digest(reviewed)==lock["reviewed_sha256"]
    adapter=BASE/"20261120T143131Z/scripts/text_encoder.py"
    assert digest(adapter)==source["sources"][str(adapter)]
    spec=importlib.util.spec_from_file_location("frozen_text_encoder",adapter)
    text=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text)
    tokenizer=RobertaTokenizer.from_pretrained(PARENT/"local_only/text_model/best_classifier",local_files_only=True)
    data=torch.load(PRIVATE/"source_data.pt",map_location="cpu",weights_only=True)
    cache=torch.load(PRIVATE/"source_cache.pt",map_location="cpu",weights_only=True)
    originals={f"MELD:train:{d['dialogue_id']}:{uid}":original
               for split in data["splits"].values() for d in split
               for uid,original in zip(d["utterance_ids"],d["texts"])}
    counts=Counter()
    responses=defaultdict(set)
    for row in jsonl_read(reviewed):
        sid,role=row["sample_id"],row["role"]
        if row["review_status"]!="PASS":
            continue
        counts[role+"/review_pass"]+=1
        ids,stats=text.tokenize(tokenizer,[originals[sid],row["processed_text"]],"MELD")
        assert ids[0]==cache["clean_token_ids"][sid]
        if ids[0]!=ids[1] and not any(r["truncated"] for r in stats) and not stats[1]["empty_after_processing"]:
            counts[role+"/token_pass"]+=1
            responses[(sid,role)].add(tuple(ids[1]))
    for (_,role),variants in responses.items():
        counts[role+"/complete_distinct_K2"]+=len(variants)==2
    result=dict(status="TOKEN_PREFLIGHT_PASS",model_weights_loaded=False,scope="historical score_fit only",
                review_sha256=digest(reviewed),counts=dict(counts))
    dump(PROV/"TOKEN_PREFLIGHT.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
