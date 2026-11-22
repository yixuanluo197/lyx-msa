"""Choose disjoint Train-only roles, lock the typing-noise rule and prepare blind review."""
from collections import Counter, defaultdict
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re


RUN = Path(__file__).resolve().parents[1]
BASE = RUN.parent
PARENT = BASE / "20261122T041838Z_meld_real_scoring"
LAST = BASE / "20261122T151649Z_meld_dpred_corruption"
PRIVATE = RUN / "local_only"
PROV = RUN / "provenance"
ROLES = ("pilot_fit", "pilot_eval")
VARIANTS = ("corruption_a", "corruption_b")
WORD = re.compile(r"(?<![\w'’\u0092\u02bc\u2018\u201b\uff07\u2010-\u2015\u2212-])[A-Za-z]{4,}(?![\w'’\u0092\u02bc\u2018\u201b\uff07\u2010-\u2015\u2212-])")
OLD_WORD = re.compile(r"(?<![\w'’\-])[a-z]{5,}(?![\w'’\-])")
SHARDS = range(5)


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rank(*parts):
    return sha256("|".join(map(str, parts)).encode()).hexdigest()


def text_hash(value):
    return sha256(value.encode("utf-8")).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, obj):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def csv_read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, values):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def jsonl_read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_write(path, values):
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=True) + "\n")


def ranked_candidates(original, blind_id, config):
    """Return two independently ranked one-swap versions or explicit short support."""
    word_list = set(config["allowlist"])
    collisions = set(config.get("collision_words", []))
    sentence_initial = len(original) - len(original.lstrip())
    old = config.get("historical_rule", False)
    candidates = {}
    for match in (OLD_WORD if old else WORD).finditer(original):
        word = match.group()
        proper_case = word.islower() or (
            not old and match.start() == sentence_initial and word[0].isupper() and word[1:].islower()
        )
        if not proper_case or word.lower() not in word_list:
            continue
        for offset in range(1, len(word) - 2):
            position = match.start() + offset
            before = original[position:position+2]
            after = before[::-1]
            changed_word = word[:offset] + after + word[offset+2:]
            if before == after or changed_word.lower() in collisions:
                continue
            changed = original[:position] + after + original[position+2:]
            candidates.setdefault(changed, dict(position=position, before=before, after=after,
                original_word=word, edited_word=changed_word, word_start=match.start(), word_end=match.end()))
    return sorted(candidates.items(), key=lambda pair: (
        rank(config["seed"], blind_id, pair[1]["position"], pair[0]), pair[0]))[:2]


def role_assignment(design):
    path = PARENT / "local_only/SPLIT_DIALOGUES.csv"
    split = load(PARENT / "SPLIT_LOCK.json")
    assert split["hashes"][path.name] == digest(path)
    ids = sorted({row["dialogue_id"] for row in csv_read(path) if row["role"] == design["source_role"]},
                 key=lambda x: (rank(design["split_seed"], "dialogue", x), x))
    assert len(ids) == 155
    n_engineering, n_fit = design["engineering_dialogues"], design["pilot_fit_dialogues"]
    roles = {did: ("engineering" if i < n_engineering else
                  ("pilot_fit" if i < n_engineering+n_fit else "pilot_eval"))
             for i, did in enumerate(ids)}
    assert Counter(roles.values()) == {"engineering":20,"pilot_fit":75,"pilot_eval":60}
    return roles


def target_index(dialogue, seed):
    did = str(dialogue["dialogue_id"])
    utterances = dialogue["utterance_ids"]
    return min(range(len(utterances)), key=lambda i: (rank(seed, "target", did, str(utterances[i])), i))


def verify_source(file, binding):
    evidence = binding["inputs"].get(str(file.resolve()))
    assert evidence and digest(file) == evidence, file
    return evidence


def load_historical(name, binding):
    import torch
    path = PARENT / "local_only" / name
    verify_source(path, binding)
    return torch.load(path, map_location="cpu", weights_only=False)


def develop_rule():
    gate = load(PROV / "INDEPENDENCE_GATE.json")
    assert gate["status"] == "CONFIRMATION_PAUSED" and not gate["independent_evaluation"]
    design = load(RUN / "ENGINEERING_DESIGN.json")
    roles = role_assignment(design)
    csv_write(PRIVATE / "role_assignment.csv", [dict(dialogue_id=k,role=v) for k,v in roles.items()])
    prior = load(PARENT / "results/score_prediction_verification.json")
    bundle = load_historical("score_data.pt", prior)
    engineering = [dialogue for dialogue in bundle["splits"]["score_fit"]
                   if roles[str(dialogue["dialogue_id"])] == "engineering"]
    assert len(engineering) == design["engineering_dialogues"]
    old = load(LAST / "EXPERIMENT_PROTOCOL.json")["text"]["corruption"]["allowlist"]
    allowed = sorted(set(old+design["allowlist_additions"])-set(design["protected_even_if_listed"]))
    observed = {match.lower() for dialogue in engineering for line in dialogue["texts"]
                for match in re.findall(r"[A-Za-z]+", line)}
    collision = sorted(observed | set(allowed+design["protected_even_if_listed"]+design["extra_collision_words"]))
    options = dict(seed=design["candidate_seed"],allowlist=allowed,collision_words=collision)
    old_options = dict(seed=design["candidate_seed"],allowlist=old,historical_rule=True)
    results = []
    for dialogue in engineering:
        selected = target_index(dialogue, design["split_seed"])
        for i, line in enumerate(dialogue["texts"]):
            key = rank("engineering", dialogue["dialogue_id"], i)
            old_n = len(ranked_candidates(line,key,old_options))
            new_n = len(ranked_candidates(line,key,options))
            why = ("HISTORICAL_K2" if old_n == 2 else
                   ("NO_LOWERCASE_WHOLE_WORD_LENGTH_GE5" if not OLD_WORD.search(line) else
                    ("NO_OLD_ALLOWLIST_WORD" if not any(m.group() in old for m in OLD_WORD.finditer(line))
                     else "FEWER_THAN_TWO_INTERNAL_SWAPS")))
            results.append(dict(engineering_row_hash=key,selected=int(i==selected),
                                old_candidates=old_n,new_candidates=new_n,old_limiting_reason=why))
    csv_write(RUN / "results/engineering_candidate_coverage.csv",results)
    def summary(rows):
        return dict(n=len(rows), old_K2=sum(r["old_candidates"]==2 for r in rows),
                    new_K2=sum(r["new_candidates"]==2 for r in rows),
                    old_limit_reasons=dict(Counter(r["old_limiting_reason"] for r in rows)))
    counts = dict(all_engineering_utterances=summary(results),
                  one_per_engineering_dialogue=summary([r for r in results if r["selected"]]))
    dump(PROV / "ENGINEERING_COUNTS.json",counts)
    if counts["all_engineering_utterances"]["new_K2"] <= counts["all_engineering_utterances"]["old_K2"]:
        raise RuntimeError("The prespecified rule failed the engineering coverage check")
    collision_file = PRIVATE / "collision_words.json"
    dump(collision_file,collision)
    protocol = {
        "dataset":"MELD","model_seed":67137,"source_commit":design["source_commit"],
        "source_role":"score_fit","evaluation_scope":"TRAIN_ONLY_EXPLORATORY",
        "historical_score_eval_used":False,"new_independent_evaluation":False,
        "confirmatory_scoring":"PAUSED_NO_UNTOUCHED_DIALOGUES",
        "test_access":False,"official_validation_access":False,
        "role_assignment_sha256":digest(PRIVATE/"role_assignment.csv"),
        "engineering_design_sha256":digest(RUN/"ENGINEERING_DESIGN.json"),
        "engineering_counts_sha256":digest(PROV/"ENGINEERING_COUNTS.json"),
        "cohort":{"engineering_dialogues":20,"pilot_fit_dialogues":75,"pilot_eval_dialogues":60,
                  "target_dialogues":135,"one_target_per_dialogue":True,
                  "target_rule":design["target_rule"],"split_seed":design["split_seed"]},
        "text":{"k":2,"corruption":{"variants":list(VARIANTS),"seed":options["seed"],"allowlist":allowed,
            "collision_words_sha256":digest(collision_file),"operation":design["new_rule"],
            "collision_guard":design["collision_guard"],"selection":"SHA256(seed|blind_id|edit_position|edited_text), then text",
            "generation_source_sha256":digest(Path(__file__)),
            "review":"AI only with full original Dialogue; hidden outcomes, IDs/roles and other judges. PASS iff clear intended wording, negation, emotion, intensity, speaker stance and reference remain; otherwise FAIL or UNCERTAIN. No alternate candidate after review.",
            "no_candidate":"Record unchanged fallback with null edit as procedural FAIL. Do not infer it."}},
        "behavior":{"D_pred":"mean(p_clean(c)-p_k(c))","D_abs":"mean(abs(p_clean(c)-p_k(c)))",
                    "clean_class":"fixed clean argmax c","valid":"Both reviewed PASS examples must have distinct nontruncated changed token sequences and successful frozen forward. Otherwise both features missing."},
        "scoring":{"arms":{"E0":["logit_confidence"],"E1":["logit_confidence","D_pred"],
                          "E2":["logit_confidence","D_abs"],"E3":["logit_confidence","D_pred","D_abs"]},
                   "logistic_l2_C":1,"bootstrap_dialogue_replicates":2000,"bootstrap_seed":design["scorer_seed"],
                   "metrics":["binary_nll","brier","auroc_error","auprc_error"],
                   "primary":"E2 vs E1 Binary NLL, identical K2-eligible fit and eval for all four models",
                   "fallback":"On complete pilot_eval, each enhanced model uses the full-fit E0 prediction exactly when K2 is missing. All-fit E0 versus restricted-fit augmentations is a secondary policy comparison.",
                   "scorer_sha256":"22acb70db16cd1013fbc5b60dccc707d7382653a39cfedbd7d442b51aa8d9049"}
    }
    dump(RUN / "EXPERIMENT_PROTOCOL.json",protocol)
    print(json.dumps(counts))


def export_and_generate():
    import torch
    protocol=load(RUN/"EXPERIMENT_PROTOCOL.json")
    assert protocol["role_assignment_sha256"]==digest(PRIVATE/"role_assignment.csv")
    design=load(RUN/"ENGINEERING_DESIGN.json")
    role_rows=csv_read(PRIVATE/"role_assignment.csv")
    role_map={row["dialogue_id"]:row["role"] for row in role_rows}
    assert role_map==role_assignment(design)
    options=protocol["text"]["corruption"]
    assert digest(PRIVATE/"collision_words.json")==options["collision_words_sha256"]
    assert digest(Path(__file__))==options["generation_source_sha256"]
    options={**options,"collision_words":load(PRIVATE/"collision_words.json")}
    source_report=PARENT/"results/score_prediction_verification.json"
    prior=load(source_report)
    assert prior["status"]=="COMPLETE" and prior["no_test"] and prior["model_state_before"]==prior["model_state_after"]
    data=load_historical("score_data.pt",prior)
    audio=load_historical("score_audio_features.pt",prior)
    cache_file=PARENT/"local_only/score_text_features.pt"
    assert digest(cache_file)==prior["encoded_features_sha256"]
    cache=torch.load(cache_file,map_location="cpu",weights_only=False)
    clean_file=PARENT/"results/score_inputs.csv"
    assert digest(clean_file)==prior["outputs"][clean_file.name]
    clean={row["sample_id"]:row for row in csv_read(clean_file) if row["role"]=="score_fit"}
    assert len(clean)==1569
    dialogue_ids={str(row["dialogue_id"]) for row in data["splits"]["score_fit"]}
    assert dialogue_ids==set(role_map)
    groups={key:[d for d in data["splits"]["score_fit"] if role_map[str(d["dialogue_id"])]==key] for key in ROLES}
    assert [len(groups[k]) for k in ROLES]==[75,60]
    cohort=[]; roster=[]; blind=[]; clean_rows=[]; clean_tokens={}
    for role in ROLES:
        for dialogue in groups[role]:
            did=str(dialogue["dialogue_id"])
            target=target_index(dialogue,design["split_seed"])
            for i,uid0 in enumerate(dialogue["utterance_ids"]):
                uid=str(uid0);sid=f"MELD:train:{did}:{uid}"
                identity=dict(sample_id=sid,role=role,dialogue_id=did,utterance_index=i,utterance_id=uid)
                cohort.append(identity)
                source=clean[sid]
                assert all(str(identity[k])==source[k] for k in ("dialogue_id","utterance_index","utterance_id"))
                kept=("original_prediction","original_correct","original_confidence","clean_margin","clean_entropy",
                      "original_label","original_probabilities")
                clean_rows.append({**identity,"original_role":"score_fit",**{k:source[k] for k in kept}})
                if i!=target:
                    continue
                blind_id=rank(design["candidate_seed"],"blind",sid)[:24]
                roster.append({**identity,"blind_id":blind_id})
                clean_tokens[sid]=cache["clean_token_ids"][sid]
                original=dialogue["texts"][i]
                offered=ranked_candidates(original,blind_id,options)
                candidates={}
                for v,variant in enumerate(VARIANTS):
                    if v<len(offered):
                        processed,edit=offered[v]
                        status="GENERATED"
                    else:
                        processed,edit=original,None
                        status="NO_ELIGIBLE_EDIT" if not offered else "NO_SECOND_DISTINCT_EDIT"
                    candidates[variant]=dict(text=processed,edit=edit,generation_status=status)
                context=[dict(turn=t,speaker=int(s),text=x) for t,(s,x) in enumerate(
                         zip(dialogue["speaker"].argmax(-1).tolist(),dialogue["texts"]))]
                blind.append(dict(blind_id=blind_id,original_text=original,target_index=i,context=context,candidates=candidates))
    assert len(roster)==len({r["dialogue_id"] for r in roster})==135
    assert len(cohort)==len({r["sample_id"] for r in cohort})==len(clean_rows)
    files={"source_data.pt":{**{k:data[k] for k in ("dataset","seed","class_names")},"splits":groups},
           "source_audio.pt":{k:{str(d["dialogue_id"]):audio["score_fit"][str(d["dialogue_id"])] for d in groups[k]} for k in ROLES},
           "source_cache.pt":{"clean":{k:{str(d["dialogue_id"]):cache["clean"]["score_fit"][str(d["dialogue_id"])]
                                          for d in groups[k]} for k in ROLES},
                              "clean_token_ids":clean_tokens,"text_model_sha256":cache["text_model_sha256"]}}
    for name,value in files.items():
        with (PRIVATE/name).open("xb") as handle:
            torch.save(value,handle)
    for name,value in (("source_clean.csv",clean_rows),("pilot_cohort.csv",cohort),("target_manifest.csv",roster)):
        csv_write(PRIVATE/name,value)
    selected_paths=[PRIVATE/name for name in (*files,"source_clean.csv","pilot_cohort.csv","target_manifest.csv")]
    paths={str(path.resolve()):value for path,value in [
        (source_report,digest(source_report)),(clean_file,prior["outputs"][clean_file.name]),
        (cache_file,prior["encoded_features_sha256"])]}
    for key,value in prior["inputs"].items():
        paths[key]=value
    dump(PROV/"SOURCE_LOCK.json",dict(status="SOURCE_EXPORT_VERIFIED",original_role="score_fit",
         historical_score_eval_used=False,old_score_eval_exported=False,n_targets=135,
         n_dialogues={"pilot_fit":75,"pilot_eval":60},counts=dict(Counter(r["role"] for r in cohort)),
         new_independent_evaluation=False,protocol_sha256=digest(RUN/"EXPERIMENT_PROTOCOL.json"),
         exports={str(p.relative_to(RUN)):digest(p) for p in selected_paths},sources=paths))
    blind.sort(key=lambda row:rank(design["candidate_seed"],"shard",row["blind_id"]))
    folder=PRIVATE/"review"
    folder.mkdir()
    for shard in SHARDS:
        jsonl_write(folder/f"source_{shard}.jsonl",blind[shard::len(SHARDS)])
    generation=Counter(v["generation_status"] for row in blind for v in row["candidates"].values())
    dump(PROV/"GENERATION_LOCK.json",dict(n_targets=135,n_versions=270,
         generation_counts=dict(generation),hashes={str(p.relative_to(RUN)):digest(p) for p in sorted(folder.glob("source_*.jsonl"))}))
    print(json.dumps(dict(pilot_n=dict(Counter(r["role"] for r in cohort)),generation=dict(generation))))


def assemble():
    source_lock=load(PROV/"SOURCE_LOCK.json")
    protocol=RUN/"EXPERIMENT_PROTOCOL.json"
    assert digest(protocol)==source_lock["protocol_sha256"]
    roster_path=PRIVATE/"target_manifest.csv"
    assert digest(roster_path)==source_lock["exports"]["local_only/target_manifest.csv"]
    targets={r["blind_id"]:r for r in csv_read(roster_path)}
    generation=load(PROV/"GENERATION_LOCK.json")
    reviewers=load(PRIVATE/"review/assignments.json")
    assert set(reviewers)==set(map(str,SHARDS)) and all(reviewers.values())
    seen=set();output=[];counts=Counter();complete=Counter();review_input={}
    for shard in SHARDS:
        src=PRIVATE/f"review/source_{shard}.jsonl"
        review=PRIVATE/f"review/reviewed_{shard}.jsonl"
        assert digest(src)==generation["hashes"][str(src.relative_to(RUN))]
        review_input[str(review.relative_to(RUN))]=digest(review)
        a,b=jsonl_read(src),jsonl_read(review)
        checked={r["blind_id"]:r["variants"] for r in b}
        assert len(a)==len(b)==len(checked)==27 and {r["blind_id"] for r in a}==set(checked)
        for row in a:
            bid=row["blind_id"];identity=targets[bid]
            assert bid not in seen
            seen.add(bid)
            statuses=[]
            assert set(checked[bid])==set(VARIANTS)
            for version in VARIANTS:
                c=row["candidates"][version];v=checked[bid][version]
                assert v["status"] in {"PASS","FAIL","UNCERTAIN"} and v["notes"].strip()
                assert v["text_sha256"]==text_hash(c["text"])
                if c["generation_status"]!="GENERATED":
                    assert c["edit"] is None and c["text"]==row["original_text"] and v["status"]=="FAIL"
                method="AI_CONTEXT_ORTHOGRAPHIC_REVIEW" if c["edit"] else "NO_CANDIDATE"
                output.append({**identity,"variant_id":version,"processed_text":c["text"],"edit":c["edit"],
                    "generation_status":c["generation_status"],"review_status":v["status"],"review_method":method,
                    "review_notes":v["notes"],"reviewer":reviewers[str(shard)],
                    "original_text_sha256":text_hash(row["original_text"]),"processed_text_sha256":v["text_sha256"]})
                counts["/".join((identity["role"],c["generation_status"],v["status"]))]+=1
                statuses.append(v["status"])
            complete[identity["role"]]+=statuses==["PASS","PASS"]
    assert set(targets)==seen and len(output)==270
    path=PRIVATE/"reviewed.jsonl"
    jsonl_write(path,output)
    dump(PROV/"REVIEW_LOCK.json",dict(status="AI_REVIEW_LOCKED",reviewed_sha256=digest(path),human_review=False,
         n_targets=135,n_versions=270,protocol_sha256=digest(protocol),counts=dict(counts),
         complete_review_pass=dict(complete),review_input_sha256=review_input))
    print(json.dumps(dict(counts=counts,complete=complete)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("develop","export","assemble"))
    {"develop":develop_rule,"export":export_and_generate,"assemble":assemble}[parser.parse_args().action]()
