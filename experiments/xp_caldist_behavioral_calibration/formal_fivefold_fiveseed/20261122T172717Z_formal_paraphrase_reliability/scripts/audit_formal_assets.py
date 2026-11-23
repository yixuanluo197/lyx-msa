"""Audit frozen Protocol B metadata and checkpoint bytes. No evaluation feature or label loader."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import csv
import json
import re
import subprocess


RUN=Path(__file__).resolve().parents[1]
SOURCE=RUN.parent/"20261109T174024Z"
PROJECT=RUN.parents[3]
OUT=RUN/"results"
ROLES=("training","development","calibration","evaluation")
DATASETS=("IEMOCAP","MELD")


def digest(path):
    result=sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")


def metadata_only(manifest):
    projection=r'''
        .protocols | with_entries(
            .value |= {group_unit, folds: (.folds | with_entries(
                .value |= .roles | .value |= with_entries(
                    .value |= {dialogue_ids,group_ids,dialogue_count,utterance_count,supervision_eligible_count,unknown_masked_count,sha256}
                )
            ))}
        )
    '''
    return json.loads(subprocess.check_output(["jq","-c",projection,str(manifest)]))


def audit_splits(metadata,readiness):
    result=[];counts={}
    for ds in DATASETS:
        entry=metadata[ds];folds=entry["folds"]
        assert len(folds)==5
        expected={row["fold"]:row for row in readiness["datasets"][ds]["folds"]}
        heldout=Counter();pool=set();test_aliases=Counter()
        for name,roles in folds.items():
            assert set(roles)==set(ROLES)
            ids={r:set(roles[r]["dialogue_ids"]) for r in ROLES}
            groups={r:set(roles[r]["group_ids"]) for r in ROLES}
            assert all(len(ids[r])==len(roles[r]["dialogue_ids"])==roles[r]["dialogue_count"] for r in ROLES)
            assert all(ids[a].isdisjoint(ids[b]) for a,b in combinations(ROLES,2))
            for role in ("calibration","evaluation"):
                assert all(groups[role].isdisjoint(groups[a]) for a in ROLES if a!=role)
            planned=expected[name]
            assert ids["evaluation"]==set(planned["dialogue_ids"])
            assert groups["evaluation"]==set(planned["group_ids"])
            assert roles["evaluation"]["sha256"]==planned["role_sha256"]
            assert roles["evaluation"]["utterance_count"]==planned["utterance_count"]
            assert roles["evaluation"]["supervision_eligible_count"]==planned["supervision_eligible_count"]
            heldout.update(ids["evaluation"]);pool.update(set().union(*ids.values()))
            for role,spec in roles.items():
                aliases=Counter(re.fullmatch(r"MELD_(train|validation|test)_\d+",key).group(1)
                                for key in ids[role]) if ds=="MELD" else {}
                if role=="evaluation":
                    test_aliases.update(aliases)
                result.append(dict(dataset=ds,fold=name,role=role,group_unit=entry["group_unit"],
                         dialogue_count=spec["dialogue_count"],group_count=len(groups[role]),
                         utterance_count=spec["utterance_count"],eligible_count=spec["supervision_eligible_count"],
                         unknown_masked_count=spec["unknown_masked_count"],role_sha256=spec["sha256"],
                         official_train_alias_count=aliases.get("train"),
                         official_validation_alias_count=aliases.get("validation"),
                         official_test_alias_count=aliases.get("test"),
                         within_fold_dialogue_disjoint=True,heldout_group_disjoint=True))
        assert set(heldout)==pool and all(n==1 for n in heldout.values())
        assert len(heldout)==readiness["datasets"][ds]["evaluation_dialogues"]
        counts[ds]=dict(evaluation_dialogues=len(heldout),exact_once=True,
                        evaluation_original_split_aliases=dict(test_aliases))
    return result,counts


def audit_weights(ready,verification,source_hashes,protocol):
    rows=csv_rows(SOURCE/"results/cmerc_runs.csv")
    assert len(rows)==len({r["task"] for r in rows})==50
    reference={row["task"]:row for row in ready["model_parameters"]}
    reload={row["task"]:row for row in verification["checkpoint_reload"] if row["task"].startswith("cmerc:")}
    combinations_expected={(d,str(f),str(s)) for d in DATASETS for f in range(1,6) for s in protocol["cmerc_seeds"]}
    assert {(r["dataset"],r["fold"],r["seed"]) for r in rows}==combinations_expected
    assert len(reference)==len(reload)==len(rows)

    def check(row):
        earlier=reference[row["task"]]
        expected=row["best_checkpoint_sha256"]
        assert earlier["best_checkpoint_sha256"]==expected and earlier["output"]==row["output"]
        r=reload[row["task"]]
        assert r["strict_reload"] is True and r["forward_executed"] is False
        assert r["model_state_sha256"]==row["model_state_sha256"]
        rel=Path(row["output"])/"best.pt"
        path=PROJECT/rel
        assert source_hashes[str(rel)]==expected
        actual=digest(path) if path.is_file() else None
        return dict(dataset=row["dataset"],fold=row["fold"],seed=row["seed"],checkpoint=str(rel),
                    expected_sha256=expected,computed_sha256=actual,
                    verified_size_bytes=path.stat().st_size if actual is not None else None,
                    sha256_match=actual==expected,prior_cpu_strict_reload=True,new_deserialization=False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(check,rows))


def audit_role_indexes(split_rows):
    lookup={(r["dataset"],r["fold"],r["role"]):r for r in split_rows}
    output=[]
    for row in csv_rows(SOURCE/"results/shared_role_artifacts.csv"):
        artifact_role=row["artifact_role"]
        assert artifact_role in {"teacher_training","training","development","calibration"}
        role="training" if artifact_role=="teacher_training" else artifact_role
        planned=lookup[(row["dataset"],row["fold"],role)]
        assert row["evaluation_artifact"].lower()=="false" and row["exact_manifest_match"].lower()=="true"
        assert int(row["dialogues"])==planned["dialogue_count"] and int(row["utterances"])==planned["utterance_count"]
        assert int(row["supervision_eligible"])==planned["eligible_count"] and int(row["unknown_masked"])==planned["unknown_masked_count"]
        p=PROJECT/row["path"]
        assert p.is_file()
        output.append(dict(dataset=row["dataset"],fold=row["fold"],role=artifact_role,path=row["path"],
                    expected_sha256=row["sha256"],exists=True,previous_exact_manifest_match=True,
                    reopened_or_rehashed=False))
    return output


def main():
    manifest=SOURCE/"configs/approved_split_manifests.json"
    protocol_path=SOURCE/"configs/formal_protocol.json"
    readiness_path=SOURCE/"results/final_verification/evaluation_readiness_manifest.json"
    verification_path=SOURCE/"verification.json"
    hash_path=SOURCE/"provenance/source_hashes.json"
    protocol=load(protocol_path);ready=load(readiness_path);prior=load(verification_path)
    frozen_digest=digest(manifest)
    assert frozen_digest==protocol["manifest"]["sha256"]==ready["manifest_sha256"]==prior["manifest_sha256"]
    assert ready["status"]=="frozen_pending_researcher_approval" and not any(protocol["evaluation"].values())
    assert prior["status"]=="passed" and prior["counts"]=={"teacher":10,"difficulty":10,"cmerc":50,"calibration":50}
    split_rows,counts=audit_splits(metadata_only(manifest),ready)
    indexed_roles=audit_role_indexes(split_rows)
    hashes=audit_weights(ready,prior,load(hash_path),protocol)
    write_csv(OUT/"frozen_split_metadata.csv",split_rows)
    write_csv(OUT/"checkpoint_integrity.csv",hashes)
    write_csv(OUT/"role_artifact_index_audit.csv",indexed_roles)
    fail=[r for r in hashes if not r["sha256_match"]]
    inputs=[manifest,protocol_path,readiness_path,verification_path,hash_path,
            SOURCE/"APPROVED_PROTOCOL.md",SOURCE/"results/cmerc_runs.csv",
            SOURCE/"results/shared_role_artifacts.csv",SOURCE/"scripts/build_role_artifact.py"]
    result=dict(status="INTEGRITY_PASS_AUDIT_ONLY" if not fail else "INTEGRITY_FAILED_AUDIT_ONLY",
        approved_manifest_sha256=frozen_digest,checkpoint_files_checked=len(hashes),
        checkpoint_hash_matches=len(hashes)-len(fail),
        checkpoint_bytes_checked=sum(r["verified_size_bytes"] or 0 for r in hashes),
        per_dataset_checkpoint_hash_matches=dict(Counter(r["dataset"] for r in hashes if r["sha256_match"])),
        non_evaluation_role_indexes_checked=len(indexed_roles),
        role_artifacts_deserialized=False,checkpoint_deserialized=False,
        evaluation_role_artifact_built=False,evaluation_or_test_standalone_label_files_opened=False,
        this_script_metadata_projection_only=True,evaluation=counts,
        input_sha256={str(p.relative_to(PROJECT)):digest(p) for p in inputs})
    write_json(OUT/"asset_audit_summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="input_sha256"}))
    if fail:
        raise SystemExit("Checkpoint missing or changed; stop")


if __name__=="__main__":
    main()
