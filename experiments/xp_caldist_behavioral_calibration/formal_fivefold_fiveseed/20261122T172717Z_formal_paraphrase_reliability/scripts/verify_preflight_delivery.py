"""Verify the stopped audit delivery without constructing Evaluation data."""
from audit_formal_assets import OUT, PROJECT, RUN, csv_rows, digest, load, write_csv, write_json


def main():
    audit=load(OUT/"asset_audit_summary.json")
    protocol=load(RUN/"EXPERIMENT_PROTOCOL.json")
    for name, expected in audit["input_sha256"].items():
        assert digest(PROJECT/name)==expected
    assert protocol["status"]=="STOPPED_AT_PREFLIGHT"
    assert protocol["manifest_sha256"]==audit["approved_manifest_sha256"]
    assert not protocol["formal_inference_authorized_by_this_record"]
    assert not protocol["planned_condition"]["fit_or_infer_completed"] and not protocol["planned_scoring"]["performed"]
    weights=csv_rows(OUT/"checkpoint_integrity.csv")
    assert len(weights)==50 and all(r["sha256_match"]=="True" and r["new_deserialization"]=="False" for r in weights)
    exposure=load(OUT/"exposure/exposure_summary.json")
    for ds in ("IEMOCAP","MELD"):
        assert exposure[ds]["evaluation_union_dialogues"]==audit["evaluation"][ds]["evaluation_dialogues"]
    for name in ("FINAL_DECISION.md","TEXT_ENCODER_COMPATIBILITY_AUDIT.md","HISTORICAL_EXPOSURE_AUDIT.md"):
        assert (RUN/name).is_file()
    files=sorted(p for p in RUN.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p.name not in {"SHA256SUMS.csv","verification.json"})
    manifest=RUN/"SHA256SUMS.csv"
    write_csv(manifest,[dict(path=str(p.relative_to(RUN)),sha256=digest(p),bytes=p.stat().st_size) for p in files])
    write_json(RUN/"verification.json",dict(status="STOPPED_AT_PREFLIGHT",wait_reason="Compatible original frozen text encoder unavailable",
        checkpoint_hash_matches=50,split_metadata_matches=True,independent_confirmation_established=False,
        evaluation_artifact_built=False,standalone_evaluation_or_test_label_files_opened=False,
        item_level_mask_metadata_exposed_in_audit_output=True,mask_metadata_used_for_results=False,
        paraphrase_generated=False,new_inference=False,new_cmerc_training=False,scoring_performed=False,
        files=len(files),sha256_manifest=digest(manifest)))
    print("Verified stopped audit delivery and frozen source hashes")


if __name__=="__main__":
    main()
