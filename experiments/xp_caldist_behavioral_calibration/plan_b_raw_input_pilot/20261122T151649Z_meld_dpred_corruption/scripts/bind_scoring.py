"""Independently check preserved semantic rows before fitting the pilot scores."""
import json

from audit_inputs import RUN, SCORE, digest, read_csv, read_json


def main():
    results = RUN / "results/text_comparison"
    protocol = RUN / "EXPERIMENT_PROTOCOL.json"
    new = results / "score_inputs.csv"
    old = SCORE / "results/score_inputs.csv"
    inference_path = results / "inference_verification.json"
    lock = RUN / "provenance/CORRUPTION_REVIEW_LOCK.json"
    verification = read_json(inference_path)
    assert verification["status"] == "COMPLETE"
    assert verification["outputs"][new.name] == digest(new)
    assert verification["protocol_sha256"] == digest(protocol)
    before, after = read_csv(old), read_csv(new)
    assert len(before) == len(after) == 2976
    for a, b in zip(before, after):
        assert a == {key: b[key] for key in a}, "Historical clean or semantic output changed"
        assert b["sample_id"].startswith("MELD:train:")
    roster_rows = read_csv(SCORE / "local_only/perturbation_attempt_manifest.csv")
    roster = {row["sample_id"] for row in roster_rows}
    assert len(roster) == 192
    assert all(row["sample_id"] in roster for row in after if row["corruption_d_pred"])
    required = [
        inference_path, results / "corruption_variants.csv", old,
        SCORE / "results/score_transformations.csv", SCORE / "results/score_prediction_verification.json",
        lock, RUN / "provenance/INPUT_AUDIT.json",
    ]
    declaration = {
        "data_kind": "declared_real",
        "declarant": "LYX-MSA MELD orthographic corruption pilot",
        "declaration_note": "Both frozen score roles are from Train. Reused score_eval is exploratory. "
                            "Semantic probabilities are reused byte-for-byte per field. "
                            "Corruption variants are generated deterministically and have context-only AI review.",
        "confidence_definition": "original_max_softmax_probability",
        "outcome_definition": "original_correct_1",
        "split_frozen": True,
        "input_sha256": digest(new),
        "dialogue_manifest_sha256": digest(SCORE / "local_only/SPLIT_DIALOGUES.csv"),
        "utterance_manifest_sha256": digest(SCORE / "local_only/SPLIT_UTTERANCES.csv"),
        "split_lock_sha256": digest(SCORE / "SPLIT_LOCK.json"),
        "protocol_sha256": digest(protocol),
        "source_bindings": {str(path.resolve()): digest(path) for path in required},
        "historical_rows_bit_exact": len(before),
        "human_confirmation": False,
    }
    path = RUN / "provenance/SCORING_INPUT_PROVENANCE.json"
    with path.open("x") as stream:
        json.dump(declaration, stream, indent=2)
    print(f"Preserved all {len(before)} clean/semantic rows and bound corruption scoring inputs.")


if __name__ == "__main__":
    main()
