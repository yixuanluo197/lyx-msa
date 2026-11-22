"""Verify existing experiment artifacts before selecting or running any new edit."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess


RUN = Path(__file__).resolve().parents[1]
BASE = RUN.parent
SCORE = BASE / "20261122T041838Z_meld_real_scoring"
AUDIO = BASE / "20261120T143131Z"


def digest(path):
    with path.open("rb") as stream:
        result = sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    bindings = {}

    def verify(path, expected=None):
        path = path.resolve(strict=True)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Historical artifact hash mismatch: {path.relative_to(BASE)}")
        bindings[str(path.relative_to(BASE))] = actual

    verify(RUN / "EXPERIMENT_PROTOCOL.json")
    protocol = read_json(RUN / "EXPERIMENT_PROTOCOL.json")
    repo = BASE.parents[2] / "CMERC"

    def verify_published(path):
        committed = subprocess.check_output(["git", "-C", str(repo), "show",
            f"{protocol['source_commit']}:{path.relative_to(repo.parent)}"])
        verify(path, sha256(committed).hexdigest())

    check = read_json(SCORE / "verification.json")
    assert check["status"] == "COMPLETE_VERIFIED" and not check["official_validation_used"] and not check["test_used"]
    verify_published(SCORE / "verification.json")
    for rel, expected in check["artifacts"].items():
        verify(SCORE / rel, expected)
    path = SCORE / "results/score_prediction_verification.json"
    predictions = read_json(path)
    assert predictions["status"] == "COMPLETE" and predictions["no_test"] and predictions["no_model_fit"]
    verify(path)
    for file, expected in predictions["inputs"].items():
        verify(Path(file), expected)
    for file, expected in predictions["outputs"].items():
        verify(SCORE / "results" / file, expected)
    verify(SCORE / "local_only/score_text_features.pt", predictions["encoded_features_sha256"])
    verify(SCORE / "local_only/score_text_features.manifest.json", predictions["encoder_manifest_sha256"])
    split = read_json(SCORE / "SPLIT_LOCK.json")
    assert split["source_split"] == "Train" and split["no_test_data"] and split["official_validation_rows_used"] == 0
    for name in ("SPLIT_DIALOGUES.csv", "SPLIT_UTTERANCES.csv"):
        verify(SCORE / "local_only" / name, split["hashes"][name])
    roster = SCORE / "provenance/ai_attempt_roster_lock.json"
    verify(roster)
    for rel, expected in read_json(roster)["hashes"].items():
        verify(SCORE / rel, expected)
    ids = read_csv(SCORE / "local_only/perturbation_attempt_manifest.csv")
    assert len(ids) == len({row["sample_id"] for row in ids}) == len({row["dialogue_id"] for row in ids}) == 192
    assert {row["role"] for row in ids} == {"score_fit", "score_eval"}
    assert all(sum(row["role"] == role for row in ids) == 96 for role in ("score_fit", "score_eval"))
    assert all(row["sample_id"].startswith("MELD:train:") for row in ids)
    manifest = AUDIO / "provenance/FINAL_ARTIFACT_MANIFEST.json"
    verify(manifest)
    audio = read_json(manifest)["code_configs_reports_and_statistics"]
    files = ["results/analysis/feature_response_per_variant.csv", "results/analysis/feature_clean_repeat_distances.csv"]
    for dataset in ("IEMOCAP", "MELD"):
        p = AUDIO / "results/pilot" / dataset
        verify_published(p / "inference_both.json")
        assert read_json(p / "inference_both.json")["test_accessed"] is False
        for name in ("original_predictions.csv", "perturbed_predictions_both.csv",
                     "transformation_manifest_both.csv", "clean_reextraction_controls_both.csv"):
            files.append(str((p / name).relative_to(AUDIO)))
    for rel in files:
        verify(AUDIO / rel, audio[rel])
    scorer = BASE / "20261121T175233Z_reliability_readiness/scripts/score_heldout.py"
    verify(scorer, protocol["scoring"]["scorer_sha256"])
    verify(Path(__file__))
    output = {
        "status": "INPUTS_VERIFIED",
        "scores_existing_train_only_roles": True,
        "score_eval_previously_viewed": True,
        "fresh_independent_evaluation_available": False,
        "audio_sources_saved_analyses_only": True,
        "n_fixed_targets": len(ids),
        "sha256_relative_to_pilot_parent": bindings,
    }
    with (RUN / "provenance/INPUT_AUDIT.json").open("x") as stream:
        json.dump(output, stream, indent=2)
    print(f"Verified {len(bindings)} files; 192 fixed targets; existing score_eval is exploratory.")


if __name__ == "__main__":
    main()
