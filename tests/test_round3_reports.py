import json
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.round3_reports import context_share, new_iemocap_sensitivity, render, sha, validate_configurations
from scripts.round3_statistics import main as statistics_main
from test_round3_statistics import fixture_matrix


def setup_inputs(tmp_path, actual_optimizer=False, dataset="IEMOCAP"):
    matrix_path = fixture_matrix(tmp_path / "inputs", dataset=dataset)
    matrix = json.loads(matrix_path.read_text())
    if actual_optimizer:
        for variant in ("baseline", "cmerc"):
            path = matrix_path.parent / f"{variant}_optimizer.json"
            epochs = 80 if dataset == "IEMOCAP" else 40
            path.write_text(json.dumps({"optimizer_steps_total": 2 * epochs,
                "expected_optimizer_steps_total": 2 * epochs, "optimizer_tensors_finite": True,
                "full_rng_state_present": True, "completed_epochs": epochs}))
            matrix["pairs"][0][f"{variant}_optimizer"] = path.name
        matrix_path.write_text(json.dumps(matrix))
    stats = tmp_path / "statistics"
    statistics_main(["--matrix", str(matrix_path), "--output-dir", str(stats)])
    batch = tmp_path / "batch"
    batch.mkdir()
    protocol = batch / "THIRD_ROUND_PROTOCOL.md"
    protocol.write_bytes(b"# Frozen protocol\n\nNo post-result configuration changes.\n")
    (batch / "PLAN.json").write_text(json.dumps({"commit": "c" * 40, "protocol_sha256": sha(protocol)}))
    return stats, batch, matrix_path


def test_partial_reports_retain_missing_status_and_copy_protocol_exactly(tmp_path):
    stats, batch, matrix = setup_inputs(tmp_path)
    output = tmp_path / "reports"
    result = render(stats, batch, matrix, output)
    assert result["status"] == "incomplete" and result["reports"] == 8
    expected = {"THIRD_ROUND_PROTOCOL.md", "MELD_BASELINE_VS_CMERC.md", "MULTISEED_PAIRED_RESULTS.md",
        "CALIBRATION_DIAGNOSTIC.md", "CLASS_AND_POSITION_ANALYSIS.md", "LOSS_GRADIENT_ANALYSIS.md",
        "NEXT_RESEARCH_CANDIDATES.md", "FINAL_EXECUTION_STATUS.md"}
    assert expected <= {path.name for path in output.iterdir()}
    assert (output / "THIRD_ROUND_PROTOCOL.md").read_bytes() == (batch / "THIRD_ROUND_PROTOCOL.md").read_bytes()
    assert "No completed MELD" in (output / "MELD_BASELINE_VS_CMERC.md").read_text()
    assert "minimum p=0.25" in (output / "CALIBRATION_DIAGNOSTIC.md").read_text()
    assert "not complete modality removal" in (output / "CLASS_AND_POSITION_ANALYSIS.md").read_text()
    assert "not independently audited here" in (output / "FINAL_EXECUTION_STATUS.md").read_text()
    assert len(json.loads((output / "loss_gradient_trajectories.json").read_text())) == 160
    assert "first batch" in (output / "LOSS_GRADIENT_ANALYSIS.md").read_text()
    manifest = json.loads((output / "REPORT_MANIFEST.json").read_text())
    assert all(sha(output / name) == digest for name, digest in manifest.items())
    with pytest.raises(FileExistsError):
        render(stats, batch, matrix, output)


def test_actual_adam_audit_is_checked_separately_from_derived_counts(tmp_path):
    stats, batch, matrix = setup_inputs(tmp_path, actual_optimizer=True)
    output = tmp_path / "reports"
    render(stats, batch, matrix, output)
    evidence = json.loads((output / "REPORT_EVIDENCE.json").read_text())
    audit = evidence["execution"]["optimizer_audits"]["IEMOCAP/1937"]["cmerc"]
    assert audit["actual_adam_steps"] == 160
    assert audit["verification"] == "actual_adam_state_verified"
    path = matrix.parent / "cmerc_optimizer.json"
    record = json.loads(path.read_text())
    record["optimizer_steps_total"] = 159
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Adam count"):
        render(stats, batch, matrix, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_meld_report_contains_actual_classes_and_per_utterance_cel(tmp_path):
    stats, batch, matrix = setup_inputs(tmp_path, dataset="MELD", actual_optimizer=True)
    output = tmp_path / "reports"
    render(stats, batch, matrix, output)
    meld = (output / "MELD_BASELINE_VS_CMERC.md").read_text()
    assert "f1.anger" in meld and "40" in meld
    assert "CEL per utterance" in meld and "cel.inter_context" in meld
    saved = json.loads((output / "paired_statistics.json").read_text())
    assert saved["report_additions"]["actual_optimizer_audits"]["MELD/1937"]["baseline"]["actual_adam_steps"] == 80


def test_public_copy_redacts_nested_paths_and_ids_preserving_private_original_and_hashes(tmp_path):
    stats, batch, matrix = setup_inputs(tmp_path)
    path = stats / "paired_statistics.json"
    value = json.loads(path.read_text())
    artifact = {"sha256": "d" * 64, "source_checkpoint": {"path": "/private/teacher/best.pt", "sha256": "e" * 64},
                "manifest": {"train_ids": ["private_dialogue"]}}
    value["training"]["IEMOCAP/1937"]["cmerc"]["config"]["difficulty_artifact"] = artifact
    path.write_text(json.dumps(value))
    original_sha = sha(path)
    output = tmp_path / "reports"
    render(stats, batch, matrix, output)
    assert sha(path) == original_sha
    assert "/private/teacher" in path.read_text()
    for public in output.iterdir():
        assert "/private/teacher" not in public.read_text()
        assert "private_dialogue" not in public.read_text()
    evidence = json.loads((output / "REPORT_EVIDENCE.json").read_text())
    assert evidence["input_sha256"]["statistics_sha256"] == original_sha
    public_artifact = evidence["public_run_configurations"]["IEMOCAP/1937"]["cmerc"]["difficulty_artifact"]
    assert public_artifact["sha256"] == "d" * 64
    assert public_artifact["source_checkpoint"]["sha256"] == "e" * 64


def test_full_cross_seed_config_admission_and_new_seed_only_sensitivity(tmp_path):
    stats, _, _ = setup_inputs(tmp_path)
    value = json.loads((stats / "paired_statistics.json").read_text())
    source = value["training"]["IEMOCAP/1937"]
    for seed in (2025, 3407):
        value["training"][f"IEMOCAP/{seed}"] = copy.deepcopy(source)
        for run in value["training"][f"IEMOCAP/{seed}"].values():
            run["config"]["seed"] = seed
            run["config"]["difficulty_artifact"] = {"sha256": str(seed)}
    assert validate_configurations(value)["IEMOCAP/cmerc"]["seeds"] == [1937, 2025, 3407]
    for seed, delta in ((2025, 2.), (3407, 4.)):
        entry = copy.deepcopy(value["per_seed"][0])
        entry["seed"] = seed
        entry["overall"]["weighted_f1"] = {"baseline": 50., "cmerc": 50. + delta, "delta": delta}
        value["per_seed"].append(entry)
    sensitivity = next(row for row in new_iemocap_sensitivity(value) if row["metric"] == "weighted_f1")
    assert sensitivity["n"] == 2 and sensitivity["delta"]["mean"] == 3.
    assert sensitivity["delta"]["sample_sd"] == pytest.approx(2 ** .5)
    value["training"]["IEMOCAP/2025"]["cmerc"]["config"]["auxiliary_policy"] = "same_step_same_mode"
    with pytest.raises(ValueError, match="cmerc protocols differ"):
        validate_configurations(value)


def test_context_share_suppresses_near_zero_or_nonreducing_denominators():
    entry = {"overall": {"cel.total": {"baseline": 100., "delta": -10.}},
             "diagnostics": {"context_cel_delta": -12.}}
    assert context_share(entry) == 120.
    for delta in (-1e-12, 0., 10.):
        entry["overall"]["cel.total"]["delta"] = delta
        assert context_share(entry) is None


def test_public_metadata_redacts_paths_embedded_after_punctuation():
    from scripts.round3_reports import public_metadata
    value = {"claim": "pair:artifact:/data2/private/server/best.pt",
             "evidence": "source='/home/private/experiment.json'", "sha256": "a" * 64}
    public = public_metadata(value)
    assert "/data2/" not in json.dumps(public)
    assert "/home/" not in json.dumps(public)
    assert public["sha256"] == value["sha256"]


def test_failure_attempts_and_partial_epochs_retained_without_private_errors(tmp_path):
    stats, batch, matrix = setup_inputs(tmp_path)
    pair = batch / "pairs/MELD_seed1937"
    pair.mkdir(parents=True)
    state = {"dataset": "MELD", "seed": 1937, "status": "failed", "error_type": "RuntimeError",
        "error": "private_dialogue /private/server OOM", "stages": {"baseline_training": [
            {"status": "failed", "exit_code": 1, "command": ["/private/python", "private_dialogue"],
             "elapsed_seconds": 3.5, "started_utc": "start", "ended_utc": "end"}]},
        "recovery_candidates": {"baseline": {"latest_checkpoint": "/private/latest.pt", "sha256": "f" * 64,
                                               "command": ["private_dialogue"]}}}
    (pair / "STATE.json").write_text(json.dumps(state))
    run = pair / "runs/baseline/some_run"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"epoch":1}\n{"epoch":2}\n')
    (run / "latest.pt").write_bytes(b"checkpoint fixture")
    output = tmp_path / "reports"
    render(stats, batch, matrix, output)
    evidence = json.loads((output / "REPORT_EVIDENCE.json").read_text())
    saved = evidence["execution"]["states"]["MELD/1937"]
    assert saved["partial_training"]["baseline"]["last_logged_epoch"] == 2
    assert saved["stages"][0]["returncode"] == 1
    assert saved["stages"][0]["elapsed_seconds"] == 3.5
    assert saved["recovery_candidates"]["baseline"]["sha256"] == "f" * 64
    assert "--resume-baseline" in (output / "FINAL_EXECUTION_STATUS.md").read_text()
    for path in output.iterdir():
        assert "/private" not in path.read_text()
        assert "private_dialogue" not in path.read_text()


@pytest.mark.parametrize("mutation,error", [("matrix", "matrix differs"), ("protocol", "protocol differs")])
def test_renderer_rejects_changed_inputs_before_creating_output(tmp_path, mutation, error):
    stats, batch, matrix = setup_inputs(tmp_path)
    path = matrix if mutation == "matrix" else batch / "THIRD_ROUND_PROTOCOL.md"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match=error):
        render(stats, batch, matrix, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
