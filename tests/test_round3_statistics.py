import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_metrics import calibration_metrics
from scripts.round2_analysis import compare_evaluations, file_sha256
from scripts.round3_statistics import (
    analyze_matrix, holm_adjust, main, paired_statistics, signflip_pvalue,
)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def fixture_matrix(tmp_path, dataset="IEMOCAP", seed=1937):
    tmp_path.mkdir(parents=True, exist_ok=True)
    epochs = 80 if dataset == "IEMOCAP" else 40
    classes = 6 if dataset == "IEMOCAP" else 7
    provenance = {"dataset": dataset, "feature_files": {"features": "f" * 64},
                  "train_ids_sha256": "a" * 64, "validation_ids_sha256": "b" * 64}
    pair = {"dataset": dataset, "seed": seed, "evaluations": {}}
    runs = {}
    for variant in ("baseline", "cmerc"):
        run = tmp_path / variant
        run.mkdir()
        runs[variant] = run
        full = variant == "cmerc"
        config = {"Dataset": dataset, "seed": seed, "split_seed": 1475, "batch_size": 16,
            "causal": True, "selection_protocol": "validation", "replacement_policy": "prefix_mean_v1",
            "auxiliary_policy": "legacy_epoch_snapshot", "data_provenance": provenance,
            "courselearning": full, "contrastlearning": full, "calibrate": full,
            "rank_coff": .005 if full else 0, "mscl_coff": .05 if full else 0,
            "cscl_coff": .05 if full else 0}
        write_json(run / "run_manifest.json", {"config": config, "config_sha256": variant * 8})
        write_json(run / "initial_state.json", {"model_state_sha256": "1" * 64})
        for name in ("best", "latest"):
            (run / f"{name}.pt").write_bytes(f"fixture {variant} {name}".encode())
        records, diagnostics = [], []
        for epoch in range(epochs):
            records.append({"epoch": epoch + 1, "selection_split": "validation",
                "selection_weighted_f1": 50. + epoch, "elapsed_seconds": 2., "max_gpu_memory_mb": 100.,
                "classification_loss": 1., "cc_loss": 2. if full else 0., "mscl_loss": 3. if full else 0.,
                "cscl_loss": 4. if full else 0., "train_dialogues": 17 if full else 32})
            coefficients = {"classification_loss": 1., "cc_loss": config["rank_coff"],
                            "mscl_loss": config["mscl_coff"], "cscl_loss": config["cscl_coff"]}
            losses = {name: records[-1][name] for name in coefficients}
            losses["total_loss"] = sum(coefficients[name] * losses[name] for name in coefficients)
            records[-1]["total_loss"] = losses["total_loss"]
            norms = {name: 2. for name in coefficients if coefficients[name]}
            diagnostics.append({"epoch": epoch, "batch": 0, "finite_losses": True,
                "finite_gradients": True, "losses": losses, "gradient_coefficients": coefficients,
                "gradient_norms": norms, "weighted_gradient_norms":
                    {name: norm * coefficients[name] for name, norm in norms.items()}})
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records))
        (run / "training_diagnostics.jsonl").write_text("\n".join(json.dumps(record) for record in diagnostics))
        verified = {"status": "passed", "variant": variant, "completed_epochs": epochs, "best_epoch": epochs,
            "best_validation_weighted_f1": 49. + epochs, "config_sha256": variant * 8,
            "run_manifest_sha256": file_sha256(run / "run_manifest.json"),
            "initial_model_sha256": "1" * 64, "data_provenance": provenance,
            "best_sha256": file_sha256(run / "best.pt"), "latest_sha256": file_sha256(run / "latest.pt"),
            "optimizer_steps": 2 * epochs}
        if full:
            verified["difficulty"] = {"teacher_best_sha256": file_sha256(runs["baseline"] / "best.pt")}
        write_json(tmp_path / f"{variant}_verification.json", verified)
        pair[f"{variant}_run"] = variant
        pair[f"{variant}_verification"] = f"{variant}_verification.json"
    for split in ("validation", "test"):
        pair["evaluations"][split] = {}
        for variant in runs:
            directory = tmp_path / f"{split}_{variant}"
            directory.mkdir()
            labels = np.array([0, 1, 0, 1, 0, 1])
            probabilities = np.full((6, classes), .02)
            probabilities[:, 0] = .5
            probabilities[:, 1] = .5 - .02 * (classes - 2)
            if variant == "cmerc":
                probabilities[labels == 1, 0] = .5 - .02 * (classes - 2)
                probabilities[labels == 1, 1] = .5
            logp = np.log(probabilities)
            np.savez_compressed(directory / "predictions.npz", labels=labels,
                log_probabilities=logp, dialogue_ids=np.array(["A"] * 3 + ["B"] * 3),
                utterance_indices=np.array([0, 1, 2, 0, 1, 2]),
                **{f"removed_{view}_log_probabilities": logp
                   for view in ("text", "audio", "visual", "intra_context", "inter_context")})
            calibration = calibration_metrics(labels, probabilities)
            for key in ("auroc_success", "auprc_success"):
                if not math.isfinite(calibration[key]):
                    calibration[key] = None
            metadata = {"dataset": dataset, "split": split, "split_seed": 1475,
                "valid_fraction": .1, "causal": True, "replacement_policy": "prefix_mean_v1",
                "architecture": {"base_model": "GRU", "norm": "LN"},
                "checkpoint": str(runs[variant] / "best.pt"), "calibration": calibration,
                "evaluation_data": {"source_data_provenance": provenance,
                                     "verification": "checkpoint_provenance_verified"}}
            write_json(directory / "metrics.json", metadata)
            pair["evaluations"][split][variant] = directory.name
    write_json(tmp_path / "validation_freeze.json",
        compare_evaluations(tmp_path / "validation_baseline", tmp_path / "validation_cmerc"))
    pair["evaluations"]["test"]["validation_freeze"] = "validation_freeze.json"
    matrix = {"schema_version": 1, "planned_seeds": [1937, 2025, 3407],
              "planned_datasets": ["IEMOCAP", "MELD"],
              "known_test_exposure": [{"dataset": "IEMOCAP", "seed": 1937, "reason": "round 2 observed"}],
              "pairs": [pair]}
    write_json(tmp_path / "matrix.json", matrix)
    return tmp_path / "matrix.json"


def test_exact_signflip_exhausts_all_eight_assignments_and_ties():
    assert signflip_pvalue([1, 2, 3]) == .25
    assert signflip_pvalue([-1, -2, -3]) == .25
    assert signflip_pvalue([1, 1, -1]) == 1.
    assert signflip_pvalue([0, 0, 0]) == 1.


def test_paired_sample_sd_and_df2_interval_use_deltas_not_independent_groups():
    result = paired_statistics([100, 200, 300], [101, 202, 303])
    assert result["delta"]["mean"] == 2
    assert result["delta"]["sample_sd"] == 1
    assert result["paired_t_95_interval_df2"] == pytest.approx([-.4841377117195456, 4.484137711719546])
    partial = paired_statistics([1], [2])
    assert partial["exact_two_sided_signflip_p"] is None
    assert partial["paired_t_95_interval_df2"] is None
    assert partial["delta"]["sample_sd"] is None


def test_holm_is_monotone_and_keeps_full_family():
    adjusted = holm_adjust({"a": .01, "b": .04, "c": .03, "d": .8, "e": .7, "f": 1.})
    assert adjusted == {"a": .06, "c": .15, "b": .16, "e": 1., "d": 1., "f": 1.}


def test_complete_matrix_uses_six_hypotheses_per_split_and_all_three_seeds(tmp_path):
    pairs = []
    for dataset in ("IEMOCAP", "MELD"):
        for seed in (1937, 2025, 3407):
            directory = tmp_path / f"{dataset}_{seed}"
            matrix_path = fixture_matrix(directory, dataset, seed)
            pair = json.loads(matrix_path.read_text())["pairs"][0]
            for key in ("baseline_run", "cmerc_run", "baseline_verification", "cmerc_verification"):
                pair[key] = str(directory / pair[key])
            for evaluation in pair["evaluations"].values():
                for key, value in evaluation.items():
                    evaluation[key] = str(directory / value)
            pairs.append(pair)
    matrix = {"schema_version": 1, "planned_seeds": [1937, 2025, 3407],
              "planned_datasets": ["IEMOCAP", "MELD"], "pairs": pairs,
              "known_test_exposure": [{"dataset": "IEMOCAP", "seed": 1937}]}
    write_json(tmp_path / "matrix.json", matrix)
    result, _, _, _ = analyze_matrix(tmp_path / "matrix.json")
    assert result["status"] == "complete"
    for split in ("validation", "test"):
        family = result["primary_families"][split]
        assert family["status"] == "complete"
        assert len(family["holm_adjusted_p"]) == 6
        assert all(value == 1. for value in family["holm_adjusted_p"].values())
    wf1 = next(row for row in result["statistics"] if row["metric"] == "weighted_f1" and
               row["dataset"] == "IEMOCAP" and row["split"] == "test" and row["segment"] == "overall")
    assert wf1["seeds"] == [1937, 2025, 3407]
    assert wf1["exact_two_sided_signflip_p"] == .25
    assert wf1["delta"]["sample_sd"] == 0.


def test_partial_matrix_keeps_missing_seeds_and_no_interim_holm(tmp_path):
    matrix = fixture_matrix(tmp_path)
    result, rows, gradients, curves = analyze_matrix(matrix)
    assert result["status"] == "incomplete"
    assert len(result["missing_evaluations"]) == 10
    assert len(result["per_seed"]) == 2
    assert len(curves) == 160 and len(gradients) == 160
    assert result["training"]["IEMOCAP/1937"]["baseline"]["optimizer_steps"] == 160
    assert result["primary_families"]["test"]["status"] == "incomplete_no_holm_test"
    assert all(row["exact_two_sided_signflip_p"] is None for row in result["statistics"])
    assert next(row for row in rows if row["metric"] == "auroc_success")["cmerc"] is None
    assert all("private" not in json.dumps(row) for row in rows)


@pytest.mark.parametrize("mutation,error", [
    ("teacher", "teacher"), ("checkpoint", "checkpoint changed"),
    ("init", "initial fingerprint"), ("gradient", "weighted gradient norm"),
    ("freeze", "validation artifacts differ"), ("exposure", "previously observed"),
    ("duplicate", "duplicate"), ("incomplete", "budget incomplete"),
])
def test_admission_rejects_invalid_evidence(tmp_path, mutation, error):
    matrix_path = fixture_matrix(tmp_path)
    if mutation == "checkpoint":
        (tmp_path / "baseline/best.pt").write_bytes(b"replacement")
    elif mutation == "gradient":
        path = tmp_path / "cmerc/training_diagnostics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["weighted_gradient_norms"]["cc_loss"] = 200.
        path.write_text("\n".join(json.dumps(row) for row in rows))
    elif mutation == "incomplete":
        path = tmp_path / "baseline/metrics.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    elif mutation == "init":
        write_json(tmp_path / "cmerc/initial_state.json", {"model_state_sha256": "2" * 64})
    elif mutation == "teacher":
        path = tmp_path / "cmerc_verification.json"
        record = json.loads(path.read_text())
        record["difficulty"]["teacher_best_sha256"] = "2" * 64
        write_json(path, record)
    elif mutation == "freeze":
        path = tmp_path / "validation_freeze.json"
        record = json.loads(path.read_text())
        record["input_sha256"]["baseline"]["metrics.json"] = "2" * 64
        write_json(path, record)
    else:
        record = json.loads(matrix_path.read_text())
        if mutation == "exposure":
            record["known_test_exposure"] = []
        else:
            record["pairs"].append(record["pairs"][0])
        write_json(matrix_path, record)
    with pytest.raises(ValueError, match=error):
        analyze_matrix(matrix_path)


def test_cli_outputs_serializable_full_trajectories_and_refuses_overwrite(tmp_path):
    matrix = fixture_matrix(tmp_path)
    destination = tmp_path / "output"
    args = ["--matrix", str(matrix), "--output-dir", str(destination)]
    assert main(args) == 0
    result = json.loads((destination / "paired_statistics.json").read_text())
    assert len(result["training"]["IEMOCAP/1937"]["cmerc"]["gradient_trajectory"]) == 80
    assert "wholly blind" in (destination / "MULTISEED_PAIRED_RESULTS.md").read_text()
    assert (destination / "per_seed_metrics.csv").is_file()
    with pytest.raises(FileExistsError):
        main(args)
