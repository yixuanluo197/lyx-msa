import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.round2_analysis import compare_evaluations, main, relative_position_masks, summarize_training


def evaluation_dir(path, *, improved=False, replacement_policy="prefix_mean_v1"):
    path.mkdir()
    labels = np.array([0, 1, 0, 1])
    probabilities = np.full((4, 6), .02)
    probabilities[:, 0] = .5
    probabilities[:, 1] = .42
    if improved:
        probabilities[labels == 1, 0] = .42
        probabilities[labels == 1, 1] = .5
    values = np.log(probabilities)
    np.savez_compressed(path / "predictions.npz", labels=labels,
        log_probabilities=values, dialogue_ids=np.array(["private_A"] * 3 + ["private_B"]),
        utterance_indices=np.array([0, 1, 2, 0]),
        **{f"removed_{view}_log_probabilities": values for view in
           ("text", "audio", "visual", "intra_context", "inter_context")})
    metadata = {"dataset": "IEMOCAP", "split": "validation", "split_seed": 1475,
        "valid_fraction": .1, "causal": True, "replacement_policy": replacement_policy,
        "architecture": {"base_model": "GRU", "norm": "LN"},
        "evaluation_data": {"verification": "checkpoint_provenance_verified",
                            "source_data_provenance": {"feature_files": {"x": "abc"}}},
        "calibration": {"n_bins": 15}, "checkpoint": "/private/server/best.pt"}
    (path / "metrics.json").write_text(json.dumps(metadata))
    return path


def test_position_thirds_includes_length_one_in_middle():
    masks = relative_position_masks(np.array(["A"] * 3 + ["B"]), np.array([0, 1, 2, 0]))
    assert masks["early"].tolist() == [True, False, False, False]
    assert masks["middle"].tolist() == [False, True, False, True]
    assert masks["late"].tolist() == [False, False, True, False]


def test_analysis_computes_paired_deltas_and_keeps_ids_private(tmp_path):
    baseline = evaluation_dir(tmp_path / "base")
    cmerc = evaluation_dir(tmp_path / "full", improved=True)
    result = compare_evaluations(baseline, cmerc)
    assert result["overall"]["accuracy"]["baseline"] == 50
    assert result["overall"]["accuracy"]["cmerc"] == 100
    assert result["overall"]["accuracy"]["delta"] == 50
    assert result["overall"]["cel.total"]["delta"] == 0
    assert result["position_thirds"]["middle"]["num_samples"] == 2
    assert "private_" not in json.dumps(result)
    assert "/private" not in json.dumps(result)


@pytest.mark.parametrize("field,value", [("replacement_policy", "zero_v1"),
    ("split", "test"), ("split_seed", 42), ("causal", False)])
def test_analysis_rejects_incompatible_protocol(tmp_path, field, value):
    baseline = evaluation_dir(tmp_path / "base")
    cmerc = evaluation_dir(tmp_path / "full")
    path = cmerc / "metrics.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        compare_evaluations(baseline, cmerc)


def test_position_masks_reject_missing_or_duplicate_indices():
    with pytest.raises(ValueError, match="contiguous"):
        relative_position_masks(np.array(["A", "A"]), np.array([0, 2]))
    with pytest.raises(ValueError, match="contiguous"):
        relative_position_masks(np.array(["A", "A"]), np.array([0, 0]))


def test_analysis_rejects_same_size_different_dialogues(tmp_path):
    baseline = evaluation_dir(tmp_path / "base")
    cmerc = evaluation_dir(tmp_path / "full")
    path = cmerc / "predictions.npz"
    with np.load(path) as original:
        arrays = dict(original)
    arrays["dialogue_ids"] = np.array(["different_A"] * 3 + ["different_B"])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="membership"):
        compare_evaluations(baseline, cmerc)


def test_cli_exports_public_aggregates_and_refuses_overwrite(tmp_path):
    baseline = evaluation_dir(tmp_path / "base")
    cmerc = evaluation_dir(tmp_path / "full")
    output = tmp_path / "comparison"
    args = ["--baseline-eval", str(baseline), "--cmerc-eval", str(cmerc), "--output-dir", str(output)]
    assert main(args) == 0
    assert "private" not in (output / "comparison.json").read_text()
    assert "weighted_f1" in (output / "comparison.md").read_text()
    with pytest.raises(FileExistsError):
        main(args)


def test_test_reporting_requires_a_validation_freeze(tmp_path):
    baseline = evaluation_dir(tmp_path / "base")
    cmerc = evaluation_dir(tmp_path / "full")
    for directory in (baseline, cmerc):
        path = directory / "metrics.json"
        metadata = json.loads(path.read_text())
        metadata["split"] = "test"
        path.write_text(json.dumps(metadata))
    args = ["--baseline-eval", str(baseline), "--cmerc-eval", str(cmerc),
            "--output-dir", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="validation-freeze"):
        main(args)


def test_training_summary_preserves_curve_and_first_best(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [{"epoch": e, "selection_split": "validation", "selection_weighted_f1": score,
        "validation_weighted_f1": score, "train_weighted_f1": score + 1,
        "elapsed_seconds": 10, "max_gpu_memory_mb": 100 + e,
        "classification_loss": 1 / e, "cc_loss": 0., "mscl_loss": 0.,
        "cscl_loss": 0., "total_loss": 1 / e} for e, score in enumerate([50., 60., 60.], 1)]
    path.write_text("\n".join(json.dumps(record) for record in records))
    result = summarize_training(path)
    assert result["completed_epochs"] == 3
    assert result["best_epoch"] == 2
    assert result["elapsed_seconds"] == 30
    assert result["max_gpu_memory_mb"] == 103
    assert len(result["convergence"]) == 3
    path.write_text("\n".join(json.dumps(record) for record in records[1:]))
    with pytest.raises(ValueError, match="contiguous"):
        summarize_training(path)
