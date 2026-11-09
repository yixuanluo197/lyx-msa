"""Regression guards for the independent real-model parity gate."""

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/real_removal_parity.py"
spec = importlib.util.spec_from_file_location("real_removal_parity", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_parity_rejects_a_changed_removal_view():
    expected = [torch.zeros(3, 6) for _ in range(6)]
    actual = [value.clone() for value in expected]
    actual[4][1, 2] = 0.01
    with pytest.raises(AssertionError, match="intra"):
        gate.compare_views(expected, actual, 1e-6, "log_prob")


def test_combined_tolerance_accepts_reproduced_total_gap_error():
    reference = 2.450494170188904
    observed = reference + 1.6689300537109375e-6
    result = gate.compare_values(reference, observed, 1e-6, 1e-5, "total_gap")
    assert result["max_absolute_error"] > 1e-6
    assert result["max_absolute_error"] <= result["effective_threshold_at_max_error"]
    assert result["tolerance_relative"] == 1e-5
    with pytest.raises(AssertionError, match="total_gap"):
        gate.compare_values(reference, reference + 1e-3, 1e-6, 1e-5, "total_gap")


def test_vector_tolerance_uses_each_reference_value():
    reference = [torch.tensor([[0., 2.45]], dtype=torch.float64) for _ in range(6)]
    observed = [value.clone() for value in reference]
    observed[0][0, 1] += 1.6689300537109375e-6
    assert gate.compare_views(reference, observed, 1e-6, "log_prob")["full"]["passed"]
    observed[0][0, 0] += 2e-6
    with pytest.raises(AssertionError, match="full"):
        gate.compare_views(reference, observed, 1e-6, "log_prob")


def test_reference_masks_unknown_before_true_class_lookup():
    full = torch.tensor([[-2., -1.], [-1., -2.], [-2., -1.]])
    removed = [full + 0.5 for _ in range(5)]
    record = {"labels": [0, -100, 1], "supervision_mask": [True, False, True],
              "qmask": [[1, 0], [0, 1], [1, 0]]}
    result = gate.reference_decomposition([full] + removed, record)
    assert result["valid_label_count"] == 2
    assert result["speaker_term"] == 2
    assert result["text_gap"] == 1
    assert result["final_difficulty"] == 7 / 4
    assert result["difficulty_total_length"] == 7 / 5


def test_reference_extracts_six_author_forward_calls():
    function, metadata = gate.author_reference(gate.OUT / "source_snapshot/cal_diff.py")
    assert callable(function)
    assert metadata["forward_calls"] == ["model", "model_mm", "model_mm", "model_mm",
                                         "model_intra", "model_inter"]
    assert len(metadata["mean_replacement_assignments"]) == 7


def test_strict_load_rejects_missing_parameters():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="Missing key"):
        gate.strict_copy_state([model], {"weight": model.weight.detach().clone()})
