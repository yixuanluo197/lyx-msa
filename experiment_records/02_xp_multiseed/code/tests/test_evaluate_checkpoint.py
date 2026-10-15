import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("evaluate_checkpoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dataset_config_matches_official_readme():
    iemocap = MODULE.dataset_config("iemocap")
    meld = MODULE.dataset_config("MELD")

    assert (iemocap.dropout, iemocap.num_classes, iemocap.num_k, iemocap.seed) == (
        0.5,
        6,
        4,
        1475,
    )
    assert (meld.dropout, meld.num_classes, meld.num_k, meld.seed) == (
        0.4,
        7,
        3,
        67137,
    )


def test_extract_state_dict_accepts_nested_checkpoint_and_strips_module_prefix():
    state = MODULE.extract_state_dict(
        {"epoch": 4, "state_dict": {"module.layer.weight": "weights", "layer.bias": "bias"}}
    )

    assert state == {"layer.weight": "weights", "layer.bias": "bias"}


def test_expand_utterance_ids_follows_dialogue_major_model_output_order():
    dialogue_ids, utterance_indices = MODULE.expand_utterance_ids(
        ["dialogue-a", "dialogue-b"], [2, 3]
    )

    assert dialogue_ids == ["dialogue-a", "dialogue-a", "dialogue-b", "dialogue-b", "dialogue-b"]
    assert utterance_indices == [0, 1, 0, 1, 2]


def test_classification_metrics_include_fixed_class_order_and_confusion_matrix():
    result = MODULE.classification_metrics(
        labels=np.array([0, 0, 1, 2]),
        predictions=np.array([0, 1, 1, 1]),
        class_names=["zero", "one", "two"],
    )

    assert result["accuracy"] == 50.0
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert [row["class_name"] for row in result["per_class"]] == ["zero", "one", "two"]
    assert result["per_class"][2]["support"] == 1


def test_validate_log_probabilities_rejects_non_normalized_values():
    invalid = np.log(np.array([[0.2, 0.2]], dtype=np.float64))

    try:
        MODULE.validate_log_probabilities(invalid)
    except ValueError as error:
        assert "normalized" in str(error)
    else:
        raise AssertionError("expected non-normalized log probabilities to be rejected")
