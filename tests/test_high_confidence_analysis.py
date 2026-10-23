"""CPU-only regression checks for error-analysis arithmetic and sample alignment."""

import csv
import importlib
import importlib.util

import numpy as np
import pytest


def module():
    name = "scripts.high_confidence_analysis"
    assert importlib.util.find_spec(name) is not None, "Diagnostic implementation is missing"
    return importlib.import_module(name)


def records(probabilities=None, labels=None, ids=None, positions=None, texts=None, model="baseline"):
    if probabilities is None:
        probabilities = [[0.8, 0.2], [0.1, 0.9], [0.95, 0.05], [0.6, 0.4]]
    n = len(probabilities)
    return module().make_records(
        "MELD", model, [1, 0, 0, 1][:n] if labels is None else labels,
        np.log(probabilities), ["d"] * n if ids is None else ids,
        list(range(n)) if positions is None else positions,
        [f"text {i}" for i in range(n)] if texts is None else texts, ["neutral", "sad"],
    )


def test_log_probabilities_are_exponentiated_and_preserve_original_text():
    result = records(texts=['Hi, "friend"\nnext\tline\x00', "b", "c", "d"])
    assert result[0]["probability_neutral"] == pytest.approx(0.8)
    assert result[0]["true_class_probability"] == pytest.approx(0.2)
    assert result[0]["max_probability"] == pytest.approx(0.8)
    assert result[0]["is_error"] is True
    assert result[0]["text"] == 'Hi, "friend"\nnext\tline\x00'


def test_thresholds_include_equal_boundary_and_use_all_high_confidence_predictions():
    summary = module().summarize_records(records(), ["neutral", "sad"])
    assert summary["num_samples"] == 4
    assert summary["error_count"] == 3
    assert summary["accuracy_percent"] == 25
    assert summary["thresholds"]["0.8"] == {
        "high_confidence_count": 3, "high_confidence_error_count": 2,
        "error_rate_among_high_confidence": pytest.approx(2 / 3),
    }
    assert summary["thresholds"]["0.9"]["high_confidence_error_count"] == 1
    assert summary["thresholds"]["0.9"]["high_confidence_count"] == 2
    assert summary["class_error_distribution"]["sad"]["error_count"] == 2
    assert summary["confusion_pairs"][0]["count"] == 2


def test_no_high_confidence_ratio_is_undefined():
    summary = module().summarize_records(records([[0.6, 0.4]], [1]), ["neutral", "sad"])
    assert summary["thresholds"]["0.8"]["error_rate_among_high_confidence"] is None


def test_empty_records_produce_zero_counts_and_undefined_accuracy():
    summary = module().summarize_records([], ["neutral", "sad"])
    assert summary["num_samples"] == 0
    assert summary["accuracy_percent"] is None


@pytest.mark.parametrize("change", [
    {"labels": [-1]}, {"labels": [2]}, {"labels": [0.5]},
    {"labels": [True]}, {"ids": []}, {"texts": []}, {"positions": [-1]},
    {"positions": [0.5]}, {"positions": [True]},
])
def test_invalid_metadata_is_rejected(change):
    kwargs = {"probabilities": [[0.7, 0.3]], "labels": [0]}
    kwargs.update(change)
    with pytest.raises(ValueError):
        records(**kwargs)


@pytest.mark.parametrize("logp", [
    [[np.nan, -1]], [[np.inf, -1]], [[-np.inf, 0]],
    [[0.8, 0.2]], [[-2, -2]], [[0]], [-1, -1],
])
def test_invalid_probabilities_or_dimensions_are_rejected(logp):
    with pytest.raises(ValueError):
        module().make_records("MELD", "baseline", [0], logp, ["d"], [0], ["text"], ["n", "s"])


def test_duplicate_normalized_sample_keys_are_rejected():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        records([[0.7, 0.3], [0.7, 0.3]], [0, 0], ids=[1, "1"], positions=[0, 0])


def test_duplicate_class_names_are_rejected():
    with pytest.raises(ValueError):
        module().make_records("MELD", "baseline", [0], [[-0.69314718] * 2], ["d"], [0], ["t"], ["n", "n"])


def test_csv_roundtrip_and_error_ordering(tmp_path):
    original = records(texts=['Hi, "friend"\r\nnext\tline\x01', "b", "c", "d"])
    summary = module().write_model_outputs(tmp_path, original, ["neutral", "sad"])
    assert summary["error_count"] == 3
    with (tmp_path / "all_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    assert all_rows[0]["text"] == original[0]["text"]
    assert [int(r["utterance_index"]) for r in all_rows] == [0, 1, 2, 3]
    with (tmp_path / "errors_all.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [int(r["utterance_index"]) for r in csv.DictReader(stream)] == [1, 0, 3]
    with (tmp_path / "errors_ge_0p8.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with (tmp_path / "errors_ge_0p9.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1
    assert (tmp_path / "class_error_distribution.csv").exists()


def test_equal_confidence_errors_sort_by_dialogue_and_index(tmp_path):
    result = records([[0.8, 0.2]] * 3, [1] * 3, ids=["b", "a", "a"], positions=[0, 3, 1])
    module().write_model_outputs(tmp_path, result, ["neutral", "sad"])
    with (tmp_path / "errors_all.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [(r["dialogue_id"], int(r["utterance_index"])) for r in csv.DictReader(stream)] == [("a", 1), ("a", 3), ("b", 0)]


def test_optional_original_utterance_ids_are_written_to_predictions_and_errors(tmp_path):
    result = records()
    for i, row in enumerate(result):
        row["source_utterance_id"] = f"dia_42_utt_{i}"
    module().write_model_outputs(tmp_path, result, ["neutral", "sad"])
    for filename in ("all_predictions.csv", "errors_all.csv", "errors_ge_0p8.csv", "errors_ge_0p9.csv"):
        with (tmp_path / filename).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        for row in rows:
            assert row["source_utterance_id"] == f'dia_42_utt_{row["utterance_index"]}'


def test_comparison_preserves_and_validates_original_utterance_ids():
    baseline, cmerc = records(), records(model="cmerc")
    for collection in (baseline, cmerc):
        for i, row in enumerate(collection):
            row["source_utterance_id"] = f"original_{i}"
    compared = module().compare_records(baseline, cmerc[::-1])
    assert compared[0]["source_utterance_id"] == "original_0"
    cmerc[0]["source_utterance_id"] = "wrong_source_id"
    with pytest.raises(ValueError, match="metadata differ"):
        module().compare_records(baseline, cmerc)


def test_comparison_joins_keys_even_when_input_order_differs():
    baseline = records()
    cmerc = records([[0.1, 0.9], [0.95, 0.05], [0.95, 0.05], [0.9, 0.1]], model="cmerc")
    compared = module().compare_records(baseline, cmerc[::-1])
    assert compared[0]["utterance_index"] == 0
    assert compared[0]["error_transition"] == "baseline_wrong_cmerc_correct"
    assert compared[2]["error_transition"] == "both_correct"
    assert compared[3]["error_transition"] == "both_wrong"
    assert compared[0]["cmerc_probability_sad"] == pytest.approx(0.9)
    assert compared[0]["baseline_probability_sad"] == pytest.approx(0.2)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "label", "text", "dataset"])
def test_comparison_rejects_missing_duplicate_or_misaligned_samples(corruption):
    baseline, cmerc = records(), records(model="cmerc")
    if corruption == "missing":
        cmerc.pop()
    elif corruption == "duplicate":
        cmerc.append(cmerc[0].copy())
    elif corruption == "label":
        cmerc[0]["true_label"] = 0
    elif corruption == "text":
        cmerc[0]["text"] = "other text"
    else:
        cmerc[0]["dataset"] = "IEMOCAP"
    with pytest.raises(ValueError):
        module().compare_records(baseline, cmerc)
