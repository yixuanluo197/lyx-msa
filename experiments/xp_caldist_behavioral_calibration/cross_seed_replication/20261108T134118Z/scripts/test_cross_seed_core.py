import json
from pathlib import Path

import numpy as np
import pytest

from cross_seed_core import (behavior, contract, donor_index, donor_modality_missing,
                             interpolate, native_key_map, target_indices)
from run_cross_seed import donor_pool


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/preregistered_config.json").read_text())


def test_frozen_condition_contract():
    main, donor = contract(config())
    assert len(main) == 10
    assert {(x["target"], x["strength"]) for x in main} == {
        ("T-All", .5), ("T-All", 1), ("A", .5), ("A", 1), ("V", .5), ("V", 1),
        ("T1", 1), ("T2", 1), ("T3", 1), ("T4", 1)}
    assert [(x["target"], x["repeats"]) for x in donor] == [("T-All", 5), ("A", 5), ("V", 5)]


def test_donor_selection_is_model_seed_independent_and_identity_sensitive():
    # No model-seed argument exists: identical target identity must give the same donor.
    first = donor_index(9989, "MELD", "dia1", 3, "V", 1, 2)
    assert first == donor_index(9989, "MELD", "dia1", 3, "V", 1, 2)
    changed = {donor_index(9989, "MELD", "dia1", 3, "V", 1, repeat) for repeat in range(5)}
    assert len(changed) > 1


def test_target_specific_donor_missingness():
    zero, nonzero = np.zeros(3, np.float32), np.array([0, 1, 0], np.float32)
    assert donor_modality_missing("A", [zero])
    assert not donor_modality_missing("A", [nonzero])
    assert donor_modality_missing("V", [zero])
    assert donor_modality_missing("T-All", [zero, zero, zero, zero])
    assert not donor_modality_missing("T-All", [zero, zero, nonzero, zero])
    with pytest.raises(ValueError):
        donor_modality_missing("T-All", [zero])


def test_interpolation_endpoints_and_fp32():
    x = [np.array([1, 3], np.float32)]
    reference = [np.array([5, 7], np.float32)]
    assert np.array_equal(interpolate(x, reference, 0)[0], x[0])
    assert np.array_equal(interpolate(x, reference, 1)[0], reference[0])
    assert np.array_equal(interpolate(x, reference, .5)[0], np.array([3, 5], np.float32))
    assert interpolate(x, reference, .5)[0].dtype == np.float32


def test_behavior_metrics_are_label_free_and_repeat_aware():
    p = np.array([[.8, .2], [.4, .6]])
    q = np.array([[[.7, .3], [.2, .8]], [[.6, .4], [.45, .55]]])
    result = behavior(p, q)
    assert np.allclose(result["flip_rate"], [.5, .5])
    assert np.all(result["prediction_instability_mu"] > 0)
    assert np.all(result["absolute_confidence_change"] >= 0)
    assert np.all(result["js_divergence"] >= 0)


def test_stored_text_array_labels_are_only_order_identifiers():
    assert target_indices("T-All") == [0, 1, 2, 3]
    assert target_indices("T4") == [3]
    with pytest.raises(ValueError):
        target_indices("RoBERTa-Layer-4")


def test_serialized_validation_ids_resolve_to_native_integer_or_string_keys():
    meld = native_key_map([1039, 1040])
    iemocap = native_key_map(["Ses01", "Ses02"])
    assert meld["1039"] == 1039
    assert iemocap["Ses01"] == "Ses01"
    with pytest.raises(ValueError):
        native_key_map([1, "1"])


def test_donor_pool_preserves_native_dictionary_keys():
    class Dataset:
        keys = [0]
    dataset = Dataset()
    for name in ("roberta1", "roberta2", "roberta3", "roberta4", "videoVisual", "videoAudio"):
        setattr(dataset, name, {0: np.zeros((2, 3), np.float32)})
    collections, records = donor_pool(dataset)
    assert len(collections) == 6
    assert records == [(0, 0), (0, 1)]
