import copy
import math

import numpy as np
import pytest

from scripts.analyze_official_reliability import (
    error_distribution, generate_dataset_tables, high_confidence_counts, validate_arrays, write_csv,
)


def record(confidence, true=1):
    return dict(dataset='test', model='Official CMERC', dialogue_id=str(confidence), utterance_index=0,
        source_utterance_id='raw', utterance_text='text', true_label=true,
        true_emotion=['a', 'b'][true], predicted_label=0, predicted_emotion='a',
        predicted_confidence=confidence, true_label_probability=[confidence, 1-confidence][true],
        input_condition='full', probability_a=confidence, probability_b=1-confidence,
        correct=true == 0, true_label_log_probability=math.log([confidence, 1-confidence][true])
        if [confidence, 1-confidence][true] else -math.inf)


def test_confidence_boundaries_partition_without_overlap():
    rows = [record(v) for v in [.5, .6, .7, .8, .9, .95, 1]]
    bins = error_distribution(rows)
    assert [b['error_count'] for b in bins] == [0, 1, 1, 1, 1, 1, 2]
    assert sum(b['error_count'] for b in bins) == len(rows)
    assert sum(b['ratio_of_all_errors'] for b in bins) == pytest.approx(1)


def test_high_confidence_error_denominator_includes_correct_predictions():
    rows = [record(.9), record(.95, true=0), record(.6)]
    counts = high_confidence_counts(rows)
    assert counts['errors_ge_0p9'] == 1
    assert counts['high_confidence_predictions_ge_0p9'] == 2
    assert counts['high_confidence_error_rate_ge_0p9'] == .5
    assert high_confidence_counts([record(.6)])['high_confidence_error_rate_ge_0p8'] is None


def test_saved_array_order_and_duplicate_ids_are_rejected():
    rows = [record(.8), record(.9, true=0)]
    p = np.array([[.8, 1-.8], [.9, 1-.9]])
    validate_arrays(rows, p, np.log(p), np.array([1, 0]), np.array([0, 0]))
    with pytest.raises(ValueError, match='order'):
        validate_arrays(rows, p, np.log(p), np.array([0, 1]), np.array([0, 0]))
    duplicate = copy.deepcopy(rows)
    duplicate[1]['dialogue_id'] = duplicate[0]['dialogue_id']
    with pytest.raises(ValueError, match='Duplicate'):
        validate_arrays(duplicate, p, np.log(p), np.array([1, 0]), np.array([0, 0]))


def test_nll_brier_and_empty_calibration_bins_keep_established_definitions():
    rows = [record(.8), record(.9, true=0)]
    metrics = dict(class_names=['a', 'b'], accuracy=50, weighted_f1=100/3, macro_f1=100/3)
    summary, tables = generate_dataset_tables(rows, metrics)
    assert summary['nll'] == pytest.approx(-sum(math.log(p) for p in [.2, .9])/2)
    assert summary['brier_score'] == pytest.approx((2*.8**2 + 2*.1**2)/2)
    assert [b['count'] for b in tables['calibration_bins']] == [0]*8+[1, 1]
    assert tables['calibration_bins'][0]['mean_confidence'] is None
    assert len(tables['high_confidence_errors']) == 1


def test_no_high_confidence_errors_still_emits_schema_and_undefined_rates(tmp_path):
    rows = [record(.6), record(.7, true=0)]
    metrics = dict(class_names=['a', 'b'], accuracy=50, weighted_f1=100/3, macro_f1=100/3)
    summary, tables = generate_dataset_tables(rows, metrics)
    assert tables['high_confidence_errors'] == []
    assert summary['errors_ge_0p9'] == 0
    assert summary['high_confidence_error_rate_ge_0p9'] is None
    output = tmp_path/'empty.csv'
    write_csv(output, [], fields=['dataset', 'threshold'])
    assert output.read_text(encoding='utf-8-sig').strip() == 'dataset,threshold'


def test_original_log_score_mismatch_is_rejected():
    rows = [record(.8), record(.9, true=0)]
    rows[0]['true_label_log_probability'] = -10
    metrics = dict(class_names=['a', 'b'], accuracy=50, weighted_f1=100/3, macro_f1=100/3)
    with pytest.raises(ValueError, match='NLL'):
        generate_dataset_tables(rows, metrics)
