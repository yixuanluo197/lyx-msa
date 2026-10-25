"""Read-only, CPU calibration metrics for fixed full-input predictions.

ECE/MCE use maximum predicted class probability and prediction correctness.
All reported accuracies, gaps, and weights are fractions. The multiclass Brier
score sums squared errors across classes before averaging across utterances;
it is not divided by the class count or by two. NLL uses the natural logarithm
of the original true-class probability, without clipping or renormalization.
"""
import bisect
import math

from scripts.high_confidence_followup_stats import _validate


def _bin_index(confidence, n_bins):
    """Left-closed bins with mathematically explicit edges; include p=1 last.

    Explicit i/n boundaries avoid accumulating a floating-point step that could
    put exactly 0.3 into the bin ending at 0.30000000000000004.
    """
    return min(bisect.bisect_right([i / n_bins for i in range(n_bins + 1)], confidence) - 1,
               n_bins - 1)


def summarize_calibration(records, n_bins=10):
    """Return (summary_dict, bin_rows) for one nonempty dataset/model group.

    ``records`` are canonical rows from ``load_full_records``. Validation reuses
    the existing full-input metadata/class/probability checks, also rejecting
    non-finite scalar confidence fields and empty IDs. Nothing in records is
    mutated. Empty input is rejected because dataset/model identity is absent.

    Bins are [i/n_bins, (i+1)/n_bins), with the final upper endpoint inclusive.
    Empty bins have count, weight, and ECE contribution zero; their confidence,
    accuracy, and gaps are None. If any true probability is exactly zero, NLL is
    positive infinity and zero_true_probability_count records the affected rows.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError('n_bins must be a positive integer')
    rows = list(records)
    if not rows:
        raise ValueError('Calibration requires a nonempty dataset/model group')
    try:
        for row in rows:
            for field in ('dataset', 'model', 'dialogue_id'):
                if not isinstance(row[field], str) or not row[field]:
                    raise ValueError(f'{field} must be a nonempty string')
            for field in ('predicted_confidence', 'true_label_probability'):
                if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                    raise ValueError(f'{field} must be a finite probability')
        dataset, model = rows[0]['dataset'], rows[0]['model']
        _validate(rows, (dataset, model), 'full')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f'Invalid canonical prediction record: {exc}') from exc

    fields = [field for field in rows[0] if field.startswith('probability_')]
    groups = [[] for _ in range(n_bins)]
    brier_values, nll_values, confidences = [], [], []
    zero_count = 0
    for row in rows:
        probabilities = [row[field] for field in fields]
        confidence = probabilities[row['predicted_label']]
        confidences.append(confidence)
        groups[_bin_index(confidence, n_bins)].append((confidence, row['correct']))
        brier_values.append(math.fsum((probability - int(index == row['true_label'])) ** 2
                                      for index, probability in enumerate(probabilities)))
        true_probability = probabilities[row['true_label']]
        if true_probability == 0:
            zero_count += 1
        else:
            nll_values.append(-math.log(true_probability))

    total = len(rows)
    bin_rows = []
    for index, group in enumerate(groups):
        count = len(group)
        bin_confidence = math.fsum(p for p, _ in group) / count if count else None
        accuracy = sum(correct for _, correct in group) / count if count else None
        gap = bin_confidence - accuracy if count else None
        absolute_gap = abs(gap) if count else None
        weight = count / total
        bin_rows.append(dict(dataset=dataset, model=model, bin_index=index,
                             lower_bound=index / n_bins, upper_bound=(index + 1) / n_bins,
                             count=count, mean_confidence=bin_confidence, accuracy=accuracy,
                             signed_gap=gap, absolute_gap=absolute_gap, weight=weight,
                             ece_contribution=weight * absolute_gap if count else 0.0))
    summary = dict(dataset=dataset, model=model, input_condition='full', valid_utterances=total,
                   accuracy=sum(row['correct'] for row in rows) / total,
                   mean_confidence=math.fsum(confidences) / total,
                   ece=math.fsum(row['ece_contribution'] for row in bin_rows),
                   mce=max(row['absolute_gap'] for row in bin_rows if row['count']),
                   brier_score=math.fsum(brier_values) / total,
                   nll=math.inf if zero_count else math.fsum(nll_values) / total,
                   n_bins=n_bins, zero_true_probability_count=zero_count)
    return summary, bin_rows
