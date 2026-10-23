"""CPU-only statistics for full-input predictions; never selects a checkpoint.

Inputs to ``make_records`` must already be aligned, one row per valid utterance.
The official models emit log probabilities, so probabilities are obtained by
exponentiation, with no second softmax or renormalization.
"""

from collections import Counter
import csv
from pathlib import Path

import numpy as np


THRESHOLDS = (0.8, 0.9)
RECORD_FIELDS = (
    "dataset", "model", "dialogue_id", "utterance_index", "text", "true_label",
    "true_emotion", "predicted_label", "predicted_emotion", "max_probability",
    "true_class_probability", "is_error",
)


def _class_names(class_names):
    names = list(class_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Class names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    return names


def _integer_vector(values, name, size):
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if size and array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers, not booleans or floats")
    return array


def _index_records(records):
    indexed = {}
    for record in records:
        key = (str(record["dialogue_id"]), record["utterance_index"])
        if key in indexed:
            raise ValueError(f"Duplicate utterance key: {key!r}")
        indexed[key] = record
    return indexed


def make_records(dataset, model_name, labels, log_probabilities, dialogue_ids,
                 utterance_indices, texts, class_names):
    """Validate aligned model outputs and return JSON/CSV-friendly records.

    Probability normalization tolerance accommodates float32 log-softmax output;
    accepted probabilities are preserved exactly as ``exp(log_probabilities)``.
    Sample keys are unique after dialogue IDs are converted to strings.
    """
    names = _class_names(class_names)
    logp = np.asarray(log_probabilities, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != len(names):
        raise ValueError("Log probabilities must have shape (utterances, classes)")
    if not np.all(np.isfinite(logp)):
        raise ValueError("Log probabilities must be finite")
    with np.errstate(over="ignore", under="ignore"):
        probabilities = np.exp(logp)
    if (not np.all(np.isfinite(probabilities))
            or np.any(logp > 1e-6)
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)):
        raise ValueError("Exponentiated log probabilities must be normalized")
    size = logp.shape[0]
    labels = _integer_vector(labels, "labels", size)
    indices = _integer_vector(utterance_indices, "utterance_indices", size)
    if np.any(labels < 0) or np.any(labels >= len(names)):
        raise ValueError("Labels are outside the class range")
    if np.any(indices < 0):
        raise ValueError("Utterance indices must be zero based and nonnegative")
    if len(dialogue_ids) != size or len(texts) != size:
        raise ValueError("IDs and texts must match the number of utterances")
    if any(value is None for value in dialogue_ids):
        raise ValueError("Dialogue IDs cannot be missing")
    if any(not isinstance(value, str) for value in texts):
        raise ValueError("Original texts must be strings")

    records = []
    for i, row in enumerate(probabilities):
        true_label, predicted_label = int(labels[i]), int(np.argmax(row))
        record = {
            "dataset": str(dataset), "model": str(model_name),
            "dialogue_id": str(dialogue_ids[i]), "utterance_index": int(indices[i]),
            "text": texts[i], "true_label": true_label, "true_emotion": names[true_label],
            "predicted_label": predicted_label, "predicted_emotion": names[predicted_label],
            "max_probability": float(row[predicted_label]),
            "true_class_probability": float(row[true_label]),
            "is_error": bool(predicted_label != true_label),
        }
        record.update({f"probability_{name}": float(row[j]) for j, name in enumerate(names)})
        records.append(record)
    _index_records(records)
    return records


def _threshold_statistics(records, threshold):
    high = [row for row in records if row["max_probability"] >= threshold]
    errors = sum(bool(row["is_error"]) for row in high)
    return {
        "high_confidence_count": len(high),
        "high_confidence_error_count": errors,
        "error_rate_among_high_confidence": errors / len(high) if high else None,
    }


def summarize_records(records, class_names):
    """Summarize errors; high-confidence error fractions use ALL high predictions."""
    names = _class_names(class_names)
    records = list(records)
    _index_records(records)
    error_count = sum(bool(row["is_error"]) for row in records)
    by_class = {}
    for label, name in enumerate(names):
        rows = [row for row in records if row["true_label"] == label]
        errors = sum(bool(row["is_error"]) for row in rows)
        by_class[name] = {
            "true_label": label, "true_emotion": name, "num_samples": len(rows),
            "error_count": errors, "error_rate": errors / len(rows) if rows else None,
            "share_of_all_errors": errors / error_count if error_count else None,
            "high_confidence_errors_ge_0p8": sum(row["is_error"] and row["max_probability"] >= 0.8 for row in rows),
            "high_confidence_errors_ge_0p9": sum(row["is_error"] and row["max_probability"] >= 0.9 for row in rows),
        }
    pairs = Counter((row["true_emotion"], row["predicted_emotion"])
                    for row in records if row["is_error"])
    return {
        "num_samples": len(records), "error_count": error_count,
        "accuracy_percent": 100.0 * (len(records) - error_count) / len(records) if records else None,
        "thresholds": {str(t): _threshold_statistics(records, t) for t in THRESHOLDS},
        "class_error_distribution": by_class,
        "confusion_pairs": [
            {"true_emotion": true, "predicted_emotion": predicted, "count": count}
            for (true, predicted), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _write_csv(path, records, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(records)


def write_model_outputs(output_dir, records, class_names):
    """Write original-order predictions and confidence-ranked error subsets."""
    names = _class_names(class_names)
    records = list(records)
    summary = summarize_records(records, names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(RECORD_FIELDS) + [f"probability_{name}" for name in names]
    if any("source_utterance_id" in row for row in records):
        fields.insert(fields.index("utterance_index") + 1, "source_utterance_id")
    _write_csv(output_dir / "all_predictions.csv", records, fields)
    errors = sorted((row for row in records if row["is_error"]),
                    key=lambda row: (-row["max_probability"], row["dialogue_id"], row["utterance_index"]))
    _write_csv(output_dir / "errors_all.csv", errors, fields)
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        _write_csv(output_dir / f"errors_ge_{suffix}.csv",
                   (row for row in errors if row["max_probability"] >= threshold), fields)
    distribution = list(summary["class_error_distribution"].values())
    _write_csv(output_dir / "class_error_distribution.csv", distribution, distribution[0].keys())
    return summary


def compare_records(baseline, cmerc):
    """Strictly join identical test samples, preserving Baseline input order.

    Any missing/duplicate key, changed dataset, truth or text is an error, rather
    than silently dropping samples or comparing rows in unrelated batch order.
    """
    baseline = _index_records(baseline)
    cmerc = _index_records(cmerc)
    if baseline.keys() != cmerc.keys():
        missing = len(baseline.keys() - cmerc.keys())
        extra = len(cmerc.keys() - baseline.keys())
        raise ValueError(f"Test sample keys differ: {missing} missing and {extra} extra in CMERC")
    compared = []
    for key, before in baseline.items():
        after = cmerc[key]
        shared = ("dataset", "true_label", "true_emotion", "text")
        if "source_utterance_id" in before and "source_utterance_id" in after:
            shared += ("source_utterance_id",)
        if any(before[field] != after[field] for field in shared):
            raise ValueError(f"Test sample metadata differ at {key!r}")
        probabilities = {field for field in before if field.startswith("probability_")}
        if probabilities != {field for field in after if field.startswith("probability_")}:
            raise ValueError(f"Class probability schemas differ at {key!r}")
        row = {field: before[field] for field in ("dataset", "dialogue_id", "utterance_index", "text", "true_label", "true_emotion")}
        if "source_utterance_id" in shared:
            row["source_utterance_id"] = before["source_utterance_id"]
        model_fields = ("model", "predicted_label", "predicted_emotion", "max_probability", "true_class_probability", "is_error")
        for prefix, source in (("baseline", before), ("cmerc", after)):
            for field in model_fields + tuple(sorted(probabilities)):
                row[f"{prefix}_{field}"] = source[field]
        if before["is_error"] and not after["is_error"]:
            transition = "baseline_wrong_cmerc_correct"
        elif not before["is_error"] and after["is_error"]:
            transition = "baseline_correct_cmerc_wrong"
        else:
            transition = "both_wrong" if before["is_error"] else "both_correct"
        row["error_transition"] = transition
        compared.append(row)
    return compared
