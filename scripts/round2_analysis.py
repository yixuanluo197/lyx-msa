#!/usr/bin/env python3
"""Compare paired causal evaluations without exporting dialogue IDs or paths.

This script reads existing evaluator outputs; it never runs a model or selects a
checkpoint. Classification/calibration differences use percentage points. CEL
retains the evaluator's summed positive log-probability difference scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import calibration_metrics, confidence_enhancement_level
from scripts.evaluate_checkpoint import classification_metrics, dataset_config, validate_log_probabilities

VIEWS = ("text", "audio", "visual", "intra_context", "inter_context")
POSITION_RULE = {
    "relative_position": "(zero_based_utterance_index + 0.5) / dialogue_length",
    "early": "[0, 1/3)", "middle": "[1/3, 2/3)", "late": "[2/3, 1]",
    "length_one": "middle",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_position_masks(dialogue_ids, utterance_indices):
    """Require complete dialogues, then assign each utterance once to a third."""
    ids, indices = np.asarray(dialogue_ids), np.asarray(utterance_indices)
    if ids.ndim != 1 or indices.shape != ids.shape or ids.size == 0:
        raise ValueError("dialogue IDs and utterance indices must be nonempty matching vectors")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("utterance indices must be integers")
    positions = np.empty(ids.size, dtype=np.float64)
    for dialogue_id in np.unique(ids):
        selected = ids == dialogue_id
        length = int(selected.sum())
        if not np.array_equal(np.sort(indices[selected]), np.arange(length)):
            raise ValueError("each dialogue must have unique contiguous indices starting at zero")
        positions[selected] = (indices[selected] + .5) / length
    return {"early": positions < 1 / 3,
            "middle": (positions >= 1 / 3) & (positions < 2 / 3),
            "late": positions >= 2 / 3}


def _load_evaluation(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metrics.json").read_text())
    keys = ("labels", "log_probabilities", "dialogue_ids", "utterance_indices")
    keys += tuple(f"removed_{view}_log_probabilities" for view in VIEWS)
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in keys}
    labels = arrays["labels"]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer vector")
    classes = dataset_config(metadata["dataset"]).class_names
    shape = (labels.size, len(classes))
    for key in ("log_probabilities",) + tuple(f"removed_{view}_log_probabilities" for view in VIEWS):
        if arrays[key].shape != shape:
            raise ValueError("prediction arrays do not match dataset classes/sample count")
        validate_log_probabilities(arrays[key])
    if arrays["dialogue_ids"].shape != labels.shape:
        raise ValueError("dialogue IDs do not match labels")
    relative_position_masks(arrays["dialogue_ids"], arrays["utterance_indices"])
    return metadata, arrays


def _flatten_metrics(arrays, class_names, n_bins, selected=None, include_cel=True):
    selected = np.ones(arrays["labels"].size, dtype=bool) if selected is None else selected
    labels = arrays["labels"][selected]
    if labels.size == 0:
        return None
    full = arrays["log_probabilities"][selected]
    probabilities = np.exp(full)
    classification = classification_metrics(labels, full.argmax(axis=1), class_names)
    calibration = calibration_metrics(labels, probabilities, n_bins=n_bins)
    flat = {key: classification[key] for key in ("accuracy", "weighted_f1", "macro_f1")}
    flat.update({f"f1.{item['class_name']}": item["f1"] for item in classification["per_class"]})
    flat.update({key: calibration[key] for key in ("ece", "mce", "rmsce")})
    if include_cel:
        cel = confidence_enhancement_level(labels, full,
            {view: arrays[f"removed_{view}_log_probabilities"][selected] for view in VIEWS})
        flat["cel.total"] = cel["total"]
        flat.update({f"cel.{view}": cel["by_removal"][view] for view in VIEWS})
    return flat


def _paired_values(baseline, cmerc):
    if baseline is None or cmerc is None:
        return None
    return {key: {"baseline": float(value), "cmerc": float(cmerc[key]),
                  "delta": float(cmerc[key] - value)} for key, value in baseline.items()}


def compare_evaluations(baseline_dir, cmerc_dir):
    """Recompute metrics, enforce paired membership/protocol, export aggregates."""
    base_meta, baseline = _load_evaluation(baseline_dir)
    full_meta, cmerc = _load_evaluation(cmerc_dir)
    fields = ("dataset", "split", "split_seed", "valid_fraction", "replacement_policy", "architecture")
    for field in fields:
        if field not in base_meta or field not in full_meta or base_meta[field] != full_meta[field]:
            raise ValueError(f"paired evaluation protocol differs or is missing: {field}")
    if base_meta.get("causal") is not True or full_meta.get("causal") is not True:
        raise ValueError("both evaluations must be causal")
    if base_meta["replacement_policy"] != "prefix_mean_v1":
        raise ValueError("round 2 analysis requires prefix_mean_v1")
    if base_meta["split"] not in ("validation", "test"):
        raise ValueError("round 2 analysis requires validation or test")
    provenance = base_meta.get("evaluation_data", {}).get("source_data_provenance")
    full_provenance = full_meta.get("evaluation_data", {}).get("source_data_provenance")
    if not provenance or provenance != full_provenance:
        raise ValueError("paired evaluations require identical source data provenance")
    if base_meta["split"] == "validation":
        for metadata in (base_meta, full_meta):
            if metadata.get("evaluation_data", {}).get("verification") != "checkpoint_provenance_verified":
                raise ValueError("validation requires verified checkpoint provenance")
    n_bins = base_meta.get("calibration", {}).get("n_bins")
    if type(n_bins) is not int or n_bins <= 0 or n_bins != full_meta.get("calibration", {}).get("n_bins"):
        raise ValueError("paired calibration bin definitions differ or are missing")
    # Exact order is part of this frozen evaluator contract; never compare
    # differing dialogue membership merely because the array lengths match.
    for key in ("labels", "dialogue_ids", "utterance_indices"):
        if not np.array_equal(baseline[key], cmerc[key]):
            raise ValueError(f"paired evaluation membership/order differs: {key}")
    classes = dataset_config(base_meta["dataset"]).class_names
    positions = relative_position_masks(baseline["dialogue_ids"], baseline["utterance_indices"])
    result = {"schema_version": 1, "dataset": base_meta["dataset"], "split": base_meta["split"],
        "split_seed": base_meta["split_seed"], "replacement_policy": "prefix_mean_v1",
        "replacement_semantics": "prefix smoothing view; not complete modality removal",
        "delta_definition": "CMERC - Baseline (absolute difference; classification/calibration in percentage points)",
        "cel_scale": "sum of positive true-class log-probability differences",
        "num_samples": int(baseline["labels"].size), "num_dialogues": int(np.unique(baseline["dialogue_ids"]).size),
        "calibration_bins": n_bins, "position_rule": POSITION_RULE,
        "input_sha256": {name: {filename: file_sha256(Path(directory) / filename)
            for filename in ("metrics.json", "predictions.npz")} for name, directory in
            (("baseline", baseline_dir), ("cmerc", cmerc_dir))},
        "overall": _paired_values(_flatten_metrics(baseline, classes, n_bins),
                                    _flatten_metrics(cmerc, classes, n_bins)),
        "position_thirds": {}}
    for name, selected in positions.items():
        result["position_thirds"][name] = {"num_samples": int(selected.sum()),
            "metrics": _paired_values(_flatten_metrics(baseline, classes, n_bins, selected, False),
                                       _flatten_metrics(cmerc, classes, n_bins, selected, False))}
    return result


def summarize_training(path):
    """Preserve epoch curves and aggregate elapsed time and peak allocated memory."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records or [record["epoch"] for record in records] != list(range(1, len(records) + 1)):
        raise ValueError("training epochs must be contiguous starting at one")
    if any(record.get("selection_split") != "validation" for record in records):
        raise ValueError("formal runs must select checkpoints using validation")
    keys = ("epoch", "train_accuracy", "train_weighted_f1", "validation_accuracy", "validation_weighted_f1",
        "selection_weighted_f1", "classification_loss", "cc_loss", "mscl_loss", "cscl_loss", "total_loss",
        "elapsed_seconds", "max_gpu_memory_mb", "sample_ratio", "train_dialogues")
    for record in records:
        for key in keys:
            if key in record and not math.isfinite(float(record[key])):
                raise ValueError(f"training record contains nonfinite {key}")
    best = max(records, key=lambda item: item["selection_weighted_f1"])
    return {"completed_epochs": len(records), "best_epoch": best["epoch"],
        "best_validation_weighted_f1": best["selection_weighted_f1"],
        "elapsed_seconds": sum(record["elapsed_seconds"] for record in records),
        "max_gpu_memory_mb": max(record["max_gpu_memory_mb"] for record in records),
        "metrics_sha256": file_sha256(path),
        "convergence": [{key: record[key] for key in keys if key in record} for record in records]}


def markdown_report(result):
    lines = [f"# {result['dataset']} {result['split']}: Causal Baseline vs CMERC", "",
        f"Samples: {result['num_samples']}; dialogues: {result['num_dialogues']}. Delta = CMERC − Baseline.", "",
        "Classification and calibration use percentages; their deltas are percentage points. CEL uses summed log-probability differences.",
        "`prefix_mean_v1` is a prefix smoothing view, not complete modality removal.", ""]
    def table(metrics):
        if metrics is None:
            return ["No utterances in this preregistered segment.", ""]
        return ["| Metric | Baseline | CMERC | Delta |", "|---|---:|---:|---:|"] + [
            f"| {name} | {values['baseline']:.6f} | {values['cmerc']:.6f} | {values['delta']:+.6f} |"
            for name, values in metrics.items()] + [""]
    lines += table(result["overall"])
    lines += ["## Preregistered relative-position thirds", "",
        "Position = (zero-based utterance index + 0.5) / complete dialogue length. Early [0, 1/3), middle [1/3, 2/3), late [2/3, 1]. Length-one dialogues belong to middle.", ""]
    for name, segment in result["position_thirds"].items():
        lines += [f"### {name} ({segment['num_samples']} utterances)", ""] + table(segment["metrics"])
    if "training" in result:
        lines += ["## Training resource summary", "", "| Run | Epochs | Best epoch | Validation W-F1 | Seconds | Peak allocated GPU MB |", "|---|---:|---:|---:|---:|---:|"]
        for name, run in result["training"].items():
            lines.append(f"| {name} | {run['completed_epochs']} | {run['best_epoch']} | {run['best_validation_weighted_f1']:.6f} | {run['elapsed_seconds']:.2f} | {run['max_gpu_memory_mb']:.2f} |")
        lines += ["", "Resource differences (CMERC − Baseline):", ""]
        lines += table(result["training_comparison"])
        lines += ["Full per-epoch loss and convergence curves are retained in the companion JSON. Recorded loss components are raw terms; coefficient-weighted optimization contributions must be read from the training diagnostics.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--cmerc-eval", type=Path, required=True)
    parser.add_argument("--baseline-training", type=Path)
    parser.add_argument("--cmerc-training", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-freeze", type=Path,
        help="previously frozen validation comparison JSON (required for test reporting)")
    args = parser.parse_args(argv)
    result = compare_evaluations(args.baseline_eval, args.cmerc_eval)
    if result["split"] == "test":
        if args.validation_freeze is None:
            raise ValueError("test reporting requires --validation-freeze")
        freeze = json.loads(args.validation_freeze.read_text())
        for key in ("dataset", "split_seed", "replacement_policy"):
            if freeze.get(key) != result[key]:
                raise ValueError(f"validation freeze differs: {key}")
        if freeze.get("split") != "validation":
            raise ValueError("validation freeze must describe validation results")
        result["validation_freeze_sha256"] = file_sha256(args.validation_freeze)
    if bool(args.baseline_training) != bool(args.cmerc_training):
        raise ValueError("supply both training metrics files or neither")
    if args.baseline_training:
        result["training"] = {"baseline": summarize_training(args.baseline_training),
                              "cmerc": summarize_training(args.cmerc_training)}
        result["training_comparison"] = _paired_values(
            {key: result["training"]["baseline"][key] for key in ("elapsed_seconds", "max_gpu_memory_mb")},
            {key: result["training"]["cmerc"][key] for key in ("elapsed_seconds", "max_gpu_memory_mb")})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "comparison.md").write_text(markdown_report(result))
    print(json.dumps({"split": result["split"], "num_samples": result["num_samples"],
                      "comparison_sha256": file_sha256(args.output_dir / "comparison.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
