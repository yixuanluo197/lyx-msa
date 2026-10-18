#!/usr/bin/env python3
"""Read completed paired runs; never train, infer, select, or overwrite outputs.

Input schema: schema_version=1, planned_seeds=[1937,2025,3407],
planned_datasets=["IEMOCAP","MELD"], known_test_exposure=[...], pairs=[{
dataset, seed, baseline_run, cmerc_run, baseline_verification,
cmerc_verification, evaluations: {validation: {baseline, cmerc},
test: {baseline, cmerc, validation_freeze}}}]. Paths are relative to the matrix.
Missing pairs/splits remain explicitly incomplete, never silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from evaluation_metrics import calibration_metrics
from scripts.round2_analysis import (
    POSITION_RULE, _load_evaluation, compare_evaluations, file_sha256,
    relative_position_masks, summarize_training,
)

SEEDS = (1937, 2025, 3407)
DATASETS = ("IEMOCAP", "MELD")
PRIMARY = ("weighted_f1", "ece", "cel.total")
T975_DF2 = 4.302652729696142


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/missing {name}")
    return float(value)


def signflip_pvalue(deltas):
    """Exact two-sided test of absolute paired mean over all 2**n sign flips.

    Includes the observed assignment and ties. No Monte Carlo correction or
    pseudoreplicating utterances as independently randomized seeds.
    """
    values = [finite(value, "paired delta") for value in deltas]
    require(0 < len(values) <= 3, "sign-flip test expects one to three frozen seeds")
    observed = abs(math.fsum(values))
    tolerance = 1e-12 * max(1., math.fsum(abs(value) for value in values))
    extreme = sum(abs(math.fsum(sign * value for sign, value in zip(signs, values)))
                  >= observed - tolerance
                  for signs in itertools.product((-1, 1), repeat=len(values)))
    return extreme / (2 ** len(values))


def holm_adjust(pvalues):
    """Holm adjustment; keys identify hypotheses, values must be finite p-values."""
    require(bool(pvalues), "Holm family is empty")
    for key, value in pvalues.items():
        require(0 <= finite(value, key) <= 1, "p-value outside [0,1]")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted, previous = {}, 0.
    for index, key in enumerate(ordered):
        previous = min(1., max(previous, (len(ordered) - index) * pvalues[key]))
        adjusted[key] = previous
    return adjusted


def describe(values):
    values = [finite(value, "statistic") for value in values]
    require(bool(values), "no observations")
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def paired_statistics(baseline, cmerc):
    require(len(baseline) == len(cmerc), "unpaired statistic")
    deltas = [finite(full, "cmerc") - finite(base, "baseline")
              for base, full in zip(baseline, cmerc)]
    result = {"baseline": describe(baseline), "cmerc": describe(cmerc), "delta": describe(deltas)}
    result["complete_three_seeds"] = len(deltas) == 3
    result["exact_two_sided_signflip_p"] = signflip_pvalue(deltas) if len(deltas) == 3 else None
    # No unplanned interim inferential tests when some seeds are missing.
    if len(deltas) == 3:
        margin = T975_DF2 * result["delta"]["sample_sd"] / math.sqrt(3)
        result["paired_t_95_interval_df2"] = [result["delta"]["mean"] - margin,
                                                result["delta"]["mean"] + margin]
    else:
        result["paired_t_95_interval_df2"] = None
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def training_evidence(run, verification_path, dataset, seed, variant):
    """Admit a complete run using immutable verifier hashes and raw trajectories."""
    manifest_path = run / "run_manifest.json"
    manifest = read_json(manifest_path)
    config = manifest["config"]
    require(config.get("Dataset") == dataset and config.get("seed") == seed,
            "run dataset/training seed differs from matrix")
    require(config.get("causal") is True and config.get("selection_protocol") == "validation",
            "formal run must be causal and validation selected")
    require(config.get("replacement_policy") == "prefix_mean_v1" and
            config.get("auxiliary_policy") == "legacy_epoch_snapshot", "run policy differs")
    require(all(config.get(key) is (variant == "cmerc")
                for key in ("courselearning", "contrastlearning", "calibrate")), "variant flags differ")
    require(dataset != "IEMOCAP" or config.get("split_seed") == 1475, "IEMOCAP split seed differs")
    summary = summarize_training(run / "metrics.jsonl")
    epochs = 80 if dataset == "IEMOCAP" else 40
    require(summary["completed_epochs"] == epochs, "formal training budget incomplete")
    verification = read_json(verification_path)
    require(verification.get("status") == "passed" and verification.get("variant") == variant,
            "run requires passed matching formal verification")
    for key, expected in (("completed_epochs", epochs), ("best_epoch", summary["best_epoch"]),
                          ("best_validation_weighted_f1", summary["best_validation_weighted_f1"]),
                          ("config_sha256", manifest["config_sha256"]),
                          ("run_manifest_sha256", file_sha256(manifest_path))):
        require(verification.get(key) == expected, f"run verification differs: {key}")
    for name in ("best", "latest"):
        require(file_sha256(run / f"{name}.pt") == verification.get(f"{name}_sha256"),
                f"{name} checkpoint changed after verification")
    initial = read_json(run / "initial_state.json").get("model_state_sha256")
    require(initial and initial == verification.get("initial_model_sha256"), "initial fingerprint differs")
    require(config.get("data_provenance") == verification.get("data_provenance") and
            bool(config.get("data_provenance")), "verified run data provenance differs")
    records = read_jsonl(run / "metrics.jsonl")
    batch_size = config["batch_size"]
    require(type(batch_size) is int and batch_size > 0, "invalid batch size")
    steps = sum(math.ceil(record["train_dialogues"] / batch_size) for record in records)
    # The verified production loader has drop_last=False and one update per batch.
    summary["optimizer_steps"] = steps
    summary["optimizer_steps_source"] = "sum(ceil(epoch.train_dialogues / batch_size)); production drop_last=False"
    if "optimizer_steps" in verification:
        require(verification["optimizer_steps"] == steps, "verified Adam steps differ from epoch records")
    summary["initial_model_sha256"] = initial
    summary["best_sha256"] = verification["best_sha256"]
    summary["latest_sha256"] = verification["latest_sha256"]
    summary["verification_sha256"] = file_sha256(verification_path)
    summary["run_manifest_sha256"] = file_sha256(manifest_path)
    summary["data_provenance"] = config["data_provenance"]
    summary["config"] = config
    diagnostics = read_jsonl(run / "training_diagnostics.jsonl")
    require(bool(diagnostics), "missing actual weighted-gradient diagnostics")
    first_epochs = [row["epoch"] for row in diagnostics if row.get("batch") == 0]
    require(first_epochs == list(range(epochs)), "first-batch gradient coverage differs from full epoch history")
    coefficients = {"classification_loss": 1., "cc_loss": config["rank_coff"],
                    "mscl_loss": config["mscl_coff"], "cscl_loss": config["cscl_coff"]}
    for epoch in summary["convergence"]:
        weighted_losses = {key: finite(epoch[key], key) * coefficient for key, coefficient in coefficients.items()}
        require(math.isclose(sum(weighted_losses.values()), finite(epoch["total_loss"], "total_loss"),
                             rel_tol=1e-6, abs_tol=1e-6), "weighted epoch losses do not reconstruct total")
        epoch.update({f"weighted_{key}": value for key, value in weighted_losses.items()})
    gradient_rows = []
    for row in diagnostics:
        require(row.get("finite_gradients") is True and row.get("finite_losses") is True,
                "nonfinite actual gradient/loss diagnostic")
        require(row.get("gradient_coefficients") == coefficients, "gradient coefficient differs from run")
        losses = row["losses"]
        weighted_losses = {key: finite(losses[key], key) * coefficient
                           for key, coefficient in coefficients.items()}
        require(math.isclose(sum(weighted_losses.values()), finite(losses["total_loss"], "total_loss"),
                             rel_tol=1e-6, abs_tol=1e-6), "weighted losses do not reconstruct total")
        weighted_norms = row["weighted_gradient_norms"]
        for key, norm in row["gradient_norms"].items():
            require(math.isclose(finite(weighted_norms[key], key),
                                 finite(norm, key) * abs(coefficients[key]), rel_tol=1e-6, abs_tol=1e-9),
                    "weighted gradient norm differs from coefficient times raw norm")
        gradient_rows.append({key: row.get(key) for key in
            ("epoch", "batch", "gradient_coefficients", "gradient_norms", "weighted_gradient_norms",
             "finite_gradients", "finite_losses", "main_aux_max_abs_diff", "main_training", "aux_training")}
            | {"raw_losses": losses, "weighted_losses": weighted_losses})
    summary["gradient_trajectory"] = gradient_rows
    summary["gradient_diagnostics_sha256"] = file_sha256(run / "training_diagnostics.jsonl")
    summary["gradient_sampling_scope"] = "recorded batches only; first batch of every epoch; no whole-epoch gradient claim"
    summary["weighted_gradient_norm_summary"] = {
        key: describe([row["weighted_gradient_norms"][key] for row in gradient_rows
                       if key in row["weighted_gradient_norms"]])
        for key in coefficients if any(key in row["weighted_gradient_norms"] for row in gradient_rows)}
    return summary


def add_success_metrics(comparison, baseline_dir, cmerc_dir):
    """Recompute confidence ranking metrics on the exact paired predictions."""
    evidence = [_load_evaluation(path) for path in (baseline_dir, cmerc_dir)]
    for segment in ("overall", "early", "middle", "late"):
        target = comparison["overall"] if segment == "overall" else comparison["position_thirds"][segment]["metrics"]
        if target is None:
            continue
        paired = []
        for metadata, arrays in evidence:
            selected = (np.ones(len(arrays["labels"]), dtype=bool) if segment == "overall" else
                        relative_position_masks(arrays["dialogue_ids"], arrays["utterance_indices"])[segment])
            calibration = calibration_metrics(arrays["labels"][selected],
                np.exp(arrays["log_probabilities"][selected]), n_bins=comparison["calibration_bins"])
            for name in ("auroc_success", "auprc_success"):
                if not math.isfinite(calibration[name]):
                    calibration[name] = None
            if segment == "overall":
                for name in ("auroc_success", "auprc_success"):
                    saved = metadata["calibration"][name]
                    current = calibration[name]
                    require((saved is None and current is None) or
                            (saved is not None and current is not None and
                             math.isclose(saved, current, rel_tol=1e-9, abs_tol=1e-7)),
                            f"saved success metric differs from predictions: {name}")
            paired.append(calibration)
        for metric in ("auroc_success", "auprc_success"):
            base, full = (item[metric] for item in paired)
            target[metric] = {"baseline": base, "cmerc": full,
                              "delta": None if base is None or full is None else full - base}


def diagnostics(comparison):
    overall = comparison["overall"]
    context_delta = overall["cel.intra_context"]["delta"] + overall["cel.inter_context"]["delta"]
    total_delta = overall["cel.total"]["delta"]
    return {"angry_f1": overall.get("f1.angry") or overall.get("f1.anger"),
            "context_cel_delta": context_delta,
            "total_cel_delta": total_delta,
            "context_share_of_net_cel_reduction_percent":
                100 * context_delta / total_delta if total_delta < 0 else None,
            "context_share_caution": "net ratio can exceed 100% if other views worsen; undefined without total reduction",
            "ece_delta": overall["ece"]["delta"],
            "auroc_success_delta": overall["auroc_success"]["delta"],
            "ece_improves_auroc_does_not": overall["ece"]["delta"] < 0 and
                overall["auroc_success"]["delta"] is not None and overall["auroc_success"]["delta"] <= 0}


def analyze_matrix(matrix_path):
    matrix_path = Path(matrix_path).resolve()
    matrix = read_json(matrix_path)
    require(matrix.get("schema_version") == 1, "unsupported matrix schema")
    require(matrix.get("planned_seeds") == list(SEEDS) and matrix.get("planned_datasets") == list(DATASETS),
            "matrix must retain all frozen datasets and seeds in declared order")
    exposure = matrix.get("known_test_exposure", [])
    require(any(row.get("dataset") == "IEMOCAP" and row.get("seed") == 1937 for row in exposure),
            "must disclose the previously observed IEMOCAP 1937 test result")
    resolve = lambda value: (matrix_path.parent / value).resolve()
    results, training, seen, rows, gradients, curves = [], {}, set(), [], [], []
    for pair in matrix.get("pairs", []):
        dataset, seed = pair["dataset"], pair["seed"]
        require(dataset in DATASETS and seed in SEEDS, "unplanned dataset/seed")
        require((dataset, seed) not in seen, "duplicate dataset/seed pair")
        seen.add((dataset, seed))
        runs = {variant: resolve(pair[f"{variant}_run"]) for variant in ("baseline", "cmerc")}
        evidence = {variant: training_evidence(runs[variant], resolve(pair[f"{variant}_verification"]),
                    dataset, seed, variant) for variant in runs}
        base, full = evidence["baseline"], evidence["cmerc"]
        require(base["initial_model_sha256"] == full["initial_model_sha256"], "paired initializations differ")
        require(base["data_provenance"] == full["data_provenance"], "paired training provenance differs")
        verification = read_json(resolve(pair["cmerc_verification"]))
        require(verification.get("difficulty", {}).get("teacher_best_sha256") == base["best_sha256"],
                "difficulty teacher differs from paired causal baseline best")
        key = f"{dataset}/{seed}"
        training[key] = evidence
        for variant, value in evidence.items():
            for epoch in value["convergence"]:
                curves.append({"dataset": dataset, "seed": seed, "variant": variant, **epoch})
            for gradient in value["gradient_trajectory"]:
                gradients.append({"dataset": dataset, "seed": seed, "variant": variant, **gradient})
        evaluations = pair.get("evaluations", {})
        require(set(evaluations) <= {"validation", "test"}, "unplanned evaluation split")
        for split, locations in evaluations.items():
            evaldirs = {variant: resolve(locations[variant]) for variant in runs}
            comparison = compare_evaluations(evaldirs["baseline"], evaldirs["cmerc"])
            require(comparison["dataset"] == dataset and comparison["split"] == split,
                    "evaluation dataset/split differs from matrix")
            for variant, evaldir in evaldirs.items():
                metadata = read_json(evaldir / "metrics.json")
                require(Path(metadata["checkpoint"]).resolve() == (runs[variant] / "best.pt").resolve(),
                        "evaluation does not identify the admitted best checkpoint")
                require(metadata["evaluation_data"]["source_data_provenance"] == evidence[variant]["data_provenance"],
                        "evaluation provenance differs from admitted run")
            if split == "test":
                require("validation_freeze" in locations, "test analysis requires frozen validation comparison")
                frozen_path = resolve(locations["validation_freeze"])
                frozen = read_json(frozen_path)
                require(frozen.get("dataset") == dataset and frozen.get("split") == "validation" and
                        frozen.get("replacement_policy") == "prefix_mean_v1", "invalid validation freeze")
                require("validation" in evaluations, "test pair must also identify its validation evaluation")
                for variant in runs:
                    directory = resolve(evaluations["validation"][variant])
                    for filename in ("metrics.json", "predictions.npz"):
                        require(frozen["input_sha256"][variant][filename] == file_sha256(directory / filename),
                                "validation artifacts differ from freeze")
                comparison["validation_freeze_sha256"] = file_sha256(frozen_path)
            add_success_metrics(comparison, evaldirs["baseline"], evaldirs["cmerc"])
            comparison.update(seed=seed, diagnostics=diagnostics(comparison))
            results.append(comparison)
            segments = {"overall": comparison["overall"]} | {
                name: segment["metrics"] for name, segment in comparison["position_thirds"].items()}
            for segment, metrics in segments.items():
                for metric, values in (metrics or {}).items():
                    rows.append({"dataset": dataset, "seed": seed, "split": split,
                                 "segment": segment, "metric": metric, **values})
            for metric in ("elapsed_seconds", "max_gpu_memory_mb", "optimizer_steps"):
                rows.append({"dataset": dataset, "seed": seed, "split": split, "segment": "training",
                             "metric": metric, "baseline": base[metric], "cmerc": full[metric],
                             "delta": full[metric] - base[metric]})
    # A dataset's fixed split/provenance and backbone must remain the same across seeds.
    for dataset in DATASETS:
        evidence = [training[f"{dataset}/{seed}"]["baseline"] for seed in SEEDS if f"{dataset}/{seed}" in training]
        if evidence:
            anchor = dict(evidence[0]["config"])
            for row in evidence[1:]:
                current = dict(row["config"])
                for config in (anchor, current):
                    config.pop("seed", None)
                require(anchor == current, "backbone/protocol/data differ between seeds")
    aggregated = {}
    for row in rows:
        group = (row["dataset"], row["split"], row["segment"], row["metric"])
        aggregated.setdefault(group, []).append(row)
    statistics_rows = []
    for key, members in sorted(aggregated.items()):
        members.sort(key=lambda item: SEEDS.index(item["seed"]))
        finite_members = [item for item in members if item["baseline"] is not None and item["cmerc"] is not None]
        if not finite_members:
            continue
        statistics_rows.append(dict(zip(("dataset", "split", "segment", "metric"), key)) |
            {"seeds": [row["seed"] for row in finite_members],
             "undefined_seeds": [row["seed"] for row in members if row not in finite_members],
             **paired_statistics([row["baseline"] for row in finite_members], [row["cmerc"] for row in finite_members])})
        if key[2] != "overall" or key[3] not in PRIMARY:
            statistics_rows[-1]["exact_two_sided_signflip_p"] = None
    families = {}
    for split in ("validation", "test"):
        hypotheses = [row for row in statistics_rows if row["split"] == split and
                      row["segment"] == "overall" and row["metric"] in PRIMARY]
        complete = len(hypotheses) == 6 and all(row["seeds"] == list(SEEDS) for row in hypotheses)
        adjusted = holm_adjust({f"{row['dataset']}/{row['metric']}": row["exact_two_sided_signflip_p"]
                               for row in hypotheses}) if complete else {}
        families[split] = {"status": "complete" if complete else "incomplete_no_holm_test",
                           "family_size": 6, "holm_adjusted_p": adjusted,
                           "role": "primary" if split == "validation" else "replication"}
        for row in hypotheses:
            row["holm_adjusted_p"] = adjusted.get(f"{row['dataset']}/{row['metric']}")
    completed = {(row["dataset"], row["seed"], row["split"]) for row in results}
    missing = [{"dataset": dataset, "seed": seed, "split": split}
               for dataset in DATASETS for seed in SEEDS for split in ("validation", "test")
               if (dataset, seed, split) not in completed]
    return {"schema_version": 1, "matrix_sha256": file_sha256(matrix_path),
        "analysis_source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in
            (Path(__file__).resolve(), ROOT / "scripts/round2_analysis.py", ROOT / "evaluation_metrics.py")},
        "status": "complete" if not missing else "incomplete", "missing_evaluations": missing,
        "known_test_exposure": exposure, "position_rule": POSITION_RULE,
        "replacement_policy": "prefix_mean_v1",
        "replacement_semantics": "prefix smoothing view, not complete modality removal",
        "statistical_protocol": {"unit": "paired training seed; fixed data split",
            "delta": "CMERC minus Baseline; percentages differ in percentage points; CEL summed units",
            "sd": "sample standard deviation (ddof=1)", "interval": "paired t 95%, df=2; normality-dependent descriptive interval",
            "test": "exact two-sided sign-flip test on paired mean, all 8 flips; null sign exchangeability required",
            "minimum_n3_two_sided_p": 0.25, "primary_hypotheses": "dataset x (W-F1, ECE, CEL total), Holm across 6 per split",
            "split_roles": "validation primary; test replication, separate families",
            "other_metrics": "descriptive; no multiplicity-adjusted significance claims",
            "caution": "three seeds cannot establish robust significance; no utterance-level pseudoreplication"},
        "primary_families": families, "per_seed": results, "training": training,
        "statistics": statistics_rows}, rows, gradients, curves


def markdown_report(result):
    lines = ["# Third-round paired statistics", "", f"Status: {result['status']}.", "",
        "Validation is the primary analysis; test is a separately reported replication family.",
        "IEMOCAP seed 1937 test results were already observed in round 2; this is not a wholly blind study.",
        "Three paired training seeds on a fixed split give uncertain estimates. The exact two-sided sign-flip test",
        "has only 8 assignments and minimum p = 0.25; no robust significance claim is supported.",
        "95% paired t intervals use df = 2 and a normality assumption; all other metrics are descriptive.",
        "`prefix_mean_v1` is prefix smoothing, not complete modality removal. CEL uses summed log-probability differences.", "",
        "| Dataset | Split | Metric | n | Baseline mean ± SD | CMERC mean ± SD | Paired delta ± SD | 95% t interval | Exact p | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
    def number(value):
        return "NA" if value is None else f"{value:.6f}"
    for row in result["statistics"]:
        if row["segment"] != "overall" or row["metric"] not in (*PRIMARY, "auroc_success", "macro_f1", "accuracy"):
            continue
        mean_sd = lambda name: f"{number(row[name]['mean'])} ± {number(row[name]['sample_sd'])}"
        interval = row["paired_t_95_interval_df2"]
        interval_text = "NA" if interval is None else f"[{number(interval[0])}, {number(interval[1])}]"
        lines.append(f"| {row['dataset']} | {row['split']} | {row['metric']} | {row['delta']['n']} | "
                     f"{mean_sd('baseline')} | {mean_sd('cmerc')} | {mean_sd('delta')} | {interval_text} | "
                     f"{number(row['exact_two_sided_signflip_p'])} | {number(row.get('holm_adjusted_p'))} |")
    lines += ["", "## All-seed diagnostic findings", "",
        "Context CEL share is the combined intra/inter GNN-edge-ablation delta divided by net CEL delta, only when net CEL falls.",
        "These ablations remove selected GNN temporal edges; encoders and hypergraphs may retain speaker history.",
        "Share can exceed 100% when modality views worsen; it does not prove a causal mechanism.", ""]
    for row in result["per_seed"]:
        diag = row["diagnostics"]
        angry = diag["angry_f1"]
        lines.append(f"- {row['dataset']} seed {row['seed']} {row['split']}: angry F1 delta "
            f"{number(angry['delta'] if angry else None)}; ECE delta {number(diag['ece_delta'])}; "
            f"AUROC-Success delta {number(diag['auroc_success_delta'])}; context share of net CEL reduction "
            f"{number(diag['context_share_of_net_cel_reduction_percent'])}%.")
    lines += ["", "Full class and position metrics, all epoch loss/convergence/resource trajectories, and every recorded raw/weighted gradient norm",
              "are retained in JSON/CSV. Gradient samples cover the recorded batches (first batch each epoch), not the entire epoch.",
              "Per-component norm magnitudes do not add to a total gradient norm; directions may reinforce or cancel.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result, rows, gradients, curves = analyze_matrix(args.matrix)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write_json("paired_statistics.json", result)
    write_json("per_seed_metrics.json", rows)
    write_json("loss_gradient_trajectories.json", gradients)
    write_json("epoch_trajectories.json", curves)
    for name, records in (("per_seed_metrics.csv", rows), ("epoch_trajectories.csv", curves)):
        if records:
            with (args.output_dir / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in records for key in row)))
                writer.writeheader()
                writer.writerows(records)
    (args.output_dir / "MULTISEED_PAIRED_RESULTS.md").write_text(markdown_report(result))
    print(json.dumps({"status": result["status"], "completed_evaluations": len(result["per_seed"]),
                      "statistics_sha256": file_sha256(args.output_dir / "paired_statistics.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
