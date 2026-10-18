#!/usr/bin/env python3
"""Render public-safe third-round reports from existing aggregate statistics.

No training, inference, checkpoint deserialization, or checkpoint selection.
Paths in ANALYSIS_MATRIX are relative to that file. Optional per-pair
baseline_optimizer/cmerc_optimizer identify actual Adam-step audit JSON; otherwise
verification/{variant}_optimizer.json beside the pair's runs/ is discovered.
Incomplete matrices remain explicitly incomplete. Output directories are exclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics


DATASETS = ("IEMOCAP", "MELD")
SEEDS = (1937, 2025, 3407)
CALIBRATION = ("ece", "mce", "rmsce", "auroc_success", "auprc_success")
CLASSIFICATION = ("accuracy", "weighted_f1", "macro_f1")
CEL = ("cel.total", "cel.text", "cel.audio", "cel.visual", "cel.intra_context", "cel.inter_context")
LOSSES = ("classification_loss", "cc_loss", "mscl_loss", "cscl_loss")
COPIES = ("paired_statistics.json", "per_seed_metrics.json", "per_seed_metrics.csv", "epoch_trajectories.json",
          "epoch_trajectories.csv", "loss_gradient_trajectories.json", "MULTISEED_PAIRED_RESULTS.md")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_metadata(value):
    """Keep numeric aggregates/hashes, redact nested paths and private vectors.

    Original statistics and manifests stay untouched in the private directory;
    REPORT_EVIDENCE records their SHA256. Publication is a separate artifact.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("dialogue_ids", "train_ids", "validation_ids", "test_ids", "labels",
                       "predictions", "log_probabilities", "utterance_indices"):
                result[key] = "[private values retained locally]"
                continue
            public_key = ("private_artifact_" + hashlib.sha256(key.encode()).hexdigest()[:12]
                          if key.startswith("/") else key)
            result[public_key] = public_metadata(item)
        return result
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, str):
        if value.startswith("/") or re.search(r"(?:^|\s)/\S+", value):
            return "[private path retained locally]"
        return re.sub(r"/(?:data2|home|tmp|opt)/[^\s\"']+", "[private path retained locally]", value)
    return value


def validate_configurations(stats):
    """Compare both student and teacher protocols across every completed seed.

    Per-seed difficulty identity must differ, so omit that artifact identity and
    seed only. Source architecture/selection/policy remain in the separately
    compared causal-baseline teacher config. Frozen statistics already admitted
    each difficulty against that exact baseline best and data provenance.
    """
    result = {}
    for dataset in DATASETS:
        for variant in ("baseline", "cmerc"):
            anchor = None
            seeds = []
            for seed in SEEDS:
                run = stats["training"].get(f"{dataset}/{seed}", {}).get(variant)
                if run is None:
                    continue
                config = dict(run["config"])
                config.pop("seed", None)
                config.pop("difficulty_artifact", None)
                if anchor is None:
                    anchor = config
                require(config == anchor, f"{dataset} {variant} protocols differ across seeds")
                seeds.append(seed)
            result[f"{dataset}/{variant}"] = {"seeds": seeds,
                "status": "identical_across_available_seeds" if seeds else "no_completed_runs",
                "excluded_identity_fields": ["seed", "difficulty_artifact"],
                "teacher_architecture_selection_policy": "checked in baseline configuration and frozen provenance admission"}
    return result


def new_iemocap_sensitivity(stats):
    """Prespecified descriptive sensitivity; never replace the three-seed table."""
    rows = []
    metrics = (*CLASSIFICATION, *CALIBRATION, *CEL, "f1.angry")
    for split in ("validation", "test"):
        entries = [entry for entry in stats["per_seed"] if entry["dataset"] == "IEMOCAP"
                   and entry["split"] == split and entry["seed"] in (2025, 3407)]
        for metric in metrics:
            values = [entry["overall"][metric] for entry in entries if metric in entry["overall"]
                      and entry["overall"][metric]["delta"] is not None]
            if not values:
                continue
            summarize = lambda key: {"mean": statistics.mean(row[key] for row in values),
                "sample_sd": statistics.stdev(row[key] for row in values) if len(values) > 1 else None}
            rows.append({"split": split, "metric": metric, "n": len(values),
                         "baseline": summarize("baseline"), "cmerc": summarize("cmerc"), "delta": summarize("delta")})
    return rows


def context_share(entry):
    total = entry["overall"]["cel.total"]
    delta = total["delta"]
    scale = max(1., abs(total["baseline"]))
    return (100 * entry["diagnostics"]["context_cel_delta"] / delta
            if delta < -1e-12 * scale else None)


def num(value):
    return "NA" if value is None else f"{value:.6f}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
                "| " + " | ".join(map(cell, row)) + " |" for row in rows] + [""]


def preamble(title, stats):
    return [f"# {title}", "", f"Execution coverage: **{stats['status']}**; missing evaluations: {len(stats['missing_evaluations'])}.", "",
        "Validation is the primary analysis. Test is the prespecified replication analysis and is reported separately.",
        "IEMOCAP seed 1937 test results were already observed in round 2 and were reused without retraining or reselection.",
        "MELD seed 1937 is the technical pilot; its test may precede the remaining fixed matrix. Configurations do not change after that exposure.",
        "Deltas are CMERC minus Baseline. Classification/calibration are percentages, with deltas in percentage points.",
        "CEL is a sum of positive true-class log-probability differences under `prefix_mean_v1`, a prefix smoothing view, not complete modality removal.",
        "Three seeds on a fixed split give uncertain estimates. Sample SD uses ddof=1; df=2 t intervals rely on normality and are descriptive.",
        "The exact two-sided sign-flip test has minimum p=0.25 for three seeds. No robust statistical significance or proven innovation is claimed.", ""]


def per_seed_table(stats, *, dataset=None, metrics=None, segments=False):
    rows = []
    for entry in stats["per_seed"]:
        if dataset and entry["dataset"] != dataset:
            continue
        selected = {"overall": entry["overall"]}
        if segments:
            selected.update({name: segment["metrics"] for name, segment in entry["position_thirds"].items()})
        for segment, values in selected.items():
            for metric, pair in (values or {}).items():
                if metrics is None or metrics(metric):
                    rows.append([entry["dataset"], entry["seed"], entry["split"], segment, metric,
                                 num(pair["baseline"]), num(pair["cmerc"]), num(pair["delta"])])
    return table(["Dataset", "Seed", "Split", "Segment", "Metric", "Baseline", "CMERC", "Delta"], rows)


def aggregate_table(stats, predicate, dataset=None):
    rows = []
    for entry in stats["statistics"]:
        if (dataset and entry["dataset"] != dataset) or not predicate(entry):
            continue
        summary = lambda variant: f"{num(entry[variant]['mean'])} ± {num(entry[variant]['sample_sd'])}"
        interval = entry["paired_t_95_interval_df2"]
        rows.append([entry["dataset"], entry["split"], entry["segment"], entry["metric"],
                     entry["delta"]["n"], summary("baseline"), summary("cmerc"), summary("delta"),
                     "NA" if interval is None else f"[{num(interval[0])}, {num(interval[1])}]"])
    return table(["Dataset", "Split", "Segment", "Metric", "n", "Baseline mean ± SD", "CMERC mean ± SD",
                  "Paired delta mean ± SD", "95% t interval"], rows)


def optimizer_evidence(matrix, matrix_path, stats):
    """Read saved Adam audits only, clearly distinguishing derived step counts."""
    result = {}
    for pair in matrix.get("pairs", []):
        key = f"{pair['dataset']}/{pair['seed']}"
        if key not in stats["training"]:
            continue
        result[key] = {}
        for variant, training in stats["training"][key].items():
            explicit = pair.get(f"{variant}_optimizer")
            if explicit:
                path = (matrix_path.parent / explicit).resolve()
                require(path.is_file(), "explicit optimizer evidence is missing")
            else:
                run = (matrix_path.parent / pair[f"{variant}_run"]).resolve()
                path = run.parents[2] / "verification" / f"{variant}_optimizer.json"
            derived = training["optimizer_steps"]
            evidence = {"derived_steps": derived, "actual_adam_steps": None,
                        "verification": "derived_from_epoch_batch_counts_only", "sha256": None}
            if path.is_file():
                saved = read(path)
                require(saved.get("optimizer_steps_total") == derived and
                        saved.get("expected_optimizer_steps_total") == derived,
                        "actual Adam count differs from completed epoch batch counts")
                require(saved.get("optimizer_tensors_finite") is True and
                        saved.get("full_rng_state_present") is True,
                        "actual optimizer/RNG audit did not pass")
                require(saved.get("completed_epochs") == training["completed_epochs"],
                        "optimizer epoch coverage differs from training")
                evidence.update(actual_adam_steps=derived, verification="actual_adam_state_verified", sha256=sha(path))
            result[key][variant] = evidence
    return result


def execution_evidence(batch, stats, optimizer):
    """Export stage status/error type and hashes, never local commands or paths."""
    states = {}
    for path in sorted((batch / "pairs").glob("*/STATE.json")):
        state = read(path)
        key = f"{state['dataset']}/{state['seed']}"
        attempts = []
        for stage, records in state.get("stages", {}).items():
            for index, row in enumerate(records, 1):
                attempts.append({"stage": stage, "attempt": index, "status": row.get("status"),
                    "started_utc": row.get("started_utc"), "ended_utc": row.get("ended_utc"),
                    "returncode": row.get("exit_code", row.get("returncode")),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "artifact_count": len(row.get("artifact_sha256", {}))})
        recovery = {variant: {"checkpoint_name": Path(value["latest_checkpoint"]).name,
                              "sha256": value["sha256"],
                              "resume_flag": f"--resume-{variant}"}
                    for variant, value in state.get("recovery_candidates", {}).items()}
        partial = {}
        for variant in ("baseline", "cmerc"):
            candidates = list((path.parent / "runs" / variant).glob("*/metrics.jsonl"))
            if len(candidates) == 1:
                contents = candidates[0].read_bytes()
                records = []
                for line in contents.splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A live trainer may be writing the final line. Keep only
                        # complete records and report an observation, not completion.
                        break
                latest = candidates[0].parent / "latest.pt"
                partial[variant] = {"last_logged_epoch": records[-1]["epoch"] if records else 0,
                    "metrics_snapshot_sha256": hashlib.sha256(contents).hexdigest(),
                    "latest_checkpoint_present": latest.is_file(),
                    "latest_checkpoint_sha256": sha(latest) if latest.is_file() and state.get("status") != "running" else None}
        states[key] = {"state_sha256": sha(path), "status": state.get("status"),
            "error_type": state.get("error_type"), "stages": attempts,
            "partial_training": partial,
            "phase_attempts": [{key: row.get(key) for key in
                                ("phase", "status", "started_utc", "ended_utc")}
                               for row in state.get("phase_attempts", [])],
            "recovery_candidates": recovery}
    return {"status": stats["status"], "missing_evaluations": stats["missing_evaluations"],
            "known_test_exposure": stats["known_test_exposure"], "states": states,
            "optimizer_audits": optimizer}


def meld_report(stats, optimizer):
    lines = preamble("MELD Causal Baseline versus CMERC", stats)
    completed = [entry for entry in stats["per_seed"] if entry["dataset"] == "MELD"]
    if not completed:
        lines += ["**No completed MELD paired evaluations are available in this statistics snapshot. No MELD performance conclusion is drawn.**", ""]
    lines += ["## All completed paired validation and test results", ""]
    lines += per_seed_table(stats, dataset="MELD")
    lines += ["## Supplementary CEL per utterance", "",
        "Divide summed CEL by the identical paired split's utterance count for a scale diagnostic. The main CEL definition remains the sum.", ""]
    cel_rows = []
    for entry in completed:
        for metric in CEL:
            pair = entry["overall"][metric]
            cel_rows.append([entry["seed"], entry["split"], entry["num_samples"], metric,
                *[num(pair[key] / entry["num_samples"]) for key in ("baseline", "cmerc", "delta")]])
    lines += table(["Seed", "Split", "Utterances", "Metric", "Baseline / utterance", "CMERC / utterance", "Delta / utterance"], cel_rows)
    lines += ["## Paired seed summaries", ""]
    lines += aggregate_table(stats, lambda row: row["segment"] == "overall", dataset="MELD")
    lines += ["## Training budgets and resource use", "",
        "Both methods use 40 epochs. Curriculum sampling changes actual optimizer update counts; equal epoch budgets do not imply equal steps.", ""]
    rows = []
    for seed in SEEDS:
        key = f"MELD/{seed}"
        for variant, run in stats["training"].get(key, {}).items():
            audit = optimizer[key][variant]
            rows.append([seed, variant, run["completed_epochs"], run["best_epoch"],
                         num(run["best_validation_weighted_f1"]), num(run["elapsed_seconds"]),
                         num(run["max_gpu_memory_mb"]), audit["derived_steps"],
                         audit["actual_adam_steps"] if audit["actual_adam_steps"] is not None else "unverified"])
    lines += table(["Seed", "Model", "Epochs", "Best epoch", "Selection W-F1 (rounded)", "Epoch seconds sum",
                    "Peak allocated MB", "Derived steps", "Actual Adam steps"], rows)
    lines += ["Shared-server timing is descriptive and can reflect concurrency. No hyperparameters are changed to improve results.", ""]
    return "\n".join(lines)


def calibration_report(stats):
    lines = preamble("Calibration and confidence-ranking diagnostic", stats)
    lines += ["ECE/MCE/RMSCE use the frozen 15 equal-width confidence bins. AUROC/AUPRC-Success rank prediction correctness",
              "using maximum softmax confidence. Calibration and correctness ranking are distinct properties.",
              "All-correct/all-wrong segments have undefined success AUROC/AUPRC and remain NA, never zero.", ""]
    lines += per_seed_table(stats, metrics=lambda name: name in CALIBRATION, segments=True)
    lines += aggregate_table(stats, lambda row: row["metric"] in CALIBRATION)
    lines += ["## Supplemental CEL per utterance", "",
        "CEL remains a summed primary metric. Per-utterance values diagnose sample-count scale only; cross-dataset differences do not establish better causal calibration.", ""]
    cel_rows = []
    for entry in stats["per_seed"]:
        for metric in CEL:
            pair = entry["overall"][metric]
            cel_rows.append([entry["dataset"], entry["seed"], entry["split"], entry["num_samples"], metric,
                *[num(pair[key] / entry["num_samples"]) for key in ("baseline", "cmerc", "delta")]])
    lines += table(["Dataset", "Seed", "Split", "Utterances", "Metric", "Baseline / utterance", "CMERC / utterance", "Delta / utterance"], cel_rows)
    rows = []
    for entry in stats["per_seed"]:
        diag = entry["diagnostics"]
        rows.append([entry["dataset"], entry["seed"], entry["split"], num(diag["ece_delta"]),
                     num(diag["auroc_success_delta"]), diag["ece_improves_auroc_does_not"]])
    lines += ["## ECE versus AUROC direction for every available seed", ""]
    lines += table(["Dataset", "Seed", "Split", "ECE delta", "AUROC-Success delta", "ECE down but AUROC not up"], rows)
    lines += ["Improved ECE alone does not establish improved selective prediction or error detection. These are observations, not a causal attribution to CC.", ""]
    return "\n".join(lines)


def class_position_report(stats):
    lines = preamble("Class and preregistered dialogue-position analysis", stats)
    lines += ["Position = (zero-based utterance index + 0.5) / complete dialogue length: early [0,1/3), middle [1/3,2/3), late [2/3,1].",
              "Length-one dialogues belong to middle. Every class and segment is retained, including negative deltas.", ""]
    chosen = lambda name: name in CLASSIFICATION or name in CALIBRATION or name.startswith("f1.")
    lines += per_seed_table(stats, metrics=chosen, segments=True)
    lines += ["## Across-seed descriptive summaries", ""]
    lines += aggregate_table(stats, lambda row: chosen(row["metric"]))
    lines += ["## IEMOCAP Angry and MELD anger", ""]
    lines += per_seed_table(stats, metrics=lambda name: name in ("f1.angry", "f1.anger"), segments=True)
    lines += ["## Prespecified IEMOCAP new-seed sensitivity (2025 and 3407 only)", "",
        "This descriptive sensitivity excludes the already observed seed 1937. It never replaces the full three-seed main table and has no inferential p-value.", ""]
    sensitivity = new_iemocap_sensitivity(stats)
    if not sensitivity:
        lines += ["Neither new IEMOCAP seed has a completed paired evaluation in this snapshot; sensitivity is unavailable.", ""]
    else:
        summary = lambda row, model: f"{num(row[model]['mean'])} ± {num(row[model]['sample_sd'])}"
        lines += table(["Split", "Metric", "Available new seeds n", "Baseline mean ± SD", "CMERC mean ± SD", "Delta mean ± SD"],
            [[row["split"], row["metric"], row["n"], summary(row, "baseline"), summary(row, "cmerc"), summary(row, "delta")]
             for row in sensitivity])
    lines += ["The second-round Angry decline is a prespecified diagnostic. A repeat across seeds is a robustness observation; a class-specific mechanism requires a new controlled ablation.",
              "Fixed-split seed variation does not measure split/session uncertainty or establish deployment generalization.", ""]
    return "\n".join(lines)


def loss_report(stats):
    lines = preamble("Loss and actual weighted-gradient analysis", stats)
    lines += ["Gradients are from the actual training loss, with normal dropout and `legacy_epoch_snapshot`. The main model trains while read-only auxiliary snapshots evaluate.",
              "Each epoch's first batch is instrumented; these samples do not estimate the complete epoch's gradient distribution.",
              "Weighted component norm = abs(coefficient) × raw component norm. Norms do not sum to a total gradient norm because directions may cancel or reinforce.", ""]
    rows = []
    for key, variants in stats["training"].items():
        dataset, seed = key.split("/")
        for variant, training in variants.items():
            for component, distribution in training["weighted_gradient_norm_summary"].items():
                rows.append([dataset, seed, variant, component, distribution["n"],
                    num(min(distribution["values"])), num(distribution["mean"]),
                    num(distribution["sample_sd"]), num(max(distribution["values"]))])
    lines += table(["Dataset", "Seed", "Model", "Weighted component gradient norm", "Recorded batches",
                    "Minimum", "Mean", "Sample SD", "Maximum"], rows)
    lines += ["## Full epoch loss trajectories", "",
              "Raw terms and actual coefficient-weighted terms are both shown. A large raw CC term is not by itself a large optimization contribution.", ""]
    rows = []
    for key, variants in stats["training"].items():
        dataset, seed = key.split("/")
        for variant, training in variants.items():
            coefficients = training["config"]
            for epoch in training["convergence"]:
                weighted = [epoch.get(f"weighted_{name}", epoch[name] * coefficient) for name, coefficient in
                    zip(LOSSES, (1., coefficients["rank_coff"], coefficients["mscl_coff"], coefficients["cscl_coff"]))]
                rows.append([dataset, seed, variant, epoch["epoch"], *[num(epoch[name]) for name in LOSSES],
                             *map(num, weighted), num(epoch["total_loss"]),
                             num(epoch.get("sample_ratio")), epoch.get("train_dialogues", "NA")])
    lines += table(["Dataset", "Seed", "Model", "Epoch", "Raw class", "Raw CC", "Raw MSCL", "Raw CSCL",
                    "Weighted class", "Weighted CC", "Weighted MSCL", "Weighted CSCL", "Total", "CL ratio", "Train dialogues"], rows)
    lines += ["## Every recorded gradient sample", ""]
    rows = []
    for key, variants in stats["training"].items():
        dataset, seed = key.split("/")
        for variant, training in variants.items():
            for sample in training["gradient_trajectory"]:
                raw, weighted = sample["gradient_norms"], sample["weighted_gradient_norms"]
                rows.append([dataset, seed, variant, sample["epoch"] + 1, sample["batch"],
                             *[num(raw.get(name, 0.)) for name in LOSSES],
                             *[num(weighted.get(name, 0.)) for name in LOSSES],
                             sample["finite_gradients"], num(sample.get("main_aux_max_abs_diff"))])
    lines += table(["Dataset", "Seed", "Model", "Epoch", "Batch", "Raw class norm", "Raw CC norm", "Raw MSCL norm",
                    "Raw CSCL norm", "Weighted class norm", "Weighted CC norm", "Weighted MSCL norm", "Weighted CSCL norm",
                    "Finite gradients", "Main/aux max weight difference"], rows)
    lines += ["Zero entries for disabled components indicate no enabled term, not a measured gradient for an active branch.",
              "All raw diagnostic records are retained in loss_gradient_trajectories.json. No favorable-epoch filtering is applied.", ""]
    return "\n".join(lines)


def candidate_report(stats):
    lines = preamble("Next research candidates: hypotheses, not established innovations", stats)
    lines += ["## Measured diagnostic evidence", ""]
    rows = []
    for entry in stats["per_seed"]:
        overall, diag = entry["overall"], entry["diagnostics"]
        delta = overall["cel.total"]["delta"]
        share = context_share(entry)
        rows.append([entry["dataset"], entry["seed"], entry["split"], num(diag["angry_f1"]["delta"] if diag["angry_f1"] else None),
                     num(delta), num(diag["context_cel_delta"]), num(share), num(diag["ece_delta"]), num(diag["auroc_success_delta"])])
    lines += table(["Dataset", "Seed", "Split", "Angry/anger F1 delta", "CEL total delta", "Context CEL delta",
                    "Context share of net reduction %", "ECE delta", "AUROC-Success delta"], rows)
    lines += ["Context share is defined only for a meaningful net CEL reduction; near-zero threshold is 1e-12 × max(1, |baseline CEL|).",
              "The ratio can exceed 100% when other views worsen. It is not a causal attribution. CEL magnitudes are not directly comparable across dataset sample counts.", "",
              "## Candidates requiring a separately frozen follow-up", "",
              "1. **Class trade-offs:** test whether CC/HSCL/CL change Angry/anger errors systematically. Compare paired component ablations with per-class recall/precision; retain the current negative results.",
              "2. **Context-view sensitivity:** investigate why GNN-edge ablations contribute to CEL changes. Encoders and hypergraphs retain history, so this is not complete context removal. Use explicit edge and representation controls before mechanism claims.",
              "3. **Calibration versus correctness ranking:** study ECE improvements alongside AUROC/AUPRC-Success and selective-risk curves. A lower ECE does not prove better error detection.",
              "4. **Auxiliary snapshot semantics:** independently compare legacy_epoch_snapshot and same_step_same_mode with matched confidence-pass weights/modes and explicit gradient rules; do not silently substitute the auxiliary policy.",
              "5. **Replacement semantics:** prefix_mean_v1 is invertible prefix smoothing. Strict missing-modality diagnostics need an explicitly named fixed replacement and cannot directly reuse cross-policy CEL comparisons.", "",
              "These are candidate questions. This round does not establish novelty, mechanism, statistical significance, or cross-split robustness and does not launch any ablation.", ""]
    return "\n".join(lines)


def status_report(stats, evidence, plan, hashes):
    lines = preamble("Final execution status", stats)
    lines += [f"Frozen training commit: `{plan.get('commit', 'not recorded')}`.",
              f"Protocol SHA256: `{hashes['protocol_sha256']}`.",
              f"Analysis matrix SHA256: `{hashes['matrix_sha256']}`.",
              f"Statistics SHA256: `{hashes['statistics_sha256']}`.", ""]
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            key = f"{dataset}/{seed}"
            runs = stats["training"].get(key, {})
            validation = any(entry["dataset"] == dataset and entry["seed"] == seed and entry["split"] == "validation"
                             for entry in stats["per_seed"])
            test = any(entry["dataset"] == dataset and entry["seed"] == seed and entry["split"] == "test"
                       for entry in stats["per_seed"])
            state = evidence["states"].get(key, {})
            rows.append([dataset, seed, "round 2 reuse" if dataset == "IEMOCAP" and seed == 1937 else state.get("status", "not included"),
                         runs.get("baseline", {}).get("completed_epochs", state.get("partial_training", {}).get("baseline", {}).get("last_logged_epoch", "not started")),
                         runs.get("cmerc", {}).get("completed_epochs", state.get("partial_training", {}).get("cmerc", {}).get("last_logged_epoch", "not started")), validation, test])
    lines += table(["Dataset", "Seed", "Execution status", "Baseline epochs", "CMERC epochs", "Validation analyzed", "Test analyzed"], rows)
    lines += ["## Complete available paired outcomes", ""]
    lines += aggregate_table(stats, lambda row: row["segment"] == "overall" and
                             row["metric"] in ("weighted_f1", "ece", "cel.total"))
    lines += ["Positive W-F1 deltas and negative ECE/CEL deltas indicate the prespecified favorable directions; opposite directions remain in the same tables.", ""]
    lines += ["## All recorded attempts, including failures and recovery", ""]
    rows = []
    for key, state in evidence["states"].items():
        for attempt in state["stages"]:
            rows.append([key, attempt["stage"], attempt["attempt"], attempt["status"], attempt["returncode"],
                         attempt["started_utc"], attempt["ended_utc"], num(attempt["elapsed_seconds"])])
    lines += table(["Pair", "Stage", "Attempt", "Status", "Return code", "Started UTC", "Ended UTC", "Wall seconds"], rows)
    lines += ["Epoch counts for unadmitted/running runs are last complete log entries, not proof of checkpoint completeness. Exact state/metric snapshot hashes are retained in REPORT_EVIDENCE.json.", ""]
    for key, state in evidence["states"].items():
        if state["error_type"]:
            lines += [f"- {key}: recorded error type `{state['error_type']}`; full private error/command retained locally; STATE SHA256 `{state['state_sha256']}`."]
        for variant, recovery in state["recovery_candidates"].items():
            lines += [f"- {key}/{variant}: recoverable checkpoint `{recovery['checkpoint_name']}`, SHA256 `{recovery['sha256']}`; use the preserved local command with `{recovery['resume_flag']}`. This is not a completed run."]
    lines += ["", "## Checkpoint and optimizer evidence", ""]
    rows = []
    for key, variants in stats["training"].items():
        for variant, run in variants.items():
            audit = evidence["optimizer_audits"][key][variant]
            rows.append([key, variant, run["best_epoch"], run["best_sha256"], run["latest_sha256"],
                         run["initial_model_sha256"], audit["derived_steps"],
                         audit["actual_adam_steps"] if audit["actual_adam_steps"] is not None else "not independently audited here"])
    lines += table(["Pair", "Model", "Best epoch", "Best SHA256", "Latest SHA256", "Initial state SHA256", "Derived updates", "Actual Adam updates"], rows)
    lines += ["## Evidence boundary", "",
        "Completed formal budgets and finite/update/provenance checks are distinct from performance improvement. Negative results are retained.",
        "Missing runs/evaluations are not replaced with smoke tests. No formal performance conclusion is drawn for missing cells.",
        "All model weights, features, dialogue IDs, predictions and large training logs remain on the experiment server. Public outputs contain aggregate values and hashes.",
        "Pre-extracted feature temporal causality is the audited boundary; upstream feature extraction remains unaudited.",
        "Synthetic causality tests were not repeated in this round. Three fixed-split seeds do not establish broad generalization.", ""]
    return "\n".join(lines)


def render(statistics_dir, batch, matrix_path, output):
    statistics_dir, batch, matrix_path, output = map(Path, (statistics_dir, batch, matrix_path, output))
    stats_path = statistics_dir / "paired_statistics.json"
    stats, matrix, plan = read(stats_path), read(matrix_path), read(batch / "PLAN.json")
    protocol = batch / "THIRD_ROUND_PROTOCOL.md"
    require(stats.get("schema_version") == 1, "unsupported statistics schema")
    require(stats.get("matrix_sha256") == sha(matrix_path), "statistics matrix differs from supplied matrix")
    require(plan.get("protocol_sha256") == sha(protocol), "frozen protocol differs from PLAN")
    require(stats.get("known_test_exposure") == matrix.get("known_test_exposure"), "test exposure differs from matrix")
    consistency = validate_configurations(stats)
    optimizer = optimizer_evidence(matrix, matrix_path.resolve(), stats)
    execution = execution_evidence(batch, stats, optimizer)
    hashes = {"statistics_sha256": sha(stats_path), "matrix_sha256": sha(matrix_path),
              "protocol_sha256": sha(protocol), "plan_sha256": sha(batch / "PLAN.json"),
              "renderer_sha256": sha(__file__)}
    reports = {"MELD_BASELINE_VS_CMERC.md": meld_report(stats, optimizer),
        "CALIBRATION_DIAGNOSTIC.md": calibration_report(stats),
        "CLASS_AND_POSITION_ANALYSIS.md": class_position_report(stats),
        "LOSS_GRADIENT_ANALYSIS.md": loss_report(stats),
        "NEXT_RESEARCH_CANDIDATES.md": candidate_report(stats),
        "FINAL_EXECUTION_STATUS.md": status_report(stats, execution, plan, hashes)}
    # Validate everything before creating a destination; no partial report after admission failure.
    output.mkdir(parents=True, exist_ok=False)
    for filename, content in reports.items():
        (output / filename).write_text(content)
    shutil.copyfile(protocol, output / protocol.name)
    for filename in COPIES:
        if (statistics_dir / filename).is_file():
            if filename.endswith(".json"):
                aggregate = read(statistics_dir / filename)
                if filename == "paired_statistics.json":
                    for entry in aggregate["per_seed"]:
                        entry["diagnostics"]["context_share_of_net_cel_reduction_percent"] = context_share(entry)
                    aggregate["report_additions"] = {"actual_optimizer_audits": optimizer,
                        "configuration_consistency": consistency,
                        "iemocap_new_seed_descriptive_sensitivity": new_iemocap_sensitivity(stats),
                        "context_share_near_zero_guard": "None unless net CEL delta < -1e-12 * max(1, abs(baseline CEL)); private original remains hash-pinned"}
                (output / filename).write_text(json.dumps(public_metadata(aggregate), indent=2, allow_nan=False) + "\n")
            else:
                shutil.copyfile(statistics_dir / filename, output / filename)
    public = {"input_sha256": hashes, "execution": execution,
              "configuration_consistency": consistency,
              "iemocap_new_seed_descriptive_sensitivity": new_iemocap_sensitivity(stats),
              "public_run_configurations": {key: {variant: run["config"] for variant, run in variants.items()}
                  for key, variants in stats["training"].items()},
              "difficulty_teacher_provenance": {f"{pair['dataset']}/{pair['seed']}":
                  read((matrix_path.resolve().parent / pair["cmerc_verification"]).resolve()).get("difficulty")
                  for pair in matrix.get("pairs", []) if f"{pair['dataset']}/{pair['seed']}" in stats["training"]},
              "training_artifact_sha256": {key: {variant: {name: run[name] for name in
                  ("best_sha256", "latest_sha256", "initial_model_sha256", "verification_sha256", "run_manifest_sha256",
                   "gradient_diagnostics_sha256", "metrics_sha256")} for variant, run in variants.items()}
                   for key, variants in stats["training"].items()}}
    (output / "REPORT_EVIDENCE.json").write_text(json.dumps(public_metadata(public), indent=2, allow_nan=False) + "\n")
    manifest = {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": stats["status"], "reports": len(reports) + 2, "report_manifest_sha256": sha(output / "REPORT_MANIFEST.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, help="defaults to BATCH/ANALYSIS_MATRIX.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = render(args.statistics_dir, args.batch_dir,
                    args.matrix or args.batch_dir / "ANALYSIS_MATRIX.json", args.output_dir)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
