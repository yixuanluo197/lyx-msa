#!/usr/bin/env python3
"""CPU-only, score-independent acceptance of the complete XP mainline batch.

Usage: CUDA_VISIBLE_DEVICES='' python tools/finalize_mainline.py [--check-only]
Incomplete queues exit 2 without publishing acceptance or a baseline registry.
The existing scheduler validators are imported; no scheduler/GPU command runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import tempfile

BATCH = Path(__file__).resolve().parents[1]
CMERC = BATCH.parents[2]
SEEDS = [1475, 67137, 2024, 3407, 42]
DATASETS = ("IEMOCAP", "MELD")
VARIANTS = ("baseline", "cmerc")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "cel", "ece")
REMOVALS = ("text", "audio", "visual", "intra_context", "inter_context")
TOLERANCE = 1e-6
TITLE = "XP 主研究基线：非因果、验证集选模、固定外部课程、五 seed 配对"
PAPER = {("IEMOCAP", "baseline"): (69.61, 1772.54),
         ("IEMOCAP", "cmerc"): (71.98, 1143.50),
         ("MELD", "baseline"): (65.37, 799.58),
         ("MELD", "cmerc"): (66.85, 605.05)}


class NotReady(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    write_atomic(path, json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + "\n")


def close(actual, expected, label, tolerance=TOLERANCE):
    require(not isinstance(actual, bool) and isinstance(actual, (int, float))
            and math.isfinite(actual), f"{label}: nonfinite/non-numeric {actual!r}")
    require(math.isfinite(float(expected)) and abs(actual - expected) <= tolerance,
            f"{label}: {actual} differs from independently recomputed {expected}")


def describe(values):
    values = list(map(float, values))
    require(len(values) == 5 and all(math.isfinite(x) for x in values),
            "statistics require all five finite planned seeds")
    mean, std = statistics.mean(values), statistics.stdev(values)
    return {"n": 5, "mean": mean, "std": std, "ddof": 1,
            "minimum": min(values), "maximum": max(values),
            "approximate_t95_interval_df4": [mean - 2.7764451051977987 * std / math.sqrt(5),
                                             mean + 2.7764451051977987 * std / math.sqrt(5)]}


def import_scheduler():
    code = BATCH / "code"
    sys.path.insert(0, str(code))
    name = "_xp_mainline_acceptance_scheduler"
    spec = importlib.util.spec_from_file_location(name, code / "scripts/multiseed_scheduler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def matrix_gate():
    path = BATCH / "state/queue.json"
    if not path.is_file():
        raise NotReady("state/queue.json does not yet exist")
    queue = read_json(path)
    jobs = queue.get("jobs", [])
    if queue.get("phase") != "completed" or len(jobs) != 20 \
            or any(job.get("status") != "completed" for job in jobs):
        raise NotReady(f"queue phase={queue.get('phase')}; completed "
                       f"{sum(j.get('status') == 'completed' for j in jobs)}/20")
    config = queue["config"]
    require(config["first_seeds"] + config["additional_seeds"] == SEEDS,
            "queue seeds differ from the preregistered ordered five seeds")
    expected = {(ds, variant, seed) for ds in DATASETS for variant in VARIANTS for seed in SEEDS}
    require({(j["dataset"], j["variant"], j["seed"]) for j in jobs} == expected,
            "queue is not the exact 20-run paired matrix")
    require(len({job["id"] for job in jobs}) == 20, "duplicate queue job IDs")
    for job in jobs:
        for field, root in (("run_dir", BATCH / "runs"), ("result_dir", BATCH / "results")):
            require(Path(job[field]).resolve().is_relative_to(root.resolve()),
                    f"{job['id']}: {field} escapes this batch")
        for field in ("training_exit_code", "evaluation_exit_code"):
            require(job.get(field) == 0, f"{job['id']}: {field} is not zero")
        require(job.get("server") == "user23" and config["allowed_gpu_uuids"].get(
                str(job.get("host_gpu_index"))) == job.get("gpu_uuid"),
                f"{job['id']}: saved server/GPU UUID identity is not an approved local XP")
    summary = read_json(BATCH / "results/summary.json")
    require(summary.get("complete") is True and summary.get("valid_runs") == 20
            and summary.get("planned_runs") == 20, "summary is not complete 20/20")
    require(summary.get("planned_seeds") == SEEDS and not summary.get("unplanned_seeds"),
            "summary seeds mismatch/unplanned seeds")
    require(len(summary.get("groups", [])) == 4 and len(summary.get("pairs", [])) == 10
            and len(summary.get("runs", [])) == 20, "summary groups/pairs/runs incomplete")
    require({(r["dataset"], r["variant"], r["seed"]) for r in summary["runs"]} == expected,
            "summary run identities differ from queue")
    return queue, jobs, summary


def finite_tree(value, label, torch):
    if torch.is_tensor(value):
        require(value.device.type == "cpu", f"{label}: checkpoint escaped CPU map_location")
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), f"{label}: nonfinite tensor")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{label}.{key}", torch)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{label}[{index}]", torch)
    elif isinstance(value, float):
        require(math.isfinite(value), f"{label}: nonfinite scalar")


def checkpoint_audit(job, manifest, records):
    import numpy as np
    import torch

    selected = max(records, key=lambda r: r["selection_weighted_f1"])
    details, signatures = {}, []
    for filename, epoch_record in (("latest.pt", records[-1]), ("best.pt", selected)):
        path = Path(job["run_dir"]) / filename
        checkpoint = torch.load(path, map_location="cpu")
        required = {"model_state_dict", "optimizer_state_dict", "rng_state", "args", "epoch",
                    "best_epoch", "best_metric", "provenance", "sampling_metadata", "causal"}
        require(required <= set(checkpoint), f"{path}: incomplete checkpoint schema")
        require(checkpoint["epoch"] + 1 == epoch_record["epoch"], f"{path}: wrong saved epoch")
        require(checkpoint["best_epoch"] + 1 == selected["epoch"], f"{path}: wrong selected epoch")
        close(checkpoint["best_metric"], selected["selection_weighted_f1"], f"{path}.best_metric")
        require(checkpoint["causal"] is False and checkpoint["args"] == manifest["args"],
                f"{path}: effective configuration differs from manifest")
        require(checkpoint["provenance"] == manifest["provenance"], f"{path}: provenance mismatch")
        for key in ("batch_policy", "sampler_retries", "max_batch_edges", "max_batch_nodes"):
            require(checkpoint["sampling_metadata"].get(key) == epoch_record[key],
                    f"{path}: sampling_metadata.{key} mismatch")
        model = checkpoint["model_state_dict"]
        require(isinstance(model, dict) and len(model) == (85 if job["dataset"] == "IEMOCAP" else 83),
                f"{path}: unexpected main-model state structure")
        require(all(torch.is_tensor(t) and (not t.is_floating_point() or t.dtype == torch.float32)
                    for t in model.values()), f"{path}: non-FP32 model state")
        finite_tree(model, f"{path}.model", torch)
        signatures.append([(k, tuple(v.shape), str(v.dtype)) for k, v in model.items()])
        optimizer = checkpoint["optimizer_state_dict"]
        require(optimizer.get("state") and len(optimizer.get("param_groups", [])) == 1,
                f"{path}: missing/nonstandard Adam state")
        group = optimizer["param_groups"][0]
        require(group.get("params") and group.get("lr") == 1e-4
                and group.get("weight_decay") == manifest["args"]["l2"]
                and tuple(group.get("betas", [])) == (.9, .999) and group.get("eps") == 1e-8,
                f"{path}: Adam configuration mismatch")
        for key, expected in (("amsgrad", False), ("maximize", False), ("capturable", False),
                              ("differentiable", False), ("foreach", None), ("fused", False)):
            require(group.get(key) == expected, f"{path}: Adam {key} differs")
        require(set(optimizer["state"]) <= set(group["params"]), f"{path}: orphan Adam state")
        for param_id, state in optimizer["state"].items():
            require({"step", "exp_avg", "exp_avg_sq"} <= set(state),
                    f"{path}: incomplete Adam state for {param_id}")
            require(float(state["step"]) > 0, f"{path}: Adam state has no completed step")
        finite_tree(optimizer, f"{path}.optimizer", torch)
        rng = checkpoint["rng_state"]
        require({"python", "numpy", "torch", "cuda"} <= set(rng), f"{path}: incomplete RNG state")
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator(device="cpu").set_state(rng["torch"])
        require(len(rng["cuda"]) == 1 and all(torch.is_tensor(t) and t.device.type == "cpu"
                and t.dtype == torch.uint8 and t.ndim == 1 and t.numel() > 0 for t in rng["cuda"]),
                f"{path}: incomplete visible-device CUDA RNG byte states")
        finite_tree(rng, f"{path}.rng", torch)
        details[filename] = {"sha256": sha256(path), "epoch": checkpoint["epoch"] + 1,
                             "best_epoch": selected["epoch"], "state_tensors": len(model),
                             "adam_parameter_states": len(optimizer["state"]),
                             "sampling_metadata": checkpoint["sampling_metadata"],
                             "rng_keys": sorted(rng)}
        del checkpoint, model, optimizer, rng
        gc.collect()
    require(signatures[0] == signatures[1], f"{job['id']}: best/latest structures differ")
    return details, selected


def expected_test_rows(dataset, split):
    import numpy as np

    filename = "IEMOCAP_features.pkl" if dataset == "IEMOCAP" else "MELD_features_raw1.pkl"
    with (BATCH / "code/data" / filename).open("rb") as handle:
        labels_by_dialogue = pickle.load(handle, encoding="latin1")[2]
    ids, labels, indices = [], [], []
    for dialogue in split["test_ids"]:
        values = labels_by_dialogue[dialogue]
        ids.extend([str(dialogue)] * len(values))
        labels.extend(values)
        indices.extend(range(len(values)))
    return np.asarray(labels), np.asarray(ids), np.asarray(indices)


def independent_metrics(path, dataset, expected):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    needed = {"labels", "predictions", "probabilities", "log_probabilities", "dialogue_ids",
              "utterance_indices", *(f"removed_{name}_log_probabilities" for name in REMOVALS)}
    require(needed <= set(arrays), f"{path}: missing saved prediction arrays")
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.number):
            require(bool(np.isfinite(values).all()), f"{path}.{name}: nonfinite values")
    labels, predicted, log_prob = arrays["labels"], arrays["predictions"], arrays["log_probabilities"]
    classes = 6 if dataset == "IEMOCAP" else 7
    n = 1623 if dataset == "IEMOCAP" else 2610
    require(labels.shape == predicted.shape == (n,) and log_prob.shape == (n, classes),
            f"{path}: invalid test shape")
    require(np.issubdtype(labels.dtype, np.integer) and np.issubdtype(predicted.dtype, np.integer),
            f"{path}: class labels must be integers")
    for actual, reference in zip((labels, arrays["dialogue_ids"].astype(str), arrays["utterance_indices"]), expected):
        require(np.array_equal(actual, reference), f"{path}: predictions are not in the actual test split/order")
    probabilities = np.exp(log_prob)
    require(bool(np.allclose(probabilities.sum(axis=1), 1., rtol=1e-4, atol=1e-5)),
            f"{path}: log probabilities are not normalized")
    require(arrays["probabilities"].shape == probabilities.shape
            and bool(np.allclose(arrays["probabilities"], probabilities, rtol=1e-6, atol=1e-7)),
            f"{path}: saved probabilities differ from saved log probabilities")
    require(np.array_equal(predicted, probabilities.argmax(axis=1)), f"{path}: inconsistent argmax predictions")
    result = {"accuracy": float(accuracy_score(labels, predicted) * 100),
              "weighted_f1": float(f1_score(labels, predicted, labels=range(classes), average="weighted", zero_division=0) * 100),
              "macro_f1": float(f1_score(labels, predicted, labels=range(classes), average="macro", zero_division=0) * 100)}
    true_full = log_prob.astype(np.float64)[np.arange(n), labels]
    by_removal = {}
    for name in REMOVALS:
        removed = arrays[f"removed_{name}_log_probabilities"]
        require(removed.shape == log_prob.shape, f"{path}: removal shape mismatch: {name}")
        require(bool(np.allclose(np.exp(removed).sum(axis=1), 1., rtol=1e-4, atol=1e-5)),
                f"{path}: removal log probabilities not normalized: {name}")
        by_removal[name] = float(np.maximum(0., removed.astype(np.float64)[np.arange(n), labels] - true_full).sum())
    result["cel"] = sum(by_removal.values())
    # Independent explicit ((k-1)/15,k/15] bin accumulation in percentage points.
    confidence = probabilities.astype(np.float64).max(axis=1)
    correctness = (predicted == labels).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0., 1., 16)
    for index in range(15):
        included = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if included.any():
            ece += float(included.sum()) / n * abs(float(correctness[included].mean() - confidence[included].mean()))
    result["ece"] = 100 * ece
    return result, by_removal


def capacity_index(audit):
    require(audit.get("batch_policy") == "capacity_safe" and audit.get("causal") is False,
            "capacity preflight policy mismatch")
    config = audit["configuration"]
    require(config["seeds"] == SEEDS and config["split_seed"] == 1475 and config["batch_size"] == 16
            and config["edge_capacity"] == 1000 and config["node_capacity"] == 5200
            and config["batch_max_retries"] == 10000, "capacity audit configuration mismatch")
    indexed = {}
    for dataset, data in audit["datasets"].items():
        require(data.get("split_disjoint") is True, f"{dataset}: capacity split not certified disjoint")
        for name in ("fixed_validation", "fixed_test"):
            require(data[name]["status"] == "passed", f"{dataset}: fixed {name} capacity not passed")
        for run in data["runs"]:
            key = (dataset, run["variant"], run["seed"])
            require(key not in indexed and run.get("status") == "passed"
                    and run.get("all_eligible_ids_once_every_epoch") is True,
                    f"{key}: duplicate/failed/incomplete capacity audit")
            indexed[key] = run["epoch_records"]
    require(len(indexed) == 20, "capacity audit does not contain 20 runs")
    return indexed


def audit_run(job, audit_records, hash_cache, truth_cache):
    run_dir, result_dir = Path(job["run_dir"]), Path(job["result_dir"])
    manifest = read_json(run_dir / "run_manifest.json")
    expected_path = BATCH / "configs/effective" / (job["id"] + ".json")
    expected_args = read_json(expected_path)
    actual_args = manifest["args"]
    differences = sorted(key for key in set(expected_args) | set(actual_args)
                         if expected_args.get(key) != actual_args.get(key)
                         or (key in expected_args) != (key in actual_args))
    require(not differences, f"{job['id']}: approved full effective args differ: {differences}")
    provenance = manifest["provenance"]
    for name, digest in provenance["data"].items():
        path = BATCH / "code" / name
        if str(path) not in hash_cache:
            hash_cache[str(path)] = sha256(path)
        actual = hash_cache[str(path)]
        require(actual == digest, f"{job['id']}: current input SHA mismatch for {name}")
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    require(len(records) == len(audit_records) == job["epochs"], f"{job['id']}: capacity epoch budget mismatch")
    for record, audited in zip(records, audit_records):
        require(record["epoch"] == audited["epoch"], f"{job['id']}: capacity epoch identity mismatch")
        require({"test_accuracy", "test_weighted_f1"} <= set(record)
                and all(v is None for k, v in record.items() if k.startswith("test_")),
                f"{job['id']}: per-epoch test field is populated")
        close(record["selection_weighted_f1"], record["validation_weighted_f1"], "validation selection")
        for key in ("sample_ratio", "sampler_retries", "max_batch_edges", "max_batch_nodes"):
            close(record[key], audited[key], f"{job['id']}: epoch{record['epoch']}.{key}")
        require(record["train_dialogues"] == audited["eligible_dialogues"]
                and audited.get("each_eligible_id_once") is True, f"{job['id']}: curriculum sample count mismatch")
    require(provenance["split"] == read_json(BATCH / "preflight/data_and_capacity/batch_capacity_audit.json")["datasets"][job["dataset"]]["split"],
            f"{job['id']}: capacity preflight split differs")
    checkpoints, selected = checkpoint_audit(job, manifest, records)
    if job["dataset"] not in truth_cache:
        truth_cache[job["dataset"]] = expected_test_rows(job["dataset"], provenance["split"])
    numbers, removals = independent_metrics(result_dir / "predictions.npz", job["dataset"], truth_cache[job["dataset"]])
    metrics = read_json(result_dir / "metrics.json")
    for name in METRICS:
        saved = metrics["cel"]["total"] if name == "cel" else metrics["calibration"]["ece"] if name == "ece" else metrics[name]
        close(saved, numbers[name], f"{job['id']}: independently recomputed {name}")
    require(metrics["calibration"]["n_bins"] == 15, f"{job['id']}: ECE does not use 15 bins")
    for name, value in removals.items():
        close(metrics["cel"]["by_removal"][name], value, f"{job['id']}: CEL.{name}")
    final = read_json(run_dir / "final_test.json")
    require(metrics["selected_epoch"] == selected["epoch"] == final["selected_epoch"],
            f"{job['id']}: validation-selected epoch mismatch")
    close(final["selection_weighted_f1"], selected["selection_weighted_f1"], "final selection metric")
    close(final["test_weighted_f1"], numbers["weighted_f1"], "final test F1 (two-decimal log)", .011)
    close(final["test_accuracy"], numbers["accuracy"], "final test accuracy (two-decimal log)", .011)
    effective = manifest["args"]
    require(effective["num_workers"] == 0 and effective["l2"] == 3e-5
            and effective["class_weight"] is True, f"{job['id']}: unapproved effective configuration")
    artifact_hashes = {name: sha256(path) for name, path in {
        "run_manifest.json": run_dir / "run_manifest.json", "metrics.jsonl": run_dir / "metrics.jsonl",
        "final_test.json": run_dir / "final_test.json", "independent_metrics.json": result_dir / "metrics.json",
        "predictions.npz": result_dir / "predictions.npz"}.items()}
    return {"id": job["id"], "dataset": job["dataset"], "variant": job["variant"], "seed": job["seed"],
            "selected_epoch": selected["epoch"], "split_hash": provenance["split"]["hash"], **numbers,
            "cel_by_removal": removals, "effective_config": effective, "original_provenance": provenance,
            "approved_effective_config_sha256": sha256(expected_path),
            "queue_job": job, "checkpoint_checks": checkpoints, "artifact_sha256": artifact_hashes,
            "capacity_epochs_verified": len(records), "elapsed_seconds": sum(r["elapsed_seconds"] for r in records),
            "max_gpu_memory_mb": max(r["max_gpu_memory_mb"] for r in records)}


def audit_statistics(summary, runs):
    indexed = {(run["dataset"], run["variant"], run["seed"]): run for run in runs}
    supplied_runs = {(r["dataset"], r["variant"], r["seed"]): r for r in summary["runs"]}
    groups, pairs, paired = [], [], []
    supplied_groups = {(g["dataset"], g["variant"]): g for g in summary["groups"]}
    require(set(supplied_groups) == {(d, v) for d in DATASETS for v in VARIANTS}, "wrong summary group identities")
    for key, run in indexed.items():
        for metric in METRICS:
            close(supplied_runs[key][metric], run[metric], f"summary {key}.{metric}")
    for dataset in DATASETS:
        for variant in VARIANTS:
            supplied = supplied_groups[dataset, variant]
            require(supplied["planned_n"] == supplied["valid_n"] == 5, "group not complete five seeds")
            group = {"dataset": dataset, "variant": variant, "metrics": {}}
            for metric in METRICS:
                stats = describe(indexed[dataset, variant, seed][metric] for seed in SEEDS)
                supplied_stats = supplied["metrics"][metric]
                require(supplied_stats["n"] == 5, "summary metric n mismatch")
                for field in ("mean", "std"):
                    close(supplied_stats[field], stats[field], f"group {dataset}/{variant}/{metric}/{field}")
                group["metrics"][metric] = stats
            groups.append(group)
        for seed in SEEDS:
            base, cmerc = indexed[dataset, "baseline", seed], indexed[dataset, "cmerc", seed]
            pairs.append({"dataset": dataset, "seed": seed, "split_hash": base["split_hash"],
                          **{metric + "_change": cmerc[metric] - base[metric] for metric in METRICS}})
        paired.append({"dataset": dataset, "n": 5, "metrics": {
            metric: describe(p[metric + "_change"] for p in pairs if p["dataset"] == dataset)
            for metric in METRICS}})
    supplied_pairs = {(p["dataset"], p["seed"]): p for p in summary["pairs"]}
    require(set(supplied_pairs) == {(ds, seed) for ds in DATASETS for seed in SEEDS}, "summary pair identities mismatch")
    for pair in pairs:
        supplied = supplied_pairs[pair["dataset"], pair["seed"]]
        close(supplied["weighted_f1_gain"], pair["weighted_f1_change"], "paired F1 gain")
        close(supplied["cel_change"], pair["cel_change"], "paired CEL change")
    supplied_paired = {p["dataset"]: p for p in summary["paired_statistics"]}
    require(set(supplied_paired) == set(DATASETS), "paired summary missing dataset")
    for group in paired:
        for saved_name, metric in (("weighted_f1_gain", "weighted_f1"), ("cel_change", "cel")):
            for field in ("mean", "std"):
                close(supplied_paired[group["dataset"]][saved_name][field], group["metrics"][metric][field],
                      "paired summary statistic")
    return groups, pairs, paired


def history():
    old_xp = CMERC / "experiment_records/02_xp_multiseed"
    old_v100 = CMERC / "experiment_records/03_v100_multiseed/results/multiseed_validation_v2/summary.json"
    xp = read_json(old_xp / "summary.json")
    v100 = read_json(old_v100)
    require(v100.get("complete") is True and v100.get("valid_runs") == 20,
            "historical V100 V2 reference is incomplete")
    comparison = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            vgroup = next(g for g in v100["groups"] if (g["dataset"], g["variant"]) == (dataset, variant))
            if variant == "cmerc":
                x = xp["groups"][dataset]
                require(x["completed_n"] == 5, "historical XP CMERC group incomplete")
                xmetric = {"n": 5, "mean": x["mean"], "std": x["sample_std"]}
                xcel = describe(read_json(old_xp / f"results/cmerc/{dataset}/official_test/seed{seed}/metrics.json")["cel"]["total"]
                                for seed in x["seeds"])
            else:
                seed = 1475 if dataset == "IEMOCAP" else 67137
                path = old_xp / f"results/baseline/{dataset}/official_test/seed{seed}/metrics.json"
                xmetric = {"n": 1, "mean": read_json(path)["weighted_f1"], "std": None}
                xcel = {"n": 1, "mean": read_json(path)["cel"]["total"], "std": None}
            comparison[dataset + "/" + variant] = {"old_xp02_official_test": xmetric,
                "old_xp02_cel": xcel, "v100v2_03_cel": vgroup["metrics"]["cel"],
                "v100v2_03_validation": vgroup["metrics"]["weighted_f1"],
                "paper": {"weighted_f1": PAPER[dataset, variant][0], "cel": PAPER[dataset, variant][1]}}
    return comparison


def fmt(stats):
    return f"{stats['mean']:.4f} ± {stats['std']:.4f}" if stats.get("std") is not None else f"{stats['mean']:.4f}（n={stats['n']}）"


def reproduction_report(payload):
    lines = ["# XP 主线五 seed 配对复现", "", TITLE, "",
             f"批次 `{BATCH.name}`；验收时间 {payload['accepted_at_utc']}。20/20完整训练和独立测试评估通过技术验收；验收没有性能分数门槛。", "",
             "训练 seeds 固定为1475、67137、2024、3407、42；IEMOCAP固定split1475，MELD官方split；每seed两模型配对。以下均为单模型测试指标，未使用预测集成。", "",
             "## 20 个逐 seed 结果", "",
             "| 数据集 | 模型 | seed | validation选中epoch | Acc | W-F1 | Macro-F1 | CEL | ECE |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in payload["runs"]:
        lines.append(f"| {run['dataset']} | {run['variant']} | {run['seed']} | {run['selected_epoch']} | "
                     + " | ".join(f"{run[m]:.4f}" for m in METRICS) + " |")
    lines += ["", "## 四组完整五 seed 均值 ± 样本标准差", "", "全部Std使用ddof=1，指标为百分点，CEL为五路log概率正差总和。", "",
              "| 数据集 | 模型 | Acc | W-F1 | Macro-F1 | CEL | ECE |", "|---|---|---:|---:|---:|---:|---:|"]
    for group in payload["groups"]:
        lines.append(f"| {group['dataset']} | {group['variant']} | "
                     + " | ".join(fmt(group["metrics"][m]) for m in METRICS) + " |")
    lines += ["", "## 10 个配对差值", "", "变化=CMERC−baseline；Acc/F1正值较好，CEL/ECE负值较好。", "",
              "| 数据集 | seed | ΔAcc | ΔW-F1 | ΔMacro-F1 | ΔCEL | ΔECE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for pair in payload["pairs"]:
        lines.append(f"| {pair['dataset']} | {pair['seed']} | "
                     + " | ".join(f"{pair[m + '_change']:+.4f}" for m in METRICS) + " |")
    lines += ["", "| 数据集 | 配对指标 | n | 差值均值 ± 样本Std | 近似95% t区间 |", "|---|---|---:|---:|---|" ]
    for group in payload["paired_statistics"]:
        for name, stats in group["metrics"].items():
            low, high = stats["approximate_t95_interval_df4"]
            lines.append(f"| {group['dataset']} | {name} | 5 | {fmt(stats)} | [{low:.4f}, {high:.4f}] |")
    lines += ["", "n=5的区间只作描述，依赖差值近似正态，未做多重比较修正；跨零或非显著不能解释成复现等价。", "",
              "## 历史与论文独立分列（W-F1）", "",
              "| 数据集/模型 | 本批XP04 validation | 旧XP02 official_test | V100 V2 03 validation | 论文 |",
              "|---|---:|---:|---:|---:|"]
    for group in payload["groups"]:
        key = group["dataset"] + "/" + group["variant"]
        old = payload["historical_comparison"][key]
        lines.append(f"| {key} | {fmt(group['metrics']['weighted_f1'])} | {fmt(old['old_xp02_official_test'])} | "
                     f"{fmt(old['v100v2_03_validation'])} | {old['paper']['weighted_f1']:.2f} |")
    lines += ["", "旧XP02的baseline仅单seed，CMERC五seed集和测试集选模协议也与本批不同，不进入本批统计/配对；V100 V2为同主协议的独立历史硬件组。论文完整五seed未知，模型/实现与原论文声明仍有边界；本文不以数值接近代替协议等价。", "",
              "| 数据集/模型 | 本批XP04 CEL | 旧XP02 CEL | V100 V2 03 CEL | 论文CEL |",
              "|---|---:|---:|---:|---:|"]
    for group in payload["groups"]:
        key = group["dataset"] + "/" + group["variant"]
        old = payload["historical_comparison"][key]
        lines.append(f"| {key} | {fmt(group['metrics']['cel'])} | {fmt(old['old_xp02_cel'])} | "
                     f"{fmt(old['v100v2_03_cel'])} | {old['paper']['cel']:.2f} |")
    environment = payload["environment"]
    runtime = payload["effective_queue_config"]["runtime_environment"]
    lines += ["",
              "## 证据、环境与限制", "",
              f"冻结源摘要 `{payload['source_snapshot_sha256']}`；来源与本批修改见 provenance/origin.json、快照和有效配置；补充逐run证据位于 provenance/run_supplements/，未改写原run_manifest。", "",
              "Python/torch/PyG/scatter/sparse等实际环境见 preflight/environment.json；硬件为本地user23 TITAN Xp，论文RTX4090，历史组V100。只使用配置指定UUID的0/1/3，设备号与进程内cuda:0分开记录，GPU2未作为本批候选。", "",
              "模型为FP32、原BN非因果，batch16。continuous RNG推进训练随机过程，评估保存恢复状态；GRU/LSTM使用ATen以支持完整epoch边界恢复，未将其数值/随机执行差异隐藏成旧cuDNN结果。", "",
              "capacity_safe仅按对话长度重抽超容量的整轮排列，所有eligible样本每轮一次，未按标签或分数筛选；条件重排改变batch联合分布。本验收逐run逐epoch与CPU容量预检核对ratio、样本数、重排次数及最大边/节点。", "",
              "curriculum_source=fixed_author：固定作者外部difficulty，训练前缀只从本轮train IDs选择。文件包含当前valid键且生成过程未知，不能称为train-only curriculum或证明整个预处理validation-free；此限制与已声明协议一致。", "",
              "测试不用于选epoch/seed/超参数；训练JSONL每epoch test字段全部null。训练结束一次final_test及同一best.pt的独立审计评估不形成测试调参。未追加种子或隐藏失败/低分运行。", "",
              "UUID/NVML枚举补丁仅服务设备识别；XP CUDA allocator/缓存清理设置与wrapper以有效配置和源hash为准，分配峰值与reserved/context不同，不能由allocated推断整卡容量。采样探针仅用于预检，正式训练不得启用梯度observer。", "",
              "验收在CPU加载checkpoint并检查完整/有限模型、Adam、RNG字节及元数据；CUDA RNG的可恢复性和真实GPU数值边界由本批预检提供，不能从CPU结构检查推出跨GPU逐位一致。", "",
              "完整机器证据：results/acceptance.json、results/summary.json、逐seed predictions.npz/metrics.json、state/queue.json；本报告全部五指标由保存预测独立重算核对。"]
    lines += ["", "环境包版本：`" + json.dumps(environment.get("packages", {}), sort_keys=True, ensure_ascii=False) + "`。", "",
              "冻结运行环境变量：`" + json.dumps(runtime, sort_keys=True, ensure_ascii=False) + "`。", "",
              "| Dataset | train / valid / test对话数 | 固定split SHA-256 |", "|---|---|---|"]
    for dataset in DATASETS:
        split = next(run["original_provenance"]["split"] for run in payload["runs"] if run["dataset"] == dataset)
        lines.append(f"| {dataset} | " + " / ".join(str(len(split[k])) for k in ("train_ids", "validation_ids", "test_ids"))
                     + f" | `{split['hash']}` |")
    lines += ["", "实际各run进程的UUID、启动命令与训练/评估峰值allocated/reserved见逐run supplement；不将这些耗时与旧并发V100或论文4090直接对比。"]
    return "\n".join(lines) + "\n"


def accept(check_only=False):
    queue, jobs, summary = matrix_gate()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run finalizer with CUDA_VISIBLE_DEVICES='' (CPU only)")
    require(read_json(BATCH / "preflight/status.json").get("status") == "passed",
            "canonical batch preflight/status.json is not passed")
    scheduler = import_scheduler()
    scheduler.validate_config(queue["config"])
    snapshot = scheduler.verify_source_snapshot()
    require((BATCH / "provenance/immutable_inputs.json").is_file(),
            "immutable input/tool/environment-patch manifest is missing")
    require(queue.get("source_snapshot_sha256") == snapshot, "queue source snapshot differs from current frozen source")
    stages = {}
    for stage in (1, 2):
        subset = [job for job in jobs if job["stage"] == stage]
        require(len(subset) == (12 if stage == 1 else 8), "incorrect stage matrix")
        stages[str(stage)] = scheduler.validate_stage(subset, inspect_checkpoints=False)
    # Full matrix validation also compares source/data/split identities across stages.
    stages["all"] = scheduler.validate_stage(jobs, inspect_checkpoints=False)
    capacity_path = BATCH / "preflight/data_and_capacity/batch_capacity_audit.json"
    capacities = capacity_index(read_json(capacity_path))
    require(read_json(BATCH / "preflight/data_and_capacity/input_audit.json").get("status") == "passed",
            "input preflight is not passed")
    cache, truth = {}, {}
    runs = [audit_run(job, capacities[job["dataset"], job["variant"], job["seed"]], cache, truth) for job in jobs]
    for run in runs:
        run["runtime_environment"] = queue["config"]["runtime_environment"]
        run["batch_source_snapshot_sha256"] = snapshot
        run["memory_profiles"] = {}
        for phase, log_field in (("training", "train_log"), ("evaluation", "evaluation_log")):
            path = scheduler.memory_profile_path(run["queue_job"][log_field])
            if path.is_file():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                require(rows and all(row.get("visible_device") in queue["config"]["allowed_gpu_uuids"].values()
                        for row in rows), f"{run['id']}: memory profile contains an unapproved GPU UUID")
                run["memory_profiles"][phase] = {"path": str(path), "sha256": sha256(path),
                    "samples": len(rows), "visible_uuids": sorted({row["visible_device"] for row in rows}),
                    "peak_allocated_mib": max(row["peak_allocated_mib"] for row in rows),
                    "peak_reserved_mib": max(row["peak_reserved_mib"] for row in rows)}
    groups, pairs, paired = audit_statistics(summary, runs)
    payload = {"schema_version": 1, "status": "passed", "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
               "baseline_name": TITLE, "batch": str(BATCH), "runs_completed": 20,
               "score_threshold_used": False, "planned_seeds": SEEDS, "ddof": 1,
               "source_snapshot_sha256": snapshot, "stage_checks": stages,
               "independent_metric_absolute_tolerance": TOLERANCE,
               "capacity_audit_sha256": sha256(capacity_path),
               "canonical_preflight_sha256": sha256(BATCH / "preflight/status.json"),
               "immutable_manifest_sha256": sha256(BATCH / "provenance/immutable_inputs.json"),
               "input_sha256_verified": cache, "finalizer_sha256": sha256(__file__),
               "queue_sha256": sha256(BATCH / "state/queue.json"),
               "summary_sha256": sha256(BATCH / "results/summary.json"),
               "effective_queue_config": queue["config"], "environment": read_json(BATCH / "preflight/environment.json"),
               "runs": runs, "groups": groups, "pairs": pairs, "paired_statistics": paired,
               "historical_comparison": history()}
    if check_only:
        return payload
    for run in runs:
        write_json(BATCH / "provenance/run_supplements" / (run["id"] + ".json"), run)
    write_json(BATCH / "provenance/effective_config.json", queue["config"])
    write_atomic(BATCH / "reports/XP_MAINLINE_REPRODUCTION.md", reproduction_report(payload))
    acceptance_report = (f"# XP 主线验收\n\n通过：20/20，两个阶段及全矩阵协议验收、40个checkpoint、20份保存预测独立重算、1200个epoch容量/课程重放全部通过。\n\n"
                         f"验收时间：{payload['accepted_at_utc']}；源摘要 `{snapshot}`。\n\n"
                         "验收不设分数阈值；模型/Adam有限、epoch与validation选模对齐、每epoch test字段为空、完整随机状态及sampling元数据均核对。\n\n"
                         "CPU检查不证明跨GPU逐位复现；fixed_author生成独立性未知，不宣称train-only。完整证据见 ../results/acceptance.json。\n")
    write_atomic(BATCH / "reports/XP_MAINLINE_ACCEPTANCE.md", acceptance_report)
    write_json(BATCH / "results/acceptance.json", payload)
    # Register last: no incomplete/failed run or metric threshold can grant this name.
    write_atomic(BATCH / "BASELINE_REGISTRY.md", f"# {TITLE}\n\n"
                 f"批次 `{BATCH.name}`；20/20技术验收通过，时间 {payload['accepted_at_utc']}。\n\n"
                 "登记以 results/acceptance.json 当前passed状态和冻结哈希为准；五seed结果见 reports/XP_MAINLINE_REPRODUCTION.md。\n\n"
                 "本文件只登记本批XP研究基线，不覆盖根registry或历史批次。技术验收不意味着优于baseline或论文等价。\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="perform all completed-batch checks without writing outputs")
    options = parser.parse_args()
    try:
        payload = accept(options.check_only)
    except NotReady as error:
        print(json.dumps({"status": "not_ready", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except Exception as error:
        failure = {"status": "failed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                   "batch": str(BATCH), "error": f"{type(error).__name__}: {error}", "baseline_registered_this_attempt": False}
        if not options.check_only:
            write_json(BATCH / "results/acceptance.json", failure)
            write_atomic(BATCH / "reports/XP_MAINLINE_ACCEPTANCE.md", "# XP 主线验收\n\n未通过："
                         + failure["error"] + "\n\n本次不登记基线，不按分数放宽技术条件。\n")
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "runs": 20, "check_only": options.check_only,
                      "source_snapshot_sha256": payload["source_snapshot_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
