#!/usr/bin/env python3
"""Combine training records and independent evaluations into one JSON summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("IEMOCAP", "MELD")
VARIANTS = ("baseline", "cmerc")
PROTOCOLS = ("official_test", "validation")
SEEDS = {"IEMOCAP": 1475, "MELD": 67137}
MULTISEED_SEEDS = (1475, 67137, 2024, 3407, 42)
METRIC_NAMES = ("accuracy", "weighted_f1", "macro_f1", "cel", "ece")
BATCH_POLICY_CONFIG = {"batch_policy": "capacity_safe", "edge_capacity": 1000,
                       "node_capacity": 5200, "batch_max_retries": 10000}
PAPER_REFERENCES = {
    ("IEMOCAP", "baseline"): {"weighted_f1": 69.61, "cel": 1772.54},
    ("IEMOCAP", "cmerc"): {"weighted_f1": 71.98, "cel": 1143.50},
    ("MELD", "baseline"): {"weighted_f1": 65.37, "cel": 799.58},
    ("MELD", "cmerc"): {"weighted_f1": 66.85, "cel": 605.05},
}


def paper_reference(dataset: str, variant: str) -> dict[str, float]:
    return dict(PAPER_REFERENCES[(dataset, variant)])


def best_training_record(
    records: Iterable[dict[str, Any]], protocol: str
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol}")
    records = list(records)
    if not records:
        raise ValueError("training records are empty")
    return max(records, key=lambda record: float(record["selection_weighted_f1"]))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_run(
    run_root: Path,
    result_root: Path,
    dataset: str,
    variant: str,
    protocol: str,
) -> dict[str, Any]:
    seed = SEEDS[dataset]
    run_dir = run_root / variant / f"{dataset}_{protocol}_seed{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    evaluation_path = result_root / variant / dataset / protocol / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)

    records = read_jsonl(metrics_path)
    selected = best_training_record(records, protocol)
    evaluation = read_json(evaluation_path)
    reference = paper_reference(dataset, variant)
    summary = {
        "dataset": dataset,
        "variant": variant,
        "selection_protocol": protocol,
        "seed": seed,
        "epochs_completed": len(records),
        "selected_epoch": int(selected["epoch"]),
        "selection_weighted_f1": float(selected["selection_weighted_f1"]),
        "accuracy": float(evaluation["accuracy"]),
        "weighted_f1": float(evaluation["weighted_f1"]),
        "macro_f1": float(evaluation["macro_f1"]),
        "cel": float(evaluation["cel"]["total"]),
        "calibration": evaluation["calibration"],
        "paper_reference": reference,
    }
    summary["weighted_f1_gap_to_paper"] = summary["weighted_f1"] - reference["weighted_f1"]
    summary["cel_gap_to_paper"] = summary["cel"] - reference["cel"]
    return summary


def finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} contains nonfinite values")
    return number


def describe(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "std_definition": "sample standard deviation, ddof=1"}


def collect_multiseed(run_root: Path, result_root: Path, seeds: Iterable[int],
                      protocol: str = "validation") -> dict[str, Any]:
    """Validate the entire preregistered matrix; never select seeds by test score."""
    seeds = list(seeds)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("planned seeds must be nonempty and unique")
    if protocol != "validation":
        raise ValueError("multiseed primary statistics require validation selection")
    registry, splits, sources, data_hashes = [], {}, set(), {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in seeds:
                run_dir = run_root / variant / f"{dataset}_{protocol}_seed{seed}"
                evaluation_path = result_root / variant / dataset / protocol / f"seed{seed}" / "metrics.json"
                entry = {"dataset": dataset, "variant": variant, "seed": seed,
                         "selection_protocol": protocol, "run_dir": str(run_dir),
                         "evaluation_path": str(evaluation_path), "status": "missing"}
                registry.append(entry)
                if not run_dir.exists():
                    entry["reason"] = "training directory does not exist"
                    continue
                try:
                    manifest = read_json(run_dir / "run_manifest.json")
                    arguments = manifest["args"]
                    provenance = manifest.get("provenance", manifest)
                    split = provenance["split"]
                    if arguments.get("causal") is not False:
                        raise ValueError("explicit noncausal metadata is required")
                    if arguments.get("seed") != seed or arguments.get("Dataset") != dataset:
                        raise ValueError("manifest training seed/dataset does not match path")
                    if arguments.get("selection_protocol") != protocol:
                        raise ValueError("manifest selection protocol mismatch")
                    if arguments.get("rng_policy") != "continuous":
                        raise ValueError("strict group requires continuous RNG policy")
                    if arguments.get("rnn_backend") != "aten":
                        raise ValueError("strict resumable group requires recorded ATen recurrent backend")
                    for key, value in BATCH_POLICY_CONFIG.items():
                        if arguments.get(key) != value:
                            raise ValueError(f"V2 {key} must be recorded as {value!r}; legacy/V1 runs cannot be mixed")
                    if dataset == "IEMOCAP" and arguments.get("split_seed") != 1475:
                        raise ValueError("IEMOCAP split_seed must be fixed at 1475")
                    if not provenance.get("source") or not provenance.get("data"):
                        raise ValueError("source/data provenance missing")
                    sources.add(json.dumps(provenance["source"], sort_keys=True))
                    for filename, digest in provenance["data"].items():
                        data_hashes.setdefault(filename, set()).add(digest)
                    ids = [list(split[key]) for key in ("train_ids", "validation_ids", "test_ids")]
                    if not split.get("hash") or any(not group for group in ids):
                        raise ValueError("split hash/IDs missing")
                    recomputed_hash = hashlib.sha256(json.dumps({key: split[key] for key in
                        ("train_ids", "validation_ids", "test_ids")}, sort_keys=True).encode()).hexdigest()
                    if split["hash"] != recomputed_hash:
                        raise ValueError("split hash does not match recorded IDs")
                    if any(len(group) != len(set(group)) for group in ids):
                        raise ValueError("duplicate split IDs")
                    if any(set(ids[a]) & set(ids[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
                        raise ValueError("split overlap")
                    signature = (split["hash"], *(tuple(sorted(group)) for group in ids))
                    splits.setdefault(dataset, []).append((entry, signature))
                    entry.update({"split_hash": split["hash"], "split_seed": arguments.get("split_seed"),
                                  "rng_policy": arguments["rng_policy"], "provenance": provenance})
                    entry["rnn_backend"] = arguments["rnn_backend"]
                    entry.update({key: arguments[key] for key in BATCH_POLICY_CONFIG})
                    metrics_path = run_dir / "metrics.jsonl"
                    records = read_jsonl(metrics_path) if metrics_path.is_file() else []
                    expected_epochs = 80 if dataset == "IEMOCAP" else 40
                    entry["epochs_completed"] = len(records)
                    if [record["epoch"] for record in records] != list(range(1, len(records) + 1)):
                        raise ValueError("epoch sequence duplicated, missing or reordered")
                    for record in records:
                        if record.get("causal") is not False or record.get("selection_protocol") != protocol:
                            raise ValueError("epoch protocol is causal or mismatched")
                        if record.get("test_weighted_f1") is not None or record.get("test_accuracy") is not None:
                            raise ValueError("per-epoch test scores violate validation protocol")
                        for key in ("selection_weighted_f1", "train_weighted_f1", "total_loss"):
                            finite_number(record[key], key)
                        for key in ("classification_loss", "cc_loss", "mscl_loss", "cscl_loss", "train_loss",
                                    "validation_loss", "train_accuracy", "validation_accuracy", "validation_weighted_f1"):
                            if key in record:
                                finite_number(record[key], key)
                        for key in ("sampler_retries", "max_batch_edges", "max_batch_nodes"):
                            value = finite_number(record[key], key)
                            if not value.is_integer() or value < 0:
                                raise ValueError(f"{key} must be a nonnegative integer")
                        if record["sampler_retries"] > arguments["batch_max_retries"]:
                            raise ValueError("sampler retry count exceeds declared budget")
                        if record["max_batch_edges"] > arguments["edge_capacity"] or record["max_batch_nodes"] > arguments["node_capacity"]:
                            raise ValueError("sampler admitted a batch beyond official model capacity")
                    entry.update(sampler_retries_total=sum(r["sampler_retries"] for r in records),
                                 max_batch_edges=max((r["max_batch_edges"] for r in records), default=0),
                                 max_batch_nodes=max((r["max_batch_nodes"] for r in records), default=0),
                                 max_gpu_memory_mb=max((finite_number(r.get("max_gpu_memory_mb", 0), "max_gpu_memory_mb") for r in records), default=0))
                    if len(records) > expected_epochs:
                        raise ValueError("epoch budget exceeds preregistered protocol")
                    if len(records) < expected_epochs or not (run_dir / "final_test.json").is_file():
                        entry.update(status="pending", reason="full epoch budget/final test not complete")
                        continue
                    if not all((run_dir / name).is_file() for name in ("best.pt", "latest.pt")):
                        raise ValueError("completed run has missing best/latest checkpoint")
                    if not evaluation_path.is_file():
                        entry.update(status="pending", reason="independent checkpoint evaluation missing")
                        continue
                    evaluation = read_json(evaluation_path)
                    if evaluation.get("causal") is not False or evaluation.get("effective_norm") != "BN":
                        raise ValueError("evaluation is causal or norm differs from original BN")
                    # A copied remote bundle retains the original absolute checkpoint path;
                    # the final variant/run/file identity and provenance remain invariant.
                    if Path(evaluation["checkpoint"]).parts[-3:] != (run_dir / "best.pt").parts[-3:]:
                        raise ValueError("evaluation checkpoint belongs to another run")
                    selected = best_training_record(records, protocol)
                    if evaluation.get("seed") != seed or evaluation.get("split_hash") != split["hash"]:
                        raise ValueError("checkpoint evaluation seed/split hash differs from manifest")
                    if evaluation.get("selected_epoch") != selected["epoch"]:
                        raise ValueError("evaluated checkpoint epoch differs from validation-selected epoch")
                    if evaluation.get("provenance") != provenance:
                        raise ValueError("evaluated checkpoint provenance differs from manifest")
                    if evaluation.get("rnn_backend") != arguments["rnn_backend"]:
                        raise ValueError("evaluation recurrent backend differs from training")
                    for key in BATCH_POLICY_CONFIG:
                        if evaluation.get(key) != arguments[key]:
                            raise ValueError(f"evaluation {key} differs from V2 training policy")
                    numbers = {key: finite_number(evaluation[key], key)
                               for key in ("accuracy", "weighted_f1", "macro_f1")}
                    numbers["cel"] = finite_number(evaluation["cel"]["total"], "cel")
                    numbers["ece"] = finite_number(evaluation["calibration"]["ece"], "ece")
                    reference = paper_reference(dataset, variant)
                    entry.update(numbers, status="complete", selected_epoch=selected["epoch"],
                                 paper_reference=reference,
                                 weighted_f1_gap_to_paper=numbers["weighted_f1"] - reference["weighted_f1"],
                                 cel_gap_to_paper=numbers["cel"] - reference["cel"],
                                 elapsed_seconds=sum(float(r.get("elapsed_seconds", 0)) for r in records))
                except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
                    entry.update(status="failed", reason=str(error))
    # A split conflict invalidates the affected data set, including apparent completed pairs.
    for dataset, entries in splits.items():
        if len({signature for _, signature in entries}) > 1:
            for entry in registry:
                if entry["dataset"] == dataset:
                    entry.update(status="failed", reason="different split IDs/hash across seeds or variants")
    if len(sources) > 1 or any(len(hashes) > 1 for hashes in data_hashes.values()):
        for entry in registry:
            if "provenance" in entry:
                entry.update(status="failed", reason="source snapshot or data hash differs across runs")
    complete = [entry for entry in registry if entry["status"] == "complete"]
    groups = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            runs = [entry for entry in complete if entry["dataset"] == dataset and entry["variant"] == variant]
            groups.append({"dataset": dataset, "variant": variant, "planned_n": len(seeds),
                           "valid_n": len(runs), "metrics": {key: describe(run[key] for run in runs) for key in METRIC_NAMES},
                           "selected_epochs": [run["selected_epoch"] for run in runs],
                           "paper_reference": paper_reference(dataset, variant)})
    pairs = []
    by_key = {(r["dataset"], r["variant"], r["seed"]): r for r in complete}
    for dataset in DATASETS:
        for seed in seeds:
            baseline, cmerc = by_key.get((dataset, "baseline", seed)), by_key.get((dataset, "cmerc", seed))
            if baseline and cmerc:
                pairs.append({"dataset": dataset, "seed": seed,
                              "weighted_f1_gain": cmerc["weighted_f1"] - baseline["weighted_f1"],
                              "cel_change": cmerc["cel"] - baseline["cel"], "split_hash": baseline["split_hash"]})
    paired_statistics = [{"dataset": dataset, "planned_n": len(seeds),
                          "weighted_f1_gain": describe(p["weighted_f1_gain"] for p in pairs if p["dataset"] == dataset),
                          "cel_change": describe(p["cel_change"] for p in pairs if p["dataset"] == dataset)} for dataset in DATASETS]
    discovered = sorted({int(path.name.rsplit("_seed", 1)[1]) for path in run_root.glob("*/*_validation_seed*")
                         if path.name.rsplit("_seed", 1)[1].isdigit()})
    return {"schema_version": 1, "protocol": protocol, "causal": False,
            "experiment_generation": "v2", "batch_policy_config": dict(BATCH_POLICY_CONFIG),
            "planned_seeds": seeds, "discovered_seeds": discovered,
            "unplanned_seeds": [seed for seed in discovered if seed not in seeds],
            "planned_runs": len(registry), "valid_runs": len(complete),
            "complete": len(complete) == len(registry), "registry": registry,
            "run_root": str(run_root), "result_root": str(result_root),
            "runs": complete, "groups": groups, "pairs": pairs, "paired_statistics": paired_statistics}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_multiseed_report(payload: dict[str, Any], report: Path, output: Path) -> None:
    def formatted(summary):
        if summary["mean"] is None:
            return "尚无完整运行"
        return f'{summary["mean"]:.4f} ± {summary["std"]:.4f}' if summary["std"] is not None else f'{summary["mean"]:.4f}（n=1，Std尚不可估计）'
    lines = ["# 原始非因果 CMERC 多随机种子复现", "",
             f'状态：{"全部预注册运行完成" if payload["complete"] else "部分完成，尚未完成多 seed 复现"}。有效完整运行 {payload["valid_runs"]}/{payload["planned_runs"]}。', "",
             f'预先固定训练 seeds：{payload["planned_seeds"]}。validation 选模，IEMOCAP split-seed 固定1475，MELD 使用官方划分；原始 BN 非因果路径。', "",
             "所有统计只纳入完整80/40 epoch、严格验证集选模、独立最终评估、有限指标及同一数据划分的运行。样本标准差 ddof=1，未按测试成绩筛选seed。缺失/失败项保留在注册表中。", "",
             "## 每 seed 实际分数", "", "| 数据集 | 模型 | Seed | Acc | W-F1 | Macro-F1 | CEL | ECE | 最佳epoch |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in payload["runs"]:
        lines.append(f'| {run["dataset"]} | {run["variant"]} | {run["seed"]} | ' + " | ".join(f'{run[key]:.4f}' for key in METRIC_NAMES) + f' | {run["selected_epoch"]} |')
    lines += ["", "## 多 seed Mean ± Std", "", "| 数据集 | 模型 | 有效/计划 | Acc | W-F1 | Macro-F1 | CEL | ECE | W-F1均值−论文 | CEL均值−论文 |", "|---|---|---:|---|---|---|---|---|---:|---:|"]
    for group in payload["groups"]:
        gaps = [group["metrics"][key]["mean"] - group["paper_reference"][key] if group["valid_n"] else None for key in ("weighted_f1", "cel")]
        lines.append(f'| {group["dataset"]} | {group["variant"]} | {group["valid_n"]}/{group["planned_n"]} | ' + " | ".join(formatted(group["metrics"][key]) for key in METRIC_NAMES) + " | " + " | ".join(f'{gap:+.4f}' if gap is not None else "待完成" for gap in gaps) + " |")
    lines += ["", "只有全部预注册五seed完成时，上表才是五seed结果；部分阶段的均值按有效数量标注，不外推为五seed结论。", "", "## 同 seed 成对比较", "", "| 数据集 | Seed | CMERC−Baseline W-F1 | CMERC−Baseline CEL |", "|---|---:|---:|---:|"]
    for pair in payload["pairs"]:
        lines.append(f'| {pair["dataset"]} | {pair["seed"]} | {pair["weighted_f1_gain"]:+.4f} | {pair["cel_change"]:+.4f} |')
    for stats in payload["paired_statistics"]:
        lines += ["", f'{stats["dataset"]} 成对增益：W-F1 {formatted(stats["weighted_f1_gain"])}；CEL变化 {formatted(stats["cel_change"])}（负值表示降低）。']
    lines += ["", "## V2批采样与资源记录", "",
              "| 数据集 | 模型 | Seed | 采样重排次数Σ | 最大batch边数 | 最大batch节点数 | GPU峰值MiB |", "|---|---|---:|---:|---:|---:|---:|"]
    for run in payload["runs"]:
        lines.append(f'| {run["dataset"]} | {run["variant"]} | {run["seed"]} | {run["sampler_retries_total"]} | {run["max_batch_edges"]} | {run["max_batch_nodes"]} | {run["max_gpu_memory_mb"]:.2f} |')
    lines += ["", "## 稳定性、最佳epoch与收敛", ""]
    for group in payload["groups"]:
        lines.append(f'- {group["dataset"]} / {group["variant"]}：最佳epoch {group["selected_epochs"]}；W-F1 {formatted(group["metrics"]["weighted_f1"])}，CEL {formatted(group["metrics"]["cel"])}。')
    for stats in payload["paired_statistics"]:
        gain = stats["weighted_f1_gain"]
        cel = stats["cel_change"]
        if gain["n"] >= 2:
            improvement = "提高" if gain["mean"] > 0 else "下降" if gain["mean"] < 0 else "持平"
            reliability = "降低" if cel["mean"] < 0 else "升高" if cel["mean"] > 0 else "持平"
            lines += ["", f'{stats["dataset"]} 在已完成的{gain["n"]}个配对seed中，CMERC平均W-F1{improvement}，CEL{reliability}。增益标准差{gain["std"]:.4f}个百分点；小样本均值与标准差不足以证明所有随机种子均改善，完整配对分数保留上表。']
    lines += ["", "同一数据集全部seed与Baseline/CMERC的train/validation/test IDs及hash必须一致。收敛图仅显示训练/验证Loss与W-F1；测试分数不用于选择epoch。SVG见机器汇总目录 figures/。", "",
              "## 与历史单seed及论文的差异解释", "",
              "论文参考 W-F1：IEMOCAP Baseline69.61 / CMERC71.98，MELD Baseline65.37 / CMERC66.85；CEL分别1772.54 / 1143.50 / 799.58 / 605.05。论文参考协议与validation规范协议不可视为完全等价。",
              "",
              "本组初始化时播种，训练使用continuous RNG、评估保存/恢复RNG、epoch独立DataLoader随机序列；旧单seed保留每次函数调用重新播种的legacy行为。IEMOCAP当前固定split1475，其他训练seed不改变划分。新旧分数差异还包含随机过程策略/可能划分差异，不能全部归因于纯训练随机性。原单seed成果在 ../CMERC，未覆盖。历史分数保存在JSON historical_single_seed 字段，仅作参考。",
              "",
              "PyTorch1.13的cuDNN GRU内部dropout状态无法通过公开RNG checkpoint完整恢复；为保证断点恢复一致，本组仅RNN采用ATen kernels，结构、数学公式与dropout概率不变，其他层保持原实现。checkpoint记录rnn_backend=aten，独立评估应用相同后端；历史官方checkpoint仍使用cuDNN。后端数值与随机序列差异也是新旧实验分数差异的可能原因。",
              "",
              "V1真实训练seed67137第5epoch出现batch N=1001、E=1049，超过官方hyperedge_weight的1000容量并失败；原V1输出与失败记录单独保留。V2全部五seed与Baseline/CMERC统一使用capacity_safe：固定batch size16，仅在整轮随机排列产生超容量batch时重新排列，edge_capacity1000、node_capacity5200、最大重排10000；同一轮应使用该课程阶段的全部样本一次，训练/验证split、课程样本集合与模型数学结构不变。此条件重排改变batch联合分布及随机序列，是与V1/历史单seed比较的明确协议差异。V1部分/完整成绩均不混入V2统计，重排不以测试分数为条件，不筛选seed。",
              "",
              "官方difficulty生成协议未完全公开，外部课程得分可能由包含本轮holdout的原训练数据生成。本组固定使用同一哈希difficulty并明确train/validation对话互斥，但无法宣称所有预处理阶段都不含validation信息。详细协议与资源审计见 [PROTOCOL_AND_RESOURCE_AUDIT.md](PROTOCOL_AND_RESOURCE_AUDIT.md)。",
              "", "Acc、W-F1、Macro-F1、ECE单位均为百分数；ECE采用15等宽bin、max-softmax置信度；CEL沿用原论文五种信息移除后真实类别log置信度正向增长的累计定义。CEL总量只在同一数据集/样本与移除协议下比较。", "", "## 未完成与失败注册表", "", "| 数据集 | 模型 | Seed | 状态 | 原因 |", "|---|---|---:|---|---|"]
    for entry in payload["registry"]:
        if entry["status"] != "complete":
            lines.append(f'| {entry["dataset"]} | {entry["variant"]} | {entry["seed"]} | {entry["status"]} | {entry.get("reason", "")} |')
    if payload["unplanned_seeds"]:
        lines += ["", f'发现未预注册seeds {payload["unplanned_seeds"]}，没有混入预注册统计。']
    command = " ".join(shlex.quote(str(part)) for part in (
        "python", "scripts/summarize_results.py", "--multiseed", "--run-root", payload["run_root"],
        "--result-root", payload["result_root"], "--output", output, "--report", report,
        "--seeds", ",".join(map(str, payload["planned_seeds"]))))
    lines += ["", f'机器汇总：`{output}`，同目录registry.csv、per_seed.csv、paired.csv、aggregate.csv；生成命令：', "", "```bash", command, "```", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def svg_panel(parent: ET.Element, origin: tuple[float, float], title: str,
              xlabel: str, ylabel: str, series: list[dict[str, Any]],
              categories: list[str] | None = None) -> None:
    """Render actual scalar series as editable vector primitives using stdlib."""
    namespace = "http://www.w3.org/2000/svg"
    def add(tag: str, attributes: dict[str, Any], text: str | None = None) -> ET.Element:
        node = ET.SubElement(parent, f"{{{namespace}}}{tag}", {key: str(value) for key, value in attributes.items()})
        node.text = text
        return node
    def label(x, y, text, **attributes):
        return add("text", {"x": x, "y": y, "font-size": 12, **attributes}, str(text))
    left, top = origin[0] + 64, origin[1] + 40
    width, height = 430, 230
    points = [point for curve in series for point in curve["points"]]
    xvalues = [point[0] for point in points]
    yvalues = [point[1] for point in points]
    xmin, xmax = (min(xvalues), max(xvalues)) if xvalues else (0, 1)
    ymin, ymax = (min(yvalues), max(yvalues)) if yvalues else (0, 1)
    if categories:
        xmin, xmax = -0.5, len(categories) - 0.5
    if xmax == xmin:
        xmax += 1
    pad = (ymax - ymin) * .06 if ymax != ymin else max(abs(ymin) * .05, .1)
    ymin, ymax = ymin - pad, ymax + pad
    def coordinates(x, y):
        return (left + width * (x - xmin) / (xmax - xmin), top + height * (ymax - y) / (ymax - ymin))
    label(left + width / 2, origin[1] + 22, title, **{"text-anchor": "middle", "font-weight": "bold"})
    for index in range(5):
        value = ymin + (ymax - ymin) * index / 4
        _, y = coordinates(xmin, value)
        add("line", {"x1": left, "y1": y, "x2": left + width, "y2": y, "stroke": "#dddddd"})
        label(left - 7, y + 4, f"{value:.3g}", **{"text-anchor": "end"})
    ticks = list(enumerate(categories)) if categories else [(xmin + (xmax - xmin) * index / 4, f"{xmin + (xmax - xmin) * index / 4:.3g}") for index in range(5)]
    for value, text in ticks:
        x, _ = coordinates(value, ymin)
        label(x, top + height + 19, text, **{"text-anchor": "middle", "font-size": 10})
    add("path", {"d": f"M{left},{top} V{top + height} H{left + width}", "stroke": "#333333", "fill": "none"})
    label(left + width / 2, top + height + 38, xlabel, **{"text-anchor": "middle"})
    label(left - 47, top + height / 2, ylabel, transform=f"rotate(-90 {left - 47} {top + height / 2})", **{"text-anchor": "middle"})
    for index, curve in enumerate(series):
        rendered = [coordinates(x, y) for x, y in curve["points"]]
        if curve.get("scatter"):
            for x, y in rendered:
                add("circle", {"cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": 4, "fill": curve["color"], "data-series": curve["label"]})
        else:
            attributes = {"points": " ".join(f"{x:.3f},{y:.3f}" for x, y in rendered),
                          "fill": "none", "stroke": curve["color"], "stroke-width": 1.6,
                          "data-series": curve["label"], "data-point-count": len(rendered)}
            if curve.get("dash"):
                attributes["stroke-dasharray"] = curve["dash"]
            add("polyline", attributes)
        # The five-seed matrix has at most fifteen loss curves; three legend columns.
        lx, ly = left + (index % 3) * 147, top + height + 58 + (index // 3) * 16
        add("line", {"x1": lx, "y1": ly - 4, "x2": lx + 16, "y2": ly - 4,
                     "stroke": curve["color"], "stroke-dasharray": curve.get("dash", "none")})
        label(lx + 20, ly, curve["label"], **{"font-size": 9})
    if not series:
        label(left + width / 2, top + height / 2, "No completed runs", **{"text-anchor": "middle"})


def plot_multiseed_svg(payload: dict[str, Any], output_dir: Path) -> None:
    """Dependency-free fallback: no resampling, smoothing, or synthetic points."""
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    for dataset in DATASETS:
        root = ET.Element(f"{{{namespace}}}svg", {"viewBox": "0 0 1100 880", "width": "1100", "height": "880", "font-family": "sans-serif"})
        for column, variant in enumerate(VARIANTS):
            runs = [run for run in payload["runs"] if run["dataset"] == dataset and run["variant"] == variant]
            loss_series, f1_series = [], []
            for index, run in enumerate(runs):
                records = read_jsonl(Path(run["run_dir"]) / "metrics.jsonl")
                color, seed = colors[index % len(colors)], run["seed"]
                def curve(key, label, dash=""):
                    return {"points": [(record["epoch"], finite_number(record[key], key)) for record in records],
                            "label": f"seed{seed} {label}", "color": color, "dash": dash}
                if all(record.get("train_loss") is not None for record in records):
                    loss_series.append(curve("train_loss", "train CE"))
                elif all(record.get("classification_loss") is not None for record in records):
                    loss_series.append(curve("classification_loss", "train CE"))
                if all(record.get("validation_loss") is not None for record in records):
                    loss_series.append(curve("validation_loss", "valid CE", "6,3"))
                if variant == "cmerc":
                    loss_series.append(curve("total_loss", "train total", "2,3"))
                f1_series += [curve("train_weighted_f1", "train W-F1"),
                              curve("validation_weighted_f1", "valid W-F1", "6,3")]
            svg_panel(root, (column * 550, 0), f"{dataset} {variant}: loss", "Epoch", "Loss", loss_series)
            svg_panel(root, (column * 550, 440), f"{dataset} {variant}: W-F1", "Epoch", "W-F1 (%)", f1_series)
        ET.ElementTree(root).write(output_dir / f"{dataset}_convergence.svg", encoding="utf-8", xml_declaration=True)
    root = ET.Element(f"{{{namespace}}}svg", {"viewBox": "0 0 550 460", "width": "550", "height": "460", "font-family": "sans-serif"})
    categories = [f'{group["dataset"]}/{group["variant"]}' for group in payload["groups"]]
    series = []
    for index, seed in enumerate(payload["planned_seeds"]):
        points = [(categories.index(f'{run["dataset"]}/{run["variant"]}'), run["selected_epoch"])
                  for run in payload["runs"] if run["seed"] == seed]
        if points:
            series.append({"points": points, "label": f"seed{seed}", "color": colors[index % len(colors)], "scatter": True})
    svg_panel(root, (0, 0), "Best validation epoch by seed", "Dataset / model", "Selected epoch", series, categories)
    ET.ElementTree(root).write(output_dir / "selected_epochs.svg", encoding="utf-8", xml_declaration=True)


def plot_multiseed(payload: dict[str, Any], output_dir: Path) -> None:
    if not payload["runs"]:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_multiseed_svg(payload, output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for column, variant in enumerate(VARIANTS):
            runs = [run for run in payload["runs"] if run["dataset"] == dataset and run["variant"] == variant]
            for run in runs:
                records = read_jsonl(Path(run["run_dir"]) / "metrics.jsonl")
                epochs = [record["epoch"] for record in records]
                axes[0, column].plot(epochs, [record.get("train_loss", record["total_loss"]) for record in records], label=f'seed{run["seed"]} train CE')
                if variant == "cmerc":
                    axes[0, column].plot(epochs, [record["total_loss"] for record in records], linestyle=":", alpha=.5, label=f'seed{run["seed"]} train total')
                if all(record.get("validation_loss") is not None for record in records):
                    axes[0, column].plot(epochs, [record["validation_loss"] for record in records], linestyle="--", label=f'seed{run["seed"]} valid')
                axes[1, column].plot(epochs, [record["train_weighted_f1"] for record in records], alpha=.5)
                axes[1, column].plot(epochs, [record["validation_weighted_f1"] for record in records], linestyle="--", label=f'seed{run["seed"]} valid')
            for row, metric in enumerate(("Loss", "W-F1 (%)")):
                axes[row, column].set(xlabel="Epoch", ylabel=metric, title=f"{dataset} {variant}")
                axes[row, column].grid(alpha=.2)
                if runs:
                    axes[row, column].legend(fontsize=7)
        fig.savefig(output_dir / f"{dataset}_convergence.svg")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = [f'{g["dataset"]}/{g["variant"]}' for g in payload["groups"]]
    values = [g["selected_epochs"] for g in payload["groups"]]
    for index, epochs in enumerate(values):
        ax.scatter([index] * len(epochs), epochs)
    ax.set(xticks=list(range(len(labels))), xticklabels=labels, ylabel="Selected validation epoch")
    ax.grid(alpha=.2)
    fig.savefig(output_dir / "selected_epochs.svg")
    plt.close(fig)


def save_multiseed(payload: dict[str, Any], output: Path, report: Path, historical: Path,
                   plots: bool = True) -> None:
    payload["historical_single_seed"] = read_json(historical) if historical.is_file() else {"status": "not found", "path": str(historical)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["dataset", "variant", "seed", "status", "reason", "epochs_completed", "split_hash", "split_seed", "rng_policy", "rnn_backend", *BATCH_POLICY_CONFIG,
              "sampler_retries_total", "max_batch_edges", "max_batch_nodes", "max_gpu_memory_mb",
              *METRIC_NAMES, "selected_epoch", "weighted_f1_gap_to_paper", "cel_gap_to_paper"]
    write_csv(output.parent / "registry.csv", payload["registry"], fields)
    write_csv(output.parent / "per_seed.csv", payload["runs"], fields)
    write_csv(output.parent / "paired.csv", payload["pairs"], ["dataset", "seed", "weighted_f1_gain", "cel_change", "split_hash"])
    aggregates = [{"dataset": group["dataset"], "variant": group["variant"], "metric": key,
                   "planned_n": group["planned_n"], **group["metrics"][key]} for group in payload["groups"] for key in METRIC_NAMES]
    write_csv(output.parent / "aggregate.csv", aggregates, ["dataset", "variant", "metric", "planned_n", "n", "mean", "std", "std_definition"])
    if plots:
        plot_multiseed(payload, output.parent / "figures")
    render_multiseed_report(payload, report, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--multiseed", action="store_true")
    parser.add_argument("--seeds", "--expected-seeds", default=",".join(map(str, MULTISEED_SEEDS)))
    parser.add_argument("--protocol", default="validation", choices=PROTOCOLS)
    parser.add_argument("--report", type=Path, default=Path("reports/10_MULTI_SEED_REPRODUCTION.md"))
    parser.add_argument("--historical", type=Path, default=Path("../CMERC/results/formal/summary.json"))
    parser.add_argument("--no-plots", action="store_true", help="CPU checks without optional plotting imports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = "multiseed" if args.multiseed else "formal"
    args.run_root = args.run_root or Path("runs") / mode
    args.result_root = args.result_root or Path("results") / mode
    args.output = args.output or args.result_root / "summary.json"
    if args.multiseed:
        payload = collect_multiseed(args.run_root, args.result_root, [int(seed) for seed in args.seeds.split(",")], args.protocol)
        save_multiseed(payload, args.output, args.report, args.historical, plots=not args.no_plots)
        print(f'saved multiseed registry: {payload["valid_runs"]}/{payload["planned_runs"]} valid runs -> {args.output}')
        return 0
    runs = [
        summarize_run(args.run_root, args.result_root, dataset, variant, protocol)
        for dataset in DATASETS
        for variant in VARIANTS
        for protocol in PROTOCOLS
    ]
    by_key = {
        (run["dataset"], run["variant"], run["selection_protocol"]): run
        for run in runs
    }
    for run in runs:
        baseline = by_key[(run["dataset"], "baseline", run["selection_protocol"])]
        run["weighted_f1_gain_over_baseline"] = run["weighted_f1"] - baseline["weighted_f1"]
        run["cel_reduction_from_baseline"] = baseline["cel"] - run["cel"]

    payload = {"runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"saved {len(runs)} run summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
