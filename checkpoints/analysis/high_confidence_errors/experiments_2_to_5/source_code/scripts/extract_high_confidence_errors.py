#!/usr/bin/env python3
"""Full-input, test-only diagnostics for verified M3Net/CMERC checkpoints.

This script never trains, selects checkpoints, applies modality removal, or
updates an existing prediction file. The default checkpoints are a predeclared
validation-selected pair from the completed local XP experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from high_confidence_analysis import compare_records, make_records, write_model_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = PROJECT_ROOT / "experiments/xp_mainline_validation/20261016T050823Z"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints/analysis/high_confidence_errors"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("comparison must contain the complete nonempty test set")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_identity(checkpoint, expected_model, dataset):
    saved = checkpoint.get("args")
    if not isinstance(saved, dict):
        raise ValueError("training identity is unverified: checkpoint has no args")
    flags = {name: saved.get(name) for name in
             ("calibrate", "contrastlearning", "courselearning")}
    expected_flags = expected_model == "cmerc"
    if any(value is not expected_flags for value in flags.values()):
        raise ValueError(f"checkpoint identity contradicts {expected_model}: {flags}")
    required = {"Dataset": dataset, "selection_protocol": "validation",
                "causal": False, "multi_modal": True, "modals": "avl",
                "batch_size": 16, "norm": "BN", "base_model": "GRU",
                "graph_type": "hyper", "mm_fusion_mthd": "concat_DHT"}
    for name, expected in required.items():
        if saved.get(name) != expected:
            raise ValueError(f"unsupported checkpoint {name}: {saved.get(name)!r}")
    return saved, flags


def load_frozen_evaluator(batch_root):
    code = batch_root / "code"
    path = code / "scripts/evaluate_checkpoint.py"
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("hc_frozen_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "configure_evaluation_backend"):
        raise ValueError("frozen evaluator must preserve the recorded RNN backend")
    return module


def infer_full_input(evaluator, checkpoint, dataset, device):
    """Exactly one full-input forward per test batch; no auxiliary models."""
    import torch

    config = evaluator.dataset_config(dataset)
    _, _, Model, _, _ = evaluator._import_official_modules()
    model = Model(**evaluator._model_arguments(config, device, causal=False))
    model.load_state_dict(evaluator.extract_state_dict(checkpoint), strict=True)
    backend = evaluator.configure_evaluation_backend([model], checkpoint)
    model.to(device).eval()
    loader = evaluator.build_test_loader(config, batch_size=16, num_workers=0)
    source = loader.dataset
    expected_keys = [(str(vid), index) for vid in source.keys
                     for index in range(len(source.videoLabels[vid]))]
    labels_all, log_probs_all, ids_all, indices_all, texts_all = [], [], [], [], []
    original_ids, batch_audit = [], []
    with torch.no_grad():
        for number, batch in enumerate(loader):
            # Preserve the official padded batch and dialogue order.
            lengths = [int(mask.sum().item()) for mask in batch[7]]
            flat_labels = torch.cat([batch[8][b, :length]
                                     for b, length in enumerate(lengths)])
            batch_ids, indices = evaluator.expand_utterance_ids(batch[-1], lengths)
            batch_texts, source_ids = [], []
            for b, (vid, length) in enumerate(zip(batch[-1], lengths)):
                mask = batch[7][b]
                expected_mask = torch.zeros_like(mask)
                expected_mask[:length] = 1
                if length <= 0 or not torch.equal(mask, expected_mask):
                    raise ValueError("mask is not a nonempty right-padded prefix")
                if len(source.videoLabels[vid]) != length or len(source.videoSentence[vid]) != length:
                    raise ValueError("raw labels/text length differs from batch mask")
                if not torch.equal(batch[8][b, :length],
                                   torch.as_tensor(source.videoLabels[vid], dtype=batch[8].dtype)):
                    raise ValueError("raw test labels differ from flattened batch labels")
                # Each real feature is passed through unchanged; none is replaced.
                for feature in batch[:7]:
                    if feature.shape[0] < length or feature.shape[1] != len(lengths):
                        raise ValueError("feature time/batch dimensions do not align")
                for index in range(length):
                    text = source.videoSentence[vid][index]
                    if not isinstance(text, str):
                        raise ValueError("raw text must already be a decoded string")
                    batch_texts.append(text)
                    source_ids.append(str(source.videoIDs[vid][index]))
            tensors = [tensor.to(device) for tensor in batch[:-1]]
            text1, text2, text3, text4, visual, audio, qmask, umask, _ = tensors
            log_prob = model([text1, text2, text3, text4], qmask, umask,
                             lengths, audio, visual, 0)[0]
            if tuple(log_prob.shape) != (sum(lengths), config.num_classes):
                raise ValueError("model output rows do not match valid utterances")
            values = log_prob.detach().cpu().numpy()
            evaluator.validate_log_probabilities(values)
            labels_all.append(flat_labels.numpy())
            log_probs_all.append(values)
            ids_all.extend(batch_ids)
            indices_all.extend(indices)
            texts_all.extend(batch_texts)
            original_ids.extend(source_ids)
            batch_audit.append({"batch_index": number, "dialogue_ids": [str(x) for x in batch[-1]],
                                "lengths": lengths, "padded_time": int(batch[0].shape[0]),
                                "valid_utterances": sum(lengths), "full_forward_calls": 1})
    actual_keys = list(zip(ids_all, indices_all))
    if actual_keys != expected_keys or len(set(actual_keys)) != len(expected_keys):
        raise ValueError("test membership/order or unique utterance identity mismatch")
    labels = np.concatenate(labels_all)
    log_probs = np.concatenate(log_probs_all)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return config, labels, log_probs, ids_all, indices_all, texts_all, original_ids, {
        "rnn_backend": backend, "batch_size": 16, "shuffle": False,
        "valid_utterances": len(labels), "test_dialogues": len(source.keys),
        "raw_label_and_text_alignment": "passed", "key_order_alignment": "passed",
        "mask_prefix_check": "passed", "modality_removal_calls": 0, "batches": batch_audit,
    }


def verify_existing_predictions(path, labels, log_probs, ids, indices):
    """Read-only comparison with original full-input outputs, never removed_* views."""
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    with np.load(path, allow_pickle=False) as old:
        for name, actual in [("labels", labels), ("dialogue_ids", np.asarray(ids, dtype=str)),
                             ("utterance_indices", np.asarray(indices))]:
            if not np.array_equal(old[name].astype(actual.dtype), actual):
                raise ValueError(f"saved full-input prediction {name} is misaligned")
        if not np.array_equal(old["predictions"], log_probs.argmax(axis=1)):
            raise ValueError("new predictions differ from saved full-input predictions")
        difference = float(np.max(np.abs(old["log_probabilities"] - log_probs)))
        if not np.allclose(old["log_probabilities"], log_probs, rtol=1e-5, atol=1e-5):
            raise ValueError(f"saved/new log probabilities differ: max abs={difference}")
    return {"status": "passed", "path": str(path), "labels_ids_indices_predictions": "exact",
            "log_probability_max_absolute_difference": difference,
            "log_probability_tolerance": {"rtol": 1e-5, "atol": 1e-5}}


def display_text(text):
    # Only the Markdown display is escaped/truncated; CSV retains original text.
    value = "".join(f"\\u{ord(c):04x}" if ord(c) < 32 or 127 <= ord(c) <= 159 else c
                    for c in text)
    return value[:240].replace("|", "\\|")


def render_reports(output, metadata, records_by_key, comparisons):
    lines = ["# 完整三模态高置信度错误分析", "",
             "仅分析测试集，未重新训练、未执行模态掩码或上下文移除。主分析使用已完成 XP validation 批次、预先固定的公开 seed；不按测试分数挑选 seed 或 checkpoint。", "",
             "## 总体统计", "",
             "高置信度错误比例的分母是所有达到该阈值的预测，包括正确预测；表中的概率阈值为 0–1。", "",
             "| 数据集 | 模型 | 测试话语 | 错误总数 | ≥0.8 预测数 | ≥0.8 错误数 | 错误比例 | ≥0.9 预测数 | ≥0.9 错误数 | 错误比例 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, rows in records_by_key.items():
        summary = metadata["models"][key]["statistics"]
        a, b = summary["thresholds"]["0.8"], summary["thresholds"]["0.9"]
        rate = lambda item: ("N/A" if item["error_rate_among_high_confidence"] is None
                             else f'{100 * item["error_rate_among_high_confidence"]:.2f}%')
        lines.append(f'| {rows[0]["dataset"]} | {rows[0]["model"]} | {len(rows)} | '
                     f'{summary["error_count"]} | {a["high_confidence_count"]} | '
                     f'{a["high_confidence_error_count"]} | {rate(a)} | '
                     f'{b["high_confidence_count"]} | {b["high_confidence_error_count"]} | {rate(b)} |')
    lines += ["", "## 逐话语对比", ""]
    for dataset, rows in comparisons.items():
        counts = Counter((bool(r["baseline_is_error"]), bool(r["cmerc_is_error"])) for r in rows)
        lines.append(f'- {dataset}：同一 ID/索引、相同真实标签与原文，匹配 {len(rows)} 条。'
                     f'Baseline 错而 CMERC 对 {counts[(True, False)]} 条；Baseline 对而 CMERC 错 '
                     f'{counts[(False, True)]} 条；两者均错 {counts[(True, True)]} 条；两者均对 '
                     f'{counts[(False, False)]} 条。')
        for threshold in (0.8, 0.9):
            baseline_high = [r for r in rows if r["baseline_is_error"]
                             and r["baseline_max_probability"] >= threshold]
            cmerc_high = [r for r in rows if r["cmerc_is_error"]
                          and r["cmerc_max_probability"] >= threshold]
            corrected = sum(not r["cmerc_is_error"] for r in baseline_high)
            lines.append(f'  - 阈值 {threshold}：Baseline 高置信错 {len(baseline_high)} 条，'
                         f'CMERC 高置信错 {len(cmerc_high)} 条；前者中有 {corrected} 条被 CMERC 正确分类。')
    lines += ["", "高置信错误数的变化同时受总错误数与置信度分布影响；不能只凭固定阈值下的数量判断整体准确率或校准性能。"]
    lines += ["", "## 类别错误分布与典型样本", "",
              "每个模型的 class_error_distribution.csv 提供真实情绪类别分布；statistics.json 另保留混淆对。以下按错误置信度降序展示，每组最多 3 条，不用于选择模型。CSV 保存完整原文；本页仅转义控制字符并截取前 240 字符。", ""]
    for key, rows in records_by_key.items():
        wrong = sorted((r for r in rows if r["is_error"]),
                       key=lambda r: (-r["max_probability"], r["dialogue_id"], r["utterance_index"]))
        lines += [f"### {key}", "", "| 真实情绪 | 测试支持数 | 错误数 | 类内错误率 | ≥0.8 错误数 | ≥0.9 错误数 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for item in metadata["models"][key]["statistics"]["class_error_distribution"].values():
            rate = "N/A" if item["error_rate"] is None else f'{100 * item["error_rate"]:.2f}%'
            lines.append(f'| {item["true_emotion"]} | {item["num_samples"]} | {item["error_count"]} | '
                         f'{rate} | {item["high_confidence_errors_ge_0p8"]} | '
                         f'{item["high_confidence_errors_ge_0p9"]} |')
        lines += ["", "| 对话 ID | 索引（0-based） | 真实 → 预测 | 最大概率 | 真实类概率 | 原文 |",
                  "| --- | ---: | --- | ---: | ---: | --- |"]
        for row in wrong[:3]:
            lines.append(f'| {row["dialogue_id"]} | {row["utterance_index"]} | '
                         f'{row["true_emotion"]} → {row["predicted_emotion"]} | '
                         f'{row["max_probability"]:.6f} | {row["true_class_probability"]:.6f} | '
                         f'{display_text(row["text"])} |')
        pairs = Counter((r["true_emotion"], r["predicted_emotion"]) for r in wrong)
        if pairs:
            pair, count = pairs.most_common(1)[0]
            lines += ["", f"最常见错误混淆对：{pair[0]} → {pair[1]}，{count} 条。", ""]
    lines += ["## 解释与限制", "",
              "- 这是固定 checkpoint 的描述性分析。高置信度错误体现 softmax 置信度与正确性不一致，不代表已证明某个训练模块的因果效果；两个模型不保证置信度校准尺度相同。",
              "- 统计包含全部有效测试话语。只筛选错误不会改变真实标签或模型概率，不能从此表直接推断所有预测的准确率。",
              "- 原 train.py 以及 official_test 流程存在逐轮测试集选模问题；官方 README 复现的 best 也按测试 W-F1 选择。本次排除这些 checkpoint 作为主对比，选用 metadata 明确记载 validation 的 checkpoint。",
              "- Baseline 与 CMERC 主模型结构相同，身份以内部 calibrate/contrastlearning/courselearning 开关和运行 manifest 为据。checkpoints/ 下两份作者 plain state_dict 无内部训练配置，清单标为身份待确认，没有冒充 Baseline。",
              "- MELD dialogue_id 是特征 pickle 内部全局编号，未经映射不能当作原始 CSV Dialogue_ID。utterance_index 从 0 计数，source_utterance_id 另保留 pickle 中的原始 ID。",
              "- 保留原始 batch16 和测试话语排列：未 pack 的双向 RNN、padding 和图层批次位置可能影响预测，不任意调 batch 或按长度重排。",
              "- 原文来自特征包 videoSentence，不做拼写或编码修正；MELD 部分原文带控制字符，CSV 原样保存。",
              "- 所有 checkpoint、输入和已有预测的 SHA-256 在推理前后核对一致；完整来源、环境、逐 batch 对齐及预测一致性见 run_manifest.json。", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    command = ("CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro "
               "CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 "
               "bash scripts/with_cmerc_env.sh python scripts/extract_high_confidence_errors.py "
               f"--batch-root {metadata['batch_root']} --output-dir "
               "checkpoints/analysis/high_confidence_errors_rerun --device cuda:0 "
               f"--checkpoint-inventory-json {output / 'checkpoint_inventory.json'}")
    readme = ["# 高置信度错误诊断：复现说明", "", f"生成时间：{metadata['started_at']}。服务器：{metadata['host']}。", "",
              "结果位于 checkpoints/analysis/high_confidence_errors/，对应用户要求的 checkpoint 目录内 analysis/high_confidence_errors/。", "",
              "## 执行命令", "", "在 CMERC 仓库根目录执行；先复查 GPU 资源与 UUID。下例使用本地宿主机 XP3，在进程内是 cuda:0；应根据执行时资源选择健康卡。", "", "```bash", command, "```", "",
              "诊断拒绝覆盖已有输出目录。重跑应指定新目录。TMPDIR 如根分区已满，应设为 /data2 上的独立临时目录。模型和所有文件均只读加载，未调用训练或优化器。", "",
              "## Checkpoint 来源与身份", "", "主对比固定为同一已完成 XP validation 批次的两个公开 seed，而非搜索测试集最高分。冻结代码来自该批次 code/；batch16、BN、非因果、完整 avl、ATen RNN 后端与原评估一致。", "",
              "| 数据集/模型 | checkpoint | seed | 验证集选中 epoch（1-based） |", "| --- | --- | ---: | ---: |"]
    for key, item in metadata["models"].items():
        readme.append(f"| {key} | `{item['checkpoint']}` | {item['training_args']['seed']} | {item['selected_epoch']} |")
    readme += ["", "全部实际审计的候选身份与排除原因见 checkpoint_inventory.json；未独立核验的其他 seed、attempt、latest/final 不宣称已审计。", "",
               "## 数据和列定义", "", "- IEMOCAP：data/IEMOCAP_features.pkl、data/iemocap_features_roberta.pkl；6 类按 happy/sad/neutral/angry/excited/frustrated。",
               "- MELD：data/MELD_features_raw1.pkl、data/meld_features_roberta.pkl；7 类按 neutral/surprise/fear/sadness/joy/disgust/anger。",
               "- 输入数据真实路径与 SHA-256、标签映射、代码哈希、模型参数和原评测对比均在 run_manifest.json；使用实际 DataLoader 的 test ID 顺序，不使用另一份 RoBERTa 列表补 ID。",
               "- 每个数据集/模型下 all_predictions.csv 包含全部话语，errors_all.csv 包含全部错误，errors_ge_0p8.csv / errors_ge_0p9.csv 使用 >= 阈值，错误表按置信度降序；均含所有类别 probability_* 列。",
               "- class_error_distribution.csv 与 statistics.json 保存每类统计；各数据集 baseline_vs_cmerc_all.csv 保留全量逐样本对比，baseline_vs_cmerc_errors_union.csv 为至少一模型预测错误的子集。",
               "- max_probability 是预测类别概率，true_class_probability 是真实类别概率。模型输出为 log_softmax，使用 exp 还原概率，不做温度缩放或再校准。",
               "- 高置信度错误比例 = 错误且最大概率>=阈值的条数 / 所有最大概率>=阈值的预测条数；分母为零时记 null / N/A。",
               "- CSV 使用 UTF-8 BOM，按标准 CSV 引号保留原始逗号、引号、换行和控制字符。原文包含换行时，文件物理行数不等于样本行数，请用 CSV reader 读取。", "",
               "## 只读与验证边界", "", "本次重新加载 checkpoint，执行了真实测试推理；没有使用已有预测代替推理。已有 NPZ 仅在新推理完成后用于核对完整输入输出，不读取 removed_* 作为本分析数据。未修改 checkpoint、标签、特征或旧预测；推理前后输入 SHA256 一致。没有重新训练，也未用本次测试统计选择 checkpoint、seed 或参数。", ""]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-inventory-json", type=Path)
    args = parser.parse_args()
    batch = args.batch_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(p.name != "checkpoint_inventory.json" for p in output.iterdir()):
        parser.error("output directory already contains a diagnostic run; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    inventory = args.checkpoint_inventory_json or output / "checkpoint_inventory.json"
    if not inventory.is_file():
        parser.error("provide the pre-inference identity audit via --checkpoint-inventory-json")
    if inventory.resolve() != output / "checkpoint_inventory.json":
        shutil.copyfile(inventory, output / "checkpoint_inventory.json")
    selected = [(ds, role, seed, batch / "runs" / role / f"{ds}_validation_seed{seed}" / "best.pt")
                for ds, seed in [("IEMOCAP", 1475), ("MELD", 67137)]
                for role in ["baseline", "cmerc"]]
    data_files = [batch / "code/data" / name for name in
                  ("IEMOCAP_features.pkl", "iemocap_features_roberta.pkl",
                   "MELD_features_raw1.pkl", "meld_features_roberta.pkl")]
    references = [batch / "results" / role / ds / "validation" / f"seed{seed}" / "predictions.npz"
                  for ds, role, seed, _ in selected]
    tracked = [p for *_, p in selected] + data_files + [p for p in references if p.is_file()]
    before = {str(p): {"sha256": sha256(p), "resolved_path": str(p.resolve())} for p in tracked}
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("the unchanged noncausal graph requires a healthy CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    evaluator = load_frozen_evaluator(batch)
    os.chdir(batch / "code")
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv"], text=True),
                "batch_root": str(batch), "output_dir": str(output), "input_files_before": before,
                "thresholds": [0.8, 0.9], "split": "test", "training": False,
                "checkpoint_selection_rule": "predeclared completed validation pair; original public seed per dataset; no test-score search",
                "code_sha256": {str(p): sha256(p) for p in
                                [Path(__file__), Path(__file__).with_name("high_confidence_analysis.py")]
                                + [batch / "code" / n for n in ["model.py", "model_cm.py", "model_hyper.py", "dataloader.py", "train_cm.py", "scripts/evaluate_checkpoint.py"]]},
                "models": {}}
    save_json(output / "run_manifest.json", metadata)
    records_by_key, comparisons = {}, {}
    for ds, role, seed, path in selected:
        print(f"Inferring full inputs: {ds}/{role}, checkpoint={path}", flush=True)
        payload = torch.load(path, map_location="cpu")
        saved, flags = verified_identity(payload, role, ds)
        config, labels, log_probs, ids, indices, texts, original_ids, alignment = infer_full_input(evaluator, payload, ds, device)
        records = make_records(ds, role, labels, log_probs, ids, indices, texts, config.class_names)
        for row, source_id in zip(records, original_ids):
            row["source_utterance_id"] = source_id
        model_output = output / ds / role
        model_output.mkdir(parents=True, exist_ok=True)
        stats = write_model_outputs(model_output, records, config.class_names)
        reference = batch / "results" / role / ds / "validation" / f"seed{seed}" / "predictions.npz"
        consistency = verify_existing_predictions(reference, labels, log_probs, ids, indices)
        key = f"{ds}/{role}"
        metadata["models"][key] = {"checkpoint": str(path), "checkpoint_sha256": before[str(path)]["sha256"],
                                   "training_args": saved, "identity_flags": flags,
                                   "selected_epoch": int(payload["epoch"]) + 1,
                                   "class_names": list(config.class_names), "statistics": stats,
                                   "alignment": alignment, "existing_prediction_comparison": consistency,
                                   "classification_metrics": evaluator.classification_metrics(labels, log_probs.argmax(axis=1), config.class_names)}
        records_by_key[key] = records
        save_json(model_output / "statistics.json", metadata["models"][key])
        save_json(output / "run_manifest.json", metadata)
        print(f"Completed {key}: {len(records)} utterances, {stats['error_count']} errors; reference={consistency['status']}", flush=True)
        del payload
        gc.collect()
    for ds in ["IEMOCAP", "MELD"]:
        rows = compare_records(records_by_key[f"{ds}/baseline"], records_by_key[f"{ds}/cmerc"])
        comparisons[ds] = rows
        write_csv(output / ds / "baseline_vs_cmerc_all.csv", rows)
        wrong = [r for r in rows if r["baseline_is_error"] or r["cmerc_is_error"]]
        write_csv(output / ds / "baseline_vs_cmerc_errors_union.csv", wrong)
    after = {str(p): sha256(p) for p in tracked}
    if any(after[path] != info["sha256"] for path, info in before.items()):
        raise RuntimeError("a checkpoint, input or original prediction file changed during diagnosis")
    metadata.update({"completed_at": datetime.now(timezone.utc).isoformat(), "status": "completed",
                     "input_files_after_sha256": after, "input_immutability": "passed"})
    render_reports(output, metadata, records_by_key, comparisons)
    save_json(output / "run_manifest.json", metadata)
    print(f"All diagnostics complete: {output}", flush=True)


if __name__ == "__main__":
    main()
