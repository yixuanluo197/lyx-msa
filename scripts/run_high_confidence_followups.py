#!/usr/bin/env python3
"""Reuse experiment 1; infer only three mean-replaced views for experiments 2–5."""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP1 = PROJECT_ROOT / "checkpoints/analysis/high_confidence_errors"
CONDITIONS = {"w/o_text": "text", "w/o_audio": "audio", "w/o_visual": "visual"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_hash(tensor):
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def audit_replacement(original, modified, removed):
    """Check exact official temporal means, unchanged shapes and non-target views."""
    import torch

    if removed not in ("text", "audio", "visual"):
        raise ValueError("unknown removed modality")
    changes, hashes = 0, {}
    for modality in ("text", "audio", "visual"):
        before = original[modality] if modality == "text" else [original[modality]]
        after = modified[modality] if modality == "text" else [modified[modality]]
        if len(before) != len(after):
            raise ValueError("replacement feature count changed")
        hashes[modality] = []
        for source, target in zip(before, after):
            if source.ndim != 3 or source.shape != target.shape:
                raise ValueError("replacement must preserve [time,batch,feature] shape")
            if not torch.isfinite(source).all() or not torch.isfinite(target).all():
                raise ValueError("non-finite feature input")
            if modality == removed:
                expected = source.mean(dim=0).repeat(source.shape[0], 1, 1)
                if not torch.equal(expected, target):
                    raise ValueError("replacement differs from the official temporal mean")
                changes += int(torch.count_nonzero(source != target).item())
            elif not torch.equal(source, target):
                raise ValueError("non-target modality was changed")
            hashes[modality].append(tensor_hash(target))
    return {"mean_axis": 0, "padding_included_in_mean": True,
            "non_target_modalities_unchanged": True,
            "target_changed_elements": changes, "input_sha256": hashes}


def write_tables(output, tables, fields, only_non_ablation=False):
    counts = {}
    for name, rows in tables.items():
        if only_non_ablation and name.startswith("exp4_"):
            continue
        path = output / f"{name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields[name])
            writer.writeheader()
            writer.writerows({key: ("N/A" if row.get(key) is None else row[key])
                              for key in fields[name]} for row in rows)
        counts[name] = len(rows)
    return counts


def infer_removed_views(evaluator, payload, dataset, full, device):
    """No full forward: only w/o_text, w/o_audio, w/o_visual on the original batches."""
    import torch

    config = evaluator.dataset_config(dataset)
    _, _, Model, _, _ = evaluator._import_official_modules()
    model = Model(**evaluator._model_arguments(config, device, causal=False))
    model.load_state_dict(evaluator.extract_state_dict(payload), strict=True)
    backend = evaluator.configure_evaluation_backend([model], payload)
    model.to(device).eval()
    loader = evaluator.build_test_loader(config, batch_size=16, num_workers=0)
    result = {condition: [] for condition in CONDITIONS}
    audits, offset = [], 0
    with torch.no_grad():
        for number, batch in enumerate(loader):
            lengths = [int(mask.sum().item()) for mask in batch[7]]
            ids, indices = evaluator.expand_utterance_ids(batch[-1], lengths)
            labels = torch.cat([batch[8][index, :length] for index, length in enumerate(lengths)]).numpy()
            rows = full[offset:offset + sum(lengths)]
            if len(rows) != sum(lengths):
                raise ValueError("original full prediction count differs from test loader")
            raw = loader.dataset
            for b, (vid, length) in enumerate(zip(batch[-1], lengths)):
                mask = batch[7][b]
                if not (torch.all(mask[:length] == 1) and torch.all(mask[length:] == 0)):
                    raise ValueError("padding mask is not a valid prefix")
                if len(raw.videoLabels[vid]) != length:
                    raise ValueError("raw dialogue length mismatch")
            for row, vid, index, label in zip(rows, ids, indices, labels):
                raw_vid = next(v for v in batch[-1] if str(v) == vid)
                if (row["dialogue_id"], row["utterance_index"], row["true_label"]) != (vid, index, int(label)):
                    raise ValueError("experiment 1 sample ID/label/order differs from this loader")
                if row["utterance_text"] != raw.videoSentence[raw_vid][index]:
                    raise ValueError("original text differs from experiment 1")
                if row["source_utterance_id"] != str(raw.videoIDs[raw_vid][index]):
                    raise ValueError("source utterance ID mismatch")
            tensors = [x.to(device) for x in batch[:-1]]
            t1, t2, t3, t4, visual, audio, qmask, umask, _ = tensors
            original = {"text": [t1, t2, t3, t4], "audio": audio, "visual": visual}
            original_hashes = {name: [tensor_hash(t) for t in (value if name == "text" else [value])]
                               for name, value in original.items()}
            replaced_text, replaced_visual, replaced_audio = evaluator._masked_modalities(
                original["text"], visual, audio, umask=umask, causal=False)
            replacements = {"text": replaced_text, "audio": replaced_audio, "visual": replaced_visual}
            audit = {"batch_index": number, "dialogue_ids": [str(v) for v in batch[-1]],
                     "lengths": lengths, "padded_time": int(t1.shape[0]), "conditions": {}}
            for condition, removed in CONDITIONS.items():
                view = dict(original)
                view[removed] = replacements[removed]
                check = audit_replacement(original, view, removed)
                logp = model(view["text"], qmask, umask, lengths, view["audio"], view["visual"], 0)[0]
                if tuple(logp.shape) != (sum(lengths), len(config.class_names)):
                    raise ValueError("masked output does not match valid utterances/classes")
                values = logp.detach().cpu().numpy()
                evaluator.validate_log_probabilities(values)
                probabilities = np.exp(values.astype(np.float64))
                for row, probs in zip(rows, probabilities):
                    predicted = int(probs.argmax())
                    record = dict(row)
                    record.update(input_condition=condition, predicted_label=predicted,
                                  predicted_emotion=config.class_names[predicted],
                                  predicted_confidence=float(probs[predicted]),
                                  true_label_probability=float(probs[row["true_label"]]),
                                  correct=predicted == row["true_label"])
                    record.update({f"probability_{name}": float(probs[i])
                                   for i, name in enumerate(config.class_names)})
                    result[condition].append(record)
                check.update({"forward_calls": 1, "valid_utterances": len(rows)})
                audit["conditions"][condition] = check
            for name, value in original.items():
                current = [tensor_hash(t) for t in (value if name == "text" else [value])]
                if current != original_hashes[name]:
                    raise ValueError("original full-input tensor was modified in place")
            audits.append(audit)
            offset += len(rows)
    if offset != len(full):
        raise ValueError("not all original full samples were covered")
    for condition in CONDITIONS:
        if sum(a["conditions"][condition]["target_changed_elements"] for a in audits) <= 0:
            raise ValueError("removal condition never changed its target features")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [row for condition in CONDITIONS for row in result[condition]], {
        "rnn_backend": backend, "batch_size": 16, "shuffle": False,
        "full_forward_calls": 0, "masked_forward_calls": 3 * len(audits), "batches": audits}


def markdown_table(rows, fields):
    def display(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"] + [
        "| " + " | ".join(display(row.get(field)) for field in fields) + " |" for row in rows]


def append_summary(exp1, output, tables, manifest):
    lines = ["", "", "---", "", "# 后续实验二至五", "",
             "实验一以上内容从原 SUMMARY.md 完整复制；原始文件未修改。所有比率、Accuracy 和 Weighted F1 在后续 CSV 中以 0–1 表示。",
             "沿用实验一同一四个 validation checkpoint 和原测试话语，完整输入不重新推理。只新增三种官方均值替换推理，不训练、不选模、不调参。", ""]
    sections = [
        ("实验二：Baseline 高置信错误的改善与新增错误", "exp2_summary"),
        ("实验三：CMERC 剩余高置信错误及类别分布", "exp3_class_summary"),
        ("实验四：单一模态均值替换", "exp4_modality_ablation_summary"),
        ("实验四：完整输入正确到替换后高置信错误", "exp4_full_correct_to_masked_error"),
        ("实验五：错误置信度区间分布", "exp5_error_confidence_distribution"),
        ("实验五：整体统计", "exp5_overall_summary")]
    for title, name in sections:
        rows = tables[name]
        fields = list(rows[0]) if rows else []
        lines += [f"## {title}", ""] + markdown_table(rows, fields) + [""]
    lines += ["## 分母与解释边界", "",
              "- 实验二三个 outcome 的分母是各阈值的 Baseline 高置信错误数；新增 CMERC 高置信错误比例的分母是 Baseline 原本不是高置信错误的有效话语数。",
              "- 实验三的类内错误率分母为该真实情绪类全部话语；高置信预测内错误率分母为该类中所有达到阈值的预测，包含正确预测。",
              "- 实验四新增高置信错误比例分母是该模型完整输入下全部正确话语；原本已错且仍错另表保留。",
              "- 实验五区间占比分母为全部错误；高置信预测内错误率分母为全部达到阈值的预测。分母为零统一 N/A。",
              "- 均值替换沿时间轴 dim=0，包含原 batch padding，repeat(T,1,1)。保留 batch16、原话语顺序和 ATen RNN；不改为零填充或有效时间均值。",
              "- 高置信错误减少可来自置信度降低，不能当作纠正了同等数量的分类错误；错误类别变化时，最大置信度变化也不等于同一类别的概率变化。",
              "- 删除模态后的错误类别最大概率和真实类别概率异常是两种现象；本报告不将它们混为 CEL，也不把错误直接归因为模态冲突。",
              "- 原 train.py / official_test 存在测试集选模偏差；本次仍使用实验一已验证的 validation checkpoint，不更换权重以追求测试效果。",
              "- 固定 checkpoint、单次测试集的描述性分析不能证明总体校准优劣或训练模块的因果效应。", "",
              "## 文件与来源", "", "逐话语明细没有 Top-N 截断，按数据集、实验和阈值保留全部符合条件的样本。",
              "原实验一、四个 checkpoint、数据及原预测文件的哈希在运行前后验证一致；详见 run_manifest.json。",
              f"Word 母版状态：{manifest['word_template_status']}。若母版缺失，完整版使用原 Markdown 转录作为实验一来源并明确标注，不声称保留未知 Word 版式。", ""]
    original = (exp1 / "SUMMARY.md").read_bytes()
    (output / "SUMMARY.md").write_bytes(original + "\n".join(lines).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1-root", type=Path, default=DEFAULT_EXP1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mother-docx", type=Path)
    options = parser.parse_args()
    exp1 = options.exp1_root.resolve()
    output = (options.output_dir or exp1 / "experiments_2_to_5").resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("new output directory must be empty; original experiments are never overwritten")
    if output == exp1:
        parser.error("output must be a new subdirectory, not the experiment-1 root")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((exp1 / "run_manifest.json").read_text())
    if original.get("status") != "completed" or original.get("split") != "test":
        raise ValueError("experiment 1 is not a complete test-only diagnostic")
    original_files = [p for p in exp1.rglob("*") if p.is_file() and not p.is_relative_to(output)]
    tracked = {str(p): sha256(p) for p in original_files}
    for path, entry in original["input_files_before"].items():
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"experiment 1 checkpoint/data/reference file changed: {path}")
        tracked[path] = entry["sha256"]
    for path, expected in original["code_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"experiment 1 frozen source changed: {path}")
        tracked[path] = expected
    if options.mother_docx:
        tracked[str(options.mother_docx.resolve())] = sha256(options.mother_docx)
    from high_confidence_followup_stats import load_full_records, generate_tables, TABLE_FIELDS
    from extract_high_confidence_errors import load_frozen_evaluator, verified_identity

    full = load_full_records(exp1)
    manifest = {"started_at": now(), "status": "running", "host": socket.gethostname(),
                "exp1_root": str(exp1), "output_dir": str(output), "batch_root": original["batch_root"],
                "thresholds": [0.8, 0.9, 0.95], "split": "test", "training": False,
                "full_inference_repeated": False, "input_files_before_sha256": tracked,
                "checkpoint_selection": "unchanged experiment-1 validation-selected checkpoints",
                "word_template": str(options.mother_docx.resolve()) if options.mother_docx else None,
                "word_template_status": "found" if options.mother_docx else "指定原 Word 母版未找到",
                "models": {}, "python": platform.python_version()}
    save_json(output / "run_manifest.json", manifest)
    early = generate_tables(full, None)
    write_tables(output, early, TABLE_FIELDS, only_non_ablation=True)
    full_rows = [r for rows in full.values() for r in rows]
    full_fields = list(dict.fromkeys(k for row in full_rows for k in row))
    with (output / "full_predictions_reused.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=full_fields)
        writer.writeheader()
        writer.writerows(full_rows)
    print("Experiments 2, 3, 5 computed from original full predictions", flush=True)
    import torch
    if not options.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("unchanged official noncausal graph requires a healthy CUDA GPU")
    device = torch.device(options.device)
    torch.cuda.set_device(device)
    evaluator = load_frozen_evaluator(Path(original["batch_root"]))
    os.chdir(Path(original["batch_root"]) / "code")
    manifest.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpu_name=torch.cuda.get_device_name(device),
                    gpu_inventory=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"], text=True))
    masked = {}
    for (dataset, model_name), rows in full.items():
        key = f"{dataset}/{model_name}"
        info = original["models"][key]
        payload = torch.load(info["checkpoint"], map_location="cpu")
        saved, flags = verified_identity(payload, model_name, dataset)
        if saved != info["training_args"]:
            raise ValueError("training metadata differs from experiment 1")
        print(f"Inferring three removed views: {key}", flush=True)
        masked[(dataset, model_name)], audit = infer_removed_views(evaluator, payload, dataset, rows, device)
        manifest["models"][key] = {"checkpoint": info["checkpoint"],
                                   "checkpoint_sha256": info["checkpoint_sha256"],
                                   "identity_flags": flags, "training_args": saved,
                                   "class_names": info["class_names"], "audit": audit}
        del payload
        gc.collect()
        save_json(output / "run_manifest.json", manifest)
    # Mean-replaced feature inputs must be identical between Baseline and CMERC.
    for dataset in ("IEMOCAP", "MELD"):
        left = manifest["models"][f"{dataset}/baseline"]["audit"]["batches"]
        right = manifest["models"][f"{dataset}/cmerc"]["audit"]["batches"]
        if len(left) != len(right):
            raise ValueError("model batch counts differ")
        for a, b in zip(left, right):
            for condition in CONDITIONS:
                if a["conditions"][condition]["input_sha256"] != b["conditions"][condition]["input_sha256"]:
                    raise ValueError("Baseline/CMERC replacement input tensors differ")
    tables = generate_tables(full, masked)
    manifest["table_rows"] = write_tables(output, tables, TABLE_FIELDS)
    for path, expected in tracked.items():
        if sha256(path) != expected:
            raise ValueError(f"original input changed during diagnostics: {path}")
    manifest.update(status="computations_completed", computed_at=now(),
                    original_inputs_unchanged=True, same_replacement_inputs_between_models=True,
                    full_test_utterances={ds: len(full[(ds, "baseline")]) for ds in ("IEMOCAP", "MELD")},
                    new_forward_calls=sum(m["audit"]["masked_forward_calls"] for m in manifest["models"].values()))
    append_summary(exp1, output, tables, manifest)
    save_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output": str(output), "rows": manifest["table_rows"],
                      "new_masked_forward_calls": manifest["new_forward_calls"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
