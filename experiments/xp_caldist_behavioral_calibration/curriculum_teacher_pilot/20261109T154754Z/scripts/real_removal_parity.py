#!/usr/bin/env python3
"""Batch1 parity against author caldiff statements on a real Fold1 dialogue.

This is an inference implementation gate, using newly initialized weights only.
The reference executes AST-extracted author forward/mean-replacement statements;
it never calls the new adapter's forward or reduction helpers. Unknown labels stay
in every neural forward and are excluded before the reference true-class lookup.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
VIEWS = ("full", "text", "audio", "visual", "intra", "inter")
APPROVED_SHA = "6d43928865915fe22a913c98b1165a05ba6e3e8f21d4d4c896cf51eb9c9b62be"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def author_reference(path):
    """Execute the original six calls, without importing the training entrypoint."""
    tree = ast.parse(Path(path).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "train_or_eval_graph_model")
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]
    full = next(node for node in assignments if isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "model"
                and isinstance(node.value.args[0], ast.List))
    means = [node for node in assignments if len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in {"mask_features", "mask_textf1", "mask_textf2",
                                       "mask_textf3", "mask_textf4", "mask_acouf", "mask_visuf"}]
    means.sort(key=lambda node: node.lineno)
    removed = [node for node in assignments if isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name)
               and node.value.func.id in {"model_mm", "model_intra", "model_inter"}]
    removed.sort(key=lambda node: node.lineno)
    if len(means) != 7 or len(removed) != 5:
        raise ValueError("author caldiff six-view AST structure changed")
    body = [copy.deepcopy(node) for node in [full] + means + removed]
    body += ast.parse("return [log_prob, log_prob2t, log_prob2a, log_prob2v, log_prob2intra, log_prob2inter]").body
    signature = ast.parse("def reference(model, model_mm, model_intra, model_inter, textf1, textf2, textf3, textf4, qmask, umask, lengths, acouf, visuf, epoch):\n    pass\n").body[0]
    signature.body = body
    module = ast.fix_missing_locations(ast.Module(body=[signature], type_ignores=[]))
    namespace = {"torch": torch, "args": SimpleNamespace(causal=False)}
    exec(compile(module, str(path), "exec"), namespace)
    metadata = {
        "author_source_sha256": sha256(path),
        "forward_calls": [node.value.func.id for node in [full] + removed],
        "forward_source_lines": [node.lineno for node in [full] + removed],
        "mean_replacement_assignments": [node.targets[0].id for node in means],
        "extracted_ast_sha256": hashlib.sha256(ast.dump(module).encode()).hexdigest(),
    }
    return namespace["reference"], metadata


def reference_models(author_path, source_module, device, config):
    """Construct via the four original author constructor expressions."""
    tree = ast.parse(author_path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in {"model", "model_mm", "model_intra", "model_inter"}
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id in {"Model", "Model_Intra", "Model_Inter"}]
    nodes.sort(key=lambda node: node.lineno)
    if len(nodes) != 4:
        raise ValueError("expected four author graph constructors")
    args = SimpleNamespace(
        base_model="GRU", windowp=10, windowf=10, active_listener=False,
        attention="general", dropout=config["dropout"], nodal_attention=True,
        no_cuda=device.type != "cuda", graph_type="hyper", use_topic=False,
        alpha=0.2, multiheads=6, graph_construct="direct", use_gcn=False,
        use_residue=False, modals="avl", mm_fusion_mthd="concat_DHT",
        av_using_lstm=False, Deep_GCN_nlayers=4, Dataset="IEMOCAP",
        use_speaker=True, use_modal=False, norm="BN", num_L=config["num_L"],
        num_K=config["num_K"], causal=False)
    namespace = {"args": args, "D_m": 1024, "D_g": 512, "D_p": 150,
                 "D_e": 100, "D_h": 100, "D_a": 100, "graph_h": 512,
                 "n_speakers": 2, "n_classes": 6, "D_visual": 342, "D_audio": 1582,
                 **{name: getattr(source_module, name) for name in ("Model", "Model_Intra", "Model_Inter")}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(author_path), "exec"), namespace)
    return tuple(namespace[name].to(device).eval() for name in
                 ("model", "model_mm", "model_intra", "model_inter"))


def strict_copy_state(models, state):
    reports = []
    for model in models:
        result = model.load_state_dict(state, strict=True)
        reports.append({"class": type(model).__name__, "strict": True,
                        "missing_keys": result.missing_keys,
                        "unexpected_keys": result.unexpected_keys})
        model.eval()
    return reports


def reference_decomposition(views, record):
    """Scalar author gap branch with the approved unknown-mask amendment."""
    labels = record["labels"]
    eligible = [i for i, valid in enumerate(record["supervision_mask"]) if valid]
    if not eligible:
        raise ValueError("no known labels")
    gap_sums = []
    for removed in views[1:]:
        gaps = []
        for index in eligible:
            label = int(labels[index])
            if not 0 <= label < views[0].size(1):
                raise ValueError("known label outside class range")
            original_score = views[0][index, label]
            removed_score = removed[index, label]
            if original_score <= removed_score:
                gaps.append(removed_score - original_score)
            else:
                gaps.append(original_score.new_zeros(()))
        # The author sums FP32 rank penalties; Python-double accumulation changes
        # the reference numerical protocol even when every logit is identical.
        gap_sums.append(float(torch.stack(gaps).sum().cpu()))
    qmask = torch.as_tensor(record["qmask"])
    speaker = max(index[1] for index in qmask.nonzero().tolist()) + 1
    total = sum(gap_sums)
    return {**{f"{name}_gap": value for name, value in zip(VIEWS[1:], gap_sums)},
            "valid_label_count": len(eligible), "total_length": len(labels),
            "speaker_term": speaker, "total_gap": total,
            "final_difficulty": (total + speaker) / (len(eligible) + speaker),
            "difficulty_total_length": (total + speaker) / (len(labels) + speaker)}


def compare_values(expected, actual, atol, rtol, name):
    """Use the preregistered elementwise bound relative to the reference."""
    reference = torch.as_tensor(expected, dtype=torch.float64).detach()
    observed = torch.as_tensor(actual, device=reference.device, dtype=torch.float64).detach()
    if reference.shape != observed.shape or not torch.isfinite(reference).all() or not torch.isfinite(observed).all():
        raise AssertionError(f"{name} shape/nonfinite failure")
    errors = (reference - observed).abs()
    thresholds = atol + rtol * reference.abs()
    if (errors > thresholds).any():
        raise AssertionError(f"{name} exceeds atol={atol}, rtol={rtol}: "
                             f"max error/threshold ratio {float((errors / thresholds).max().cpu())}")
    max_error_index = errors.reshape(-1).argmax()
    return {"shape": list(reference.shape),
            "max_absolute_error": float(errors.max().cpu()),
            "effective_threshold_at_max_error": float(thresholds.reshape(-1)[max_error_index].cpu()),
            "minimum_effective_threshold": float(thresholds.min().cpu()),
            "maximum_effective_threshold": float(thresholds.max().cpu()),
            "max_error_to_threshold_ratio": float((errors / thresholds).max().cpu()),
            "tolerance_absolute": atol, "tolerance_relative": rtol, "passed": True}


def compare_views(expected, actual, tolerance, kind, rtol=1e-5):
    if len(expected) != 6 or len(actual) != 6:
        raise AssertionError("exactly six views required")
    result = {}
    for name, left, right in zip(VIEWS, expected, actual):
        result[name] = compare_values(left, right, tolerance, rtol, f"{name} {kind}")
    return result


def load_real_dialogue(manifest, artifact):
    ids = manifest["protocols"]["IEMOCAP"]["folds"]["fold1"]["roles"]["training"]["dialogue_ids"]
    masked = {}
    for row in manifest["unknown_label_mask"]["utterances"]:
        masked.setdefault(row["dialogue_id"], []).append(row)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if any(payload.get(key) != value for key, value in
           {"role": "training", "dataset": "IEMOCAP", "fold": "fold1"}.items()):
        raise ValueError("parity input must be the IEMOCAP Fold1 training role artifact")
    if payload["provenance"]["approved_manifest_sha256"] != APPROVED_SHA:
        raise ValueError("training artifact approved manifest mismatch")
    if set(payload["dialogues"]) != set(ids):
        raise ValueError("training artifact dialogue coverage mismatch")
    records = payload["dialogues"]
    candidates = [key for key in ids if key in masked and len(records[key]["labels"]) > len(masked[key])]
    dialogue = min(candidates, key=lambda key: (len(records[key]["labels"]), key))
    record = records[dialogue]
    mask = [True] * len(record["labels"])
    for row in masked[dialogue]:
        index = row["utterance_index"]
        mask[index] = False
    if torch.as_tensor(record["supervision_mask"]).bool().tolist() != mask:
        raise ValueError("training role supervision mask mismatch")
    return dialogue, record, masked[dialogue]


def gpu_snapshot(path):
    output = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=True).stdout
    path.write_text(output)
    return subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader"], text=True, capture_output=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "CMERC/experiments/xp_mainline_validation/20261016T050823Z/code")
    parser.add_argument("--author-caldiff", type=Path, default=ROOT / "CMERC/caldiff/cal_diff.py")
    parser.add_argument("--adapter", type=Path, default=OUT.parents[1] / "curriculum_teacher_reconstruction/20261109T145742Z/scripts/generate_fold_local_difficulty.py")
    parser.add_argument("--approved-manifest", type=Path, default=OUT.parents[1] / "independent_training_pilot/20261109T142243Z/approved_split_manifests.json")
    parser.add_argument("--teacher-config", type=Path, default=OUT.parents[1] / "curriculum_teacher_reconstruction/20261109T145742Z/candidate_teacher_config.json")
    parser.add_argument("--training-role-artifact", type=Path, default=OUT / "runs/IEMOCAP/fold1/training_role_artifact.pt")
    parser.add_argument("--output", type=Path, default=OUT / "results/real_removal_parity.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", "--atol", dest="tolerance", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1475)
    args = parser.parse_args()
    if sha256(args.approved_manifest) != APPROVED_SHA:
        raise ValueError("approved manifest SHA mismatch")
    if not 0 < args.tolerance <= 1e-6:
        raise ValueError("parity tolerance must be positive and <= 1e-6")
    if not 0 <= args.rtol <= 1e-5:
        raise ValueError("parity relative tolerance must be nonnegative and <= 1e-5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("frozen noncausal graph code requires CUDA; no CPU semantic shim is permitted")
    before_gpu = gpu_snapshot(logs / "real_removal_parity_gpu_before.txt")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    compatibility = {"applied": False, "calls": 0, "kind": "unused_PyG_topk_import_symbol"}
    from torch_geometric.nn.pool import topk_pool
    if not hasattr(topk_pool, "topk"):
        def unused_topk(*unused_args, **unused_kwargs):
            compatibility["calls"] += 1
            raise RuntimeError("never-used topk compatibility symbol was called")
        topk_pool.topk = unused_topk
        compatibility["applied"] = True
    source_files = ["model_cm.py", "model_hyper.py", "HypergraphConv.py", "high_fre_conv.py", "model_GCN.py", "causal_graph.py"]
    source_hashes = {name: sha256(args.source_root / name) for name in source_files}
    for name in ("model_cm", "model_hyper", "HypergraphConv", "high_fre_conv", "model_GCN", "causal_graph"):
        if name in sys.modules:
            raise RuntimeError(f"source module {name} already imported; use a fresh process")
    sys.path.insert(0, str(args.source_root.resolve()))
    model_module = importlib.import_module("model_cm")
    adapter = import_file(args.adapter, "difficulty_adapter_under_test")
    config = json.loads(args.teacher_config.read_text())["datasets"]["IEMOCAP"]
    reference = reference_models(args.author_caldiff, model_module, device, config)
    state = {key: value.detach().cpu().clone() for key, value in reference[0].state_dict().items()}
    adapter_models = adapter.build_cmerc_teacher_models(args.source_root, "IEMOCAP", config, device)
    load_report = strict_copy_state(reference + adapter_models, state)
    manifest = json.loads(args.approved_manifest.read_text())
    dialogue, record, unknown_rows = load_real_dialogue(manifest, args.training_role_artifact)
    length = len(record["labels"])
    text = [torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(1) for value in record["text"]]
    qmask = torch.as_tensor(record["qmask"], dtype=torch.float32, device=device).unsqueeze(1)
    umask = torch.ones(1, length, device=device)
    audio = torch.as_tensor(record["audio"], dtype=torch.float32, device=device).unsqueeze(1)
    visual = torch.as_tensor(record["visual"], dtype=torch.float32, device=device).unsqueeze(1)
    reference_function, extraction = author_reference(args.author_caldiff)
    raw = {"reference": [], "adapter": []}
    handles = []
    for lane, models in (("reference", reference), ("adapter", adapter_models)):
        for model in models:
            handles.append(model.smax_fc.register_forward_hook(
                lambda module, inputs, output, lane=lane: raw[lane].append(output.detach().clone())))
    try:
        with torch.inference_mode():
            expected = reference_function(*reference, *text, qmask, umask, [length], audio, visual, 0)
            full, removed, labels, supervision_mask, adapter_qmask = adapter.score_complete_dialogue(adapter_models, record, device)
            actual = [full] + list(removed)
    finally:
        for handle in handles:
            handle.remove()
    scores_report = compare_views(expected, actual, args.tolerance, "log_prob", args.rtol)
    logits_report = compare_views(raw["reference"], raw["adapter"], args.tolerance, "raw_logits", args.rtol)
    expected_decomposition = reference_decomposition(expected, record)
    actual_decomposition = adapter.compute_dialogue_decomposition(dialogue, full, removed, labels, supervision_mask, adapter_qmask)
    decomposition_report = {}
    for key, expected_value in expected_decomposition.items():
        decomposition_report[key] = {"reference": expected_value, "adapter": actual_decomposition[key],
                                     **compare_values(expected_value, actual_decomposition[key],
                                                      args.tolerance, args.rtol, key)}
    sentinel_labels = labels.clone()
    sentinel_labels[~supervision_mask] = -100
    sentinel = adapter.compute_dialogue_decomposition(dialogue, full, removed, sentinel_labels, supervision_mask, adapter_qmask)
    if sentinel != actual_decomposition:
        raise AssertionError("unknown labels were not excluded before true-class lookup")
    for model in reference + adapter_models:
        if model.training or any(not torch.equal(value.cpu(), state[key]) for key, value in model.state_dict().items()):
            raise AssertionError("inference mutated frozen model state")
    if compatibility["calls"]:
        raise AssertionError("unused import compatibility symbol was executed")
    if source_hashes != {name: sha256(args.source_root / name) for name in source_files}:
        raise AssertionError("frozen source changed during parity gate")
    torch.cuda.synchronize()
    during_gpu = gpu_snapshot(logs / "real_removal_parity_gpu_during.txt")
    process_gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                                 text=True, capture_output=True, check=True).stdout
    (logs / "real_removal_parity_processes.txt").write_text(process_gpu)
    torch.save({"reference_log_prob": [value.cpu() for value in expected],
                "adapter_log_prob": [value.cpu() for value in actual],
                "reference_logits": [value.cpu() for value in raw["reference"]],
                "adapter_logits": [value.cpu() for value in raw["adapter"]]},
               args.output.with_suffix(".pt"))
    report = {
        "status": "passed", "scope": "real_model_batch1_six_view_inference_parity_only",
        "trained_teacher": False, "historical_checkpoint_access": False,
        "weights": "fresh_seeded_initialization_strict_copied_to_eight_real_models",
        "dataset": "IEMOCAP", "fold": "fold1", "role": "training", "dialogue_id": dialogue,
        "batch_size": 1, "context_length": length, "known_label_count": int(supervision_mask.sum()),
        "unknown_label_count": int((~supervision_mask).sum()), "unknown_rows": unknown_rows,
        "unknown_context_preserved": all(value.size(0) == length for value in expected + actual),
        "unknown_sentinel_minus100_invariance": True, "tolerance_absolute": args.tolerance,
        "tolerance_relative": args.rtol,
        "comparison_rule": "abs(adapter-reference) <= atol + rtol*abs(reference)",
        "six_raw_logits": logits_report, "six_log_probabilities": scores_report,
        "difficulty_decomposition": decomposition_report, "strict_state_loads": load_report,
        "model_state_unchanged": True, "source_unchanged": True, "seed": args.seed,
        "architecture": adapter.expected_teacher_architecture("IEMOCAP", config),
        "author_reference": extraction, "compatibility_shim": compatibility,
        "source_root": str(args.source_root.resolve()), "source_file_sha256": source_hashes,
        "script_sha256": sha256(__file__), "adapter_sha256": sha256(args.adapter),
        "approved_manifest_sha256": sha256(args.approved_manifest),
        "training_role_artifact": str(args.training_role_artifact.resolve()),
        "training_role_artifact_sha256": sha256(args.training_role_artifact),
        "environment": {"python": sys.executable, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid()},
        "gpu_before": before_gpu, "gpu_during": during_gpu, "gpu_processes": process_gpu,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "score_artifact_sha256": sha256(args.output.with_suffix(".pt")),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "dialogue": dialogue, "length": length,
                      "unknown": int((~supervision_mask).sum()), "output": str(args.output)}))


if __name__ == "__main__":
    main()
