#!/usr/bin/env python3
"""Leakage-guarded core for Protocol-B fold-local difficulty generation.

This module separates the frozen CMERC/M3Net adapter from reduction. It generates one
full-view and five removal-view log-probability tensors for one complete dialogue at a
time. The formal protocol fixes batch size to one; ``reduce_dialogues`` only tests
partition-independent arithmetic and is not evidence that neural forwards are batch
invariant.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


REMOVAL_NAMES = ("text", "audio", "visual", "intra", "inter")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: Path | str, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def verify_source_manifest(source_root: Path, manifest_path: Path, expected_sha256: str) -> dict:
    verify_file_hash(manifest_path, expected_sha256)
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    required_files = manifest.get("required_files")
    mandatory = {"model_cm.py", "model_hyper.py", "HypergraphConv.py", "high_fre_conv.py",
                 "model_GCN.py", "causal_graph.py"}
    if not isinstance(required_files, list) or not mandatory.issubset(set(required_files)):
        raise ValueError("source manifest lacks the declared executable dependency closure")
    for relative in required_files:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("source manifest paths must be relative and contained")
        path = source_root / relative
        expected = files.get(relative, files.get(str(path), files.get(str(path.resolve()))))
        if isinstance(expected, Mapping):
            expected = expected.get("sha256")
        if not expected:
            raise ValueError(f"source manifest is missing {relative}")
        verify_file_hash(path, expected)
    return manifest


def training_ids_from_manifest(manifest: Mapping, dataset: str, fold: str) -> list[str]:
    return list(manifest["protocols"][dataset]["folds"][fold]["roles"]["training"]["dialogue_ids"])


def assert_requested_ids_are_training(requested: Iterable[str], training_ids: Iterable[str]) -> None:
    requested_set, training_set = set(requested), set(training_ids)
    illegal = sorted(requested_set - training_set)
    if illegal:
        raise ValueError(f"requested IDs contain non-Training dialogues: {illegal[:5]}")


def validate_teacher_protocol(config: Mapping) -> None:
    if config.get("selection") != "fixed_final_epoch":
        raise ValueError("Teacher selection must be fixed_final_epoch")
    if config.get("access_roles") != ["training"]:
        raise ValueError("Teacher may access only the Training role")
    if not isinstance(config.get("epochs"), int) or config["epochs"] <= 0:
        raise ValueError("Teacher epochs must be a positive fixed integer")


def strict_load_teacher(model: torch.nn.Module, path: Path | str, expected_architecture: Mapping,
                        expected_final_epoch: int) -> Mapping:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError("Teacher checkpoint must contain model_state_dict metadata")
    if payload.get("architecture") != dict(expected_architecture):
        raise ValueError("Teacher checkpoint architecture does not match the frozen configuration")
    if payload.get("epoch") != expected_final_epoch:
        raise ValueError("Teacher checkpoint is not the required fixed final epoch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def original_rank_gap_sum(student: torch.Tensor, teacher: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
    """Preserve XP04's <= branch, including its non-zero gradient at ties."""
    rows = torch.arange(labels.numel(), device=labels.device)
    student_true = student[rows, labels]
    teacher_true = teacher[rows, labels]
    zeros = torch.zeros_like(student_true)
    return torch.where(student_true <= teacher_true, teacher_true - student_true, zeros).sum()


def _validate_dialogue_inputs(full_scores: torch.Tensor, removed_scores: Sequence[torch.Tensor],
                              labels: torch.Tensor, supervision_mask: torch.Tensor,
                              qmask: torch.Tensor) -> None:
    if len(removed_scores) != 5:
        raise ValueError("exactly five removal views are required")
    if full_scores.ndim != 2:
        raise ValueError("full scores must be [utterance, class]")
    length = full_scores.size(0)
    if labels.shape != (length,) or supervision_mask.shape != (length,):
        raise ValueError("labels and supervision mask must align with utterances")
    if qmask.ndim != 2 or qmask.size(0) != length:
        raise ValueError("qmask must be [utterance, speaker]")
    if not torch.isfinite(full_scores).all() or any(not torch.isfinite(x).all() for x in removed_scores):
        raise ValueError("all score tensors must be finite")
    if any(x.shape != full_scores.shape for x in removed_scores):
        raise ValueError("all removal views must match full score shape")


def compute_dialogue_decomposition(dialogue_id: str, full_scores: torch.Tensor,
                                   removed_scores: Sequence[torch.Tensor], labels: torch.Tensor,
                                   supervision_mask: torch.Tensor, qmask: torch.Tensor,
                                   teacher_checkpoint_sha256: str | None = None) -> dict:
    _validate_dialogue_inputs(full_scores, removed_scores, labels, supervision_mask, qmask)
    valid = supervision_mask.to(device=full_scores.device, dtype=torch.bool)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError(f"dialogue {dialogue_id!r} has no supervision-eligible labels")
    selected_labels = labels.to(full_scores.device)[valid]
    if (selected_labels < 0).any() or (selected_labels >= full_scores.size(1)).any():
        raise ValueError("eligible labels are outside the class range")
    full_valid = full_scores[valid]
    rows = torch.arange(valid_count, device=full_scores.device)
    per_view = []
    for removed in removed_scores:
        removed_valid = removed[valid]
        gaps = torch.clamp(removed_valid[rows, selected_labels] - full_valid[rows, selected_labels], min=0)
        per_view.append(float(gaps.sum().detach().cpu()))
    active = torch.nonzero(qmask, as_tuple=False)
    if active.numel() == 0:
        raise ValueError(f"dialogue {dialogue_id!r} has no active speaker in qmask")
    speaker_term = float(active[:, 1].max().item() + 1)
    total_length = int(full_scores.size(0))
    total_gap = float(sum(per_view))
    valid_difficulty = (total_gap + speaker_term) / (valid_count + speaker_term)
    total_difficulty = (total_gap + speaker_term) / (total_length + speaker_term)
    values = per_view + [speaker_term, total_gap, valid_difficulty, total_difficulty]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("difficulty decomposition must be finite")
    result = {
        "dialogue_id": dialogue_id,
        "valid_label_count": valid_count,
        "total_length": total_length,
        "speaker_term": speaker_term,
        **{f"{name}_gap": gap for name, gap in zip(REMOVAL_NAMES, per_view)},
        "total_gap": total_gap,
        "difficulty_valid_count": valid_difficulty,
        "difficulty_total_length": total_difficulty,
        "final_difficulty": valid_difficulty,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
    }
    return result


def reduce_dialogues(records: Sequence[tuple], batch_size: int = 1) -> dict[str, dict]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    output = {}
    for start in range(0, len(records), batch_size):
        for record in records[start:start + batch_size]:
            row = compute_dialogue_decomposition(*record)
            if row["dialogue_id"] in output:
                raise ValueError("duplicate dialogue ID")
            output[row["dialogue_id"]] = row
    return output


def verify_exact_coverage(results: Mapping[str, object], expected_training_ids: Iterable[str]) -> None:
    actual, expected = set(results), set(expected_training_ids)
    if actual != expected:
        raise ValueError(f"difficulty coverage mismatch: missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}")


def select_curriculum_training_ids(training_ids: Sequence[str], difficulty: Mapping[str, float],
                                   ratio: float) -> list[str]:
    training_set = set(training_ids)
    if set(difficulty) != training_set:
        raise ValueError("difficulty contains missing or non-Training IDs")
    if not 0 < ratio <= 1:
        raise ValueError("curriculum ratio must be in (0, 1]")
    count = int(ratio * len(training_ids))
    easiest = set(sorted(training_ids, key=lambda key: difficulty[key])[:count])
    return [key for key in training_ids if key in easiest]


def build_cmerc_teacher_models(source_root: Path, dataset: str, config: Mapping,
                               device: torch.device) -> tuple[torch.nn.Module, ...]:
    """Build the four frozen CMERC inference variants used by author caldiff."""
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    model_cm = importlib.import_module("model_cm")
    audio_dim = 1582 if dataset == "IEMOCAP" else 300
    graph_dim = 512 if dataset == "IEMOCAP" else 1024
    speakers = 2 if dataset == "IEMOCAP" else 9
    classes = 6 if dataset == "IEMOCAP" else 7
    common = dict(
        n_speakers=speakers, max_seq_len=200, window_past=10, window_future=10,
        n_classes=classes, listener_state=False, context_attention="general",
        dropout=float(config["dropout"]), nodal_attention=True,
        no_cuda=device.type != "cuda", graph_type="hyper", use_topic=False,
        alpha=0.2, multiheads=6, graph_construct="direct", use_GCN=False,
        use_residue=False, D_m_v=342, D_m_a=audio_dim, modals="avl",
        att_type="concat_DHT", av_using_lstm=False, Deep_GCN_nlayers=4,
        dataset=dataset, use_speaker=True, use_modal=False, norm="BN",
        num_L=int(config["num_L"]), num_K=int(config["num_K"]), causal=False,
    )
    positional = ("GRU", 1024, graph_dim, 150, 100, 100, 100, 512)
    models = (
        model_cm.Model(*positional, **common), model_cm.Model(*positional, **common),
        model_cm.Model_Intra(*positional, **common), model_cm.Model_Inter(*positional, **common),
    )
    return tuple(model.to(device).eval() for model in models)


def expected_teacher_architecture(dataset: str, config: Mapping) -> dict:
    return {
        "model_family": "M3Net", "implementation": "CMERC_model_cm_Model",
        "dataset": dataset, "base_model": "GRU", "graph_type": "hyper",
        "graph_construct": "direct", "multi_modal": True,
        "mm_fusion_mthd": "concat_DHT", "modals": "avl", "norm": "BN",
        "num_L": int(config["num_L"]), "num_K": int(config["num_K"]),
        "causal": False,
    }


def _as_float_tensor(value, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def score_complete_dialogue(models: Sequence[torch.nn.Module], record: Mapping,
                            device: torch.device) -> tuple:
    """Run the full and five author removal views for one unpadded dialogue."""
    if len(models) != 4:
        raise ValueError("full/mm/intra/inter models are required")
    text = [_as_float_tensor(x, device) for x in record["text"]]
    if len(text) != 4:
        raise ValueError("record must contain four aligned text arrays")
    audio = _as_float_tensor(record["audio"], device)
    visual = _as_float_tensor(record["visual"], device)
    qmask = _as_float_tensor(record["qmask"], device)
    length = int(audio.size(0))
    tensors = text + [audio, visual, qmask]
    if any(x.size(0) != length for x in tensors):
        raise ValueError("all dialogue inputs must have the same unpadded length")
    text_b = [x.unsqueeze(1) for x in text]
    audio_b, visual_b, qmask_b = audio.unsqueeze(1), visual.unsqueeze(1), qmask.unsqueeze(1)
    umask = torch.ones((1, length), dtype=torch.float32, device=device)
    lengths = [length]
    full_model, mm_model, intra_model, inter_model = models
    with torch.inference_mode():
        full = full_model(text_b, qmask_b, umask, lengths, audio_b, visual_b, 0)[0]
        mean_text = [x.mean(dim=0, keepdim=True).repeat(length, 1, 1) for x in text_b]
        mean_audio = audio_b.mean(dim=0, keepdim=True).repeat(length, 1, 1)
        mean_visual = visual_b.mean(dim=0, keepdim=True).repeat(length, 1, 1)
        removed = [
            mm_model(mean_text, qmask_b, umask, lengths, audio_b, visual_b, 0)[0],
            mm_model(text_b, qmask_b, umask, lengths, mean_audio, visual_b, 0)[0],
            mm_model(text_b, qmask_b, umask, lengths, audio_b, mean_visual, 0)[0],
            intra_model(text_b, qmask_b, umask, lengths, audio_b, visual_b, 0)[0],
            inter_model(text_b, qmask_b, umask, lengths, audio_b, visual_b, 0)[0],
        ]
    labels = torch.as_tensor(record["labels"], dtype=torch.long, device=device)
    mask = torch.as_tensor(record["supervision_mask"], dtype=torch.bool, device=device)
    return full, removed, labels, mask, qmask


def generate_from_role_artifact(models: Sequence[torch.nn.Module], role_payload: Mapping,
                                training_ids: Sequence[str], device: torch.device,
                                teacher_sha256: str) -> dict[str, dict]:
    if role_payload.get("role") != "training":
        raise ValueError("role-scoped artifact must declare role=training")
    records = role_payload.get("dialogues", {})
    assert_requested_ids_are_training(records, training_ids)
    if set(records) != set(training_ids):
        raise ValueError("role-scoped artifact does not exactly cover Training IDs")
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict(), strict=True)
        model.eval()
    output = {}
    for dialogue_id in training_ids:
        full, removed, labels, mask, qmask = score_complete_dialogue(models, records[dialogue_id], device)
        output[dialogue_id] = compute_dialogue_decomposition(
            dialogue_id, full, removed, labels, mask, qmask, teacher_sha256)
    verify_exact_coverage(output, training_ids)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dataset", choices=("IEMOCAP", "MELD"), required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--training-role-artifact", type=Path)
    parser.add_argument("--training-role-artifact-sha256")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--teacher-config-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise SystemExit("formal neural score generation is fixed to batch_size=1")
    verify_file_hash(args.approved_manifest, args.manifest_sha256)
    verify_file_hash(args.teacher_checkpoint, args.teacher_sha256)
    manifest = json.loads(args.approved_manifest.read_text())
    ids = training_ids_from_manifest(manifest, args.dataset, args.fold)
    if not ids:
        raise SystemExit("Training role is empty")
    if args.validate_only:
        print(json.dumps({"status": "validated_inputs_only", "training_dialogues": len(ids)}))
        return 0
    required = (args.training_role_artifact, args.training_role_artifact_sha256,
                args.source_root, args.source_manifest, args.source_manifest_sha256,
                args.teacher_config, args.teacher_config_sha256, args.output)
    if any(x is None for x in required):
        raise SystemExit("generation requires all role/config/source paths and SHA-256 values plus --output")
    verify_file_hash(args.training_role_artifact, args.training_role_artifact_sha256)
    verify_file_hash(args.teacher_config, args.teacher_config_sha256)
    verify_source_manifest(args.source_root, args.source_manifest, args.source_manifest_sha256)
    role_payload = torch.load(args.training_role_artifact, map_location="cpu")
    expected_role_metadata = {
        "dataset": args.dataset, "fold": args.fold,
        "approved_manifest_sha256": args.manifest_sha256,
    }
    for key, expected in expected_role_metadata.items():
        if role_payload.get(key) != expected:
            raise SystemExit(f"Training role artifact {key} metadata mismatch")
    config = json.loads(args.teacher_config.read_text())
    dataset_config = config["datasets"][args.dataset]
    device = torch.device(args.device)
    models = build_cmerc_teacher_models(args.source_root, args.dataset, dataset_config, device)
    architecture = expected_teacher_architecture(args.dataset, dataset_config)
    checkpoint = strict_load_teacher(models[0], args.teacher_checkpoint, architecture, dataset_config["epochs"])
    required_checkpoint_metadata = {
        "training_role_artifact_sha256": args.training_role_artifact_sha256,
        "teacher_config_sha256": args.teacher_config_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "seed": dataset_config["seed"], "access_roles": ["training"],
        "selection": "fixed_final_epoch",
    }
    for key, expected in required_checkpoint_metadata.items():
        if checkpoint.get(key) != expected:
            raise SystemExit(f"teacher checkpoint {key} metadata mismatch")
    results = generate_from_role_artifact(models, role_payload, ids, device, args.teacher_sha256)
    artifact = {
        "status": "formal_fold_local_difficulty", "dataset": args.dataset, "fold": args.fold,
        "manifest_sha256": args.manifest_sha256, "teacher_sha256": args.teacher_sha256,
        "training_role_artifact_sha256": args.training_role_artifact_sha256,
        "teacher_config_sha256": args.teacher_config_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "effective_teacher_config": dataset_config, "architecture": architecture,
        "batch_size": 1, "primary_formula": "valid_label_count_plus_speaker_term",
        "dialogues": [results[key] for key in ids],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"status": "generated", "dialogues": len(ids), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
