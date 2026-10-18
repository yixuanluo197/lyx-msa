#!/usr/bin/env python3
"""Evaluate official or causal CMERC checkpoints.

Baseline graph code requires CUDA. The causal graph also supports CPU; use
``--causal --device cpu`` with a checkpoint trained under the causal protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DatasetConfig = namedtuple(
    "DatasetConfig",
    "name dropout num_classes num_k seed class_names audio_dim visual_dim graph_dim checkpoint_name",
)


def dataset_config(dataset: str) -> DatasetConfig:
    """Return the architecture settings used by the official README commands."""
    normalized = dataset.upper()
    if normalized == "IEMOCAP":
        return DatasetConfig(
            "IEMOCAP",
            0.5,
            6,
            4,
            1475,
            ("happy", "sad", "neutral", "angry", "excited", "frustrated"),
            1582,
            342,
            512,
            "IEMOCAP_checkpoint.pkl",
        )
    if normalized == "MELD":
        return DatasetConfig(
            "MELD",
            0.4,
            7,
            3,
            67137,
            ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"),
            300,
            342,
            1024,
            "MELD_checkpoint.pkl",
        )
    raise ValueError(f"unsupported dataset {dataset!r}; expected IEMOCAP or MELD")


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a state dict from either official or common wrapped checkpoints."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state: Mapping[str, Any] = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if not state:
        raise ValueError("checkpoint state dict is empty")
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
        if isinstance(key, str)
    }


def expand_utterance_ids(
    dialogue_ids: Sequence[Any], lengths: Sequence[int]
) -> tuple[list[str], list[int]]:
    """Match the dialogue-major order produced by the official graph model."""
    if len(dialogue_ids) != len(lengths):
        raise ValueError("dialogue_ids and lengths must have equal length")
    expanded_ids: list[str] = []
    utterance_indices: list[int] = []
    for dialogue_id, length in zip(dialogue_ids, lengths):
        if int(length) < 0:
            raise ValueError("dialogue lengths cannot be negative")
        expanded_ids.extend([str(dialogue_id)] * int(length))
        utterance_indices.extend(range(int(length)))
    return expanded_ids, utterance_indices


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    """Compute the requested classification metrics on a fixed class order."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and predictions must be non-empty one-dimensional arrays")
    class_ids = np.arange(len(class_names), dtype=np.int64)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= len(class_names)):
        raise ValueError("label outside configured class range")
    if np.any(predictions < 0) or np.any(predictions >= len(class_names)):
        raise ValueError("prediction outside configured class range")
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    support = confusion.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    per_class = [
        {
            "class_id": int(class_id),
            "class_name": str(class_names[class_id]),
            "precision": float(100.0 * precision[class_id]),
            "recall": float(100.0 * recall[class_id]),
            "f1": float(100.0 * f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in class_ids
    ]
    return {
        "num_samples": int(labels.size),
        "accuracy": float(100.0 * np.mean(labels == predictions)),
        "weighted_f1": float(100.0 * np.average(f1, weights=support)),
        "macro_f1": float(100.0 * np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def validate_log_probabilities(log_probabilities: np.ndarray) -> None:
    """Reject invalid model output before deriving calibration measurements."""
    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("log probabilities must have shape [num_samples, num_classes]")
    if not np.isfinite(values).all():
        raise ValueError("log probabilities contain NaN or Inf")
    sums = np.exp(values).sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("log probabilities are not normalized")


def _import_official_modules(causal=False):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from dataloader import IEMOCAPDataset, MELDDataset
    if causal:
        from model_cm import Model
    else:
        from model import Model
    from model_cm import Model_Inter, Model_Intra

    return IEMOCAPDataset, MELDDataset, Model, Model_Intra, Model_Inter


def _model_arguments(config: DatasetConfig, device: Any, causal=False) -> dict[str, Any]:
    arguments = {
        "base_model": "GRU",
        "D_m": 1024,
        "D_g": config.graph_dim,
        "D_p": 150,
        "D_e": 100,
        "D_h": 100,
        "D_a": 100,
        "graph_hidden_size": 512,
        "n_speakers": 9 if config.name == "MELD" else 2,
        "max_seq_len": 200,
        "window_past": 10,
        "window_future": 10,
        "n_classes": config.num_classes,
        "listener_state": False,
        "context_attention": "general",
        "dropout": config.dropout,
        "nodal_attention": True,
        "no_cuda": device.type != "cuda",
        "graph_type": "hyper",
        "use_topic": False,
        "alpha": 0.2,
        "multiheads": 6,
        "graph_construct": "direct",
        "use_GCN": False,
        "use_residue": False,
        "D_m_v": config.visual_dim,
        "D_m_a": config.audio_dim,
        "modals": "avl",
        "att_type": "concat_DHT",
        "av_using_lstm": False,
        "Deep_GCN_nlayers": 4,
        "dataset": config.name,
        "use_speaker": True,
        "use_modal": False,
        "norm": "LN" if causal else "BN",
        "num_L": 3,
        "num_K": config.num_k,
    }
    if causal:
        arguments["causal"] = True
    return arguments


def build_models(config: DatasetConfig, device: Any, causal=False):
    """Construct the official full, intra-removed and inter-removed networks."""
    _, _, Model, Model_Intra, Model_Inter = _import_official_modules(causal=causal)
    arguments = _model_arguments(config, device, causal=causal)
    return (
        Model(**arguments).to(device),
        Model_Intra(**arguments).to(device),
        Model_Inter(**arguments).to(device),
    )


def build_test_loader(config: DatasetConfig, batch_size: int, num_workers: int):
    import torch

    IEMOCAPDataset, MELDDataset, _, _, _ = _import_official_modules()
    if config.name == "IEMOCAP":
        dataset = IEMOCAPDataset(train=False)
    else:
        dataset = MELDDataset("data/MELD_features_raw1.pkl", train=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def _masked_modalities(text_features, visual_features, audio_features, umask=None, causal=False):
    """Apply the exact padded-batch mean replacement used by train_cm.py."""
    if causal:
        from dataloader import causal_mean_replacement
        return ([causal_mean_replacement(feature, umask) for feature in text_features],
                causal_mean_replacement(visual_features, umask),
                causal_mean_replacement(audio_features, umask))
    masked_text = [
        feature.mean(dim=0).repeat(feature.shape[0], 1, 1)
        for feature in text_features
    ]
    masked_visual = visual_features.mean(dim=0).repeat(visual_features.shape[0], 1, 1)
    masked_audio = audio_features.mean(dim=0).repeat(audio_features.shape[0], 1, 1)
    return masked_text, masked_visual, masked_audio


def _forward_log_prob(model, text, qmask, umask, lengths, audio, visual):
    result = model(text, qmask, umask, lengths, audio, visual, 0)
    return result[0]


def run_inference(models, dataloader, device: Any) -> dict[str, Any]:
    """Run full and five information-removal passes over the real test set."""
    import torch

    model, model_intra, model_inter = models
    # train_cm.py synchronizes all removal networks from the trained model.
    model_intra.load_state_dict(model.state_dict(), strict=True)
    model_inter.load_state_dict(model.state_dict(), strict=True)
    for network in models:
        network.eval()

    labels_all: list[np.ndarray] = []
    full_all: list[np.ndarray] = []
    removed_all: dict[str, list[np.ndarray]] = {
        name: [] for name in ("text", "audio", "visual", "intra_context", "inter_context")
    }
    dialogue_ids_all: list[str] = []
    utterance_indices_all: list[int] = []

    with torch.no_grad():
        for data in dataloader:
            tensors = [value.to(device) for value in data[:-1]]
            text1, text2, text3, text4, visual, audio, qmask, umask, labels = tensors
            text = [text1, text2, text3, text4]
            lengths = [int(row.sum().item()) for row in umask]
            if any(length <= 0 for length in lengths):
                raise ValueError("encountered an empty dialogue")

            full = _forward_log_prob(model, text, qmask, umask, lengths, audio, visual)
            masked_text, masked_visual, masked_audio = _masked_modalities(text, visual, audio, umask=umask, causal=getattr(model, "causal", False))
            removal_outputs = {
                "text": _forward_log_prob(
                    model, masked_text, qmask, umask, lengths, audio, visual
                ),
                "audio": _forward_log_prob(
                    model, text, qmask, umask, lengths, masked_audio, visual
                ),
                "visual": _forward_log_prob(
                    model, text, qmask, umask, lengths, audio, masked_visual
                ),
                "intra_context": _forward_log_prob(
                    model_intra, text, qmask, umask, lengths, audio, visual
                ),
                "inter_context": _forward_log_prob(
                    model_inter, text, qmask, umask, lengths, audio, visual
                ),
            }
            flat_labels = torch.cat(
                [labels[index, :length] for index, length in enumerate(lengths)]
            )
            batch_dialogue_ids, batch_utterance_indices = expand_utterance_ids(
                data[-1], lengths
            )
            if full.shape[0] != flat_labels.shape[0]:
                raise RuntimeError(
                    f"model returned {full.shape[0]} utterances for {flat_labels.shape[0]} labels"
                )

            labels_all.append(flat_labels.cpu().numpy())
            full_all.append(full.cpu().numpy())
            for name, output in removal_outputs.items():
                removed_all[name].append(output.cpu().numpy())
            dialogue_ids_all.extend(batch_dialogue_ids)
            utterance_indices_all.extend(batch_utterance_indices)

    if not labels_all:
        raise RuntimeError("test dataloader produced no batches")
    return {
        "labels": np.concatenate(labels_all),
        "log_probabilities": np.concatenate(full_all),
        "removed_log_probabilities": {
            name: np.concatenate(parts) for name, parts in removed_all.items()
        },
        "dialogue_ids": np.asarray(dialogue_ids_all, dtype=str),
        "utterance_indices": np.asarray(utterance_indices_all, dtype=np.int64),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def save_outputs(
    output_dir: Path,
    config: DatasetConfig,
    checkpoint_path: Path,
    inference: Mapping[str, Any],
    metrics: Mapping[str, Any],
    causal=False,
    checkpoint_metadata=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_probabilities = np.asarray(inference["log_probabilities"])
    probabilities = np.exp(log_probabilities)
    predictions = probabilities.argmax(axis=1)
    removed = inference["removed_log_probabilities"]
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=inference["labels"],
        predictions=predictions,
        log_probabilities=log_probabilities,
        probabilities=probabilities,
        dialogue_ids=inference["dialogue_ids"],
        utterance_indices=inference["utterance_indices"],
        **{f"removed_{name}_log_probabilities": values for name, values in removed.items()},
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["dialogue_id", "utterance_index", "label", "prediction", "confidence"]
        )
        for row in zip(
            inference["dialogue_ids"],
            inference["utterance_indices"],
            inference["labels"],
            predictions,
            probabilities.max(axis=1),
        ):
            writer.writerow(row)
    payload = {
        "dataset": config.name,
        "checkpoint": str(checkpoint_path.resolve()),
        "causal": causal,
        "effective_norm": "LN" if causal else "BN",
        **(checkpoint_metadata or {}),
        **metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("IEMOCAP", "MELD"))
    parser.add_argument("--causal", action="store_true", help="evaluate a history/current causal checkpoint")
    parser.add_argument("--require-noncausal", action="store_true", help="strictly reject causal runs/checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (--causal only)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args(argv)
    if args.require_noncausal and args.causal:
        parser.error("--require-noncausal cannot be combined with --causal")
    return args


def checkpoint_provenance(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Export training identity without changing legacy checkpoint inference."""
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, Mapping):
        return {}
    provenance = checkpoint.get("provenance", saved_args.get("provenance", {}))
    split = provenance.get("split", {})
    metadata = {key: saved_args.get(key) for key in
                ("seed", "split_seed", "rng_policy", "selection_protocol", "rnn_backend",
                 "batch_policy", "edge_capacity", "node_capacity", "batch_max_retries")}
    metadata.update({"split_hash": split.get("hash"), "provenance": provenance,
                     "selected_epoch": int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None,
                     "training_args": dict(saved_args)})
    return metadata


def configure_evaluation_backend(models: Sequence[Any], checkpoint: Mapping[str, Any]) -> str:
    """Match recurrent kernels recorded by training; legacy weights retain cuDNN."""
    backend = checkpoint.get("args", {}).get("rnn_backend", "cudnn")
    if backend not in ("aten", "cudnn"):
        raise ValueError(f"unsupported checkpoint recurrent backend: {backend}")
    if backend == "aten":
        from train_cm import configure_recurrent_backend
        for model in models:
            configure_recurrent_backend(model, backend)
    return backend


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.n_bins <= 0:
        raise ValueError("batch-size and n-bins must be positive; num-workers cannot be negative")

    import torch

    from evaluation_metrics import calibration_metrics, confidence_enhancement_level

    config = dataset_config(args.dataset)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.causal:
        raise RuntimeError(
            "the unmodified official hypergraph implementation contains unconditional .cuda() calls; "
            "use --device cuda[:N]"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        models = build_models(config, device, causal=args.causal)
        try:
            raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch 1.13 does not expose weights_only.
            raw_checkpoint = torch.load(checkpoint_path, map_location=device)
        from dataloader import validate_checkpoint_protocol
        arguments = _model_arguments(config, device, causal=args.causal)
        expected_args = {
            "causal": args.causal, "Dataset": config.name,
            **{name: arguments[name] for name in (
                "base_model", "norm", "num_L", "num_K", "av_using_lstm",
                "use_residue", "graph_type", "modals", "use_speaker",
                "use_modal", "Deep_GCN_nlayers",
            )},
            "mm_fusion_mthd": arguments["att_type"], "use_gcn": arguments["use_GCN"],
            "multi_modal": True,
        }
        validate_checkpoint_protocol(raw_checkpoint, causal=args.causal, expected_args=expected_args)
        configure_evaluation_backend(models, raw_checkpoint)
        state = extract_state_dict(raw_checkpoint)
        models[0].load_state_dict(state, strict=True)
        loader = build_test_loader(config, args.batch_size, args.num_workers)
        inference = run_inference(models, loader, device)
    finally:
        os.chdir(original_cwd)

    labels = np.asarray(inference["labels"])
    log_probabilities = np.asarray(inference["log_probabilities"])
    validate_log_probabilities(log_probabilities)
    probabilities = np.exp(log_probabilities)
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, predictions, config.class_names)
    metrics["calibration"] = calibration_metrics(labels, probabilities, n_bins=args.n_bins)
    metrics["cel"] = confidence_enhancement_level(
        labels, log_probabilities, inference["removed_log_probabilities"]
    )
    save_outputs(args.output_dir, config, checkpoint_path, inference, metrics, causal=args.causal,
                 checkpoint_metadata=checkpoint_provenance(raw_checkpoint))
    print(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"saved evaluation artifacts to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

