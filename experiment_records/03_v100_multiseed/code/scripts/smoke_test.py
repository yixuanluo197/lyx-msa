#!/usr/bin/env python3
"""Run one real-data CMERC/M3Net training step and verify its artifacts.

This is an engineering smoke test, not a paper-result experiment.  It deliberately
uses the official dataset/model/loss code and writes a complete resumable
checkpoint plus a machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def dataset_config(name: str) -> dict[str, Any]:
    """Return the model settings from the official README commands."""
    normalized = name.strip().upper()
    configs = {
        "IEMOCAP": {
            "dataset": "IEMOCAP",
            "audio_dim": 1582,
            "visual_dim": 342,
            "text_dim": 1024,
            "n_speakers": 2,
            "n_classes": 6,
            "dropout": 0.5,
            "num_L": 3,
            "num_K": 4,
            "seed": 1475,
        },
        "MELD": {
            "dataset": "MELD",
            "audio_dim": 300,
            "visual_dim": 342,
            "text_dim": 1024,
            "n_speakers": 9,
            "n_classes": 7,
            "dropout": 0.4,
            "num_L": 3,
            "num_K": 3,
            "seed": 67137,
        },
    }
    if normalized not in configs:
        raise ValueError("dataset must be IEMOCAP or MELD")
    return dict(configs[normalized])


def select_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda device requests without silently falling back."""
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    return device


def parameter_update_summary(
    model: nn.Module, before: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Summarize finite gradients and actual optimizer-driven parameter changes."""
    trainable = 0
    finite_grad = 0
    updated = 0
    max_abs_update = 0.0
    updated_names: list[str] = []
    finite_grad_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable += 1
        if parameter.grad is not None and torch.isfinite(parameter.grad).all().item():
            finite_grad += 1
            finite_grad_names.append(name)
        delta = (parameter.detach() - before[name]).abs().max().item()
        max_abs_update = max(max_abs_update, delta)
        if delta > 0:
            updated += 1
            updated_names.append(name)
    return {
        "trainable_parameters": trainable,
        "parameters_with_finite_grad": finite_grad,
        "parameters_updated": updated,
        "max_abs_update": max_abs_update,
        "finite_grad_names": finite_grad_names,
        "updated_names": updated_names,
    }


def outputs_numerically_equal(reference: torch.Tensor, restored: torch.Tensor) -> bool:
    """Accept float32 CUDA roundoff while still detecting a changed model."""
    return torch.allclose(reference, restored, rtol=1e-5, atol=1e-7)


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Build a complete one-step resume checkpoint (not an inference-only state)."""
    return {
        "format_version": 1,
        "dataset_config": dict(config),
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


@contextlib.contextmanager
def _project_working_directory() -> Iterator[None]:
    """Satisfy the official dataloader's project-relative paths safely."""
    previous = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_dataset_and_batch(config: dict[str, Any], batch_size: int) -> list[Any]:
    from dataloader import IEMOCAPDataset, MELDDataset

    with _project_working_directory():
        if config["dataset"] == "IEMOCAP":
            dataset = IEMOCAPDataset(train=True)
        else:
            dataset = MELDDataset("data/MELD_features_raw1.pkl", train=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        return next(iter(loader))


def _validate_and_move_batch(
    batch: list[Any], config: dict[str, Any], device: torch.device
) -> tuple[
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[Any],
]:
    if len(batch) != 10:
        raise AssertionError(f"expected 10 collated fields, got {len(batch)}")
    text = [tensor.to(device) for tensor in batch[:4]]
    visual, audio, qmask, umask, labels = [tensor.to(device) for tensor in batch[4:9]]
    vids = batch[9]

    sequence, batch_count = text[0].shape[:2]
    expected = {
        "text": (sequence, batch_count, config["text_dim"]),
        "visual": (sequence, batch_count, config["visual_dim"]),
        "audio": (sequence, batch_count, config["audio_dim"]),
        "qmask": (sequence, batch_count, config["n_speakers"]),
        "umask": (batch_count, sequence),
        "labels": (batch_count, sequence),
    }
    actual = {
        "text": tuple(text[0].shape),
        "visual": tuple(visual.shape),
        "audio": tuple(audio.shape),
        "qmask": tuple(qmask.shape),
        "umask": tuple(umask.shape),
        "labels": tuple(labels.shape),
    }
    for index, tensor in enumerate(text):
        if tuple(tensor.shape) != expected["text"]:
            raise AssertionError(f"text layer {index + 1}: {tuple(tensor.shape)} != {expected['text']}")
    for name in ("visual", "audio", "qmask", "umask", "labels"):
        if actual[name] != expected[name]:
            raise AssertionError(f"{name}: {actual[name]} != {expected[name]}")
    for name, tensor in zip(
        ("text1", "text2", "text3", "text4", "visual", "audio", "qmask", "umask"),
        (*text, visual, audio, qmask, umask),
    ):
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"non-finite input in {name}")
    valid_labels = labels[umask.bool()]
    if valid_labels.numel() == 0 or valid_labels.min() < 0 or valid_labels.max() >= config["n_classes"]:
        raise AssertionError("valid labels are empty or outside the official class range")
    return text, visual, audio, qmask, umask, labels, vids


def _build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    from model import Model

    model = Model(
        "GRU",
        1024,
        512 if config["dataset"] == "IEMOCAP" else 1024,
        150,
        100,
        100,
        100,
        512,
        n_speakers=config["n_speakers"],
        max_seq_len=200,
        window_past=10,
        window_future=10,
        n_classes=config["n_classes"],
        listener_state=False,
        context_attention="general",
        dropout=config["dropout"],
        nodal_attention=True,
        no_cuda=device.type != "cuda",
        graph_type="hyper",
        use_topic=False,
        alpha=0.2,
        multiheads=6,
        graph_construct="direct",
        use_GCN=False,
        use_residue=False,
        D_m_v=config["visual_dim"],
        D_m_a=config["audio_dim"],
        modals="avl",
        att_type="concat_DHT",
        av_using_lstm=False,
        Deep_GCN_nlayers=4,
        dataset=config["dataset"],
        use_speaker=True,
        use_modal=False,
        norm="BN",
        num_L=config["num_L"],
        num_K=config["num_K"],
    )
    return model.to(device)


def _build_official_loss(config: dict[str, Any], device: torch.device) -> nn.Module:
    from model import FocalLoss

    if config["dataset"] == "MELD":
        return FocalLoss()
    weights = torch.tensor(
        [1 / 0.086747, 1 / 0.144406, 1 / 0.227883, 1 / 0.160585, 1 / 0.127711, 1 / 0.252668],
        dtype=torch.float32,
        device=device,
    )
    return nn.NLLLoss(weights)


def _forward(
    model: nn.Module,
    tensors: tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    text, visual, audio, qmask, umask, labels, _ = tensors
    lengths = [int(mask.sum().item()) for mask in umask]
    if any(length <= 0 for length in lengths):
        raise AssertionError("a real conversation has zero valid utterances")
    output = model(text, qmask, umask, lengths, audio, visual, 0)
    log_prob = output[0]
    flat_labels = torch.cat([labels[index, :length] for index, length in enumerate(lengths)])
    if tuple(log_prob.shape) != (sum(lengths), model.smax_fc.out_features):
        raise AssertionError(
            f"log_prob shape {tuple(log_prob.shape)} != {(sum(lengths), model.smax_fc.out_features)}"
        )
    return log_prob, flat_labels, lengths


def run_smoke(
    dataset: str, device_request: str, output_dir: Path, batch_size: int = 1
) -> dict[str, Any]:
    config = dataset_config(dataset)
    device = select_device(device_request)
    # The upstream hypergraph implementation constructs index tensors with
    # unconditional .cuda() calls.  Fail clearly rather than implying CPU parity.
    if device.type != "cuda":
        raise RuntimeError(
            "the official CMERC hypergraph path requires CUDA because model_hyper.py "
            "constructs graph tensors with unconditional .cuda() calls"
        )
    _seed_everything(config["seed"])
    if batch_size < 1:
        raise ValueError("batch-size must be positive")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batch = _build_dataset_and_batch(config, batch_size)
    tensors = _validate_and_move_batch(batch, config, device)
    text, visual, audio, qmask, umask, labels, vids = tensors

    model = _build_model(config, device)
    loss_function = _build_official_loss(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}

    model.train()
    optimizer.zero_grad()
    log_prob, flat_labels, lengths = _forward(model, tensors)
    loss = loss_function(log_prob, flat_labels)
    if not torch.isfinite(log_prob).all().item() or not math.isfinite(loss.item()):
        raise FloatingPointError("forward output or official classification loss is non-finite")
    loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("backward produced a NaN or Inf gradient")
    optimizer.step()
    updates = parameter_update_summary(model, before)
    expected_parameters = {
        "linear_a.weight",
        "linear_v.weight",
        "gru_l.weight_ih_l0",
        "graph_model.fc1.weight",
        "smax_fc.weight",
    }
    finite_names = set(updates["finite_grad_names"])
    updated_names = set(updates["updated_names"])
    missing_grad = sorted(expected_parameters - finite_names)
    missing_update = sorted(expected_parameters - updated_names)
    if missing_grad or missing_update:
        raise AssertionError(
            f"expected trainable path did not receive finite gradients/updates: "
            f"gradient={missing_grad}, update={missing_update}"
        )

    model.eval()
    with torch.no_grad():
        reference, _, _ = _forward(model, tensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config["dataset"].lower()
    checkpoint_path = output_dir / f"{stem}_smoke_checkpoint.pt"
    report_path = output_dir / f"{stem}_smoke.json"
    torch.save(build_checkpoint(model, optimizer, config, step=1), checkpoint_path)

    restored = _build_model(config, device)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4, weight_decay=3e-5)
    loaded = torch.load(checkpoint_path, map_location=device)
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    checkpoint_state_exact = all(
        torch.equal(value.detach().cpu(), loaded["model_state_dict"][name].detach().cpu())
        for name, value in restored.state_dict().items()
    )
    if not checkpoint_state_exact:
        raise AssertionError("strict checkpoint load succeeded but a restored tensor differs")
    restored.eval()
    with torch.no_grad():
        restored_output, _, _ = _forward(restored, tensors)
    reload_max_abs_diff = (reference - restored_output).abs().max().item()
    if not outputs_numerically_equal(reference, restored_output):
        raise AssertionError(f"checkpoint reload changed inference output by {reload_max_abs_diff}")

    report = {
        "status": "passed",
        "scope": "single-real-batch engineering smoke test; not a paper result",
        "dataset": config["dataset"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "config": config,
        "source": {
            "conversation_ids": list(vids),
            "batch_size": len(vids),
            "valid_utterances": sum(lengths),
        },
        "shapes": {
            "text_layers": [list(value.shape) for value in text],
            "visual": list(visual.shape),
            "audio": list(audio.shape),
            "speaker_mask": list(qmask.shape),
            "utterance_mask": list(umask.shape),
            "labels": list(labels.shape),
            "log_prob": list(log_prob.shape),
        },
        "checks": {
            "inputs_finite": True,
            "forward_finite": True,
            "classification_loss_finite": True,
            "backward_finite": True,
            "optimizer_step": True,
            "expected_parameters_have_gradients": True,
            "expected_parameters_updated": True,
            "checkpoint_strict_load": True,
            "checkpoint_state_tensors_exact": checkpoint_state_exact,
            "checkpoint_optimizer_load": True,
            "checkpoint_output_max_abs_diff": reload_max_abs_diff,
            "checkpoint_output_allclose_rtol": 1e-5,
            "checkpoint_output_allclose_atol": 1e-7,
        },
        "classification_loss": {
            "name": type(loss_function).__name__,
            "value": loss.item(),
        },
        "parameters": {
            "trainable_parameter_tensors": updates["trainable_parameters"],
            "finite_gradient_tensors": updates["parameters_with_finite_grad"],
            "updated_parameter_tensors": updates["parameters_updated"],
            "max_abs_update": updates["max_abs_update"],
            "explicitly_checked": sorted(expected_parameters),
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("IEMOCAP", "MELD", "iemocap", "meld"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "smoke")
    parser.add_argument("--batch-size", type=int, default=1, help="real conversations per smoke batch")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_smoke(args.dataset, args.device, args.output_dir, args.batch_size)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
