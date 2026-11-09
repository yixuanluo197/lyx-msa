#!/usr/bin/env python3
"""Experiment-local Protocol-B post-forward loss runner and CPU tensor audit.

The caller supplies full-dialogue student and five removal forward closures. No
labels or supervision mask enter these closures. The frozen CMERC graph, removal
construction, FocalLoss/NLL choice, and teacher refresh remain caller-owned.
This module does not launch CMERC training or load any research dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = EXPERIMENT / "source_snapshot"
REFERENCE_TRAIN = ROOT / "CMERC/experiment_records/04_xp_mainline_validation/code/train_cm.py"
REMOVAL_NAMES = ("text", "audio", "visual", "intra", "inter")
_FROZEN_CLLOSS = None


def load_frozen_clloss():
    global _FROZEN_CLLOSS
    if _FROZEN_CLLOSS is None:
        expected = json.loads((EXPERIMENT / "provenance/source_manifest.json").read_text())["files"]["clloss.py"]
        if hashlib.sha256((FROZEN_SOURCE / "clloss.py").read_bytes()).hexdigest() != expected:
            raise ValueError("frozen contrastive source hash mismatch")
        spec = importlib.util.spec_from_file_location("frozen_xp04_clloss", FROZEN_SOURCE / "clloss.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FROZEN_CLLOSS = module
    return _FROZEN_CLLOSS


def flatten_supervision_mask(mask: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    if mask.ndim != 2 or mask.size(0) != len(lengths):
        raise ValueError("mask must have dialogue-major [batch, time] shape")
    selected = []
    for row, length in zip(mask.bool(), lengths):
        if not 0 <= length <= row.numel():
            raise ValueError("invalid dialogue length")
        if row[length:].any():
            raise ValueError("supervision mask is true at padding")
        selected.append(row[:length])
    return torch.cat(selected) if selected else mask.new_zeros(0, dtype=torch.bool)


def _eligible(log_prob: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    if log_prob.ndim != 2 or labels.shape != (log_prob.size(0),) or mask.shape != labels.shape:
        raise ValueError("flattened scores, labels, and supervision mask must align")
    mask = mask.to(device=log_prob.device, dtype=torch.bool)
    selected_labels = labels.to(log_prob.device)[mask]
    if selected_labels.dtype != torch.long:
        raise ValueError("eligible labels must be int64")
    if ((selected_labels < 0) | (selected_labels >= log_prob.size(1))).any():
        raise ValueError("eligible label is outside the class range")
    return mask, selected_labels


def training_class_weights(labels: torch.Tensor, mask: torch.Tensor, classes: int) -> torch.Tensor:
    """Call on complete eligible Training labels, before curriculum subsampling."""
    if labels.ndim != 1 or mask.shape != labels.shape or classes <= 0:
        raise ValueError("Training labels and supervision mask must align")
    selected = labels[mask.to(device=labels.device, dtype=torch.bool)]
    if selected.dtype != torch.long or ((selected < 0) | (selected >= classes)).any():
        raise ValueError("eligible Training labels are outside the class range")
    counts = torch.bincount(selected, minlength=classes).to(torch.float32)
    if (counts == 0).any():
        raise ValueError("every class must have eligible Training support")
    return selected.numel() / counts


def relation_class_weights(labels: torch.Tensor) -> list[float]:
    """Frozen create_class_weight_SCL: N/count, absent binary classes get zero."""
    if not labels.numel():
        return []
    counts = torch.bincount(labels, minlength=int(labels.max()) + 1).tolist()
    return [labels.numel() / count if count else 0.0 for count in counts]


def masked_objective(
    log_prob: torch.Tensor,
    removal_log_prob: Sequence[torch.Tensor],
    hidden: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    criterion: Callable,
    *,
    calibrate: bool = True,
    contrastlearning: bool = True,
    rank_coff: float = .005,
    mscl_coff: float = .05,
    cscl_coff: float = .05,
) -> dict:
    """Filter every supervised axis after all full-context forwards complete.

    Teachers are detached. Hidden remains attached to the student graph. Frozen
    CKSCL receives ONLY eligible hidden rows, excluding unknown rows as anchors,
    positives, and negatives. For N<2 its undefined ratio is replaced by an
    attached zero. Empty supervision produces attached zeros and no step.
    """
    mask, target = _eligible(log_prob, labels, supervision_mask)
    if len(removal_log_prob) != 5 or any(view.shape != log_prob.shape for view in removal_log_prob):
        raise ValueError("exactly five aligned removal views are required")
    if hidden.ndim != 2 or hidden.size(0) != log_prob.size(0):
        raise ValueError("hidden rows must align with full forward scores")
    student, features = log_prob[mask], hidden[mask]
    teachers = [view.detach().to(log_prob.device)[mask] for view in removal_log_prob]
    zero = student.sum() * 0.0 + features.sum() * 0.0
    classification = criterion(student, target) if target.numel() else zero
    rows = torch.arange(target.numel(), device=log_prob.device)
    student_true = student[rows, target]
    student_correct = student.argmax(1).eq(target)
    rank, relations, weights, contrastive, roles = {}, {}, {}, {}, {}
    eligible_indices = torch.where(mask)[0].tolist()
    frozen = load_frozen_clloss()
    for name, teacher in zip(REMOVAL_NAMES, teachers):
        teacher_true = teacher[rows, target]
        # torch.relu has a different derivative at exact equality than XP04.
        rank[name] = torch.where(student_true <= teacher_true, teacher_true - student_true,
                                 torch.zeros_like(student_true)).sum()
        relation = (student_correct & (~teacher.argmax(1).eq(target) | (student_true > teacher_true))).long()
        relations[name] = relation
        weights[name] = relation_class_weights(relation)
        if target.numel() < 2:
            contrastive[name] = zero
        else:
            # Use the actual frozen pairwise kernel, including its epsilon mask.
            pair_logits = frozen.CKSCL_base()(features, labels=relation)
            term = -(pair_logits * features.new_tensor(weights[name])[relation]).mean()
            # Preserve the frozen aggregate's nonfinite fallback.
            contrastive[name] = term
        roles[name] = {"anchors": eligible_indices, "positive_candidates": eligible_indices,
                       "negative_candidates": eligible_indices}
    rank_loss = sum(rank.values(), zero)
    mscl = sum((contrastive[name] for name in REMOVAL_NAMES[:3]), zero)
    cscl = sum((contrastive[name] for name in REMOVAL_NAMES[3:]), zero)
    if not torch.isfinite(mscl):
        mscl = zero
    if not torch.isfinite(cscl):
        cscl = zero
    total = classification
    if calibrate:
        total = total + rank_coff * rank_loss
    if contrastlearning:
        total = total + mscl_coff * mscl + cscl_coff * cscl
    return {"classification_loss": classification, "rank_loss": rank_loss,
            "mscl_loss": mscl, "cscl_loss": cscl, "total_loss": total,
            "rank_by_view": rank, "relation_labels": relations, "relation_weights": weights,
            "contrastive_by_view": contrastive, "contrastive_roles": roles,
            "eligible_indices": eligible_indices, "supervised_count": int(target.numel())}


def run_masked_batch(full_forward: Callable, removal_forwards: Sequence[Callable],
                     labels: torch.Tensor, supervision_mask: torch.Tensor,
                     criterion: Callable, optimizer=None, **loss_options) -> dict:
    """Execute all six full-context forwards before supervised selection."""
    if len(removal_forwards) != 5:
        raise ValueError("five removal forward closures are required")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    scores, hidden = full_forward()
    with torch.no_grad():
        removals = [forward()[0] for forward in removal_forwards]
    result = masked_objective(scores, removals, hidden, labels, supervision_mask,
                              criterion, **loss_options)
    result["optimizer_step"] = False
    if optimizer is not None and result["supervised_count"]:
        result["total_loss"].backward()
        optimizer.step()
        result["optimizer_step"] = True
    return result


def development_metrics(log_prob: torch.Tensor, labels: torch.Tensor,
                         supervision_mask: torch.Tensor) -> dict:
    mask, labels = _eligible(log_prob, labels, supervision_mask)
    count = labels.numel()
    if not count:
        return {"count": 0, "accuracy": None, "weighted_f1": None, "nll": None}
    scores = log_prob[mask].detach()
    prediction = scores.argmax(1)
    weighted_f1 = 0.0
    for target in labels.unique():
        support = int(labels.eq(target).sum())
        predicted = int(prediction.eq(target).sum())
        true_positive = int((prediction.eq(target) & labels.eq(target)).sum())
        weighted_f1 += support * 2.0 * true_positive / (support + predicted)
    return {"count": int(count), "accuracy": float(prediction.eq(labels).double().mean()),
            "weighted_f1": weighted_f1 / count, "nll": float(F.nll_loss(scores, labels))}


def curriculum_training_ids(training_ids: Sequence[str], difficulty: Mapping[str, float],
                             ratio: float) -> list[str]:
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("duplicate Training IDs are not allowed")
    if set(difficulty) != set(training_ids):
        raise ValueError("difficulty must contain exact Training IDs; no fallback is allowed")
    if not all(math.isfinite(value) for value in difficulty.values()):
        raise ValueError("difficulty values must be finite")
    if not 0 < ratio <= 1:
        raise ValueError("curriculum ratio must be in (0, 1]")
    selected = set(sorted(training_ids, key=difficulty.__getitem__)[:int(ratio * len(training_ids))])
    return [key for key in training_ids if key in selected]


def tensor_audit_report() -> dict:
    """Measured per-branch gradients; this is a synthetic CPU integration audit."""
    torch.manual_seed(142)
    torch.set_num_threads(1)
    scores = F.log_softmax(torch.randn(8, 3), dim=1).detach().requires_grad_()
    teachers = [F.log_softmax(torch.randn(8, 3), dim=1).requires_grad_() for _ in range(5)]
    hidden = torch.randn(8, 7, requires_grad=True)
    labels = torch.tensor([0, 999, 1, -40, 2, 0, 1, 2])
    mask = torch.tensor([1, 0, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    result = masked_objective(scores, teachers, hidden, labels, mask, F.nll_loss)
    gradients = {}
    terms = {"classification": result["classification_loss"],
             **{f"rank_{name}": term for name, term in result["rank_by_view"].items()},
             **{f"{'MSCL' if name in REMOVAL_NAMES[:3] else 'CSCL'}_{name}": term
                for name, term in result["contrastive_by_view"].items()}}
    for name, term in terms.items():
        source = hidden if name.startswith(("MSCL", "CSCL")) else scores
        gradient, = torch.autograd.grad(term, source, retain_graph=True)
        excluded_max = float(gradient[~mask].abs().max())
        valid_sum = float(gradient[mask].abs().sum())
        assert excluded_max == 0 and valid_sum > 0 and torch.isfinite(gradient).all()
        gradients[name] = {"value": float(term.detach()), "excluded_direct_gradient_abs_max": excluded_max,
                           "eligible_gradient_abs_sum": valid_sum}
    result["total_loss"].backward()
    assert all(view.grad is None for view in teachers)
    return {"status": "passed", "scope": "synthetic CPU tensor runner integration; no CMERC training",
            "torch_version": torch.__version__, "device": "cpu", "full_rows": 8,
            "eligible_rows": 6, "unknown_rows": 2, "label_dependent_paths": gradients,
            "teachers_detached": True, "total_loss": float(result["total_loss"].detach()),
            "frozen_sources_sha256": {
                "clloss.py": hashlib.sha256((FROZEN_SOURCE / "clloss.py").read_bytes()).hexdigest(),
                "train_cm.py": hashlib.sha256(REFERENCE_TRAIN.read_bytes()).hexdigest() if REFERENCE_TRAIN.is_file() else None},
            "contrastive_runtime_source": "source_snapshot/clloss.py",
            "limitations": ["Full frozen CMERC graph forward is not exercised by this audit.",
                            "Caller must pass complete Training labels when constructing class weights.",
                            "Forward closures must preserve original dialogue features and structure."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", required=True, type=Path)
    arguments = parser.parse_args()
    report = tensor_audit_report()
    arguments.audit_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.audit_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "paths": len(report["label_dependent_paths"]),
                      "output": str(arguments.audit_output)}))
