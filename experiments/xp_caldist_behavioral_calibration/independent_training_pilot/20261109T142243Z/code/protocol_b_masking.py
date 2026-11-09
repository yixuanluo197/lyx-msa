"""Post-forward supervision filtering for Protocol B Unknown-Label Mask Policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def flatten_supervision_mask(mask: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    if mask.ndim != 2 or mask.size(0) != len(lengths):
        raise ValueError("supervision mask must be [batch, time] and match lengths")
    selected = []
    for row, length in zip(mask.bool(), lengths):
        if length < 0 or length > row.numel():
            raise ValueError("invalid dialogue length")
        if row[length:].any():
            raise ValueError("supervision mask must be false at padding positions")
        selected.append(row[:length])
    return torch.cat(selected) if selected else mask.new_zeros(0, dtype=torch.bool)


def select_after_full_forward(tensor: torch.Tensor, supervision_mask: torch.Tensor) -> torch.Tensor:
    mask = supervision_mask.to(device=tensor.device, dtype=torch.bool)
    if tensor.size(0) != mask.numel():
        raise ValueError("mask and flattened forward output are misaligned")
    return tensor[mask]


def _relation_label(student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(labels.numel(), device=labels.device)
    student_prediction = student.argmax(dim=1)
    teacher_prediction = teacher.argmax(dim=1)
    student_correct = student_prediction.eq(labels)
    teacher_correct = teacher_prediction.eq(labels)
    student_true = student[rows, labels]
    teacher_true = teacher[rows, labels]
    return (student_correct & (~teacher_correct | (student_true > teacher_true))).long()


def prepare_supervised_views(
    student_log_prob: torch.Tensor,
    teacher_log_prob: Sequence[torch.Tensor],
    hidden: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> dict:
    if len(teacher_log_prob) != 5:
        raise ValueError("CMERC requires three modality and two context teacher views")
    mask = supervision_mask.to(device=student_log_prob.device, dtype=torch.bool)
    student = select_after_full_forward(student_log_prob, mask)
    selected_hidden = select_after_full_forward(hidden, mask)
    selected_labels = select_after_full_forward(labels, mask)
    teachers = [select_after_full_forward(view.detach(), mask) for view in teacher_log_prob]
    if selected_labels.numel():
        rows = torch.arange(selected_labels.numel(), device=selected_labels.device)
        rank_loss = sum(
            torch.relu(view[rows, selected_labels] - student[rows, selected_labels]).sum()
            for view in teachers
        )
        relations = [_relation_label(student, view, selected_labels) for view in teachers]
    else:
        rank_loss = student.sum() * 0.0
        relations = [selected_labels.clone() for _ in teachers]
    return {
        "student_log_prob": student,
        "teacher_log_prob": teachers,
        "hidden": selected_hidden,
        "labels": selected_labels,
        "rank_loss": rank_loss,
        "relation_labels": relations,
    }


def masked_classification_loss(
    log_prob: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    mask = supervision_mask.to(device=log_prob.device, dtype=torch.bool)
    if not mask.any():
        return log_prob.sum() * 0.0
    return criterion(log_prob[mask], labels.to(log_prob.device)[mask])


def fold_local_class_weights(labels: torch.Tensor, supervision_mask: torch.Tensor, classes: int) -> torch.Tensor:
    mask = supervision_mask.to(device=labels.device, dtype=torch.bool)
    selected = labels[mask]
    counts = torch.bincount(selected, minlength=classes).to(torch.float32)
    if counts.numel() != classes or (counts == 0).any():
        raise ValueError("every class must have eligible Training support")
    return selected.numel() / counts


def should_optimizer_step(supervision_mask: torch.Tensor) -> bool:
    return bool(supervision_mask.bool().any().item())


def safe_contrastive_loss(features: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if features.size(0) < 2:
        return features.sum() * 0.0
    return loss_fn(features)
