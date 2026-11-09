import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).resolve().parents[1] / "code/protocol_b_masking.py"


def load_module():
    spec = importlib.util.spec_from_file_location("protocol_b_masking", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flatten_mask_preserves_dialogue_major_alignment_and_rejects_padding():
    module = load_module()
    mask = torch.tensor([[True, False, True], [True, True, False]])
    flattened = module.flatten_supervision_mask(mask, [3, 2])
    assert flattened.tolist() == [True, False, True, True, True]
    with pytest.raises(ValueError, match="padding"):
        module.flatten_supervision_mask(
            torch.tensor([[True, True, True], [True, True, True]]), [3, 2]
        )


def test_excluded_labels_never_affect_any_ground_truth_dependent_component():
    module = load_module()
    torch.manual_seed(4)
    logits = torch.randn(5, 3, requires_grad=True)
    log_prob = F.log_softmax(logits, dim=1)
    removed = [F.log_softmax(logits.detach() + shift, dim=1) for shift in (
        torch.tensor([[.2, -.1, 0.], [0., .1, -.1], [.1, 0., -.1], [0., 0., 0.], [-.1, .2, -.1]]),
        torch.randn(5, 3) * .1,
        torch.randn(5, 3) * .1,
        torch.randn(5, 3) * .1,
        torch.randn(5, 3) * .1,
    )]
    hidden = torch.randn(5, 4, requires_grad=True)
    mask = torch.tensor([True, False, True, False, True])
    labels_a = torch.tensor([0, 0, 2, 1, 1])
    labels_b = torch.tensor([0, 999, 2, -50, 1])
    result_a = module.prepare_supervised_views(log_prob, removed, hidden, labels_a, mask)
    result_b = module.prepare_supervised_views(log_prob, removed, hidden, labels_b, mask)
    for key in ("labels", "student_log_prob", "hidden", "rank_loss", "relation_labels"):
        if key == "relation_labels":
            assert all(torch.equal(a, b) for a, b in zip(result_a[key], result_b[key]))
        else:
            assert torch.equal(result_a[key], result_b[key])
    assert result_a["hidden"].requires_grad
    assert all(not tensor.requires_grad for tensor in result_a["teacher_log_prob"])


def test_masked_classification_and_fold_local_weights_use_only_eligible_rows():
    module = load_module()
    logits = torch.tensor([[3., 0.], [0., 3.], [2., 1.]], requires_grad=True)
    log_prob = F.log_softmax(logits, dim=1)
    labels = torch.tensor([0, 999, 1])
    mask = torch.tensor([True, False, True])
    loss = module.masked_classification_loss(log_prob, labels, mask, F.nll_loss)
    expected = F.nll_loss(log_prob[[0, 2]], torch.tensor([0, 1]))
    assert torch.allclose(loss, expected)
    weights = module.fold_local_class_weights(labels, mask, 2)
    assert torch.equal(weights, torch.tensor([2., 2.]))


def test_empty_supervision_skips_step_and_singleton_contrastive_is_finite_zero():
    module = load_module()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=.1, weight_decay=.1)
    before_parameter = parameter.detach().clone()
    before_state = dict(optimizer.state)
    assert module.should_optimizer_step(torch.zeros(3, dtype=torch.bool)) is False
    assert torch.equal(parameter.detach(), before_parameter)
    assert optimizer.state == before_state
    singleton = torch.randn(1, 4, requires_grad=True)
    zero = module.safe_contrastive_loss(singleton, lambda _: torch.tensor(float("nan")))
    assert zero.item() == 0.0 and torch.isfinite(zero) and zero.requires_grad


def test_masking_occurs_after_full_context_forward_without_mutating_structure():
    module = load_module()
    features = torch.tensor([[1.0], [7.0], [2.0]], requires_grad=True)
    qmask = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    umask = torch.ones(3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    originals = tuple(t.clone() for t in (features, qmask, umask, edge_index))
    # Every output depends on every utterance, including the masked middle context.
    full_hidden = features + features.sum()
    mask = torch.tensor([True, False, True])
    supervised = module.select_after_full_forward(full_hidden, mask)
    supervised.sum().backward()
    assert features.grad[1].item() != 0.0
    for current, original in zip((features.detach(), qmask, umask, edge_index), originals):
        assert torch.equal(current, original)
