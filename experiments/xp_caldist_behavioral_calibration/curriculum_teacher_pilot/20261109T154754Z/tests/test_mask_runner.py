"""CPU tensor regression audit; no dataset, checkpoint, or training run."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/protocol_b_mask_runner.py"


def load_runner():
    assert SCRIPT.is_file(), "experiment-local mask runner is not implemented"
    spec = importlib.util.spec_from_file_location("protocol_b_mask_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors():
    torch.manual_seed(142)
    scores = F.log_softmax(torch.randn(8, 3), dim=1).detach().requires_grad_()
    teachers = [F.log_softmax(torch.randn(8, 3), dim=1).requires_grad_() for _ in range(5)]
    hidden = torch.randn(8, 7, requires_grad=True)
    labels = torch.tensor([0, 999, 1, -40, 2, 0, 1, 2])
    mask = torch.tensor([1, 0, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    return scores, teachers, hidden, labels, mask


def reference_relation(student, teacher, labels):
    result = torch.zeros_like(labels)
    for index, target in enumerate(labels):
        correct = student[index].argmax() == target
        teacher_correct = teacher[index].argmax() == target
        if correct and not teacher_correct:
            result[index] = 1
        elif correct and teacher_correct and student[index, target] > teacher[index, target]:
            result[index] = 1
    return result


def test_all_components_match_frozen_xp04_equations_and_aggregate_losses():
    runner = load_runner()
    scores, teachers, hidden, labels, mask = tensors()
    weights = runner.training_class_weights(labels, mask, 3)
    result = runner.masked_objective(scores, teachers, hidden, labels, mask,
                                     torch.nn.NLLLoss(weight=weights))
    selected_labels, selected_scores = labels[mask], scores[mask]
    torch.testing.assert_close(result["classification_loss"],
                               F.nll_loss(selected_scores, selected_labels, weight=weights))
    relations, relation_weights = [], []
    for name, teacher in zip(runner.REMOVAL_NAMES, teachers):
        expected_rank = sum((teacher[i, labels[i]] - scores[i, labels[i]])
                            for i in range(len(labels))
                            if mask[i] and scores[i, labels[i]] <= teacher[i, labels[i]])
        torch.testing.assert_close(result["rank_by_view"][name], expected_rank)
        relation = reference_relation(selected_scores, teacher[mask], selected_labels)
        assert torch.equal(result["relation_labels"][name], relation)
        relations.append(relation)
        relation_weights.append(runner.relation_class_weights(relation))
    frozen = runner.load_frozen_clloss()
    mscl = frozen.CKSCL_tav()(hidden[mask], *relations[:3], *relation_weights[:3])
    cscl = frozen.CKSCL_context()(hidden[mask], *relations[3:], *relation_weights[3:])
    torch.testing.assert_close(result["mscl_loss"], mscl)
    torch.testing.assert_close(result["cscl_loss"], cscl)
    torch.testing.assert_close(result["total_loss"], result["classification_loss"]
                               + .005 * result["rank_loss"] + .05 * mscl + .05 * cscl)


@pytest.mark.parametrize("component", ["classification_loss", "rank_loss", "mscl_loss", "cscl_loss"])
def test_masked_rows_have_zero_direct_gradient_and_eligible_rows_backpropagate(component):
    runner = load_runner()
    scores, teachers, hidden, labels, mask = tensors()
    result = runner.masked_objective(scores, teachers, hidden, labels, mask, F.nll_loss)
    result[component].backward()
    tensor = hidden if component in ("mscl_loss", "cscl_loss") else scores
    assert tensor.grad is not None
    assert torch.equal(tensor.grad[~mask], torch.zeros_like(tensor.grad[~mask]))
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad[mask].abs().sum() > 0
    assert all(teacher.grad is None for teacher in teachers)
    if component in ("mscl_loss", "cscl_loss"):
        for name, roles in result["contrastive_roles"].items():
            assert roles["anchors"] == torch.where(mask)[0].tolist()
            assert set(roles["positive_candidates"]) == set(roles["anchors"])
            assert set(roles["negative_candidates"]) == set(roles["anchors"])


def test_each_of_five_rank_branches_preserves_equality_gradient():
    runner = load_runner()
    scores = torch.tensor([[-.4, -1.1], [-.8, -.7]], requires_grad=True)
    teachers = [scores.detach().clone().requires_grad_() for _ in range(5)]
    result = runner.masked_objective(scores, teachers, torch.randn(2, 3, requires_grad=True),
                                     torch.tensor([0, -999]), torch.tensor([True, False]), F.nll_loss)
    for loss in result["rank_by_view"].values():
        gradient, = torch.autograd.grad(loss, scores, retain_graph=True)
        assert gradient.tolist() == [[-1., 0.], [0., 0.]]
    result["rank_loss"].backward()
    assert scores.grad.tolist() == [[-5., 0.], [0., 0.]]


def test_sentinel_replacement_cannot_change_any_label_dependent_component():
    runner = load_runner()
    scores, teachers, hidden, labels, mask = tensors()
    other_labels = labels.clone()
    other_labels[~mask] = torch.tensor([-999999, 999999])
    first = runner.masked_objective(scores, teachers, hidden, labels, mask, F.nll_loss)
    second = runner.masked_objective(scores, teachers, hidden, other_labels, mask, F.nll_loss)
    for key in ("classification_loss", "rank_loss", "mscl_loss", "cscl_loss", "total_loss"):
        assert torch.equal(first[key], second[key])
    assert torch.equal(runner.training_class_weights(labels, mask, 3),
                       runner.training_class_weights(other_labels, mask, 3))
    assert runner.development_metrics(scores, labels, mask) == runner.development_metrics(scores, other_labels, mask)


def test_training_weights_use_complete_eligible_training_support_before_curriculum():
    runner = load_runner()
    assert runner.training_class_weights(torch.tensor([0, 0, 1, 999, 2, 2, 2]),
                                        torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype=torch.bool), 3).tolist() == [3., 6., 2.]
    with pytest.raises(ValueError, match="every class"):
        runner.training_class_weights(torch.tensor([0, -999]), torch.tensor([True, False]), 3)


def test_development_metrics_exclude_unknown_rows_from_nll_accuracy_and_weighted_f1():
    runner = load_runner()
    scores = torch.tensor([[.8, .2], [.99, .01], [.3, .7], [.6, .4]]).log()
    result = runner.development_metrics(scores, torch.tensor([0, 999, 1, 1]),
                                         torch.tensor([True, False, True, True]))
    assert result["count"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["weighted_f1"] == pytest.approx(2 / 3)
    assert result["nll"] == pytest.approx(float(-torch.tensor([.8, .7, .4]).log().mean()))
    assert runner.development_metrics(scores, torch.full((4,), -999), torch.zeros(4, dtype=torch.bool)) == {
        "count": 0, "accuracy": None, "weighted_f1": None, "nll": None}


def test_full_forward_precedes_mask_and_unknown_context_remains_differentiable():
    runner = load_runner()
    parameter = torch.nn.Parameter(torch.tensor([[.7, -.3], [.2, .5]]))
    features = torch.tensor([[1., .1], [7., -.3], [2., .8]], requires_grad=True)
    qmask = torch.eye(3)
    umask = torch.ones(3)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    before = [x.detach().clone() for x in (features, qmask, umask, edges)]
    calls = []

    def forward():
        calls.append(features.size(0))
        hidden = features @ parameter + (features @ parameter).mean(0)
        return hidden.log_softmax(1), hidden

    result = runner.run_masked_batch(forward, [forward] * 5, torch.tensor([0, -999, 1]),
                                     torch.tensor([True, False, True]), F.nll_loss)
    result["total_loss"].backward()
    assert calls == [3] * 6
    assert features.grad[1].abs().sum() > 0
    assert parameter.grad.abs().sum() > 0
    for actual, original in zip((features.detach(), qmask, umask, edges), before):
        assert torch.equal(actual, original)


def test_empty_batch_does_not_advance_populated_adam_state_or_weight_decay():
    runner = load_runner()
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(layer.parameters(), lr=.1, weight_decay=.2)
    features = torch.tensor([[1., 2.], [3., 1.]])

    def forward():
        hidden = layer(features)
        return hidden.log_softmax(1), hidden

    runner.run_masked_batch(forward, [forward] * 5, torch.tensor([0, 1]),
                            torch.ones(2, dtype=torch.bool), F.nll_loss, optimizer)
    before_parameters = copy.deepcopy(layer.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    result = runner.run_masked_batch(forward, [forward] * 5, torch.tensor([-999, 999]),
                                     torch.zeros(2, dtype=torch.bool), F.nll_loss, optimizer)
    assert not result["optimizer_step"]
    assert result["total_loss"].item() == 0
    assert all(torch.equal(value, before_parameters[key]) for key, value in layer.state_dict().items())
    for key, value in optimizer.state_dict()["state"].items():
        assert all(torch.equal(tensor, before_optimizer["state"][key][name]) for name, tensor in value.items())


def test_singleton_contrastive_terms_are_finite_differentiable_zero():
    runner = load_runner()
    scores, teachers, hidden, labels, _ = tensors()
    mask = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    result = runner.masked_objective(scores, teachers, hidden, labels, mask, F.nll_loss)
    for key in ("mscl_loss", "cscl_loss"):
        assert result[key].requires_grad and result[key].item() == 0
    result["total_loss"].backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))


def test_curriculum_sampler_requires_exact_training_keys_and_restores_source_order():
    runner = load_runner()
    ids = ["train-c", "train-a", "train-b", "train-d", "train-e"]
    difficulty = {"train-a": .1, "train-b": .2, "train-c": .3, "train-d": .4, "train-e": .5}
    assert runner.curriculum_training_ids(ids, difficulty, .6) == ["train-c", "train-a", "train-b"]
    assert runner.curriculum_training_ids(ids, difficulty, 1) == ids
    for invalid in ({**difficulty, "development": .0}, {key: value for key, value in difficulty.items() if key != "train-a"}):
        with pytest.raises(ValueError, match="exact Training"):
            runner.curriculum_training_ids(ids, invalid, .6)
    with pytest.raises(ValueError, match="duplicate"):
        runner.curriculum_training_ids(ids + [ids[0]], difficulty, 1)
    with pytest.raises(ValueError, match="finite"):
        runner.curriculum_training_ids(ids, {**difficulty, "train-a": float("nan")}, 1)


def test_dialogue_major_mask_rejects_padding_and_does_not_change_lengths():
    runner = load_runner()
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    assert runner.flatten_supervision_mask(mask, [3, 2]).tolist() == [True, False, True, True, True]
    with pytest.raises(ValueError, match="padding"):
        runner.flatten_supervision_mask(torch.ones(2, 3, dtype=torch.bool), [3, 2])


def test_every_individual_loss_path_has_audited_real_tensor_gradients():
    report = load_runner().tensor_audit_report()
    assert report["status"] == "passed"
    assert len(report["label_dependent_paths"]) == 11
    for path in report["label_dependent_paths"].values():
        assert path["excluded_direct_gradient_abs_max"] == 0
        assert path["eligible_gradient_abs_sum"] > 0


@pytest.mark.parametrize("dataset", ["IEMOCAP", "MELD"])
def test_sampler_exact_fold1_manifest_training_ids_without_reading_heldout_labels(dataset):
    runner = load_runner()
    manifest_path = SCRIPT.parents[1] / "configs/approved_split_manifests.reference.json"
    manifest = json.loads(manifest_path.read_text())
    roles = manifest["protocols"][dataset]["folds"]["fold1"]["roles"]
    ids = roles["training"]["dialogue_ids"]
    # Synthetic scores audit only membership/order, never stand in for a trained teacher artifact.
    difficulty = {key: float(index) for index, key in enumerate(reversed(ids))}
    assert runner.curriculum_training_ids(ids, difficulty, 1) == ids
    ratio = .15 if dataset == "IEMOCAP" else .40
    chosen = runner.curriculum_training_ids(ids, difficulty, ratio)
    assert len(chosen) == int(ratio * len(ids))
    assert set(chosen).issubset(ids)
    assert chosen == [key for key in ids if key in set(chosen)]
    development_id = roles["development"]["dialogue_ids"][0]
    assert development_id not in ids
    with pytest.raises(ValueError, match="exact Training"):
        runner.curriculum_training_ids(ids, {**difficulty, development_id: -.1}, ratio)
