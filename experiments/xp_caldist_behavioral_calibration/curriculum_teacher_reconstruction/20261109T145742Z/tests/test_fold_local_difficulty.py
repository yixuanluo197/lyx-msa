import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_fold_local_difficulty.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fold_difficulty", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return load_module()


def fixture_dialogue(mask=(True, False, True)):
    full = torch.log_softmax(torch.tensor([[3.0, 1.0], [1.0, 2.0], [2.0, 1.0]]), 1)
    removals = []
    for shift in (0.1, 0.2, 0.3, 0.4, 0.5):
        altered = full.clone()
        altered[:, 0] += shift
        removals.append(altered)
    qmask = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    return full, removals, torch.tensor([0, -1, 0]), torch.tensor(mask), qmask


def test_masked_label_is_never_gathered(mod):
    full, removals, labels, mask, qmask = fixture_dialogue()
    row = mod.compute_dialogue_decomposition("d", full, removals, labels, mask, qmask)
    assert row["valid_label_count"] == 2
    assert row["total_gap"] == pytest.approx(3.0)


def test_valid_count_formula(mod):
    full, removals, labels, mask, qmask = fixture_dialogue()
    row = mod.compute_dialogue_decomposition("d", full, removals, labels, mask, qmask)
    assert row["speaker_term"] == 2.0
    assert row["difficulty_valid_count"] == pytest.approx(5 / 4)


def test_total_length_control_formula(mod):
    full, removals, labels, mask, qmask = fixture_dialogue()
    row = mod.compute_dialogue_decomposition("d", full, removals, labels, mask, qmask)
    assert row["difficulty_total_length"] == pytest.approx(1.0)


def test_all_unknown_dialogue_fails(mod):
    full, removals, labels, _, qmask = fixture_dialogue(mask=(False, False, False))
    with pytest.raises(ValueError, match="no supervision-eligible"):
        mod.compute_dialogue_decomposition("d", full, removals, labels, torch.zeros(3, dtype=torch.bool), qmask)


def test_finite_guard(mod):
    full, removals, labels, mask, qmask = fixture_dialogue()
    removals[0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mod.compute_dialogue_decomposition("d", full, removals, labels, mask, qmask)


def test_exact_training_coverage(mod):
    mod.verify_exact_coverage({"a": {}, "b": {}}, ["a", "b"])
    with pytest.raises(ValueError, match="coverage"):
        mod.verify_exact_coverage({"a": {}}, ["a", "b"])
    with pytest.raises(ValueError, match="coverage"):
        mod.verify_exact_coverage({"a": {}, "b": {}, "heldout": {}}, ["a", "b"])


def test_training_role_extraction_excludes_heldout(mod):
    manifest = {"protocols": {"IEMOCAP": {"folds": {"fold1": {"roles": {
        "training": {"dialogue_ids": ["train"]}, "development": {"dialogue_ids": ["dev"]},
        "calibration": {"dialogue_ids": ["cal"]}, "evaluation": {"dialogue_ids": ["eval"]}
    }}}}}}
    assert mod.training_ids_from_manifest(manifest, "IEMOCAP", "fold1") == ["train"]


def test_reject_requested_nontraining_ids(mod):
    with pytest.raises(ValueError, match="non-Training"):
        mod.assert_requested_ids_are_training(["train", "cal"], ["train"])


def test_fixed_final_epoch_protocol(mod):
    cfg = {"selection": "fixed_final_epoch", "epochs": 80, "access_roles": ["training"]}
    mod.validate_teacher_protocol(cfg)
    for bad in ({**cfg, "selection": "best_dev"}, {**cfg, "access_roles": ["training", "development"]}):
        with pytest.raises(ValueError):
            mod.validate_teacher_protocol(bad)


def test_strict_checkpoint_loading_and_architecture(mod, tmp_path):
    model = torch.nn.Linear(3, 2)
    cfg = {"name": "linear", "in": 3, "out": 2}
    path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": model.state_dict(), "architecture": cfg, "epoch": 80}, path)
    target = torch.nn.Linear(3, 2)
    meta = mod.strict_load_teacher(target, path, cfg, 80)
    assert meta["epoch"] == 80
    bad = torch.nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        mod.strict_load_teacher(bad, path, cfg, 80)


def test_checkpoint_architecture_mismatch(mod, tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": model.state_dict(), "architecture": {"a": 1}, "epoch": 40}, path)
    with pytest.raises(ValueError, match="architecture"):
        mod.strict_load_teacher(model, path, {"a": 2}, 40)


def test_checkpoint_final_epoch_mismatch(mod, tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": model.state_dict(), "architecture": {}, "epoch": 39}, path)
    with pytest.raises(ValueError, match="final epoch"):
        mod.strict_load_teacher(model, path, {}, 40)


def test_batch_partition_invariance_of_reducer(mod):
    records = []
    for name in ("a", "b", "c"):
        full, removals, labels, mask, qmask = fixture_dialogue()
        records.append((name, full, removals, labels, mask, qmask))
    one = mod.reduce_dialogues(records, batch_size=1)
    sixteen = mod.reduce_dialogues(records, batch_size=16)
    assert one == sixteen


def test_speaker_term_uses_full_context(mod):
    full, removals, labels, mask, qmask = fixture_dialogue(mask=(True, False, False))
    row = mod.compute_dialogue_decomposition("d", full, removals, labels, mask, qmask)
    assert row["speaker_term"] == 2.0
    assert row["total_length"] == 3


def test_curriculum_sampler_only_selects_training_and_keeps_source_order(mod):
    keys = ["z", "a", "m"]
    chosen = mod.select_curriculum_training_ids(keys, {"z": 3.0, "a": 1.0, "m": 2.0}, 2 / 3)
    assert chosen == ["a", "m"]
    with pytest.raises(ValueError, match="non-Training"):
        mod.select_curriculum_training_ids(keys, {**{"z": 3.0, "a": 1.0, "m": 2.0}, "eval": 0.0}, 1.0)


def test_context_tensors_are_not_mutated(mod):
    full, removals, labels, mask, qmask = fixture_dialogue()
    before = qmask.clone()
    mod.compute_dialogue_decomposition("d", full, removals, labels, mask, qmask)
    assert torch.equal(qmask, before)


def test_artifact_hash_is_reproducible(mod, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"b": 2, "a": 1}, sort_keys=True) + "\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert mod.sha256_file(path) == expected == mod.sha256_file(path)


def test_manifest_hash_guard(mod, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}\n")
    good = mod.sha256_file(path)
    mod.verify_file_hash(path, good)
    with pytest.raises(ValueError, match="SHA-256"):
        mod.verify_file_hash(path, "0" * 64)


def test_original_tie_gradient_semantics(mod):
    student = torch.tensor([[0.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, -1.0]])
    loss = mod.original_rank_gap_sum(student, teacher, torch.tensor([0]))
    loss.backward()
    assert student.grad[0, 0].item() == -1.0


def test_complete_role_artifact_generator_uses_six_full_context_forwards(mod):
    class FakeModel(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(float(bias)))
            self.calls = 0

        def forward(self, text, qmask, umask, lengths, audio, visual, epoch):
            self.calls += 1
            length = lengths[0]
            logits = torch.stack((torch.ones(length) + self.bias, torch.zeros(length)), 1)
            return (torch.log_softmax(logits, 1), None, None, None, None, logits)

    models = [FakeModel(0.0) for _ in range(4)]
    record = {
        "text": [torch.ones(2, 3) for _ in range(4)], "audio": torch.ones(2, 2),
        "visual": torch.ones(2, 2), "qmask": torch.tensor([[1, 0], [0, 1]]),
        "labels": torch.tensor([0, -1]), "supervision_mask": torch.tensor([True, False]),
    }
    payload = {"role": "training", "dialogues": {"d": record}}
    result = mod.generate_from_role_artifact(models, payload, ["d"], torch.device("cpu"), "abc")
    assert set(result) == {"d"}
    assert sum(x.calls for x in models) == 6
    assert result["d"]["teacher_checkpoint_sha256"] == "abc"
