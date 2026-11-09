"""Bounded Teacher regressions; synthetic batches and CPU construction only."""
import copy
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_training_role_artifact as builder
import teacher_runner as teacher


def record(key, labels=(0, 1, -100)):
    labels = torch.tensor(labels)
    n = len(labels)
    return dict(dialogue_id=key, text=[torch.randn(n, 1024) for _ in range(4)],
                audio=torch.randn(n, 1582), visual=torch.randn(n, 342),
                qmask=torch.tensor([[1., 0.]] * n), umask=torch.ones(n),
                labels=labels, supervision_mask=labels != -100)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1024, 6)
        self.context_seen = []

    def forward(self, text, qmask, umask, lengths, audio, visual, epoch):
        self.context_seen.append((lengths, int(umask.sum())))
        x = torch.cat([text[0][:length, index] for index, length in enumerate(lengths)])
        # Every context node, including unknowns, contributes to valid predictions.
        scores = self.projection(x + x.mean(0))
        return (torch.log_softmax(scores, dim=-1),)


def test_all_true_mask_equals_frozen_unweighted_nll():
    torch.manual_seed(2)
    x = torch.log_softmax(torch.randn(4, 6, requires_grad=True), -1)
    labels = torch.tensor([0, 1, 3, 4])
    actual = teacher.masked_classification_loss(x, labels, torch.ones(4, dtype=torch.bool), torch.nn.NLLLoss())
    assert torch.equal(actual, torch.nn.NLLLoss()(x, labels))


def test_mixed_mask_filters_before_criterion_and_preserves_gradient():
    scores = torch.randn(3, 6, requires_grad=True)
    labels = torch.tensor([1, -100, 4])
    mask = labels != -100
    def guarded(values, targets):
        assert targets.tolist() == [1, 4]
        return torch.nn.functional.cross_entropy(values, targets)
    loss = teacher.masked_classification_loss(scores, labels, mask, guarded)
    loss.backward()
    assert torch.count_nonzero(scores.grad[1]) == 0
    assert scores.grad[0].abs().sum() > 0


def test_unknown_context_retained_and_real_optimizer_updates():
    torch.manual_seed(11)
    model = TinyModel()
    before = copy.deepcopy(model.state_dict())
    batch = teacher.collate_training([record("train-a"), record("train-b", (2, 3))])
    result = teacher.train_epoch(model, [batch], torch.optim.Adam(model.parameters(), lr=.01),
                                 torch.nn.NLLLoss(), ["train-a", "train-b"], torch.device("cpu"), 0)
    assert model.context_seen == [([3, 2], 5)]
    assert result["supervised_utterances"] == 4
    assert result["exact_training_coverage"]
    assert not torch.equal(before["projection.weight"], model.projection.weight)


def test_empty_supervision_batch_skips_step_and_backward():
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters())
    before = copy.deepcopy(model.state_dict())
    batch = teacher.collate_training([record("train-a", (-100, -100))])
    result = teacher.train_epoch(model, [batch], optimizer, torch.nn.NLLLoss(),
                                 ["train-a"], torch.device("cpu"), 0)
    assert result["skipped_empty_supervision_batches"] == 1
    assert not optimizer.state
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize("batches", [ [["evaluation"]], [["a", "a"]], [["a"], ["a"]], [["a"]] ])
def test_tripwire_rejects_illegal_duplicate_and_incomplete_epochs(batches):
    guard = teacher.EpochTripwire(["a", "b"])
    with pytest.raises(ValueError):
        for ids in batches:
            guard.batch(ids)
        guard.finish()


def test_tripwire_checked_before_any_model_forward():
    model = TinyModel()
    batch = teacher.collate_training([record("calibration")])
    with pytest.raises(ValueError, match="non-Training"):
        teacher.train_epoch(model, [batch], torch.optim.Adam(model.parameters()),
                            torch.nn.NLLLoss(), ["training"], torch.device("cpu"), 0)
    assert model.context_seen == []


def test_meld_canonical_id_offsets_preserve_source_identity():
    assert builder.meld_source_id("MELD_train_0001") == 1
    assert builder.meld_source_id("MELD_validation_0001") == 1040
    assert builder.meld_source_id("MELD_test_0001") == 1154
    with pytest.raises(ValueError):
        builder.meld_source_id("MELD_unknown_0001")


def test_approved_manifest_integrity_and_training_role():
    import json
    experiment = Path(__file__).resolve().parents[1]
    manifest = json.loads((experiment / "configs/approved_split_manifests.reference.json").read_text())
    assert builder.approved_training(manifest, "IEMOCAP", "fold1")["unknown_masked_count"] == 23
    manifest["status"] = "candidate"
    with pytest.raises(ValueError, match="not approved"):
        builder.approved_training(manifest, "IEMOCAP", "fold1")


def test_builder_retains_unknown_context_and_excludes_heldout_records(tmp_path):
    import json
    import pickle
    train = record("train", (0, 0, 1, 2, 3, 4, 5))
    role = dict(dialogue_ids=["train"], group_ids=["Ses01"], dialogue_count=1,
                utterance_count=7, supervision_eligible_count=6, unknown_masked_count=1,
                class_support_supervision_eligible={name: 1 for name in teacher.CLASS_NAMES["IEMOCAP"]})
    role["sha256"] = builder.canonical_sha256(role)
    manifest = dict(status="approved", protocols={"IEMOCAP": {"folds": {"fold1": {"roles": {"training": role}}}}},
                    unknown_label_mask={"utterances": [{"dialogue_id": "train", "utterance_index": 1,
                                                       "utterance_id": "train-1"}]})
    manifest["content_sha256"] = builder.canonical_sha256(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    # Held-out values deliberately cannot be converted to numeric features.
    data = [{"train": [f"train-{i}" for i in range(7)], "evaluation": ["heldout"]},
            {"train": ["M"]*7, "evaluation": ["F"]},
            {"train": [0, 0, 1, 2, 3, 4, 5], "evaluation": [999999]}, None,
            {"train": train["audio"].numpy(), "evaluation": "never-index"},
            {"train": train["visual"].numpy(), "evaluation": "never-index"}]
    text = [{"train": x.numpy(), "evaluation": "never-index"} for x in train["text"]]
    with (tmp_path / "IEMOCAP_features.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    with (tmp_path / "iemocap_features_roberta.pkl").open("wb") as handle:
        pickle.dump([None, None] + text, handle)
    payload = builder.build_artifact(path, tmp_path, "IEMOCAP", "fold1")
    assert set(payload["dialogues"]) == {"train"}
    saved = payload["dialogues"]["train"]
    assert saved["labels"].tolist() == [0, -100, 1, 2, 3, 4, 5]
    assert torch.equal(saved["text"][0], train["text"][0])
    assert saved["umask"].sum() == 7
    assert teacher.validate_artifact(payload, manifest)["unknown"] == 1
    payload["dialogues"]["evaluation"] = saved
    with pytest.raises(ValueError, match="exactly cover"):
        teacher.validate_artifact(payload, manifest)


def test_frozen_real_model_construction_and_meld_focal_mask_parity():
    source = Path(__file__).resolve().parents[1] / "source_snapshot"
    module = teacher.load_frozen_module(source)
    torch.manual_seed(1475)
    model = teacher.build_model(module, "IEMOCAP", teacher.CONFIGS["IEMOCAP"], torch.device("cpu"))
    assert type(model).__name__ == "Model"
    assert model.causal is False
    assert len(model.state_dict()) == 85
    scores = torch.log_softmax(torch.randn(3, 7, requires_grad=True), -1)
    labels = torch.tensor([1, -100, 4])
    mask = labels != -100
    criterion = module.FocalLoss()
    assert torch.equal(teacher.masked_classification_loss(scores, labels, mask, criterion),
                       criterion(scores[mask], labels[mask]))
    teacher.masked_classification_loss(scores, labels, mask, criterion).backward()


def test_fixed_preregistered_endpoints_and_architecture():
    assert (teacher.CONFIGS["IEMOCAP"]["epochs"], teacher.CONFIGS["IEMOCAP"]["seed"]) == (80, 1475)
    assert (teacher.CONFIGS["MELD"]["epochs"], teacher.CONFIGS["MELD"]["seed"]) == (40, 67137)
    assert teacher.architecture("IEMOCAP", teacher.CONFIGS["IEMOCAP"])["causal"] is False


def test_protocol_config_validation_rejects_changed_endpoint():
    import json
    path = Path(__file__).resolve().parents[1] / "configs/teacher_protocol.json"
    protocol = json.loads(path.read_text())
    for values in protocol["datasets"].values():
        values.update(teacher.BATCH_CONFIG)
    assert teacher.validated_config(protocol, "MELD")["epochs"] == 40
    protocol["datasets"]["MELD"]["epochs"] = 15
    with pytest.raises(ValueError, match="preregistered MELD/epochs"):
        teacher.validated_config(protocol, "MELD")


def test_checkpoint_compatible_with_generator_strict_load_teacher(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilot_difficulty_contract", SCRIPTS / "generate_fold_local_difficulty.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    paths = [tmp_path / name for name in ("training.pt", "teacher_protocol.json", "source_manifest.json")]
    for index, path in enumerate(paths):
        path.write_text(str(index))
    config = teacher.CONFIGS["MELD"]
    compatibility = teacher.checkpoint_metadata(*paths, config)
    expected = {"training_role_artifact_sha256": builder.file_sha256(paths[0]),
                "teacher_config_sha256": builder.file_sha256(paths[1]),
                "source_manifest_sha256": builder.file_sha256(paths[2]),
                "seed": 67137, "access_roles": ["training"], "selection": "fixed_final_epoch"}
    assert compatibility == expected
    architecture = teacher.architecture("MELD", config)
    assert architecture == generator.expected_teacher_architecture("MELD", config)
    model, target = TinyModel(), TinyModel()
    checkpoint_path = tmp_path / "teacher_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "architecture": architecture,
                "epoch": 40, **compatibility}, checkpoint_path)
    loaded = generator.strict_load_teacher(target, checkpoint_path, architecture, 40)
    assert all(loaded[key] == value for key, value in expected.items())
    assert all(torch.equal(value, target.state_dict()[key]) for key, value in model.state_dict().items())


def test_access_audit_records_actual_union_and_epoch_coverage():
    coverage = [{"epoch": 1, "exact_training_coverage": True, "accessed_ids": ["a", "b"]}]
    audit = teacher.access_audit(["b", "a"], {"a", "b"}, "manifest-hash", coverage)
    assert audit["requested_ids"] == ["b", "a"]
    assert audit["accessed_ids"] == ["a", "b"]
    assert audit["dataset_role"] == "training"
    assert audit["rejected_ids"] == []
    assert audit["exact_final_coverage"] is True
    assert audit["per_epoch_exact_coverage"] == coverage


def test_gpu_identity_uses_actual_process_uuid(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="GPU-name", total_memory=42))
    monkeypatch.setattr(teacher.subprocess, "check_output", lambda *a, **kw: f"{teacher.os.getpid()}, GPU-actual-uuid\n9999999, GPU-other\n")
    assert teacher.gpu_identity(torch.device("cuda:0")) == {
        "gpu_uuid": "GPU-actual-uuid", "gpu_name": "GPU-name", "gpu_total_memory_bytes": 42}


def frozen_capacity_sampler():
    """Compile only the frozen sampler AST, avoiding train_cm's data/GPU imports."""
    import ast
    path = Path(__file__).resolve().parents[1] / "source_snapshot/train_cm.py"
    source = ast.parse(path.read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "CapacitySafeBatchSampler")
    namespace = {"torch": torch, "Sampler": torch.utils.data.Sampler}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["CapacitySafeBatchSampler"]


@pytest.mark.parametrize("seed", range(10))
def test_capacity_sampler_exact_frozen_math_and_rng(seed):
    lengths = [30, 30, 80, 80]
    actual_generator = torch.Generator().manual_seed(seed)
    frozen_generator = torch.Generator().manual_seed(seed)
    actual = teacher.CapacitySafeBatchSampler(lengths, 2, actual_generator, 140, 500, 10000)
    frozen = frozen_capacity_sampler()(lengths, 2, frozen_generator, 140, 500, 10000)
    before = torch.get_rng_state().clone()
    actual_batches, frozen_batches = list(actual), list(frozen)
    assert actual_batches == frozen_batches
    assert (actual.retries, actual.max_batch_edges, actual.max_batch_nodes) == (
        frozen.retries, frozen.max_batch_edges, frozen.max_batch_nodes)
    assert torch.equal(actual_generator.get_state(), frozen_generator.get_state())
    assert torch.equal(before, torch.get_rng_state())
    assert sorted(index for batch in actual_batches for index in batch) == list(range(4))
    assert actual.max_batch_edges <= 140
    assert actual.max_batch_nodes <= 500


def test_training_loader_matches_xp04_base_seed_draw_order():
    ids = [f"training-{i}" for i in range(7)]
    payload = {"dialogue_ids": ids, "dialogues": {key: record(key, (0, 1)) for key in ids}}
    config = dict(teacher.CONFIGS["IEMOCAP"], **teacher.BATCH_CONFIG)
    for epoch in (0, 1, 11):
        loader = teacher.training_loader(payload, config, epoch)
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        frozen = frozen_capacity_sampler()([2]*7, config["batch_size"], generator)
        reference = torch.utils.data.DataLoader(list(range(7)), batch_sampler=frozen, generator=generator, num_workers=0)
        before = torch.get_rng_state().clone()
        actual = [batch["ids"] for batch in loader]
        expected = [[ids[index] for index in batch.tolist()] for batch in reference]
        assert actual == expected
        assert torch.equal(before, torch.get_rng_state())


def test_sampler_uses_context_lengths_without_accessing_labels():
    payload = {"dialogue_ids": ["a", "b"], "dialogues": {
        "a": {"umask": torch.ones(3)}, "b": {"umask": torch.ones(5)}}}
    loader = teacher.training_loader(payload, dict(teacher.CONFIGS["IEMOCAP"], **teacher.BATCH_CONFIG), 0)
    assert loader.batch_sampler.lengths == [3, 5]
    assert sorted(index for batch in loader.batch_sampler for index in batch) == [0, 1]


def test_epoch_reports_sampler_capacity_and_preforward_batches():
    ids = ["train-a", "train-b"]
    payload = {"dialogue_ids": ids, "dialogues": {key: record(key) for key in ids}}
    loader = teacher.training_loader(payload, dict(teacher.CONFIGS["IEMOCAP"], **teacher.BATCH_CONFIG), 0)
    model, observed = TinyModel(), []
    def observe(row):
        assert model.context_seen == []
        observed.append(row)
    result = teacher.train_epoch(model, loader, torch.optim.Adam(model.parameters()), torch.nn.NLLLoss(),
                                 ids, torch.device("cpu"), 0, batch_observer=observe)
    assert result["sampler_retries"] == loader.batch_sampler.retries == 0
    assert result["max_batch_edges"] == loader.batch_sampler.max_batch_edges == 12
    assert result["max_batch_nodes"] == loader.batch_sampler.max_batch_nodes == 18
    assert set(observed[0]["dialogue_ids"]) == set(ids)


def test_preforward_batch_logging_and_capacity_tripwire():
    model = TinyModel()
    batch = teacher.collate_training([record("train-a")])
    batch["lengths"] = [1000]
    observed = []
    with pytest.raises(ValueError, match="capacity before forward"):
        teacher.train_epoch(model, [batch], torch.optim.Adam(model.parameters()), torch.nn.NLLLoss(),
                            ["train-a"], torch.device("cpu"), 0, batch_observer=observed.append)
    assert model.context_seen == []
    assert observed[0]["dialogue_ids"] == ["train-a"]
    assert observed[0]["edges"] == 1003
    assert observed[0]["within_capacity"] is False


def test_capacity_sampler_impossible_batches_match_frozen_error():
    for cls in (teacher.CapacitySafeBatchSampler, frozen_capacity_sampler()):
        with pytest.raises(ValueError, match="capacity impossible"):
            cls([1000], 16, torch.Generator().manual_seed(1))
