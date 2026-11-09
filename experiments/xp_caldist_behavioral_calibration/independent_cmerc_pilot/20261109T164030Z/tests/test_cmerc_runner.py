"""CPU gates for the Training/Development-only CMERC pilot runner."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import cmerc_runner as runner
from build_role_artifact import approved_role

ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "experiments/xp_caldist_behavioral_calibration/independent_training_pilot/20261109T142243Z/approved_split_manifests.json"
DIFFICULTY = ROOT / "experiments/xp_caldist_behavioral_calibration/curriculum_teacher_pilot/20261109T154754Z/results/IEMOCAP_fold1_difficulty.json"


def test_frozen_schedule_seed_and_class_weight_contract():
    iem, meld = [runner.config_for(name) for name in ("IEMOCAP", "MELD")]
    assert (iem["epochs"], meld["epochs"]) == (80, 40)
    assert iem["seed"] == meld["seed"] == 1475
    assert [runner.curriculum_ratio(iem, e) for e in range(7)] == [.15, .3, .45, .6, .75, .9, 1.]
    assert [runner.curriculum_ratio(meld, e) for e in range(3)] == [.4, .8, 1.]
    assert iem["loss"] == "fold_local_masked_NLL" and iem["class_weight"]
    weights = runner.mask_runner.training_class_weights(torch.tensor([0, -100, 1, 1]),
                                                       torch.tensor([True, False, True, True]), 2)
    assert torch.equal(weights, torch.tensor([3., 1.5]))


def test_real_difficulty_uses_dialogue_list_and_requires_exact_training():
    manifest = json.loads(MANIFEST.read_text())
    artifact = dict(dataset="IEMOCAP", fold="fold1", dialogue_ids=approved_role(manifest, "IEMOCAP", "fold1", "training")["dialogue_ids"])
    payload = json.loads(DIFFICULTY.read_text())
    values = runner.validate_difficulty(payload, artifact, runner.file_sha256(MANIFEST))
    assert len(values) == 84
    assert len(runner.mask_runner.curriculum_training_ids(artifact["dialogue_ids"], values, .15)) == 12
    for mutate in (lambda p: p["dialogues"].pop(),
                   lambda p: p["dialogues"].append(p["dialogues"][0]),
                   lambda p: p["dialogues"][0].update(final_difficulty=float("nan")),
                   lambda p: p.update(fold="fold2")):
        bad = copy.deepcopy(payload)
        mutate(bad)
        with pytest.raises(ValueError):
            runner.validate_difficulty(bad, artifact, runner.file_sha256(MANIFEST))


def test_only_allowed_roles_and_strict_rounded_earliest_selection():
    manifest = json.loads(MANIFEST.read_text())
    with pytest.raises(ValueError):
        approved_role(manifest, "IEMOCAP", "fold1", "evaluation")
    assert approved_role(manifest, "IEMOCAP", "fold1", "calibration")["dialogue_count"] == 30
    assert runner.update_selection(65.123, 1, None, None) == (65.12, 1, True)
    assert runner.update_selection(65.124, 2, 65.12, 1) == (65.12, 1, False)
    assert runner.update_selection(65.126, 3, 65.12, 1) == (65.13, 3, True)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 6)
        self.classifier = torch.nn.Linear(6, 3)
        self.calls = []
        self.refreshes = 0

    def load_state_dict(self, *args, **kwargs):
        self.refreshes += 1
        return super().load_state_dict(*args, **kwargs)

    def forward(self, text, qmask, umask, lengths, audio, visual, epoch):
        self.calls.append(dict(lengths=list(lengths), valid_context=int(umask.sum()), training=self.training))
        features = torch.cat([text[0][:n, i] for i, n in enumerate(lengths)])
        hidden = torch.tanh(self.projection(features))
        scores = F.log_softmax(self.classifier(hidden), 1)
        return scores, None, None, None, None, hidden


def toy_payload():
    records = {}
    for index, n in enumerate((4, 3)):
        key = "d" + str(index)
        valid = torch.ones(n, dtype=torch.bool)
        valid[1] = False
        labels = torch.arange(n) % 3
        labels[~valid] = -100
        records[key] = dict(dialogue_id=key, text=[torch.randn(n, 4) for _ in range(4)],
                            audio=torch.randn(n, 2), visual=torch.randn(n, 2),
                            qmask=torch.ones(n, 2), umask=torch.ones(n), labels=labels,
                            supervision_mask=valid)
    return dict(dialogue_ids=list(records), dialogues=records)


def test_six_views_keep_unknown_context_and_audit_all_eleven_losses():
    torch.manual_seed(19)
    payload = toy_payload()
    config = runner.config_for("IEMOCAP")
    config["batch_size"] = 2
    models = [ToyModel() for _ in range(4)]
    original = copy.deepcopy(models[0].state_dict())
    optimizer = torch.optim.Adam(models[0].parameters(), lr=.01)
    result = runner.train_epoch(models, runner.role_loader(payload, payload["dialogue_ids"], config, 0),
                                optimizer, torch.nn.NLLLoss(), payload["dialogue_ids"], torch.device("cpu"), 0, config)
    assert result["exact_curriculum_coverage"] and result["supervised_utterances"] == 5
    assert result["optimizer_steps"] == 1
    assert [len(model.calls) for model in models] == [1, 3, 1, 1]
    assert all(call["valid_context"] == 7 for model in models for call in model.calls)
    assert [model.refreshes for model in models] == [0, 1, 1, 1]
    assert models[0].training and all(not model.training for model in models[1:])
    assert any(not torch.equal(original[k], models[0].state_dict()[k]) for k in original)
    assert all(torch.equal(original[k], model.state_dict()[k]) for model in models[1:] for k in original)
    audit = result["direct_unknown_gradient_audit"]
    assert audit["unknown_nodes_retained"] == 2 and audit["supervised_path_count"] == 11
    assert all(row["excluded_direct_gradient_abs_max"] == 0 for row in audit["paths"].values())
    assert audit["paths"]["classification"]["eligible_gradient_abs_sum"] > 0


def test_development_calls_only_student_and_excludes_unknown():
    torch.manual_seed(21)
    payload, model = toy_payload(), ToyModel()
    result = runner.development_epoch(model, runner.role_loader(payload, payload["dialogue_ids"],
                                     runner.config_for("IEMOCAP"), 0), payload["dialogue_ids"], torch.device("cpu"), 0)
    assert result["count"] == 5 and len(model.calls) == 1
    assert result["exact_development_coverage"]
    assert all(0 <= result[k] <= 1 for k in ("accuracy", "weighted_f1", "macro_f1"))


def test_complete_rng_checkpoint_restore(tmp_path):
    torch.manual_seed(34)
    state = runner.teacher.rng_state()
    expected = torch.rand(7)
    runner.atomic_save(dict(rng_state=state), tmp_path / "latest.pt")
    runner.restore_rng(torch.load(tmp_path / "latest.pt", weights_only=False)["rng_state"])
    assert torch.equal(torch.rand(7), expected)


def test_calibration_rejects_unselected_or_incomplete_checkpoint_before_forward(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save(dict(complete=True, training_complete=True, selected_epoch=1, epoch=1), path)
    with pytest.raises(ValueError, match="selected best checkpoint"):
        runner.load_selected_model(path, "cpu")


def test_generic_artifact_extractor_preserves_unknown_context_for_each_allowed_role(tmp_path):
    import csv
    import pickle
    import numpy as np
    import build_role_artifact as builder

    root = tmp_path
    manifest_path = root / "experiments/project/pilot/stamp/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    data = root / "data"
    data.mkdir()
    keys = dict(training="Ses03_demo", development="Ses04_demo", calibration="Ses02_demo")
    source_ids = {key: [key + str(i) for i in range(7)] for key in keys.values()}
    labels = {key: [0, 1, 2, 3, 4, 5, 0] for key in keys.values()}
    speakers = {key: ["M"] * 7 for key in keys.values()}
    arrays = lambda width: {key: np.zeros((7, width), dtype=np.float32) for key in keys.values()}
    with (data / "IEMOCAP_features.pkl").open("wb") as handle:
        pickle.dump((source_ids, speakers, labels, None, arrays(1582), arrays(342)), handle)
    with (data / "iemocap_features_roberta.pkl").open("wb") as handle:
        pickle.dump((None, None, arrays(1024), arrays(1024), arrays(1024), arrays(1024)), handle)
    lineage = root / "lineage.csv"
    with lineage.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "dialogue_id", "session_id", "episode_id"])
        writer.writeheader()
        for key in keys.values():
            writer.writerow(dict(dataset="IEMOCAP", dialogue_id=key, session_id=key[:5], episode_id=""))
    roles = {}
    for role_name, key in keys.items():
        role = dict(dialogue_ids=[key], group_ids=[key[:5]], dialogue_count=1, utterance_count=7,
                    supervision_eligible_count=6, unknown_masked_count=1,
                    class_support_supervision_eligible={name: 1 for name in runner.CLASS_NAMES["IEMOCAP"]})
        role["sha256"] = builder.canonical_sha256(role)
        roles[role_name] = role
    manifest = dict(status="approved", protocols={"IEMOCAP": {"folds": {"fold1": {"roles": roles}}}},
                    source=dict(sample_lineage_path="lineage.csv", sample_lineage_sha256=builder.file_sha256(lineage)),
                    unknown_label_mask=dict(utterances=[dict(dialogue_id=key, utterance_index=6,
                                             utterance_id=source_ids[key][6]) for key in keys.values()]))
    manifest["content_sha256"] = builder.canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    for role_name, key in keys.items():
        payload = builder.build_artifact(manifest_path, data, "IEMOCAP", "fold1", role_name)
        report = runner.validate_artifact(payload, manifest, role_name, manifest_path)
        assert report["unknown"] == 1 and report["eligible"] == 6
        assert payload["dialogue_ids"] == [key]
        assert set(payload["dialogues"]) == {key}
        assert payload["dialogue_groups"] == {key: key[:5]}
        record = payload["dialogues"][key]
        assert len(record["text"][0]) == 7 and record["umask"].sum() == 7
        assert record["labels"][-1] == -100 and not record["supervision_mask"][-1]
        bad = copy.deepcopy(payload)
        bad["dialogue_groups"][key] = "bad_group"
        with pytest.raises(ValueError, match="group lineage"):
            runner.validate_artifact(bad, manifest, role_name, manifest_path)
