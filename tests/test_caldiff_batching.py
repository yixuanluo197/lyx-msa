import sys
import types
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# These tests exercise the pure batching/checkpoint helpers and should not
# require the optional graph stack to be importable.
stubbed_modules = ["dataloader", "model_cm", "clloss", "ipdb"]
previous_modules = {name: sys.modules.get(name) for name in stubbed_modules}
dataloader = types.ModuleType("dataloader")
dataloader.IEMOCAPDataset = object
dataloader.MELDDataset = object
sys.modules["dataloader"] = dataloader

model_cm = types.ModuleType("model_cm")
for name in (
    "MaskedNLLLoss",
    "LSTMModel",
    "GRUModel",
    "Model",
    "MaskedMSELoss",
    "FocalLoss",
    "Model_Inter",
    "Model_Intra",
):
    setattr(model_cm, name, object)
sys.modules["model_cm"] = model_cm

clloss = types.ModuleType("clloss")
clloss.CKSCL_context = lambda: None
clloss.CKSCL_tav = lambda: None
sys.modules["clloss"] = clloss

if "ipdb" not in sys.modules:
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))

try:
    from caldiff.cal_diff import compute_dialogue_difficulties, load_checkpoint
finally:
    for module_name in stubbed_modules:
        previous = previous_modules[module_name]
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def _scores(values, labels):
    scores = torch.zeros(len(labels), 2)
    scores[torch.arange(len(labels)), labels] = torch.tensor(values)
    return scores


def test_compute_dialogue_difficulties_maps_each_flat_slice_to_its_video():
    labels = torch.tensor([0, 1, 0])
    full = _scores([0.0, 0.0, 0.0], labels)
    removed = [_scores([1.0, 2.0, 4.0], labels)]
    removed.extend(torch.zeros_like(full) for _ in range(4))
    qmask = torch.tensor(
        [
            [[1, 0], [1, 0]],
            [[0, 1], [0, 0]],
        ]
    )

    result = compute_dialogue_difficulties(
        full,
        removed[:3],
        removed[3:],
        labels,
        lengths=[2, 1],
        qmask=qmask,
        video_ids=["dialogue-a", "dialogue-b"],
    )

    # The formula is (rank penalty + number of represented speakers) /
    # (dialogue length + number of represented speakers).
    assert result == {"dialogue-a": pytest.approx(1.25), "dialogue-b": pytest.approx(2.5)}


def test_compute_dialogue_difficulties_preserves_official_batch_one_formula():
    labels = torch.tensor([0, 1])
    full = _scores([0.0, 0.0], labels)
    removed = [_scores([1.0, 2.0], labels)]
    removed.extend(torch.zeros_like(full) for _ in range(4))
    qmask = torch.tensor([[[1, 0]], [[0, 1]]])

    result = compute_dialogue_difficulties(
        full,
        removed[:3],
        removed[3:],
        labels,
        lengths=[2],
        qmask=qmask,
        video_ids=["dialogue-a"],
    )

    assert result["dialogue-a"] == pytest.approx((3.0 + 2.0) / (2.0 + 2.0))


def test_compute_dialogue_difficulties_rejects_inconsistent_batch_metadata():
    labels = torch.tensor([0])
    full = torch.zeros(1, 2)

    with pytest.raises(ValueError, match="one video ID per dialogue"):
        compute_dialogue_difficulties(
            full,
            [full] * 3,
            [full] * 2,
            labels,
            lengths=[1, 0],
            qmask=torch.tensor([[[1, 0], [0, 0]]]),
            video_ids=["only-one-id"],
        )


def test_load_checkpoint_uses_explicit_map_location(monkeypatch):
    model = torch.nn.Linear(2, 2)
    state = model.state_dict()
    observed = {}

    def fake_load(path, map_location):
        observed.update(path=path, map_location=map_location)
        return state

    monkeypatch.setattr(torch, "load", fake_load)
    device = torch.device("cpu")
    load_checkpoint(model, "checkpoint.pkl", device)

    assert observed == {"path": "checkpoint.pkl", "map_location": device}
