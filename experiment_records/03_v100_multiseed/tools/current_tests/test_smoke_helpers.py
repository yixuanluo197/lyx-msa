import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_test import (  # noqa: E402
    build_checkpoint,
    dataset_config,
    outputs_numerically_equal,
    parameter_update_summary,
    select_device,
)


def test_dataset_config_matches_official_readme_commands():
    iemocap = dataset_config("iemocap")
    meld = dataset_config("MELD")

    assert iemocap == {
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
    }
    assert meld["dataset"] == "MELD"
    assert (meld["audio_dim"], meld["n_speakers"], meld["n_classes"]) == (300, 9, 7)
    assert (meld["dropout"], meld["num_L"], meld["num_K"], meld["seed"]) == (
        0.4,
        3,
        3,
        67137,
    )


def test_dataset_config_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="IEMOCAP or MELD"):
        dataset_config("not-a-dataset")


def test_select_device_accepts_cpu_and_rejects_unavailable_cuda(monkeypatch):
    assert select_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda")


def test_parameter_update_summary_requires_gradient_and_real_update():
    layer = torch.nn.Linear(2, 1, bias=False)
    before = {name: value.detach().clone() for name, value in layer.named_parameters()}
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss = layer(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    summary = parameter_update_summary(layer, before)

    assert summary["trainable_parameters"] == 1
    assert summary["parameters_with_finite_grad"] == 1
    assert summary["parameters_updated"] == 1
    assert summary["max_abs_update"] > 0


def test_build_checkpoint_contains_complete_resume_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint = build_checkpoint(model, optimizer, dataset_config("IEMOCAP"), step=1)

    assert set(checkpoint) == {
        "format_version",
        "dataset_config",
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    assert checkpoint["format_version"] == 1
    assert checkpoint["step"] == 1


def test_checkpoint_output_comparison_allows_only_float_roundoff():
    reference = torch.tensor([[-1.234567, -0.5]], dtype=torch.float32)
    gpu_repeat = reference + torch.tensor([[2.3841858e-7, 0.0]])
    changed_model = reference + torch.tensor([[1e-3, 0.0]])

    assert outputs_numerically_equal(reference, gpu_repeat)
    assert not outputs_numerically_equal(reference, changed_model)
