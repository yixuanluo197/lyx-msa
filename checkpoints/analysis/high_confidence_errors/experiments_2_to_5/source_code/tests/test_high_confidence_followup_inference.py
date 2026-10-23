import importlib.util
from pathlib import Path

import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_high_confidence_followups.py"
    assert path.is_file(), "The follow-up runner and removal audit are not implemented"
    spec = importlib.util.spec_from_file_location("followup_runner_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def features():
    return {"text": [torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + i
                     for i in range(4)],
            "audio": torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
            "visual": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)}


@pytest.mark.parametrize("removed", ["text", "audio", "visual"])
def test_audit_accepts_only_exact_temporal_mean_with_other_modalities_unchanged(removed):
    original = features()
    changed = dict(original)
    tensors = original[removed] if removed == "text" else [original[removed]]
    means = [x.mean(dim=0).repeat(x.shape[0], 1, 1) for x in tensors]
    changed[removed] = means if removed == "text" else means[0]
    audit = module().audit_replacement(original, changed, removed)
    assert audit["non_target_modalities_unchanged"]
    assert audit["mean_axis"] == 0
    assert audit["target_changed_elements"] > 0


def test_zero_replacement_is_rejected():
    original = features()
    changed = dict(original, audio=torch.zeros_like(original_audio := original["audio"]))
    assert original_audio.numel()
    with pytest.raises(ValueError, match="mean"):
        module().audit_replacement(original, changed, "audio")


def test_unintended_other_modality_change_is_rejected():
    original = features()
    audio = original["audio"].mean(dim=0).repeat(3, 1, 1)
    changed = dict(original, audio=audio, visual=original["visual"] + 1)
    with pytest.raises(ValueError, match="non-target"):
        module().audit_replacement(original, changed, "audio")


def test_padding_is_included_in_official_mean():
    original = features()
    original["audio"][2, 1] = 0
    changed = dict(original, audio=original["audio"].mean(dim=0).repeat(3, 1, 1))
    assert module().audit_replacement(original, changed, "audio")["padding_included_in_mean"]


def test_constant_features_may_be_unchanged_without_faking_an_effect():
    original = features()
    original["visual"].fill_(2)
    audit = module().audit_replacement(original, original, "visual")
    assert audit["target_changed_elements"] == 0
