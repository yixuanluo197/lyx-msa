"""Synthetic CPU checks only: no dataset, checkpoint, or training run."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_cm import Model, Model_Intra, Model_Inter

torch.set_num_threads(1)


def make_model(kind=Model, base="GRU", dataset="IEMOCAP", residue=False,
               causal=True, av_encoder=True, dropout=0.0):
    return kind(
        base, 1024, 8, 4, 4, 4, 4, 8, n_speakers=3,
        max_seq_len=20, window_past=10, window_future=10, n_classes=4,
        dropout=dropout, no_cuda=True, graph_type="hyper", use_residue=residue,
        D_m_v=5, D_m_a=6, modals="avl", att_type="concat_DHT",
        av_using_lstm=av_encoder, dataset=dataset, use_modal=True,
        norm="BN", num_L=2, num_K=2, causal=causal,
    )


def inputs(lengths=(4, 2), requires_grad=False):
    size, batch = max(lengths), len(lengths)
    text = [torch.randn(size, batch, 1024, requires_grad=requires_grad)
            for _ in range(4)]
    audio = torch.randn(size, batch, 6, requires_grad=requires_grad)
    visual = torch.randn(size, batch, 5, requires_grad=requires_grad)
    speakers = torch.zeros(size, batch, 3)
    valid = torch.zeros(batch, size)
    for b, length in enumerate(lengths):
        speakers[torch.arange(length), b, torch.arange(length) % 3] = 1
        valid[b, :length] = 1
    return text, speakers, valid, list(lengths), audio, visual


def forward(model, data):
    return model(*data, epoch=0)[0]


@pytest.mark.parametrize("kind", [Model, Model_Intra, Model_Inter])
@pytest.mark.parametrize("base", ["GRU", "LSTM", "Transformer"])
@pytest.mark.parametrize("dataset", ["IEMOCAP", "MELD"])
@pytest.mark.parametrize("av_encoder", [False, True])
def test_future_inputs_and_speakers_cannot_change_prefix(kind, base, dataset, av_encoder):
    torch.manual_seed(32)
    model = make_model(kind, base, dataset, av_encoder=av_encoder).eval()
    data = inputs((4,))
    reference = forward(model, data)
    for field in ["text", "audio", "visual", "speaker", "joint"]:
        text, speakers, valid, lengths, audio, visual = data
        text = [x.clone() for x in text]
        audio, visual, speakers = audio.clone(), visual.clone(), speakers.clone()
        if field in ("text", "joint"):
            for x in text:
                x[2:] = 10 * torch.randn_like(x[2:])
        if field in ("audio", "joint"):
            audio[2:] = 10 * torch.randn_like(audio[2:])
        if field in ("visual", "joint"):
            visual[2:] = 10 * torch.randn_like(visual[2:])
        if field in ("speaker", "joint"):
            speakers[2:] = speakers[2:].roll(1, dims=-1)
        changed = forward(model, (text, speakers, valid, lengths, audio, visual))
        torch.testing.assert_close(changed[:2], reference[:2], atol=2e-5, rtol=1e-5)
        assert torch.equal(changed[:2].argmax(-1), reference[:2].argmax(-1))
        assert not torch.allclose(changed[2:], reference[2:])


@pytest.mark.parametrize("base", ["GRU", "LSTM", "Transformer"])
@pytest.mark.parametrize("residue", [False, True])
def test_prefix_truncation_padding_and_batch_independence(base, residue):
    torch.manual_seed(9)
    model = make_model(base=base, residue=residue).eval()
    data = inputs()
    reference = forward(model, data)
    text, speakers, valid, lengths, audio, visual = data
    for b, length in enumerate(lengths):
        for cut in range(1, length + 1):
            prefix = ([x[:cut, b:b+1] for x in text], speakers[:cut, b:b+1],
                      valid[b:b+1, :cut], [cut], audio[:cut, b:b+1],
                      visual[:cut, b:b+1])
            offset = sum(lengths[:b])
            torch.testing.assert_close(forward(model, prefix), reference[offset:offset+cut],
                                       atol=2e-5, rtol=1e-5)
    # Padding values must not enter another valid conversation.
    for x in text + [audio, visual, speakers]:
        x[2:, 1] = 100
    torch.testing.assert_close(forward(model, data), reference, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", [Model, Model_Intra, Model_Inter])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("base", ["GRU", "LSTM", "Transformer"])
def test_future_gradients_are_zero_in_both_modes(kind, training, base):
    torch.manual_seed(17)
    model = make_model(kind, base=base).train(training)
    data = inputs((4,), requires_grad=True)
    output = forward(model, data)
    output[:2, 0].sum().backward()
    for feature in data[0] + [data[4], data[5]]:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad[2:]) == 0
        assert torch.count_nonzero(feature.grad[:2]) > 0
    assert model.norm_strategy == "LN"


def test_original_encoder_is_a_future_leakage_negative_control():
    torch.manual_seed(1)
    model = make_model(causal=False, av_encoder=False).eval()
    assert model.gru_l.bidirectional
    assert model.norm_strategy == "BN"
    original = torch.randn(4, 1, 8)
    changed = original.clone()
    changed[2:] += 10
    assert not torch.allclose(model.gru_l(original)[0][:2], model.gru_l(changed)[0][:2])


def test_causal_configuration_rejects_unimplemented_model_paths():
    with pytest.raises(ValueError, match="causal"):
        Model("GRU", 1024, 8, 4, 4, 4, 4, 8, 3, 20, 10, 10,
              graph_type="hyper", modals="al", att_type="concat_DHT", causal=True)
