"""Exercise the causal calibration/contrastive objective on synthetic CPU data."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clloss import CKSCL_context, CKSCL_tav
from dataloader import causal_mean_replacement
from model_cm import FocalLoss, Model, Model_Intra, Model_Inter
from train_cm import combine_training_losses
from test_causal_model import inputs, make_model


def test_all_causal_information_removals_and_losses_backward_on_cpu():
    torch.manual_seed(25)
    full = make_model().train()
    mm = make_model().eval()
    intra = make_model(Model_Intra).eval()
    inter = make_model(Model_Inter).eval()
    for model in (mm, intra, inter):
        model.load_state_dict(full.state_dict(), strict=True)
    data = inputs((4,))
    text, speaker, valid, lengths, audio, visual = data
    scores, *_, hidden = full(*data, epoch=0)
    with torch.no_grad():
        masked_text = [causal_mean_replacement(x, valid) for x in text]
        masked_audio = causal_mean_replacement(audio, valid)
        masked_visual = causal_mean_replacement(visual, valid)
        removed = [
            mm(masked_text, speaker, valid, lengths, audio, visual, 0)[0],
            mm(text, speaker, valid, lengths, masked_audio, visual, 0)[0],
            mm(text, speaker, valid, lengths, audio, masked_visual, 0)[0],
            intra(*data, epoch=0)[0], inter(*data, epoch=0)[0],
        ]
    labels = torch.tensor([0, 1, 2, 3])
    rows = torch.arange(4)
    cc_loss = sum(torch.relu(x[rows, labels] - scores[rows, labels]).sum() for x in removed)
    binary = torch.tensor([0, 0, 1, 1])
    weights = [2.0, 2.0]
    mscl = CKSCL_tav()(hidden, binary, binary.roll(1), binary.flip(0),
                        weights, weights, weights)
    cscl = CKSCL_context()(hidden, binary, binary.roll(1), weights, weights)
    classification = FocalLoss()(scores, labels)
    loss, components = combine_training_losses(
        classification, cc_loss, mscl, cscl, calibrate=True,
        contrastlearning=True, rank_coff=0.005, mscl_coff=0.05, cscl_coff=0.05)
    assert all(torch.isfinite(torch.tensor(x)) for x in components.values())
    loss.backward()
    for parameter in full.graph_model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert full.graph_model.fc1.weight.grad.abs().sum() > 0
    assert full.gru_l.weight_ih_l0.grad.abs().sum() > 0
    assert all(parameter.grad is None for model in (mm, intra, inter)
               for parameter in model.parameters())
