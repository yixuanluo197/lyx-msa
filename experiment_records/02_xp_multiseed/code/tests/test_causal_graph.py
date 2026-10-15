"""CPU synthetic checks; no dataset loading or model experiments."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_graph import CausalHypergraphConv, build_causal_indices
from model_hyper import HyperGCN


def make_graph(use_residue=True, num_L=3, num_K=3):
    torch.manual_seed(47)
    return HyperGCN(4, 4, 4, 4, 2, 5, 7, 0.0, 0.5, 0.1, False,
                    True, use_residue, use_modal=True, num_L=num_L,
                    num_K=num_K, causal=True).eval()


def make_inputs(lengths, requires_grad=False):
    torch.manual_seed(29)
    features = [torch.randn(sum(lengths), 4, requires_grad=requires_grad)
                for _ in range(3)]
    qmask = torch.zeros(max(lengths), len(lengths), 2)
    for batch, length in enumerate(lengths):
        for time in range(length):
            qmask[time, batch, time % 2] = 1
    return features, qmask


def test_prefix_equivalence_future_perturbation_and_no_input_mutation():
    for use_residue in (False, True):
        model = make_graph(use_residue=use_residue)
        inputs, qmask = make_inputs([5])
        originals = [x.clone() for x in inputs]
        full = model(*inputs, [5], qmask, 0)
        assert full.shape == (5, 3 * (4 + 10) if use_residue else 30)
        for prefix in (1, 2, 4):
            shortened = model(*(x[:prefix] for x in inputs), [prefix],
                              qmask[:prefix], 0)
            torch.testing.assert_close(full[:prefix], shortened, rtol=1e-5, atol=1e-6)
            changed = [x.clone() for x in inputs]
            for x in changed:
                x[prefix:] = torch.randn_like(x[prefix:]) * 100
            changed_qmask = qmask.clone()
            changed_qmask[prefix:] = qmask[prefix:].flip(-1)
            perturbed = model(*changed, [5], changed_qmask, 0)
            torch.testing.assert_close(full[:prefix], perturbed[:prefix],
                                       rtol=1e-5, atol=1e-6)
        for original, feature in zip(originals, inputs):
            assert torch.equal(original, feature)


def test_batched_single_and_reordered_dialogues_agree():
    model = make_graph()
    lengths = [3, 1, 4]
    inputs, qmask = make_inputs(lengths)
    for mask_context in (None, "intra", "inter"):
        batch = model(*inputs, lengths, qmask, 0, mask_context)
        offset = 0
        singles = []
        blocks = []
        for index, length in enumerate(lengths):
            features = [x[offset:offset + length] for x in inputs]
            speaker_mask = qmask[:length, index:index + 1]
            single = model(*features, [length], speaker_mask, 0, mask_context)
            singles.append(single)
            blocks.append(features)
            torch.testing.assert_close(batch[offset:offset + length], single,
                                       rtol=1e-5, atol=1e-6)
            offset += length
        permutation = [2, 0, 1]
        reordered_inputs = [torch.cat([blocks[i][modal] for i in permutation])
                            for modal in range(3)]
        reordered = model(*reordered_inputs, [lengths[i] for i in permutation],
                          qmask[:, permutation], 0, mask_context)
        torch.testing.assert_close(reordered,
                                   torch.cat([singles[i] for i in permutation]),
                                   rtol=1e-5, atol=1e-6)


def test_directed_edges_and_gnn_only_speaker_masks():
    speakers = torch.tensor([0, 0, 1, 1, 0])
    baseline = build_causal_indices([3, 2], speakers, torch.device("cpu"))
    node_time, node_dialogue, node_speaker, node_modality = {}, {}, {}, {}
    offset = utterance_offset = 0
    for dialogue, length in enumerate([3, 2]):
        for modality in range(3):
            for time in range(length):
                node = offset + modality * length + time
                node_time[node] = time
                node_dialogue[node] = dialogue
                node_speaker[node] = speakers[utterance_offset + time].item()
                node_modality[node] = modality
        offset += length * 3
        utterance_offset += length
    for mask in (None, "intra", "inter"):
        source, target, types, edges = build_causal_indices(
            [3, 2], speakers, torch.device("cpu"), mask)
        for actual, expected in zip((source, target, types), baseline[:3]):
            assert torch.equal(actual, expected)
        for src, dst in edges.T.tolist():
            assert node_dialogue[src] == node_dialogue[dst]
            assert node_time[src] <= node_time[dst]
            if node_modality[src] == node_modality[dst]:
                assert node_time[src] < node_time[dst]
                if mask == "intra":
                    assert node_speaker[src] != node_speaker[dst]
                if mask == "inter":
                    assert node_speaker[src] == node_speaker[dst]
            else:
                assert node_time[src] == node_time[dst]
        expected_crossmodal_count = sum([3, 2]) * 6
        assert sum(node_modality[src] != node_modality[dst]
                   for src, dst in edges.T.tolist()) == expected_crossmodal_count
        for edge, kind in enumerate(types.tolist()):
            src = source[0, source[1] == edge].tolist()
            dst = target[0, target[1] == edge].tolist()
            if kind == 0:
                assert len(dst) == 1
                assert [node_time[node] for node in src] == list(range(node_time[dst[0]] + 1))
            else:
                assert len(src) == len(dst) == 3
                assert len({node_time[node] for node in src + dst}) == 1


def test_prefix_hyperedge_never_writes_back_to_past():
    source, target, types, _ = build_causal_indices(
        [2], torch.zeros(2, dtype=torch.long), torch.device("cpu"))
    layer = CausalHypergraphConv(1)
    # l=[1,3], a=[5,7], v=[9,11]; incoming edges have equal initial weight.
    features = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]])
    output = layer(features, source, target, types)
    expected = torch.tensor([[3.0], [4.5], [5.0], [6.5], [7.0], [8.5]])
    torch.testing.assert_close(output, expected)


def test_future_gradients_zero_past_gradients_and_parameters_update():
    model = make_graph(use_residue=False)
    inputs, qmask = make_inputs([5], requires_grad=True)
    output = model(*inputs, [5], qmask, 0)
    output[2].square().sum().backward()
    for feature in inputs:
        assert torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad[3:]) == 0
        assert feature.grad[:3].abs().sum() > 0
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    for layer in model.causal_hyperconvs:
        assert layer.type_weight_logits.grad.abs().sum() > 0
    for layer in model.causal_convs:
        assert layer.gate.weight.grad.abs().sum() > 0
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert all(not torch.equal(old, parameter) for old, parameter
               in zip(before, trainable))


def test_no_fixed_capacity_or_position_parameters():
    model = make_graph(num_L=1, num_K=1)
    assert not hasattr(model, "hyperedge_weight")
    assert not hasattr(model, "EW_weight")
    assert model.causal_hyperconvs[0].type_weight_logits.shape == (2,)
    assert len(model.causal_hyperconvs) == len(model.causal_convs) == 1
