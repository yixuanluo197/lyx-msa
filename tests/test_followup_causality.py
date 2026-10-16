"""Auditable real-model causal regression; synthetic post-extraction features only.

The CLI in scripts/verify_causal_regression.py preserves numeric evidence. These
fixtures never replace Model/HyperGCN with mocks and never edit production code.
"""
import contextlib
import itertools
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_cm import Model, Model_Intra, Model_Inter
from model_hyper import HyperGCN
from causal_graph import CausalHighConv, CausalHypergraphConv, build_causal_indices
from dataloader import causal_mean_replacement

SEED = 20261016
ATOL, RTOL = 2e-5, 1e-5
KINDS = {x.__name__: x for x in (Model, Model_Intra, Model_Inter)}


def configurations():
    for kind, base, av, residue, training, dataset in itertools.product(
        KINDS, ('GRU', 'LSTM', 'Transformer'), (False, True), (False, True),
        (False, True), ('IEMOCAP', 'MELD')
    ):
        yield dict(kind=kind, base=base, av_encoder=av, residue=residue,
                   training=training, dataset=dataset, num_L=2, num_K=3)


def make_model(config, device='cpu'):
    torch.manual_seed(SEED)
    model = KINDS[config['kind']](
        config['base'], 1024, 8, 4, 4, 4, 4, 8, n_speakers=3,
        max_seq_len=20, window_past=10, window_future=10, n_classes=4,
        dropout=0., no_cuda=device == 'cpu', graph_type='hyper',
        use_residue=config['residue'], D_m_v=5, D_m_a=6, modals='avl',
        att_type='concat_DHT', av_using_lstm=config['av_encoder'],
        dataset=config['dataset'], use_modal=True, norm='LN2',
        num_L=config['num_L'], num_K=config['num_K'], causal=True)
    # Includes RNN internal dropout and attention dropout, not only nn.Dropout.
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.GRU, nn.LSTM, nn.MultiheadAttention)):
            if hasattr(module, 'p'):
                module.p = 0.
            if hasattr(module, 'dropout'):
                module.dropout = 0.
    return model.to(device).train(config['training'])


def inputs(lengths=(4, 2, 1), device='cpu'):
    generator = torch.Generator().manual_seed(SEED + 1)
    size, batch = max(lengths), len(lengths)
    features = [torch.randn(size, batch, dim, generator=generator).to(device)
                for dim in (1024, 1024, 1024, 1024, 6, 5)]
    speakers = torch.zeros(size, batch, 3, device=device)
    mask = torch.zeros(batch, size, device=device)
    for b, length in enumerate(lengths):
        for time in range(length):
            speakers[time, b, (time // 2 + b) % 3] = 1
        mask[b, :length] = 1
    return features[:4], speakers, mask, list(lengths), features[4], features[5]


def clone(data):
    return ([x.clone() for x in data[0]], data[1].clone(), data[2].clone(),
            list(data[3]), data[4].clone(), data[5].clone())


def select(data, order, lengths=None):
    lengths = [data[3][b] for b in order] if lengths is None else lengths
    n = max(lengths)
    mask = data[2][order, :n].clone()
    for b, length in enumerate(lengths):
        mask[b] = torch.arange(n, device=mask.device) < length
    return ([x[:n, order] for x in data[0]], data[1][:n, order], mask,
            list(lengths), data[4][:n, order], data[5][:n, order])


def predict(model, data):
    output = model(*data, epoch=0)
    return output[0], output[-1]


def compare(actual, expected):
    absolute = relative = 0.
    for a, b in zip(actual, expected):
        delta = (a.detach() - b.detach()).abs()
        absolute = max(absolute, delta.max().item())
        relative = max(relative, (delta / b.detach().abs().clamp_min(1e-8)).max().item())
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
    return dict(max_abs_diff=absolute, max_rel_diff=relative)


def check_config(config, device='cpu'):
    torch.set_num_threads(1)
    model = make_model(config, device)
    assert isinstance(model.graph_model, HyperGCN)
    data = inputs(device=device)
    metrics = {}
    executed_norms, transformer_masks = [], []
    hooks = []
    def norm_hook(module, _args, _out):
        executed_norms.append((type(module).__name__, list(module.normalized_shape)
                               if isinstance(module, nn.LayerNorm) else None))
    def attention_hook(module, _args, kwargs):
        future, padding = kwargs['src_mask'], kwargs['src_key_padding_mask']
        assert future.dtype == torch.bool and padding.dtype == torch.bool
        assert torch.equal(future, torch.ones_like(future).triu(1))
        assert torch.equal(padding, ~data[2].bool())
        transformer_masks.append(dict(causal=True, padding=True, shape=list(future.shape)))
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(norm_hook))
        if isinstance(module, nn.TransformerEncoderLayer):
            hooks.append(module.register_forward_pre_hook(attention_hook, with_kwargs=True))
    with torch.no_grad():
        reference = predict(model, data)
    for hook in hooks:
        hook.remove()
    assert executed_norms and all(kind == 'LayerNorm' and len(shape) == 1
                                  for kind, shape in executed_norms)
    if config['base'] == 'Transformer':
        assert len(transformer_masks) == (3 if config['av_encoder'] else 1)
    metrics['executed_norms'] = executed_norms
    metrics['executed_transformer_masks'] = transformer_masks
    with torch.no_grad():
        for field in ('text', 'audio', 'visual', 'speaker', 'joint'):
            changed = clone(data)
            for index, name in enumerate(('text', 'audio', 'visual')):
                if field in (name, 'joint'):
                    tensors = changed[0] if index == 0 else [changed[4 if index == 1 else 5]]
                    for tensor in tensors:
                        tensor[2:, 0] = tensor[2:, 0] * -7 + 13
            if field in ('speaker', 'joint'):
                changed[1][2:, 0] = changed[1][2:, 0].roll(1, dims=-1)
            metrics['future_' + field] = compare(tuple(x[:2] for x in predict(model, changed)),
                                                  tuple(x[:2] for x in reference))
        suffix_results = []
        singles = []
        for b, length in enumerate(data[3]):
            offset = sum(data[3][:b])
            singles.append(tuple(x[offset:offset + length] for x in reference))
            for cut in range(1, length + 1):
                suffix_results.append(compare(predict(model, select(data, [b], [cut])),
                                              tuple(x[:cut] for x in singles[-1])))
        metrics['all_prefixes_suffix_add_delete_single'] = suffix_results
        order = [2, 0, 1]
        metrics['batch_reordered'] = compare(predict(model, select(data, order)),
            tuple(torch.cat([singles[b][i] for b in order]) for i in (0, 1)))
        changed = clone(data)
        for tensor in changed[0] + [changed[4], changed[5]]:
            tensor[:, 1:] = tensor[:, 1:] * 17 + 23
        changed[1][:, 1:] = changed[1][:, 1:].roll(1, dims=-1)
        metrics['other_dialogues'] = compare(tuple(x[:4] for x in predict(model, changed)),
                                              tuple(x[:4] for x in reference))
        padded = clone(data)
        for tensor in padded[0] + [padded[1], padded[4], padded[5]]:
            for b, length in enumerate(data[3]):
                tensor[length:, b] = 37.
        metrics['finite_padding_values'] = compare(predict(model, padded), reference)
        extended = ([torch.cat([x, x.new_full((3, 3, x.size(-1)), -19.)]) for x in padded[0]],
                    torch.cat([padded[1], padded[1].new_full((3, 3, 3), 13.)]),
                    torch.cat([padded[2], padded[2].new_zeros(3, 3)], dim=1), padded[3],
                    torch.cat([padded[4], padded[4].new_full((3, 3, 6), -31.)]),
                    torch.cat([padded[5], padded[5].new_full((3, 3, 5), 41.)]))
        metrics['extra_right_padding'] = compare(predict(model, extended), reference)
    grad_data = select(data, [0])
    tensors = grad_data[0] + [grad_data[4], grad_data[5]]
    for tensor in tensors:
        tensor.requires_grad_()
    cuda_gradients = torch.device(device).type == 'cuda'
    # cuDNN eval RNN forward cannot be differentiated even though eval only
    # changes dropout behavior. Recompute this diagnostic forward with native
    # ATen RNN operators, keeping weights, mode, inputs, and tolerances fixed.
    gradient_context = (torch.backends.cudnn.flags(enabled=False)
                        if cuda_gradients else contextlib.nullcontext())
    with gradient_context:
        output, hidden = predict(model, grad_data)
        gradient_records = []
        for time, category in itertools.product((0, 1), (0, 3)):
            gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)
            future = max(x[time + 1:].abs().max().item() for x in gradients)
            current = [x[time].abs().max().item() for x in gradients]
            history = [x[:time + 1].abs().max().item() for x in gradients]
            past = [x[:time].abs().max().item() for x in gradients] if time else None
            assert all(torch.isfinite(x).all() for x in gradients)
            assert future == 0. and all(x > 0. for x in current + history)
            assert past is None or all(x > 0. for x in past)
            gradient_records.append(dict(output='log_probability', time=time, category=category,
                future_max=future, current_max_by_input=current, past_only_max_by_input=past, history_current_max_by_input=history))
        time = 1
        feature = hidden[time].detach().abs().argmax().item()
        gradients = torch.autograd.grad(hidden[time, feature], tensors)
        future = max(x[time + 1:].abs().max().item() for x in gradients)
        assert future == 0. and any(x[:time + 1].abs().max().item() > 0 for x in gradients)
        gradient_records.append(dict(output='hidden', time=time, feature=feature, future_max=future,
            history_current_max_by_input=[x[:time + 1].abs().max().item() for x in gradients]))
    metrics['gradient_forward_vs_default_forward'] = compare(
        (output.detach(), hidden.detach()), singles[0])
    metrics['scalar_gradients'] = gradient_records
    metrics['backend'] = dict(default_forward_cudnn_enabled=bool(torch.backends.cudnn.enabled),
        default_forward='CUDA default / cuDNN eligible RNN' if cuda_gradients else 'CPU default',
        scalar_gradient='ATen RNN; cuDNN locally disabled' if cuda_gradients else 'CPU default',
        model_mode_unchanged=True)
    return dict(config=config, seed=SEED, device=device, dtype='float32',
                atol=ATOL, rtol=RTOL, status='passed', checks=metrics)


def check_graph_structure():
    lengths = [3, 1, 2]
    speaker = torch.tensor([0, 0, 1, 2, 1, 2])
    metadata = []
    for dialogue, length in enumerate(lengths):
        for modality in range(3):
            metadata.extend((dialogue, modality, time) for time in range(length))
    baseline = build_causal_indices(lengths, speaker, torch.device('cpu'))
    records = []
    for policy in (None, 'intra', 'inter'):
        source, target, types, edges = build_causal_indices(lengths, speaker, torch.device('cpu'), policy)
        for actual, expected in zip((source, target, types), baseline[:3]):
            assert torch.equal(actual, expected)
        for src, dst in edges.T.tolist():
            sd, sm, st = metadata[src]
            td, tm, tt = metadata[dst]
            assert sd == td and st <= tt
            assert (sm == tm and st < tt) or (sm != tm and st == tt)
            if sm == tm:
                offset = sum(lengths[:sd])
                same = speaker[offset + st] == speaker[offset + tt]
                assert policy != 'intra' or not same
                assert policy != 'inter' or same
        for edge, kind in enumerate(types.tolist()):
            sources = source[0, source[1] == edge].tolist()
            targets = target[0, target[1] == edge].tolist()
            if kind == 0:
                assert len(targets) == 1
                td, tm, tt = metadata[targets[0]]
                assert [metadata[x] for x in sources] == [(td, tm, t) for t in range(tt + 1)]
            else:
                assert len(sources) == len(targets) == 3
                assert {metadata[x][2] for x in sources + targets}.__len__() == 1
        records.append(dict(policy=policy, hyperedges=len(types), gnn_edges=edges.size(1), status='passed'))
    # With one modality-independent source and a changed number of future outgoing
    # edges, only target incoming degree is allowed to normalize old targets.
    layer = CausalHighConv(1)
    with torch.no_grad():
        layer.gate.weight.zero_()
        layer.gate.bias.fill_(1.)
    small = layer(torch.tensor([[2.], [3.]]), torch.tensor([[0], [1]]))
    large = layer(torch.tensor([[2.], [3.], [7.]]), torch.tensor([[0, 0, 1], [1, 2, 2]]))
    compare((small,), (large[:2],))
    # Explicit hyperedge reading [past,current] must deliver only to current.
    source = torch.tensor([[0, 0, 1], [0, 1, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    hyper = CausalHypergraphConv(1)
    result = hyper(torch.tensor([[2.], [10.]]), source, target, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(result, torch.tensor([[2.], [6.]]))
    return dict(status='passed', policies=records, incoming_degree_prefix_invariant=True,
                hyper_source_and_target_checked_separately=True)


def check_auxiliary():
    data = inputs((4, 2, 1))
    inverse_diffs = []
    mean_diffs = []
    for feature in data[0] + [data[4], data[5]]:
        mean = causal_mean_replacement(feature, data[2])
        for b, length in enumerate(data[3]):
            for t in range(length):
                mean_diffs.append((mean[t, b] - feature[:t+1, b].mean(0)).abs().max().item())
                previous = mean[t-1, b] if t else torch.zeros_like(mean[t, b])
                inverse = (t + 1) * mean[t, b] - t * previous
                inverse_diffs.append((inverse - feature[t, b]).abs().max().item())
        changed = feature.clone()
        changed[2:, 0] *= 11
        torch.testing.assert_close(causal_mean_replacement(changed, data[2])[:2, 0], mean[:2, 0])
    assert max(mean_diffs) < ATOL and max(inverse_diffs) < ATOL
    no_ops, retained = [], []
    for base, av, residue in itertools.product(('GRU', 'LSTM', 'Transformer'), (False, True), (False, True)):
        config = dict(kind='Model', base=base, av_encoder=av, residue=residue,
                      training=False, dataset='IEMOCAP', num_L=2, num_K=3)
        model = make_model(config)
        reference = predict(model, data)[0].exp()[0]
        for modality in ('text', 'audio', 'visual'):
            changed = clone(data)
            if modality == 'text':
                changed = ([causal_mean_replacement(x, data[2]) for x in data[0]], *changed[1:])
            else:
                index = 4 if modality == 'audio' else 5
                changed = list(changed)
                changed[index] = causal_mean_replacement(changed[index], data[2])
            delta = compare((predict(model, changed)[0].exp()[0],), (reference,))
            no_ops.append(dict(base=base, av_encoder=av, residue=residue, view=modality + '_prefix_mean_v1', **delta))
        for kind in ('Model_Intra', 'Model_Inter'):
            auxiliary = make_model(dict(config, kind=kind))
            auxiliary.load_state_dict(model.state_dict(), strict=True)
            delta = compare((predict(auxiliary, data)[0].exp()[0],), (reference,))
            no_ops.append(dict(base=base, av_encoder=av, residue=residue, view=kind, **delta))
            # Even with the relevant GNN temporal edges removed, past features
            # can still flow through both causal encoder and hypergraph branch.
            cut = select(data, [0])
            features = cut[0] + [cut[4], cut[5]]
            for feature in features:
                feature.requires_grad_()
            gradients = torch.autograd.grad(predict(auxiliary, cut)[0][2, 0], features)
            past = [gradient[:2].abs().max().item() for gradient in gradients]
            assert all(x > 0. for x in past)
            retained.append(dict(base=base, av_encoder=av, residue=residue, view=kind,
                                 historical_input_gradient_max=past))
    return dict(status='passed', replacement_policy='prefix_mean_v1',
                formula_max_abs_diff=max(mean_diffs), inverse_max_abs_diff=max(inverse_diffs),
                first_utterance_probability_noops=no_ops, retained_historical_information=retained,
                interpretation='Invertible prefix smoothing; not complete modality removal. Context views ablate GNN edges only.')


def negative_controls():
    torch.manual_seed(SEED)
    x = torch.tensor([[[1., 2., 4., 8.]], [[2., 5., 1., 3.]], [[4., 2., 7., 1.]], [[3., 6., 2., 5.]]])
    changed = x.clone()
    changed[2:] = changed[2:] * -9 + 11
    recurrent = nn.GRU(4, 3, bidirectional=True).eval()
    attention = nn.TransformerEncoderLayer(4, 2, dropout=0.).eval()
    def bad_hyper(value):
        # A prefix edge is correct on its source side but copied to every
        # historical source member, introducing future delivery to old nodes.
        output = torch.zeros_like(value)
        for time in range(len(value)):
            output[:time+1] += value[:time+1].mean(0)
        return output
    def bad_outdegree(value, edges):
        source, target = edges
        degree = value.new_zeros(len(value))
        degree.index_add_(0, source, value.new_ones(len(source)))
        output = torch.zeros_like(value)
        output.index_add_(0, target, value[source] / degree[source].clamp_min(1).reshape(-1, 1, 1))
        return output
    short_edges = torch.tensor([[0], [1]])
    long_edges = torch.tensor([[0, 0, 1, 0, 1, 2], [1, 2, 2, 3, 3, 3]])
    cases = {
        'bidirectional_encoder': (recurrent(x)[0][:2], recurrent(changed)[0][:2]),
        'cross_time_normalization': (nn.functional.layer_norm(x.transpose(0, 1), (4, 4))[:, :2],
                                     nn.functional.layer_norm(changed.transpose(0, 1), (4, 4))[:, :2]),
        'missing_causal_attention_mask': (attention(x)[:2], attention(changed)[:2]),
        'hyperedge_delivery_to_past': (bad_hyper(x)[:2], bad_hyper(changed)[:2]),
        'future_outdegree_normalization': (bad_outdegree(x[:2], short_edges), bad_outdegree(x, long_edges)[:2]),
        'whole_dialogue_mean_replacement': (x.mean(0), changed.mean(0)),
    }
    records = []
    for name, (reference, bad) in cases.items():
        difference = (reference - bad).abs().max().item()
        detected = not torch.allclose(reference, bad, atol=ATOL, rtol=RTOL)
        assert detected
        records.append(dict(fixture=name, detected=detected, max_abs_diff=difference,
                            intervention='suffix deletion' if 'outdegree' in name else 'future perturbation',
                            status='passed'))
    return records


@pytest.mark.parametrize('config', list(configurations()), ids=lambda x: '-'.join(str(v) for v in x.values()))
def test_full_model_causal_matrix(config):
    check_config(config)


def test_source_delivery_and_degree_are_causal():
    check_graph_structure()


def test_auxiliary_prefix_mean_and_first_utterance_noop():
    check_auxiliary()


def test_negative_control_sensitivity():
    negative_controls()


def check_cuda_eval_rnn_backend(base):
    """Demonstrate the backend limitation and the narrowly scoped workaround."""
    torch.manual_seed(SEED)
    recurrent = getattr(nn, base)(4, 3, num_layers=2, dropout=0.).cuda().eval()
    features = torch.randn(4, 1, 4, device='cuda', requires_grad=True)
    with torch.backends.cudnn.flags(enabled=True):
        with torch.autograd.profiler.profile(use_cuda=False) as trace:
            default = recurrent(features)[0]
        operators = sorted({event.name for event in trace.function_events
                            if 'cudnn_rnn' in event.name})
        assert operators, 'This fixture must actually execute a cuDNN RNN'
        try:
            torch.autograd.grad(default.square().sum(), features)
        except RuntimeError as error:
            limitation = str(error)
            assert 'cudnn RNN backward can only be called in training mode' in limitation
        else:
            raise AssertionError('Expected eval-mode cuDNN backward restriction was not observed')
    with torch.backends.cudnn.flags(enabled=False):
        native = recurrent(features)[0]
        gradients = torch.autograd.grad(native.square().sum(), features)[0]
    assert torch.isfinite(gradients).all() and gradients.abs().max().item() > 0.
    difference = compare((default.detach(),), (native.detach(),))
    assert not recurrent.training
    return dict(base=base, status='passed', device='cuda', default_forward_operators=operators,
                default_backward_error=limitation, scalar_gradient_backend='ATen; cuDNN disabled',
                native_gradient_max=gradients.abs().max().item(), **difference)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.backends.cudnn.is_available(),
                    reason='CUDA and cuDNN required to reproduce eval RNN backend limitation')
@pytest.mark.parametrize('base', ('GRU', 'LSTM'))
def test_cuda_eval_rnn_backward_backend_limit(base):
    check_cuda_eval_rnn_backend(base)
