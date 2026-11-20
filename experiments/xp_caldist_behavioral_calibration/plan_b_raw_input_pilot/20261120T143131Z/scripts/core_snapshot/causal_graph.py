"""Directed, prefix-invariant graph operators for CMERC's causal mode.

Nodes are dialogue-major, with language/audio/vision blocks inside each
dialogue. A temporal hyperedge reads a modality prefix and writes only to its
current node. Neither normalization uses a source's future outgoing degree.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def build_causal_indices(dia_len, speakers, device, mask_context=None):
    """Return source/target hyper-incidences, edge types and directed GNN edges.

    Type 0 is a same-modality temporal prefix; type 1 is a simultaneous
    cross-modal hyperedge. Speaker masking follows the original GNN-only
    ablation: ``intra`` removes same-speaker temporal edges, ``inter`` removes
    different-speaker temporal edges. Simultaneous cross-modal edges remain.
    """
    if mask_context not in (None, "intra", "inter"):
        raise ValueError("mask_context must be None, 'intra', or 'inter'")
    source_nodes, source_edges, target_nodes, target_edges = [], [], [], []
    edge_types, gnn_source, gnn_target = [], [], []
    node_offset = utterance_offset = 0
    speaker_ids = speakers.detach().tolist()
    for length in dia_len:
        length = int(length)
        for current in range(length):
            current_nodes = [node_offset + modality * length + current
                             for modality in range(3)]
            for modality, target in enumerate(current_nodes):
                edge = len(edge_types)
                edge_types.append(0)
                for past in range(current + 1):
                    source_nodes.append(node_offset + modality * length + past)
                    source_edges.append(edge)
                target_nodes.append(target)
                target_edges.append(edge)
                for past in range(current):
                    same_speaker = (speaker_ids[utterance_offset + past] ==
                                    speaker_ids[utterance_offset + current])
                    if ((mask_context == "intra" and same_speaker) or
                            (mask_context == "inter" and not same_speaker)):
                        continue
                    gnn_source.append(node_offset + modality * length + past)
                    gnn_target.append(target)
            edge = len(edge_types)
            edge_types.append(1)
            source_nodes.extend(current_nodes)
            source_edges.extend([edge] * 3)
            target_nodes.extend(current_nodes)
            target_edges.extend([edge] * 3)
            for source in current_nodes:
                for target in current_nodes:
                    if source != target:
                        gnn_source.append(source)
                        gnn_target.append(target)
        node_offset += 3 * length
        utterance_offset += length
    return (
        torch.tensor([source_nodes, source_edges], dtype=torch.long, device=device),
        torch.tensor([target_nodes, target_edges], dtype=torch.long, device=device),
        torch.tensor(edge_types, dtype=torch.long, device=device),
        torch.tensor([gnn_source, gnn_target], dtype=torch.long, device=device),
    )


class CausalHypergraphConv(nn.Module):
    """Mean sources into edges, then positive-weight mean into current targets."""

    def __init__(self, channels):
        super().__init__()
        # Shared by edge type, independent of time, dialogue index and capacity.
        self.type_weight_logits = nn.Parameter(
            torch.full((2,), math.log(math.expm1(1.0))))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x, source_incidence, target_incidence, edge_types):
        source, source_edge = source_incidence
        target, target_edge = target_incidence
        edge_count = edge_types.numel()
        edge_features = x.new_zeros(edge_count, x.size(1))
        edge_features.index_add_(0, source_edge, x[source])
        source_count = x.new_zeros(edge_count)
        source_count.index_add_(0, source_edge, x.new_ones(source.numel()))
        edge_features = edge_features / source_count.clamp_min(1).unsqueeze(1)
        edge_weights = F.softplus(self.type_weight_logits)[edge_types]
        target_weights = edge_weights[target_edge]
        output = x.new_zeros(x.shape)
        output.index_add_(0, target,
                          edge_features[target_edge] * target_weights.unsqueeze(1))
        incoming_weight = x.new_zeros(x.size(0))
        incoming_weight.index_add_(0, target, target_weights)
        output = output / incoming_weight.clamp_min(torch.finfo(x.dtype).tiny).unsqueeze(1)
        return F.leaky_relu(output + self.bias, negative_slope=0.01)


class CausalHighConv(nn.Module):
    """Original concat/tanh gate with normalization by target incoming degree."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Linear(2 * channels, 1)

    def forward(self, x, edge_index):
        source, target = edge_index
        gate = torch.tanh(self.gate(torch.cat([x[target], x[source]], dim=1)))
        output = x.new_zeros(x.shape)
        output.index_add_(0, target, x[source] * gate)
        incoming_degree = x.new_zeros(x.size(0))
        incoming_degree.index_add_(0, target, x.new_ones(target.numel()))
        return output / incoming_degree.clamp_min(1).unsqueeze(1)
