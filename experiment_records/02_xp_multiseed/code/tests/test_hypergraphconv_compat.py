import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HypergraphConv import HypergraphConv


def test_hypergraph_conv_returns_one_embedding_per_node():
    # The official M3Net path uses equal input/output widths for this layer.
    layer = HypergraphConv(in_channels=4, out_channels=4)
    nodes = torch.randn(3, 4, requires_grad=True)
    incidence = torch.tensor(
        [
            [0, 1, 1, 2],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )

    output = layer(nodes, incidence)

    assert output.shape == (3, 4)
    output.square().mean().backward()
    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()
