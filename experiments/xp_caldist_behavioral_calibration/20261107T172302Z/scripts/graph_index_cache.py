"""Instance-local cache of XP04 graph structure, with fresh features every call.

This wraps only a supplied graph instance. Original methods remain accessible
through restore(), so callers can compare complete uncached/cached forwards.
No model parameter, global class or historical source is modified.
"""

import torch


class CachedGraphIndices:
    def __init__(self, graph):
        self.graph = graph
        self.original_hyper = graph.create_hyper_index
        self.original_gnn = graph.create_gnn_index
        self.hyper_indices = {}
        self.gnn_indices = {}
        self.stats = dict(hyper_hits=0, hyper_misses=0, gnn_hits=0,
                          gnn_misses=0, gnn_bypasses=0, feature_exact_checks=0)
        self.enable()

    @staticmethod
    def _key(a, v, l, dia_len, modals):
        lengths = tuple(int(n) for n in dia_len)
        if not lengths or any(n <= 0 for n in lengths):
            raise ValueError('Dialogue lengths must be positive')
        if any(x.ndim != 2 or x.shape[0] != sum(lengths) for x in (a, v, l)):
            raise ValueError('Features must be flattened valid utterances matching lengths')
        devices = tuple(str(x.device) for x in (a, v, l))
        return lengths, tuple(modals), devices

    @staticmethod
    def _features(a, v, l, lengths):
        offset = 0
        blocks = []
        for length in lengths:
            blocks.append(torch.cat([l[offset:offset + length],
                                     a[offset:offset + length],
                                     v[offset:offset + length]], dim=0))
            offset += length
        # The original builds the same dialogue-major L/A/V blocks using cat;
        # concatenation performs no floating-point arithmetic or normalization.
        return torch.cat(blocks, dim=0)

    def _hyper(self, a, v, l, dia_len, modals):
        key = self._key(a, v, l, dia_len, modals)
        features = self._features(a, v, l, key[0])
        if key not in self.hyper_indices:
            result = self.original_hyper(a, v, l, dia_len, modals)
            if not torch.equal(features, result[2]):
                raise AssertionError('Cached hypergraph feature reconstruction mismatch')
            self.stats['feature_exact_checks'] += 1
            self.hyper_indices[key] = (result[0], result[1], result[3], result[4])
            self.stats['hyper_misses'] += 1
        else:
            self.stats['hyper_hits'] += 1
        hyperedge_index, edge_index, batch, edge_type = self.hyper_indices[key]
        return hyperedge_index, edge_index, features, batch, edge_type

    def _gnn(self, a, v, l, dia_len, modals, mask_context=None, qmask=None):
        # Context removal depends on speaker masks and is deliberately uncached.
        if mask_context is not None:
            self.stats['gnn_bypasses'] += 1
            return self.original_gnn(a, v, l, dia_len, modals, mask_context, qmask)
        key = self._key(a, v, l, dia_len, modals)
        features = self._features(a, v, l, key[0])
        if key not in self.gnn_indices:
            edge_index, original_features = self.original_gnn(a, v, l, dia_len, modals, mask_context, qmask)
            if not torch.equal(features, original_features):
                raise AssertionError('Cached GNN feature reconstruction mismatch')
            self.stats['feature_exact_checks'] += 1
            self.gnn_indices[key] = edge_index
            self.stats['gnn_misses'] += 1
        else:
            self.stats['gnn_hits'] += 1
        return self.gnn_indices[key], features

    def restore(self):
        """Temporarily route this instance through its original bound methods."""
        self.graph.create_hyper_index = self.original_hyper
        self.graph.create_gnn_index = self.original_gnn
        return self

    def enable(self):
        """Route this instance through the cache; retain its existing entries."""
        self.graph.create_hyper_index = self._hyper
        self.graph.create_gnn_index = self._gnn
        return self


def install_graph_cache(graph):
    return CachedGraphIndices(graph)
