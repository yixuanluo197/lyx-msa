import unittest

import torch

from graph_index_cache import install_graph_cache


class FakeGraph:
    def __init__(self):
        self.calls = {'hyper': 0, 'gnn': 0}

    @staticmethod
    def features(a, v, l, lengths):
        offset = 0
        blocks = []
        for n in lengths:
            blocks.append(torch.cat([l[offset:offset+n], a[offset:offset+n], v[offset:offset+n]], 0))
            offset += n
        return torch.cat(blocks, 0)

    def create_hyper_index(self, a, v, l, dia_len, modals):
        self.calls['hyper'] += 1
        edges = torch.tensor([[0, 1], [1, 0]], device=a.device)
        batch = torch.arange(sum(dia_len), device=a.device)
        return edges, edges + 1, self.features(a, v, l, dia_len), batch, batch[:, None] % 2

    def create_gnn_index(self, a, v, l, dia_len, modals, mask_context=None, qmask=None):
        self.calls['gnn'] += 1
        edges = torch.tensor([[0, 1], [1, 0]], device=a.device)
        if mask_context is not None:
            edges = edges + 2
        return edges, self.features(a, v, l, dia_len)


class GraphCacheTests(unittest.TestCase):
    def setUp(self):
        self.graph = FakeGraph()
        self.a = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        self.v, self.l = self.a + 10, self.a + 20
        self.cache = install_graph_cache(self.graph)

    def test_cached_indices_but_fresh_features_no_input_mutation(self):
        copies = [x.clone() for x in (self.a, self.v, self.l)]
        first = self.graph.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')
        first_gnn = self.graph.create_gnn_index(self.a, self.v, self.l, [1, 2], 'avl')
        changed = self.a + 100
        second = self.graph.create_hyper_index(changed, self.v, self.l, [1, 2], 'avl')
        second_gnn = self.graph.create_gnn_index(changed, self.v, self.l, [1, 2], 'avl')
        self.assertEqual(self.graph.calls, {'hyper': 1, 'gnn': 1})
        for idx in (0, 1, 3, 4):
            self.assertTrue(torch.equal(first[idx], second[idx]))
        self.assertTrue(torch.equal(first_gnn[0], second_gnn[0]))
        self.assertFalse(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(second[2], FakeGraph.features(changed, self.v, self.l, [1, 2])))
        self.assertTrue(torch.equal(second_gnn[1], second[2]))
        for x, old in zip((self.a, self.v, self.l), copies):
            self.assertTrue(torch.equal(x, old))
        self.assertEqual(self.cache.stats['hyper_hits'], 1)
        self.assertEqual(self.cache.stats['gnn_hits'], 1)
        self.assertEqual(self.cache.stats['feature_exact_checks'], 2)

    def test_length_and_modality_key_isolation(self):
        for lengths, modals in [([1, 2], 'avl'), ([2, 1], 'avl'), ([1, 2], 'lav')]:
            self.graph.create_hyper_index(self.a, self.v, self.l, lengths, modals)
            self.graph.create_gnn_index(self.a, self.v, self.l, lengths, modals)
        self.assertEqual(self.graph.calls, {'hyper': 3, 'gnn': 3})
        self.assertEqual(self.cache.stats['hyper_misses'], 3)
        self.assertEqual(self.cache.stats['gnn_misses'], 3)

    def test_nondefault_context_uses_original_and_qmask_not_cached(self):
        full = self.graph.create_gnn_index(self.a, self.v, self.l, [1, 2], 'avl')[0]
        for _ in range(2):
            masked = self.graph.create_gnn_index(self.a, self.v, self.l, [1, 2], 'avl', 'intra', torch.ones(3, 2))[0]
            self.assertFalse(torch.equal(full, masked))
        self.assertEqual(self.graph.calls['gnn'], 3)
        self.assertEqual(self.cache.stats['gnn_bypasses'], 2)

    def test_restore_enable_and_instance_only(self):
        self.graph.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')
        self.cache.restore()
        self.graph.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')
        self.assertEqual(self.graph.calls['hyper'], 2)
        self.cache.enable()
        self.graph.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')
        self.assertEqual(self.graph.calls['hyper'], 2)
        other = FakeGraph()
        other.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')
        other.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')
        self.assertEqual(other.calls['hyper'], 2)

    def test_first_call_feature_mismatch_rejected(self):
        class IncorrectGraph(FakeGraph):
            @staticmethod
            def features(a, v, l, lengths):
                return FakeGraph.features(a, v, l, lengths) + 1
        graph = IncorrectGraph()
        install_graph_cache(graph)
        with self.assertRaises(AssertionError):
            graph.create_hyper_index(self.a, self.v, self.l, [1, 2], 'avl')


if __name__ == '__main__':
    unittest.main()
