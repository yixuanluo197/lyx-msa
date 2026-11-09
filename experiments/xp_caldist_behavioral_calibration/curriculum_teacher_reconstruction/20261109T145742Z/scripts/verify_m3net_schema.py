#!/usr/bin/env python3
"""CPU-only, no-forward M3Net -> CMERC state-dict schema audit."""

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

import torch

# PyG 2.4 removed a symbol imported but never used by the audited model source.
# A process-local placeholder permits construction-only schema inspection without
# editing either repository; calling it is a hard error.
try:
    from torch_geometric.nn.pool import topk_pool
    if not hasattr(topk_pool, "topk"):
        def _unused_topk(*args, **kwargs):
            raise RuntimeError("construction-only compatibility stub was called")
        topk_pool.topk = _unused_topk
except ImportError:
    pass


def construct(model_class, dataset, num_k):
    audio = 1582 if dataset == "IEMOCAP" else 300
    graph = 512 if dataset == "IEMOCAP" else 1024
    speakers = 2 if dataset == "IEMOCAP" else 9
    classes = 6 if dataset == "IEMOCAP" else 7
    kwargs = dict(
        n_speakers=speakers, max_seq_len=200, window_past=10, window_future=10,
        n_classes=classes, listener_state=False, context_attention="general",
        dropout=0.5 if dataset == "IEMOCAP" else 0.4, nodal_attention=True,
        no_cuda=True, graph_type="hyper", use_topic=False, alpha=0.2, multiheads=6,
        graph_construct="direct", use_GCN=False, use_residue=False, D_m_v=342,
        D_m_a=audio, modals="avl", att_type="concat_DHT", av_using_lstm=False,
        Deep_GCN_nlayers=4, dataset=dataset, use_speaker=True, use_modal=False,
        norm="BN", num_L=3, num_K=num_k,
    )
    if "causal" in inspect.signature(model_class.__init__).parameters:
        kwargs["causal"] = False
    return model_class(
        "GRU", 1024, graph, 150, 100, 100, 100, 512,
        **kwargs,
    )


def import_class(root: Path, module_name: str, class_name: str):
    for name in list(sys.modules):
        if name in {"model", "model_cm", "model_hyper", "HypergraphConv", "high_fre_conv"}:
            sys.modules.pop(name, None)
    sys.path.insert(0, str(root.resolve()))
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    finally:
        sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3net-root", type=Path, required=True)
    parser.add_argument("--cmerc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    external_class = import_class(args.m3net_root, "model", "Model")
    external = {dataset: construct(external_class, dataset, 4 if dataset == "IEMOCAP" else 3).state_dict()
                for dataset in ("IEMOCAP", "MELD")}
    cmerc_class = import_class(args.cmerc_root, "model_cm", "Model")
    results = {}
    for dataset in ("IEMOCAP", "MELD"):
        target = construct(cmerc_class, dataset, 4 if dataset == "IEMOCAP" else 3)
        state = external[dataset]
        missing = sorted(set(target.state_dict()) - set(state))
        unexpected = sorted(set(state) - set(target.state_dict()))
        shape_mismatch = sorted(k for k in set(state) & set(target.state_dict())
                                if tuple(state[k].shape) != tuple(target.state_dict()[k].shape))
        target.load_state_dict(state, strict=True)
        results[dataset] = {"tensor_count": len(state), "missing": missing,
                            "unexpected": unexpected, "shape_mismatch": shape_mismatch,
                            "strict_load": True}
    payload = {"scope": "CPU construction and strict state_dict load only; no forward/training",
               "results": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
