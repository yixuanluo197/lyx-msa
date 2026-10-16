#!/usr/bin/env python3
"""Bounded train/validation feature checks; never select test dialogues."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataloader import IEMOCAPDataset, MELDDataset, split_iemocap_dialogues


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def model_sha(model):
    """Hash actual tensor contents and schema without serialization timestamps."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def selected_dataset(name, split_seed):
    if name == "IEMOCAP":
        data = IEMOCAPDataset()
        train_ids, valid_ids = split_iemocap_dialogues(data.keys, split_seed, 0.1)
    else:
        data = MELDDataset("data/MELD_features_raw1.pkl", split="train")
        train_ids, valid_ids = list(data.keys), list(data.validIds)
    assert not set(train_ids) & set(valid_ids)
    selected = {}
    for split, ids in (("train", train_ids), ("validation", valid_ids)):
        # Deterministic size choice limits compute; no performance criterion.
        chosen = sorted((key for key in ids if len(data.videoLabels[key]) >= 3),
                        key=lambda key: (len(data.videoLabels[key]), str(key)))[:2]
        if len(chosen) < 2:
            raise RuntimeError(f"not enough nontrivial {name} {split} dialogues")
        subset = copy.copy(data)
        subset.keys, subset.len = chosen, len(chosen)
        selected[split] = subset
    return selected, {"train_ids": train_ids, "validation_ids": valid_ids,
                      "train_ids_sha256": id_sha(train_ids),
                      "validation_ids_sha256": id_sha(valid_ids)}


def audit_subset(dataset, name):
    from scripts.evaluate_checkpoint import dataset_config
    config = dataset_config(name)
    results, batches = [], []
    for index, key in enumerate(dataset.keys):
        item = dataset[index]
        length = len(dataset.videoLabels[key])
        for value in item[:6]:
            assert value.shape[0] == length and torch.isfinite(value).all()
        assert [x.shape[-1] for x in item[:4]] == [1024] * 4
        assert item[4].shape == (length, config.visual_dim)
        assert item[5].shape == (length, config.audio_dim)
        assert item[6].shape == (length, 2 if name == "IEMOCAP" else 9)
        assert torch.equal(item[6].sum(-1), torch.ones(length))
        assert ((item[6] == 0) | (item[6] == 1)).all()
        assert torch.equal(item[7], torch.ones(length))
        assert ((item[8] >= 0) & (item[8] < config.num_classes)).all()
        # Dataset indexing preserves pickle utterance order, including features.
        for tensor, source in zip(item[:4], [dataset.roberta1[key], dataset.roberta2[key],
                                             dataset.roberta3[key], dataset.roberta4[key]]):
            assert torch.equal(tensor, torch.as_tensor(np.asarray(source), dtype=torch.float32))
        assert torch.equal(item[8], torch.as_tensor(dataset.videoLabels[key]))
        batch = dataset.collate_fn([item])
        batches.append(batch)
        results.append({"id": str(key), "length": length,
                        "tensor_shapes": [list(x.shape) for x in item[:9]],
                        "utterance_order_matches_pickle": True,
                        "speaker_one_hot": True, "finite": True})
    return results, batches


def scalar_difference(left, right):
    difference = (left - right).abs()
    return {"max_abs_diff": difference.max().item(),
            "max_rel_diff": (difference / right.abs().clamp_min(1e-8)).max().item()}


def run_model_checks(name, subsets, device, seed):
    from model_cm import Model, Model_Intra, Model_Inter, FocalLoss
    from scripts.evaluate_checkpoint import dataset_config, _model_arguments
    import train_cm as training
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = dataset_config(name)
    kwargs = _model_arguments(config, device, causal=True)
    models = [kind(**kwargs).to(device) for kind in (Model, Model, Model_Intra, Model_Inter)]
    model = models[0]
    initial_model_sha256 = model_sha(model)
    diagnostics = []
    training.args = SimpleNamespace(
        seed=seed, split_seed=1475, multi_modal=True, mm_fusion_mthd="concat_DHT",
        causal=True, calibrate=True, contrastlearning=True, courselearning=False,
        rank_coff=0.005 if name == "IEMOCAP" else 0.002,
        mscl_coff=0.05 if name == "IEMOCAP" else 0.15,
        cscl_coff=0.05 if name == "IEMOCAP" else 0.15,
        auxiliary_policy="legacy_epoch_snapshot", replacement_policy="prefix_mean_v1",
        rng_policy="continuous", tensorboard=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Match train_cm.py's default class-weighted IEMOCAP objective.
    weights = torch.tensor([1 / x for x in (
        0.086747, 0.144406, 0.227883, 0.160585, 0.127711, 0.252668)], device=device)
    loss_function = FocalLoss() if name == "MELD" else torch.nn.NLLLoss(weight=weights)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    import inspect
    optional = {}
    signature = inspect.signature(training.train_or_eval_graph_model)
    if "diagnostic_callback" in signature.parameters:
        optional["diagnostic_callback"] = diagnostics.append
    elif "diagnostics_callback" in signature.parameters:
        optional["diagnostics_callback"] = diagnostics.append
    train_result = training.train_or_eval_graph_model(
        *models, loss_function, subsets["train"], epoch=0,
        cuda=device.type == "cuda", modals="avl", optimizer=optimizer,
        train=True, dataset=name, return_components=True, **optional)
    used = [(key, value) for key, value in model.named_parameters() if value.grad is not None]
    assert used and all(torch.isfinite(value.grad).all() for _, value in used)
    updated = [key for key, value in used if not torch.equal(before[key], value)]
    assert updated
    assert all(parameter.grad is None for auxiliary in models[1:] for parameter in auxiliary.parameters())
    with torch.no_grad():
        validation_result = training.train_or_eval_graph_model(
            *models, loss_function, subsets["validation"], epoch=0,
            cuda=device.type == "cuda", modals="avl", train=False,
            dataset=name, return_components=True)
    model.eval()
    batch = subsets["validation"][0]
    values = [x.to(device) for x in batch[:-1]]
    text, visual, audio = values[:4], values[4], values[5]
    speakers, mask = values[6], values[7]
    length = int(mask.sum())
    cut = max(1, length // 2)
    with torch.no_grad():
        full = model(text, speakers, mask, [length], audio, visual, 0)
        prefix = model([x[:cut] for x in text], speakers[:cut], mask[:, :cut],
                       [cut], audio[:cut], visual[:cut], 0)
    torch.testing.assert_close(full[0][:cut], prefix[0], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(full[-1][:cut], prefix[-1], atol=2e-5, rtol=1e-5)
    result = {"device": str(device), "dtype": "float32", "seed": seed,
              "initialization": "random_diagnostic_only_not_a_causal_teacher",
              "initial_model_state_sha256": initial_model_sha256,
              "final_model_state_sha256": model_sha(model),
              "production_dimensions": {key: kwargs[key] for key in ("D_g", "D_m_a", "D_m_v", "graph_hidden_size")},
              "train_batches": 2, "validation_batches": 2,
              "train_components": train_result[-1],
              "validation_components": validation_result[-1],
              "used_parameters_with_finite_gradient": len(used),
              "updated_parameters": len(updated), "auxiliary_gradients_absent": True,
              "prefix_log_probability": scalar_difference(full[0][:cut], prefix[0]),
              "prefix_hidden": scalar_difference(full[-1][:cut], prefix[-1]),
              "normal_training_dropout": kwargs["dropout"],
              "diagnostics": diagnostics}
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", choices=("IEMOCAP", "MELD", "both"), default="both")
    parser.add_argument("--seed", type=int, default=1937)
    parser.add_argument("--split-seed", type=int, default=1475)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())
    output = {"status": "running", "device": str(device), "seed": args.seed,
              "split_seed": args.split_seed, "feature_extraction": "not_audited",
              "test_dialogues_selected": 0, "datasets": {}}
    names = ("IEMOCAP", "MELD") if args.dataset == "both" else (args.dataset,)
    for name in names:
        selected, split = selected_dataset(name, args.split_seed)
        paths = (["IEMOCAP_features.pkl", "iemocap_features_roberta.pkl"] if name == "IEMOCAP"
                 else ["MELD_features_raw1.pkl", "meld_features_roberta.pkl"])
        item = {"split": split, "feature_sha256": {path: file_sha(ROOT / "data" / path) for path in paths},
                "selected": {}, "status": "passed"}
        batches = {}
        for subset_name, data in selected.items():
            item["selected"][subset_name], batches[subset_name] = audit_subset(data, name)
        if not args.audit_only:
            item["model_checks"] = run_model_checks(name, batches, device, args.seed)
        output["datasets"][name] = item
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, default=str))
        print(name, "passed", flush=True)
    output["status"] = "passed"
    args.output.write_text(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
