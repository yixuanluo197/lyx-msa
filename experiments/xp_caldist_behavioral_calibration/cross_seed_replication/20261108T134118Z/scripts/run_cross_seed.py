#!/usr/bin/env python
"""Frozen CMERC validation forward runner for mean and training-donor references."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from cross_seed_core import (contract, donor_index, donor_modality_missing,
                             interpolate, native_key_map, sha256, target_indices)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preregistered_config.json"
CFG = json.loads(CONFIG.read_text())
HIST_PILOT = Path(CFG["historical_pilot"])
HIST_EVIDENCE = Path(CFG["historical_evidence_sensitivity"])
sys.path.insert(0, str(HIST_PILOT / "scripts"))
import run_pilot as old
from graph_index_cache import install_graph_cache


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def compare(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid parity arrays: {name}")
    if not np.allclose(a, b, atol=1e-6, rtol=1e-5) or not np.array_equal(a.argmax(-1), b.argmax(-1)):
        raise ValueError(f"parity failure {name}: max error {np.max(np.abs(a-b))}")
    return float(np.max(np.abs(a-b)))


def baseline_path(dataset, seed):
    path = HIST_PILOT / f"runs/{dataset}/seed{seed}/validation_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"verified historical validation prediction missing: {path}")
    return path


def statistics_path(dataset):
    path = HIST_PILOT / f"provenance/{dataset}_train_statistics.npz"
    if not path.exists():
        raise FileNotFoundError(f"verified train-only statistics missing: {path}")
    return path


def metadata(batch, lengths, batch_index):
    values = {key: [] for key in ("labels", "dialogue_ids", "utterance_indices",
                                  "speaker_ids", "batch_indices", "dialogue_lengths")}
    for dialogue_index, (dialogue, length) in enumerate(zip(batch[-1], lengths)):
        values["labels"].extend(batch[8][dialogue_index, :length].numpy().tolist())
        values["dialogue_ids"].extend([str(dialogue)] * length)
        values["utterance_indices"].extend(range(length))
        values["speaker_ids"].extend(batch[6][:length, dialogue_index].argmax(-1).numpy().tolist())
        values["batch_indices"].extend([batch_index] * length)
        values["dialogue_lengths"].extend([length] * length)
    return {key: np.asarray(value, dtype=str if key == "dialogue_ids" else np.int64)
            for key, value in values.items()}


def donor_pool(dataset):
    """Aligned train utterance records; one row supplies all four T arrays."""
    names = ("roberta1", "roberta2", "roberta3", "roberta4", "videoVisual", "videoAudio")
    collections = [getattr(dataset, name) for name in names]
    records = []
    for dialogue in dataset.keys:
        lengths = [len(collection[dialogue]) for collection in collections]
        if len(set(lengths)) != 1:
            raise ValueError(f"unaligned training modalities in {dialogue}: {lengths}")
        for utterance_index in range(lengths[0]):
            # Preserve the dataset's native key for dictionary lookup (MELD uses
            # integers); stringify only when serializing donor identity.
            records.append((dialogue, utterance_index))
    if not records:
        raise ValueError("empty training donor pool")
    return collections, records


def run(dataset_name, seed, phase):
    if dataset_name not in CFG["datasets"] or seed not in CFG["seeds"]:
        raise ValueError("dataset/seed outside preregistration")
    if phase == "main" and seed not in CFG["new_runs"][dataset_name]:
        raise ValueError("anchor main interventions must be reused, not rerun")
    main_specs, donor_specs = contract(CFG)
    specs = main_specs if phase == "main" else donor_specs
    output = ROOT / f"runs/{dataset_name}/seed{seed}"
    conditions = output / "conditions"
    complete = output / f"{phase}_complete.json"
    if complete.exists():
        raise FileExistsError(f"refusing to overwrite completed run: {complete}")
    conditions.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if (conditions / f"{spec['id']}.npz").exists():
            raise FileExistsError(f"refusing to overwrite condition {spec['id']}")

    resource_path = output / f"resource_snapshot_{phase}.json"
    write_json(resource_path, old.resource_snapshot())
    api = old.imports()
    torch, _, dl, _, _ = api
    os.chdir(old.CODE)
    validation = old.dataset_for(dataset_name, "validation", dl)
    expected = CFG["validation_sizes"][dataset_name]
    if len(validation) != expected["dialogues"]:
        raise ValueError("validation dialogue count mismatch")
    training = old.dataset_for(dataset_name, "train", dl) if phase == "donor" else None
    pool_collections, pool_records = donor_pool(training) if training is not None else (None, None)
    with np.load(baseline_path(dataset_name, seed), allow_pickle=False) as source:
        baseline = {key: source[key] for key in source.files}
    with np.load(statistics_path(dataset_name), allow_pickle=False) as source:
        statistics = {key: source[key] for key in source.files}
    if len(baseline["labels"]) != expected["utterances"]:
        raise ValueError("validation utterance count mismatch")
    validation_native_keys = native_key_map(validation.keys)
    missing = np.asarray([np.all(np.asarray(validation.videoVisual[validation_native_keys[str(dialogue)]][int(index)]) == 0)
                          for dialogue, index in zip(baseline["dialogue_ids"], baseline["utterance_indices"])])
    if int(missing.sum()) != CFG["visual_missingness"]["expected_counts"][dataset_name]:
        raise ValueError("visual missingness count mismatch")
    local_baseline = output / "baseline_validation.npz"
    expected_baseline = dict(baseline, visual_missing=missing)
    if local_baseline.exists():
        with np.load(local_baseline, allow_pickle=False) as stored:
            if set(stored.files) != set(expected_baseline):
                raise ValueError("existing baseline schema mismatch")
            for key, value in expected_baseline.items():
                if not np.array_equal(stored[key], value):
                    raise ValueError(f"existing baseline differs: {key}")
    else:
        np.savez_compressed(local_baseline, **expected_baseline)

    model = old.load_model(dataset_name, seed, api)
    frozen = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    cache = install_graph_cache(model.graph_model)
    guard = old.GraphGuard(model, torch)
    n, classes = baseline["probabilities"].shape
    log_outputs = {spec["id"]: np.full((n, spec["repeats"], classes), np.nan, np.float32)
                   for spec in specs}
    donor_ids = {spec["id"]: np.full((n, spec["repeats"]), -1, np.int64) for spec in specs}
    stat_names = {0: "text0", 1: "text1", 2: "text2", 3: "text3", 4: "visual", 5: "audio"}
    checks, offset, forwards, skipped = [], 0, 0, 0
    start = time.monotonic()
    for batch_index, batch in enumerate(old.loader_for(validation, torch)):
        tensors = [value.cuda() for value in batch[:-1]]
        originals = [value.clone() for value in tensors]
        lengths = [int(value.sum()) for value in tensors[7]]
        count = sum(lengths)
        observed = metadata(batch, lengths, batch_index)
        for key, value in observed.items():
            if not np.array_equal(value, baseline[key][offset:offset+count]):
                raise ValueError(f"validation order mismatch: {key}")
        guard.reset()
        with torch.no_grad():
            cache.restore()
            reference = old.forward(model, tensors, lengths).cpu().numpy()
            cache.enable()
            cached = old.forward(model, tensors, lengths).cpu().numpy()
            checks.append({"batch": batch_index, "kind": "original",
                           "cache_log_error": compare(cached, reference, "cache log"),
                           "historical_log_error": compare(reference, baseline["log_probabilities"][offset:offset+count], "historical log")})
            local = 0
            for dialogue_index, (dialogue, length) in enumerate(zip(batch[-1], lengths)):
                for utterance_index in range(length):
                    row = offset + local + utterance_index
                    for spec in specs:
                        indices = target_indices(spec["target"])
                        if spec["target"] == "V" and missing[row]:
                            log_outputs[spec["id"]][row] = baseline["log_probabilities"][row]
                            skipped += spec["repeats"]
                            continue
                        vectors = [batch[index][utterance_index, dialogue_index].numpy() for index in indices]
                        for repeat in range(spec["repeats"]):
                            if phase == "main":
                                references = [statistics[stat_names[index] + "_mean"] for index in indices]
                            else:
                                selected = donor_index(len(pool_records), dataset_name, str(dialogue),
                                                       utterance_index, spec["target"], spec["strength"], repeat)
                                donor_ids[spec["id"]][row, repeat] = selected
                                donor_dialogue, donor_utterance = pool_records[selected]
                                references = [pool_collections[index][donor_dialogue][donor_utterance]
                                              for index in indices]
                            changed = interpolate(vectors, references, spec["strength"])
                            try:
                                for index, value in zip(indices, changed):
                                    tensors[index][utterance_index, dialogue_index].copy_(torch.from_numpy(value).cuda())
                                result = old.forward(model, tensors, lengths)
                                if local + utterance_index == 0 and repeat == 0:
                                    candidate = result.cpu().numpy()
                                    cache.restore()
                                    eager = old.forward(model, tensors, lengths).cpu().numpy()
                                    cache.enable()
                                    checks.append({"batch": batch_index, "kind": spec["id"],
                                                   "cache_log_error": compare(candidate, eager, spec["id"])})
                                log_outputs[spec["id"]][row, repeat] = result[local + utterance_index].cpu().numpy()
                                forwards += 1
                            finally:
                                for index in indices:
                                    tensors[index][utterance_index, dialogue_index].copy_(originals[index][utterance_index, dialogue_index])
                local += length
            if not all(torch.equal(value, original) for value, original in zip(tensors, originals)):
                raise RuntimeError("input was not restored exactly")
            for key, value in model.state_dict().items():
                if not torch.equal(value.detach().cpu(), frozen[key]):
                    raise RuntimeError(f"model state changed: {key}")
            if not all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("model parameters were trainable or acquired gradients")
        offset += count
        partial = output / "partial"
        partial.mkdir(exist_ok=True)
        for spec in specs:
            np.savez_compressed(partial / f"{spec['id']}.npz",
                                log_probabilities=log_outputs[spec["id"]], completed_utterances=offset)
        print(json.dumps({"dataset": dataset_name, "seed": seed, "phase": phase,
                          "batch": batch_index, "completed_utterances": offset,
                          "forwards": forwards, "seconds": time.monotonic()-start}), flush=True)
    if offset != n:
        raise RuntimeError("incomplete validation traversal")

    for spec in specs:
        qlog = log_outputs[spec["id"]]
        if not np.isfinite(qlog).all():
            raise ValueError(f"nonfinite output: {spec['id']}")
        probabilities = np.exp(qlog)
        if spec["target"] == "V":
            probabilities[missing] = baseline["probabilities"][missing, None, :]
            qlog[missing] = baseline["log_probabilities"][missing, None, :]
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1, atol=1e-6, rtol=1e-5):
            raise ValueError(f"invalid normalized probabilities: {spec['id']}")
        active = ~missing if spec["target"] == "V" else np.ones(n, bool)
        payload = {key: value for key, value in baseline.items()
                   if key not in ("probabilities", "log_probabilities", "predictions")}
        payload.update(original_predictions=baseline["predictions"],
                       original_probabilities=baseline["probabilities"],
                       probabilities=probabilities, log_probabilities=qlog,
                       perturbed_predictions=probabilities.argmax(-1), active=active,
                       visual_missing=missing, condition_id=np.asarray(spec["id"]),
                       target=np.asarray(spec["target"]), kind=np.asarray(spec["kind"]),
                       strength=np.asarray(spec["strength"]), repeats=np.asarray(spec["repeats"]),
                       phase=np.asarray(phase))
        if phase == "donor":
            indices = donor_ids[spec["id"]]
            target_indices_for_spec = target_indices(spec["target"])
            donor_missing = np.asarray([[donor_modality_missing(
                spec["target"], [pool_collections[j][pool_records[index][0]][pool_records[index][1]]
                                 for j in target_indices_for_spec]) if index >= 0 else False
                                         for index in row] for row in indices], bool)
            payload.update(donor_flat_indices=indices,
                           donor_dialogue_ids=np.asarray([[str(pool_records[index][0]) if index >= 0 else ""
                                                          for index in row] for row in indices]),
                           donor_utterance_indices=np.asarray([[pool_records[index][1] if index >= 0 else -1
                                                               for index in row] for row in indices], np.int64),
                           donor_modality_missing=donor_missing)
        np.savez_compressed(conditions / f"{spec['id']}.npz", **payload)

    guard.close(); cache.restore()
    checkpoint = Path(CFG["xp04"]) / f"runs/cmerc/{dataset_name}_validation_seed{seed}/best.pt"
    result = {"passed": True, "dataset": dataset_name, "seed": seed, "phase": phase,
              "seconds": time.monotonic()-start, "target_forwards": forwards,
              "skipped_visual_missing": skipped, "conditions": {
                  spec["id"]: sha256(conditions / f"{spec['id']}.npz") for spec in specs},
              "parity_checks": checks, "inputs_restored_exact": True,
              "parameters_and_buffers_unchanged": True, "gradients_absent": True,
              "split": "validation", "test_inference": False,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "validation_baseline": str(baseline_path(dataset_name, seed)),
              "validation_baseline_sha256": sha256(baseline_path(dataset_name, seed)),
              "train_statistics": str(statistics_path(dataset_name)),
              "train_statistics_sha256": sha256(statistics_path(dataset_name)),
              "resource_snapshot": str(resource_path), "resource_snapshot_sha256": sha256(resource_path),
              "config_sha256": sha256(CONFIG), "runner_sha256": sha256(__file__),
              "gpu_visible": os.getenv("CUDA_VISIBLE_DEVICES"), "server": os.uname().nodename}
    write_json(complete, result)
    print(json.dumps(result), flush=True)
    del model, validation, training
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=CFG["datasets"])
    parser.add_argument("--seed", required=True, type=int, choices=CFG["seeds"])
    parser.add_argument("--phase", required=True, choices=("main", "donor"))
    arguments = parser.parse_args()
    try:
        run(arguments.dataset, arguments.seed, arguments.phase)
    except Exception:
        failure = ROOT / f"logs/failure_{arguments.dataset}_{arguments.seed}_{arguments.phase}_{int(time.time())}.txt"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(traceback.format_exc())
        raise
