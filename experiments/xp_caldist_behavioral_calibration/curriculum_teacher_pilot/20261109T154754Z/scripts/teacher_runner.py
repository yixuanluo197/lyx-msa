#!/usr/bin/env python3
"""Training-only M3Net teacher with a preregistered fixed final epoch."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from build_training_role_artifact import approved_training, file_sha256


CONFIGS = {
    "IEMOCAP": dict(epochs=80, seed=1475, dropout=0.5, batch_size=16, lr=1e-4,
                    weight_decay=3e-5, num_L=3, num_K=4, norm="BN"),
    "MELD": dict(epochs=40, seed=67137, dropout=0.4, batch_size=16, lr=1e-4,
                 weight_decay=3e-5, num_L=3, num_K=3, norm="BN"),
}
BATCH_CONFIG = dict(batch_policy="capacity_safe", edge_capacity=1000, node_capacity=5200,
                    batch_max_retries=10000)
SOURCE_FILES = ("model_cm.py", "model_GCN.py", "model_hyper.py", "HypergraphConv.py",
                "high_fre_conv.py", "causal_graph.py")
CLASS_NAMES = {"IEMOCAP": ("happy", "sad", "neutral", "angry", "excited", "frustrated"),
               "MELD": ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")}


def validate_artifact(payload, manifest, artifact_path=None):
    if payload.get("role") != "training" or payload.get("schema") != "protocol_b_training_role_v1":
        raise ValueError("only the Training role artifact is accepted")
    dataset, fold = payload["dataset"], payload["fold"]
    role = approved_training(manifest, dataset, fold)
    ids = role["dialogue_ids"]
    if payload.get("dialogue_ids") != ids or set(payload["dialogues"]) != set(ids):
        raise ValueError("artifact must exactly cover approved Training IDs")
    if payload.get("approved_training_role") != role:
        raise ValueError("artifact Training metadata differs from approved manifest")
    if payload["provenance"]["training_role_sha256"] != role["sha256"]:
        raise ValueError("artifact Training role hash mismatch")
    unknown = {(row["dialogue_id"], row["utterance_index"]) for row in manifest["unknown_label_mask"]["utterances"]} if dataset == "IEMOCAP" else set()
    counts, total, valid_total = Counter(), 0, 0
    for dialogue_id in ids:
        r = payload["dialogues"][dialogue_id]
        n = len(r["labels"])
        expected = torch.tensor([(dialogue_id, index) not in unknown for index in range(n)])
        mask = r["supervision_mask"]
        if r["dialogue_id"] != dialogue_id or mask.dtype != torch.bool or not torch.equal(mask, expected):
            raise ValueError(f"unknown mask mismatch: {dialogue_id}")
        if n == 0 or not mask.any() or len(r["text"]) != 4:
            raise ValueError(f"empty or all-invalid Training dialogue: {dialogue_id}")
        dims = [1024] * 4 + [1582 if dataset == "IEMOCAP" else 300, 342, 2 if dataset == "IEMOCAP" else 9]
        for array, width in zip(r["text"] + [r["audio"], r["visual"], r["qmask"]], dims):
            if array.shape != (n, width) or not torch.isfinite(array).all():
                raise ValueError(f"invalid Training feature shape/value: {dialogue_id}")
        if r["umask"].shape != (n,) or not torch.equal(r["umask"], torch.ones(n)):
            raise ValueError("unknown nodes must remain in the full context umask")
        if r["labels"].dtype != torch.long or (r["labels"][~mask] != -100).any():
            raise ValueError("unknown labels must be independent -100 sentinels")
        selected = r["labels"][mask]
        if (selected < 0).any() or (selected >= len(CLASS_NAMES[dataset])).any():
            raise ValueError("invalid eligible labels")
        counts.update(CLASS_NAMES[dataset][int(label)] for label in selected)
        total += n
        valid_total += int(mask.sum())
    if (total, valid_total, total - valid_total) != (role["utterance_count"], role["supervision_eligible_count"], role["unknown_masked_count"]):
        raise ValueError("artifact utterance counts do not match Training role")
    if dict(counts) != role["class_support_supervision_eligible"]:
        raise ValueError("artifact Training class support mismatch")
    return {"dataset": dataset, "fold": fold, "dialogues": len(ids), "utterances": total,
            "eligible": valid_total, "unknown": total - valid_total,
            "artifact_sha256": file_sha256(artifact_path) if artifact_path else None}


class TrainingDataset(Dataset):
    def __init__(self, payload):
        self.records = payload["dialogues"]
        self.ids = list(payload["dialogue_ids"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        key = self.ids[index]
        if key not in self.records:
            raise ValueError("Training access tripwire")
        return self.records[key]


class CapacitySafeBatchSampler(Sampler):
    """Mirror frozen XP04 whole-permutation rejection without model RNG draws."""
    def __init__(self, lengths, batch_size, generator, edge_capacity=1000,
                 node_capacity=5200, max_retries=10000):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = batch_size
        self.generator = generator
        self.edge_capacity = edge_capacity
        self.node_capacity = node_capacity
        self.max_retries = max_retries
        self.retries = self.max_batch_edges = self.max_batch_nodes = None
        if batch_size <= 0 or min(edge_capacity, node_capacity) <= 0 or max_retries < 0:
            raise ValueError("invalid batch capacity configuration")
        if any(length <= 0 for length in self.lengths):
            raise ValueError("dialogue lengths must be positive")
        if not self.lengths:
            return
        count = len(self.lengths)
        batches = len(self)
        smallest_batch = count % batch_size or min(count, batch_size)
        ordered = sorted(self.lengths)
        minimal_longest_batch = ordered[-1] + sum(ordered[:smallest_batch - 1])
        if (sum(ordered) + 3 * count > edge_capacity * batches
                or 3 * sum(ordered) > node_capacity * batches
                or minimal_longest_batch + 3 * smallest_batch > edge_capacity
                or 3 * minimal_longest_batch > node_capacity):
            raise ValueError("batch capacity impossible for the fixed batch size and dialogue lengths")

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for attempt in range(self.max_retries + 1):
            indices = torch.randperm(len(self.lengths), generator=self.generator).tolist()
            batches = [indices[start:start + self.batch_size]
                       for start in range(0, len(indices), self.batch_size)]
            totals = [sum(self.lengths[index] for index in batch) for batch in batches]
            edges = [total + 3 * len(batch) for total, batch in zip(totals, batches)]
            nodes = [3 * total for total in totals]
            if max(edges, default=0) <= self.edge_capacity and max(nodes, default=0) <= self.node_capacity:
                self.retries = attempt
                self.max_batch_edges = max(edges, default=0)
                self.max_batch_nodes = max(nodes, default=0)
                yield from batches
                return
        raise RuntimeError(f"batch capacity rejection exceeded {self.max_retries} retries; no batches were emitted")


def training_loader(payload, config, epoch):
    dataset = TrainingDataset(payload)
    # The DataLoader iterator draws its worker/base seed before sampler randperm,
    # using the very same private generator, exactly as XP04 continuous mode.
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    lengths = [len(dataset.records[key]["umask"]) for key in dataset.ids]
    sampler = CapacitySafeBatchSampler(lengths, config["batch_size"], generator,
                                      config["edge_capacity"], config["node_capacity"],
                                      config["batch_max_retries"])
    return DataLoader(dataset, batch_sampler=sampler, generator=generator,
                      num_workers=0, collate_fn=collate_training)


def collate_training(records):
    return {"ids": [r["dialogue_id"] for r in records],
            "lengths": [len(r["labels"]) for r in records],
            "text": [pad_sequence([r["text"][i] for r in records]) for i in range(4)],
            **{key: pad_sequence([r[key] for r in records]) for key in ("audio", "visual", "qmask")},
            "umask": pad_sequence([r["umask"] for r in records], batch_first=True),
            "labels": torch.cat([r["labels"] for r in records]),
            "supervision_mask": torch.cat([r["supervision_mask"] for r in records])}


class EpochTripwire:
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.seen = Counter()

    def batch(self, ids):
        if not ids or len(set(ids)) != len(ids) or set(ids) - self.allowed:
            raise ValueError("batch contains duplicate/non-Training dialogue IDs")
        if any(self.seen[key] for key in ids):
            raise ValueError("Training dialogue repeated within epoch")
        self.seen.update(ids)

    def finish(self):
        if set(self.seen) != self.allowed or any(n != 1 for n in self.seen.values()):
            raise ValueError("epoch did not exactly cover Training dialogues")


def masked_classification_loss(scores, labels, supervision_mask, criterion):
    if scores.shape[0] != labels.numel() or labels.shape != supervision_mask.shape:
        raise ValueError("model rows, labels and supervision mask are unaligned")
    valid = supervision_mask.bool()
    if not valid.any():
        return None
    # Filter before NLL/FocalLoss gather/scatter; -100 never enters the criterion.
    return criterion(scores[valid], labels[valid])


def train_epoch(model, loader, optimizer, criterion, allowed, device, epoch, batch_observer=None):
    model.train()
    tripwire, losses, eligible, skipped = EpochTripwire(allowed), [], 0, 0
    max_edges = max_nodes = 0
    for batch_index, batch in enumerate(loader):
        tripwire.batch(batch["ids"])
        edges = sum(batch["lengths"]) + 3 * len(batch["ids"])
        nodes = 3 * sum(batch["lengths"])
        max_edges, max_nodes = max(max_edges, edges), max(max_nodes, nodes)
        batch_record = {"epoch": epoch + 1, "batch_index": batch_index,
                        "dialogue_ids": batch["ids"], "lengths": batch["lengths"],
                        "edges": edges, "nodes": nodes, "dataset_role": "training",
                        "sampler_retries": getattr(getattr(loader, "batch_sampler", None), "retries", None),
                        "within_capacity": edges <= 1000 and nodes <= 5200}
        if batch_observer is not None:
            batch_observer(batch_record)
        if not batch_record["within_capacity"]:
            raise ValueError("batch exceeds frozen hyperedge_weight/EW_weight capacity before forward")
        optimizer.zero_grad(set_to_none=True)
        text = [x.to(device) for x in batch["text"]]
        scores = model(text, batch["qmask"].to(device), batch["umask"].to(device),
                       batch["lengths"], batch["audio"].to(device), batch["visual"].to(device), epoch)[0]
        loss = masked_classification_loss(scores, batch["labels"].to(device),
                                           batch["supervision_mask"].to(device), criterion)
        if loss is None:
            skipped += 1
            continue
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite teacher loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite teacher gradient")
        optimizer.step()
        n = int(batch["supervision_mask"].sum())
        eligible += n
        losses.append((float(loss.detach()), n))
    tripwire.finish()
    sampler = getattr(loader, "batch_sampler", None)
    if isinstance(sampler, CapacitySafeBatchSampler) and (
            sampler.max_batch_edges != max_edges or sampler.max_batch_nodes != max_nodes):
        raise RuntimeError("observed batch sizes disagree with capacity sampler metadata")
    return {"epoch": epoch + 1, "training_loss": sum(x*n for x, n in losses)/eligible if eligible else None,
            "supervised_utterances": eligible, "dialogue_coverage": len(tripwire.seen),
            "batch_count": len(losses) + skipped, "skipped_empty_supervision_batches": skipped,
            "exact_training_coverage": True, "roles_accessed": ["training"],
            "accessed_ids": sorted(tripwire.seen), "sampler_retries": getattr(sampler, "retries", None),
            "max_batch_edges": getattr(sampler, "max_batch_edges", max_edges),
            "max_batch_nodes": getattr(sampler, "max_batch_nodes", max_nodes)}


def validated_config(protocol, dataset):
    amendments = protocol["protocol_b_amendments"]
    if amendments.get("selection") != "fixed_final_epoch" or amendments.get("access_roles") != ["training"]:
        raise ValueError("Teacher protocol must use Training-only fixed final epoch")
    supplied = protocol["datasets"][dataset]
    for key, expected in {**CONFIGS[dataset], **BATCH_CONFIG}.items():
        if supplied.get(key) != expected:
            raise ValueError(f"Teacher config differs from preregistered {dataset}/{key}")
    return dict(CONFIGS[dataset], **BATCH_CONFIG, selection="fixed_final_epoch", access_roles=["training"],
                initialization="random", optimizer="Adam", class_weight=False,
                loss="unweighted_NLL" if dataset == "IEMOCAP" else "frozen_FocalLoss",
                shuffle=True, drop_last=False, num_workers=0, loader_rng_policy="continuous")


def checkpoint_metadata(artifact_path, config_path, source_manifest_path, config):
    return {"training_role_artifact_sha256": file_sha256(artifact_path),
            "teacher_config_sha256": file_sha256(config_path),
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "seed": config["seed"], "access_roles": ["training"], "selection": "fixed_final_epoch"}


def gpu_identity(device):
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                          "--format=csv,noheader,nounits"], text=True)
        matches = {line.split(",", 1)[1].strip() for line in output.splitlines()
                   if line.split(",", 1)[0].strip() == str(os.getpid())}
        if len(matches) != 1:
            raise RuntimeError("cannot unambiguously identify active GPU UUID for Teacher provenance")
        uuid = matches.pop()
    return {"gpu_uuid": str(uuid), "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory}


def access_audit(requested_ids, accessed_ids, manifest_sha256, epoch_coverage):
    return {"requested_ids": list(requested_ids), "accessed_ids": sorted(accessed_ids),
            "rejected_ids": [], "dataset_role": "training", "approved_manifest_sha256": manifest_sha256,
            "per_epoch_exact_coverage": epoch_coverage,
            "exact_final_coverage": set(requested_ids) == set(accessed_ids)}


def architecture(dataset, config):
    return dict(model_family="M3Net", implementation="CMERC_model_cm_Model", dataset=dataset,
                base_model="GRU", graph_type="hyper", graph_construct="direct", multi_modal=True,
                mm_fusion_mthd="concat_DHT", modals="avl", norm="BN", num_L=config["num_L"],
                num_K=config["num_K"], causal=False)


def load_frozen_module(source_root):
    root = Path(source_root).resolve()
    # PyG removed this unused import. The shim can never participate in a forward.
    from torch_geometric.nn.pool import topk_pool
    if not hasattr(topk_pool, "topk"):
        def unused_topk(*args, **kwargs):
            raise RuntimeError("unused topk compatibility shim was called")
        topk_pool.topk = unused_topk
    for name in ("model_cm", "model_GCN", "model_hyper", "HypergraphConv", "high_fre_conv", "causal_graph"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != root:
            raise RuntimeError(f"refuse mixed model imports: {name}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("model_cm")
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != root / "model_cm.py":
        raise RuntimeError("wrong model source loaded")
    return module


def build_model(module, dataset, config, device):
    return module.Model("GRU", 1024, 512 if dataset == "IEMOCAP" else 1024, 150, 100, 100, 100, 512,
        n_speakers=2 if dataset == "IEMOCAP" else 9, max_seq_len=200, window_past=10, window_future=10,
        n_classes=6 if dataset == "IEMOCAP" else 7, listener_state=False, context_attention="general",
        dropout=config["dropout"], nodal_attention=True, no_cuda=device.type != "cuda", graph_type="hyper",
        use_topic=False, alpha=0.2, multiheads=6, graph_construct="direct", use_GCN=False, use_residue=False,
        D_m_v=342, D_m_a=1582 if dataset == "IEMOCAP" else 300, modals="avl", att_type="concat_DHT",
        av_using_lstm=False, Deep_GCN_nlayers=4, dataset=dataset, use_speaker=True, use_modal=False,
        norm=config["norm"], num_L=config["num_L"], num_K=config["num_K"], causal=False).to(device)


def state_sha256(state):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--approved-manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path, help="file-name to SHA256 frozen source map")
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    manifest = json.loads(args.approved_manifest.read_text())
    audit = validate_artifact(payload, manifest, args.artifact)
    config = validated_config(json.loads(args.teacher_config.read_text()), payload["dataset"])
    compatibility = checkpoint_metadata(args.artifact, args.teacher_config, args.source_manifest, config)
    if file_sha256(args.approved_manifest) != payload["provenance"]["approved_manifest_sha256"]:
        raise ValueError("artifact was built with a different approved manifest")
    hashes = {name: file_sha256(args.source_root / name) for name in SOURCE_FILES}
    if args.source_manifest:
        expected = json.loads(args.source_manifest.read_text()).get("files", {})
        for name, actual in hashes.items():
            value = expected.get(name, expected.get(str(args.source_root / name)))
            if isinstance(value, dict):
                value = value.get("sha256")
            if value != actual:
                raise ValueError(f"frozen source hash mismatch: {name}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.audit_only:
        print(json.dumps(audit))
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("frozen noncausal M3Net forwards require CUDA; use --audit-only for CPU")
    torch.cuda.set_device(device)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    module = load_frozen_module(args.source_root)
    model = build_model(module, payload["dataset"], config, device)
    init_hash = state_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = torch.nn.NLLLoss() if payload["dataset"] == "IEMOCAP" else module.FocalLoss()
    provenance = {**payload["provenance"], "artifact_sha256": audit["artifact_sha256"],
                  "source_root": str(args.source_root.resolve()), "source_sha256": hashes,
                  "runner_sha256": file_sha256(__file__), "initialization_state_sha256": init_hash,
                  "device": str(device), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "python_version": sys.version, "training_ids": payload["dialogue_ids"],
                  "no_historical_checkpoint_loaded": True, "roles_accessed": ["training"],
                  **compatibility, **gpu_identity(device)}
    (args.output_dir / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    accessed_ids, epoch_coverage = set(), []
    with (args.output_dir / "epochs.jsonl").open("x", buffering=1) as log, \
            (args.output_dir / "batch_access.jsonl").open("x", buffering=1) as batch_log:
        for epoch in range(config["epochs"]):
            loader = training_loader(payload, config, epoch)
            result = train_epoch(model, loader, optimizer, criterion, payload["dialogue_ids"], device, epoch,
                                 batch_observer=lambda row: batch_log.write(json.dumps(row) + "\n"))
            torch.cuda.synchronize(device)
            result["elapsed_seconds"] = time.monotonic() - started
            accessed_ids.update(result["accessed_ids"])
            epoch_coverage.append({"epoch": result["epoch"], "exact_training_coverage": result["exact_training_coverage"],
                                   "accessed_ids": result["accessed_ids"], "dialogue_coverage": result["dialogue_coverage"],
                                   "sampler_retries": result["sampler_retries"],
                                   "max_batch_edges": result["max_batch_edges"],
                                   "max_batch_nodes": result["max_batch_nodes"]})
            provenance.update(actual_accessed_ids=sorted(accessed_ids),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                              total_elapsed_seconds=result["elapsed_seconds"])
            (args.output_dir / "access_audit.json").write_text(json.dumps(access_audit(
                payload["dialogue_ids"], accessed_ids, payload["provenance"]["approved_manifest_sha256"], epoch_coverage), indent=2) + "\n")
            (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            log.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            # Recoverable complete checkpoints never serve as a selected teacher.
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch + 1, "architecture": architecture(payload["dataset"], config),
                          "config": config, "provenance": provenance, "rng_state": rng_state(),
                          **compatibility, "complete": epoch + 1 == config["epochs"]}
            temporary = args.output_dir / ".latest.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output_dir / "teacher_latest.pt")
    (args.output_dir / "teacher_latest.pt").replace(args.output_dir / "teacher_final.pt")
    provenance["total_elapsed_seconds"] = time.monotonic() - started
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    metadata = {"status": "completed", "dataset": payload["dataset"], "fold": payload["fold"],
                "epoch": config["epochs"], "selection": "fixed_final_epoch", "config": config,
                "architecture": checkpoint["architecture"], "provenance": provenance,
                "final_state_sha256": state_sha256(model.state_dict()),
                "checkpoint_sha256": file_sha256(args.output_dir / "teacher_final.pt"),
                "epoch_log_sha256": file_sha256(args.output_dir / "epochs.jsonl")}
    metadata["batch_access_log_sha256"] = file_sha256(args.output_dir / "batch_access.jsonl")
    metadata.update({key: provenance[key] for key in ("gpu_uuid", "gpu_name", "peak_allocated_bytes",
                                                     "peak_reserved_bytes", "total_elapsed_seconds", "actual_accessed_ids")})
    (args.output_dir / "teacher_checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
