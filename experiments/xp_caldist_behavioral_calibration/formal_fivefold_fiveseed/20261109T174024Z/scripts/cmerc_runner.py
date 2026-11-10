#!/usr/bin/env python3
"""Independent CMERC formal run: random student, fold-local curriculum, Development selection.

Only the role-scoped Training and Development artifacts are loaded here. The first
real epoch is the gate and its checkpoint continues unchanged into epoch two.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_role_artifact import approved_role, file_sha256

ROOT = Path(__file__).resolve().parents[5]
TEACHER_SCRIPTS = ROOT / "experiments/xp_caldist_behavioral_calibration/curriculum_teacher_pilot/20261109T154754Z/scripts"
SOURCE_ROOT = ROOT / "CMERC/experiments/xp_mainline_validation/20261016T050823Z/code"


def import_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The historical teacher uses this sibling module by its original name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import_helper("build_training_role_artifact", TEACHER_SCRIPTS / "build_training_role_artifact.py")
teacher = import_helper("pilot_teacher_helpers", TEACHER_SCRIPTS / "teacher_runner.py")
mask_runner = import_helper("pilot_protocol_b_mask", TEACHER_SCRIPTS / "protocol_b_mask_runner.py")
CapacitySafeBatchSampler = teacher.CapacitySafeBatchSampler
collate_training = teacher.collate_training
CLASS_NAMES = teacher.CLASS_NAMES
SOURCE_FILES = teacher.SOURCE_FILES + ("clloss.py", "train_cm.py")


PREREGISTERED_SEEDS = (1475, 67137, 2024, 3407, 42)


def config_for(dataset, seed):
    if seed not in PREREGISTERED_SEEDS:
        raise ValueError(f"seed {seed} is not in the preregistered seed set")
    return dict({**teacher.CONFIGS[dataset], "seed": seed}, **teacher.BATCH_CONFIG,
                rank_coff=.005 if dataset == "IEMOCAP" else .002,
                mscl_coff=.05 if dataset == "IEMOCAP" else .15,
                cscl_coff=.05 if dataset == "IEMOCAP" else .15,
                epoch_ratio=.15 if dataset == "IEMOCAP" else .4,
                scheduler_steps=1, calibrate=True, contrastlearning=True,
                initialization="random", selection="development_weighted_f1_rounded_2dp_percent",
                tie_policy="earliest_epoch", precision="FP32", class_weight=dataset == "IEMOCAP",
                optimizer="Adam", access_roles=["training", "development"],
                shuffle=True, drop_last=False, num_workers=0, loader_rng_policy="continuous",
                loss="fold_local_masked_NLL" if dataset == "IEMOCAP" else "frozen_FocalLoss")


def validate_artifact(payload, manifest, role_name, manifest_path=None):
    if role_name not in ("training", "development", "calibration") or payload.get("role") != role_name:
        raise ValueError("artifact role is not the requested Training/Development role")
    if payload.get("schema") != "protocol_b_role_v2":
        raise ValueError("invalid CMERC role artifact schema")
    dataset, fold = payload["dataset"], payload["fold"]
    role = approved_role(manifest, dataset, fold, role_name)
    ids = role["dialogue_ids"]
    if payload.get("dialogue_ids") != ids or set(payload["dialogues"]) != set(ids):
        raise ValueError("artifact must exactly cover the requested role IDs")
    if payload.get("approved_role") != role or payload["provenance"]["role_sha256"] != role["sha256"]:
        raise ValueError("artifact role provenance mismatch")
    if manifest_path and payload["provenance"]["approved_manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("artifact approved manifest file mismatch")
    from build_role_artifact import canonical_sha256
    groups = payload.get("dialogue_groups", {})
    if (set(groups) != set(ids) or set(groups.values()) - set(role["group_ids"])
            or payload["provenance"].get("sample_lineage_sha256") != manifest["source"]["sample_lineage_sha256"]
            or payload["provenance"].get("dialogue_groups_sha256") != canonical_sha256(groups)):
        raise ValueError("artifact dialogue-group lineage mismatch")
    # Reuse the established feature/mask/class-support validator against a local
    # role projection. No records or labels from any other role are inspected.
    projection = copy.deepcopy(manifest)
    projection["protocols"][dataset]["folds"][fold]["roles"]["training"] = role
    projection["content_sha256"] = canonical_sha256({k: v for k, v in projection.items() if k != "content_sha256"})
    projected = dict(payload, role="training", schema="protocol_b_training_role_v1", approved_training_role=role,
                     provenance=dict(payload["provenance"], training_role_sha256=role["sha256"]))
    result = teacher.validate_artifact(projected, projection)
    return dict(result, role=role_name)


def validate_difficulty(payload, training, manifest_sha256):
    if (payload.get("dataset"), payload.get("fold")) != (training["dataset"], training["fold"]):
        raise ValueError("difficulty dataset/fold mismatch")
    if payload.get("status") != "formal_fold_local_difficulty" or payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("difficulty must be the approved formal fold-local artifact")
    rows = payload.get("dialogues")
    if not isinstance(rows, list):
        raise ValueError("difficulty dialogues must be a list")
    values = {}
    for row in rows:
        key = row["dialogue_id"]
        value = row["final_difficulty"]
        if key in values or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("difficulty rows contain duplicate IDs or nonfinite values")
        values[key] = value
    if set(values) != set(training["dialogue_ids"]):
        raise ValueError("difficulty must cover exact Training IDs without fallback")
    return values


def curriculum_ratio(config, epoch):
    return min(1.0, round(config["epoch_ratio"] * int(epoch / config["scheduler_steps"] + 1), 2))


def role_loader(payload, ids, config, epoch):
    subset = dict(payload, dialogue_ids=list(ids))
    dataset = teacher.TrainingDataset(subset)
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    sampler = CapacitySafeBatchSampler([len(dataset.records[key]["labels"]) for key in ids],
                                      config["batch_size"], generator, config["edge_capacity"],
                                      config["node_capacity"], config["batch_max_retries"])
    return DataLoader(dataset, batch_sampler=sampler, generator=generator, num_workers=0,
                      collate_fn=collate_training)


def model_output(model, text, qmask, umask, lengths, audio, visual, epoch):
    output = model(text, qmask, umask, lengths, audio, visual, epoch)
    return output[0], output[-1]


def forward_closures(models, batch, device, epoch):
    student, multimodal, intra, inter = models
    text = [x.to(device) for x in batch["text"]]
    audio, visual, qmask, umask = [batch[key].to(device) for key in ("audio", "visual", "qmask", "umask")]
    lengths = batch["lengths"]
    # XP04 noncausal mean includes padding; retain this exact operation.
    mean = lambda feature: feature.mean(dim=0).repeat(feature.shape[0], 1, 1)
    def call(model, selected_text=text, selected_audio=audio, selected_visual=visual):
        return model_output(model, selected_text, qmask, umask, lengths, selected_audio, selected_visual, epoch)
    full = lambda: call(student)
    removals = [lambda: call(multimodal, [mean(x) for x in text]),
                lambda: call(multimodal, selected_audio=mean(audio)),
                lambda: call(multimodal, selected_visual=mean(visual)),
                lambda: call(intra), lambda: call(inter)]
    return full, removals


def refresh_removal_models(models):
    state = models[0].state_dict()
    reports = []
    for model in models[1:]:
        incompatible = model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        reports.append(dict(model=type(model).__name__, strict=True,
                            missing=list(incompatible.missing_keys), unexpected=list(incompatible.unexpected_keys)))
    return reports


def direct_gradient_audit(result, scores, hidden, mask):
    """Prove excluded rows have zero direct gradient on all 11 real loss paths."""
    mask = mask.bool()
    terms = {"classification": (result["classification_loss"], scores)}
    terms.update({"rank_" + name: (term, scores) for name, term in result["rank_by_view"].items()})
    terms.update({"contrastive_" + name: (term, hidden) for name, term in result["contrastive_by_view"].items()})
    report = {}
    for name, (term, source) in terms.items():
        gradient, = torch.autograd.grad(term, source, retain_graph=True)
        excluded = float(gradient[~mask].abs().max()) if (~mask).any() else 0.0
        eligible = float(gradient[mask].abs().sum())
        if excluded != 0.0 or not torch.isfinite(gradient).all():
            raise RuntimeError("unknown direct-gradient gate failed: " + name)
        report[name] = dict(excluded_direct_gradient_abs_max=excluded,
                            eligible_gradient_abs_sum=eligible, supervised_count=int(mask.sum()))
    if report["classification"]["eligible_gradient_abs_sum"] <= 0:
        raise RuntimeError("eligible classification rows have no direct gradient")
    return dict(status="passed", unknown_nodes_retained=int((~mask).sum()),
                supervised_path_count=len(report), paths=report)


def train_epoch(models, loader, optimizer, criterion, ids, device, epoch, config, observer=None):
    models[0].train()
    refresh = refresh_removal_models(models)  # exactly once per Training epoch
    tripwire, totals, supervised, updates = teacher.EpochTripwire(ids), Counter(), 0, 0
    gradient_audit = None
    for index, batch in enumerate(loader):
        tripwire.batch(batch["ids"])
        edges, nodes = sum(batch["lengths"]) + 3 * len(batch["ids"]), 3 * sum(batch["lengths"])
        if edges > config["edge_capacity"] or nodes > config["node_capacity"]:
            raise RuntimeError("batch exceeds frozen model capacity before forward")
        full, removals = forward_closures(models, batch, device, epoch)
        optimizer.zero_grad(set_to_none=True)
        scores, hidden = full()
        with torch.no_grad():
            removed = [forward()[0] for forward in removals]
        result = mask_runner.masked_objective(scores, removed, hidden, batch["labels"].to(device),
                      batch["supervision_mask"].to(device), criterion,
                      **{key: config[key] for key in ("rank_coff", "mscl_coff", "cscl_coff", "calibrate", "contrastlearning")})
        if epoch == 0 and gradient_audit is None and (~batch["supervision_mask"]).any():
            gradient_audit = direct_gradient_audit(result, scores, hidden, batch["supervision_mask"].to(device))
        if any(not torch.isfinite(result[key]) for key in ("classification_loss", "rank_loss", "mscl_loss", "cscl_loss", "total_loss")):
            raise RuntimeError("nonfinite CMERC loss component")
        if result["supervised_count"]:
            result["total_loss"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in models[0].parameters()):
                raise RuntimeError("nonfinite CMERC gradient")
            optimizer.step()
            updates += 1
        n = result["supervised_count"]
        supervised += n
        for key in ("classification_loss", "rank_loss", "mscl_loss", "cscl_loss", "total_loss"):
            totals[key] += float(result[key].detach()) * n
        if observer:
            observer(dict(epoch=epoch + 1, batch_index=index, role="training", ids=batch["ids"],
                          lengths=batch["lengths"], edges=edges, nodes=nodes, forwards=6,
                          supervised_count=n, optimizer_step=bool(n)))
    tripwire.finish()
    return dict(epoch=epoch + 1, **{key: value / supervised for key, value in totals.items()},
                supervised_utterances=supervised, optimizer_steps=updates,
                exact_curriculum_coverage=True, accessed_ids=sorted(tripwire.seen),
                removal_refresh_count=1, removal_refresh=refresh, direct_unknown_gradient_audit=gradient_audit,
                sampler_retries=loader.batch_sampler.retries,
                max_batch_edges=loader.batch_sampler.max_batch_edges,
                max_batch_nodes=loader.batch_sampler.max_batch_nodes)


@torch.no_grad()
def development_epoch(model, loader, ids, device, epoch, observer=None):
    model.eval()
    tripwire, scores, labels, masks = teacher.EpochTripwire(ids), [], [], []
    for index, batch in enumerate(loader):
        tripwire.batch(batch["ids"])
        full, _ = forward_closures((model, model, model, model), batch, device, epoch)
        scores.append(full()[0].cpu())
        labels.append(batch["labels"])
        masks.append(batch["supervision_mask"])
        if observer:
            observer(dict(epoch=epoch + 1, batch_index=index, role="development", ids=batch["ids"],
                          lengths=batch["lengths"], forwards=1))
    tripwire.finish()
    result = mask_runner.development_metrics(torch.cat(scores), torch.cat(labels), torch.cat(masks))
    valid = torch.cat(masks).bool()
    target, prediction = torch.cat(labels)[valid], torch.cat(scores)[valid].argmax(1)
    classes = torch.cat(scores).shape[1]
    result["macro_f1"] = sum(2.0 * int(((prediction == k) & (target == k)).sum()) /
                               max(1, int((prediction == k).sum()) + int((target == k).sum()))
                               for k in range(classes)) / classes
    result["weighted_f1_percent_rounded"] = round(100 * result["weighted_f1"], 2)
    result["exact_development_coverage"] = True
    return result


def update_selection(score, epoch, best_score, best_epoch):
    rounded = round(float(score), 2)
    if not math.isfinite(rounded):
        raise ValueError("Development selection metric must be finite")
    improved = best_score is None or rounded > best_score
    return (rounded, epoch, True) if improved else (best_score, best_epoch, False)


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_selected_model(checkpoint_path, device="cuda:0"):
    """Load only a completed Development-selected best student, strictly."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("is_best") is not True or checkpoint.get("complete") is not True
            or checkpoint.get("training_complete") is not True
            or checkpoint.get("selection_metric") != "development_weighted_f1"
            or checkpoint.get("selection_role") != "development"
            or checkpoint.get("selected_epoch") != checkpoint.get("epoch")):
        raise ValueError("a completed Development-selected best checkpoint is required")
    if checkpoint.get("schema") != "independent_cmerc_complete_checkpoint_v1":
        raise ValueError("unsupported student checkpoint schema")
    if file_sha256(__file__) != checkpoint["bindings"]["runner_sha256"]:
        raise ValueError("selected checkpoint runner version mismatch")
    for name, expected in checkpoint["bindings"]["helper_sha256"].items():
        if file_sha256(TEACHER_SCRIPTS / name) != expected:
            raise ValueError("selected checkpoint helper version mismatch: " + name)
    root = Path(checkpoint["provenance"]["source_root"])
    for name, expected in checkpoint["bindings"]["source_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("selected checkpoint source mismatch: " + name)
    if teacher.state_sha256(checkpoint["model_state_dict"]) != checkpoint["model_state_sha256"]:
        raise ValueError("selected checkpoint model state hash mismatch")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("frozen CMERC inference requires CUDA")
    torch.cuda.set_device(device)
    module = teacher.load_frozen_module(root)
    model = teacher.build_model(module, checkpoint["dataset"], checkpoint["config"], device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint


@torch.no_grad()
def predict_role_artifact(checkpoint_path, artifact_path, approved_manifest_path, device="cuda:0"):
    """Return full Calibration rows in artifact order; unknown context remains."""
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(approved_manifest_path).read_text())
    validate_artifact(payload, manifest, "calibration", approved_manifest_path)
    model, checkpoint = load_selected_model(checkpoint_path, device)
    if ((payload["dataset"], payload["fold"]) != (checkpoint["dataset"], checkpoint["fold"])
            or file_sha256(approved_manifest_path) != checkpoint["bindings"]["manifest_sha256"]):
        raise ValueError("Calibration artifact and selected student protocol mismatch")
    by_id = {}
    for batch in role_loader(payload, payload["dialogue_ids"], checkpoint["config"], checkpoint["epoch"]):
        full, _ = forward_closures((model, model, model, model), batch, torch.device(device), checkpoint["epoch"] - 1)
        scores = full()[0].cpu()
        offset = 0
        for key, length in zip(batch["ids"], batch["lengths"]):
            if key in by_id:
                raise ValueError("Calibration dialogue repeated")
            by_id[key] = scores[offset:offset + length]
            offset += length
        if offset != len(scores):
            raise ValueError("Calibration neural rows do not align with dialogue lengths")
    ids = payload["dialogue_ids"]
    if set(by_id) != set(ids):
        raise ValueError("Calibration forward did not exactly cover its role")
    return dict(logits=torch.cat([by_id[key] for key in ids]).numpy(),
                labels=torch.cat([payload["dialogues"][key]["labels"] for key in ids]).numpy(),
                supervision_mask=torch.cat([payload["dialogues"][key]["supervision_mask"] for key in ids]).numpy(),
                dialogue_ids=np.asarray([key for key in ids for _ in payload["dialogues"][key]["labels"]]),
                utterance_indices=np.asarray([i for key in ids for i in range(len(payload["dialogues"][key]["labels"]))]),
                dataset=payload["dataset"], fold=payload["fold"], selected_epoch=checkpoint["selected_epoch"],
                checkpoint_sha256=file_sha256(checkpoint_path), role="calibration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--development-artifact", type=Path, required=True)
    parser.add_argument("--approved-manifest", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=PREREGISTERED_SEEDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.approved_manifest.read_text())
    training, development = [torch.load(path, map_location="cpu", weights_only=False) for path in
                             (args.training_artifact, args.development_artifact)]
    audits = [validate_artifact(payload, manifest, role, args.approved_manifest) for payload, role in
              ((training, "training"), (development, "development"))]
    if (training["dataset"], training["fold"]) != (development["dataset"], development["fold"]):
        raise ValueError("Training and Development dataset/fold mismatch")
    if set(training["dialogue_ids"]) & set(development["dialogue_ids"]):
        raise ValueError("Training/Development overlap")
    difficulty = validate_difficulty(json.loads(args.difficulty.read_text()), training, file_sha256(args.approved_manifest))
    config = config_for(training["dataset"], args.seed)
    if training["dataset"] == "IEMOCAP":
        labels = torch.cat([training["dialogues"][key]["labels"] for key in training["dialogue_ids"]])
        valid = torch.cat([training["dialogues"][key]["supervision_mask"] for key in training["dialogue_ids"]])
        config["class_weights"] = mask_runner.training_class_weights(labels, valid, 6).tolist()
    expected_source = json.loads(args.source_manifest.read_text())["files"]
    hashes = {name: file_sha256(args.source_root / name) for name in SOURCE_FILES}
    for name, value in hashes.items():
        expected = expected_source.get(name, expected_source.get(str(args.source_root / name), expected_source.get(str((args.source_root / name).resolve()))))
        if isinstance(expected, dict):
            expected = expected["sha256"]
        if expected != value:
            raise ValueError(f"frozen executable source hash mismatch: {name}")
    # Verify the shared loss helper's contrastive kernel is the exact XP04 kernel.
    mask_runner.load_frozen_clloss()
    if file_sha256(mask_runner.FROZEN_SOURCE / "clloss.py") != hashes["clloss.py"]:
        raise ValueError("mask runner contrastive kernel differs from frozen XP04")
    bindings = {"training_artifact_sha256": file_sha256(args.training_artifact),
                "development_artifact_sha256": file_sha256(args.development_artifact),
                "manifest_sha256": file_sha256(args.approved_manifest),
                "difficulty_sha256": file_sha256(args.difficulty),
                "source_manifest_sha256": file_sha256(args.source_manifest), "source_sha256": hashes,
                "runner_sha256": file_sha256(__file__),
                "role_builder_sha256": file_sha256(Path(__file__).with_name("build_role_artifact.py")),
                "helper_sha256": {name: file_sha256(TEACHER_SCRIPTS / name) for name in
                                   ("teacher_runner.py", "protocol_b_mask_runner.py", "build_training_role_artifact.py")}}
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    (args.output_dir / "artifact_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (args.output_dir / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.audit_only:
        print(json.dumps({"artifacts": audits, "difficulty_count": len(difficulty), "bindings": bindings}))
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("frozen CMERC requires CUDA; CPU permits --audit-only")
    torch.cuda.set_device(device)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    module = teacher.load_frozen_module(args.source_root)
    student = teacher.build_model(module, training["dataset"], config, device)
    # Identical constructor contract with the two frozen context-removal classes.
    class Adapter:
        pass
    models = [student, teacher.build_model(module, training["dataset"], config, device)]
    for constructor in (module.Model_Intra, module.Model_Inter):
        adapter = Adapter()
        adapter.Model = constructor
        models.append(teacher.build_model(adapter, training["dataset"], config, device))
    initial_hash = teacher.state_sha256(student.state_dict())
    optimizer = torch.optim.Adam(student.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = torch.nn.NLLLoss(torch.tensor(config["class_weights"], device=device)) if training["dataset"] == "IEMOCAP" else module.FocalLoss()
    provenance = dict(bindings, source_root=str(args.source_root.resolve()), random_initialization_state_sha256=initial_hash,
                      historical_checkpoint_loaded=False, roles_accessed=["training", "development"],
                      device=str(device), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda, python_version=sys.version,
                      **teacher.gpu_identity(device))
    start, best_score, best_epoch, gate_passed = 0, None, None, False
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["bindings"] != bindings or checkpoint["config"] != config:
            raise ValueError("resume source/artifact/config binding mismatch")
        student.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng(checkpoint["rng_state"])
        start, best_score, best_epoch = checkpoint["epoch"], checkpoint["best_score"], checkpoint["best_epoch"]
        gate_passed = checkpoint["first_epoch_gate_passed"]
        provenance = checkpoint["provenance"]
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with (args.output_dir / "epochs.jsonl").open("a" if args.resume else "x", buffering=1) as epochs, \
         (args.output_dir / "batch_access.jsonl").open("a" if args.resume else "x", buffering=1) as batches:
        observer = lambda row: batches.write(json.dumps(row, allow_nan=False) + "\n")
        for epoch in range(start, config["epochs"]):
            before_hash = teacher.state_sha256(student.state_dict())
            before_parameter_hash = teacher.state_sha256(dict(student.named_parameters()))
            ratio = curriculum_ratio(config, epoch)
            ids = mask_runner.curriculum_training_ids(training["dialogue_ids"], difficulty, ratio)
            if not ids:
                raise ValueError("empty curriculum stage")
            train = train_epoch(models, role_loader(training, ids, config, epoch), optimizer, criterion,
                                ids, device, epoch, config, observer)
            # Isolate Development RNG consumption from the continuous Training stream.
            saved_rng = teacher.rng_state()
            try:
                dev = development_epoch(student, role_loader(development, development["dialogue_ids"], config, epoch),
                                        development["dialogue_ids"], device, epoch, observer)
            finally:
                restore_rng(saved_rng)
            after_hash = teacher.state_sha256(student.state_dict())
            after_parameter_hash = teacher.state_sha256(dict(student.named_parameters()))
            if epoch == 0:
                if before_parameter_hash == after_parameter_hash or train["optimizer_steps"] <= 0 or not dev["exact_development_coverage"]:
                    raise RuntimeError("first real epoch gate failed")
                selected_unknown = sum(int((~training["dialogues"][key]["supervision_mask"]).sum()) for key in ids)
                if selected_unknown and train["direct_unknown_gradient_audit"] is None:
                    raise RuntimeError("first real epoch did not audit its unknown supervision")
                gate_passed = True
                (args.output_dir / "first_epoch_gate.json").write_text(json.dumps(dict(
                    status="passed", epoch=1, full_real_epoch=True, continues_same_optimizer_rng=True,
                    before_state_sha256=before_hash, after_state_sha256=after_hash,
                    before_trainable_parameters_sha256=before_parameter_hash,
                    after_trainable_parameters_sha256=after_parameter_hash,
                    all_reported_loss_components_finite=True, difficulty_exact_training_membership=True,
                    selected_curriculum_exact_coverage=True, ratio=ratio,
                    capacity_bound_respected=train["max_batch_edges"] <= config["edge_capacity"] and train["max_batch_nodes"] <= config["node_capacity"],
                    train=train, development=dev), indent=2, allow_nan=False) + "\n")
            if not gate_passed:
                raise RuntimeError("first epoch gate missing")
            best_score, best_epoch, improved = update_selection(dev["weighted_f1_percent_rounded"], epoch + 1, best_score, best_epoch)
            torch.cuda.synchronize(device)
            record = dict(epoch=epoch + 1, ratio=ratio, train=train, development=dev,
                          selected=improved, best_epoch=best_epoch, best_score=best_score,
                          model_state_sha256=after_hash, elapsed_seconds=time.monotonic() - started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            epochs.write(json.dumps(record, allow_nan=False) + "\n")
            checkpoint = dict(schema="independent_cmerc_complete_checkpoint_v1", model_state_dict=student.state_dict(),
                              optimizer_state_dict=optimizer.state_dict(), rng_state=teacher.rng_state(),
                              epoch=epoch + 1, best_score=best_score, best_epoch=best_epoch,
                              first_epoch_gate_passed=gate_passed, config=config, bindings=bindings,
                              provenance=provenance, model_state_sha256=after_hash,
                              dataset=training["dataset"], fold=training["fold"], selected_epoch=best_epoch,
                              selection_metric="development_weighted_f1", selection_role="development",
                              is_best=improved, training_complete=epoch + 1 == config["epochs"],
                              complete=epoch + 1 == config["epochs"])
            atomic_save(checkpoint, args.output_dir / "latest.pt")
            if improved:
                atomic_save(checkpoint, args.output_dir / "best.pt")
            (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
    best = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    best.update(training_complete=True, complete=True, is_best=True, selected_epoch=best_epoch)
    atomic_save(best, args.output_dir / "best.pt")
    completion = dict(status="completed", dataset=training["dataset"], fold=training["fold"],
                      epochs=config["epochs"], best_epoch=best_epoch, best_score=best_score,
                      roles_accessed=["training", "development"], first_epoch_gate_passed=gate_passed,
                      hashes={name: file_sha256(args.output_dir / name) for name in
                              ("best.pt", "latest.pt", "epochs.jsonl", "batch_access.jsonl", "first_epoch_gate.json")})
    (args.output_dir / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")


if __name__ == "__main__":
    main()
