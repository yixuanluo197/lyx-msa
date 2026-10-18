#!/usr/bin/env python3
"""Persistent original CMERC queue: one experiment per idle healthy host GPU.

Run under scripts/with_cmerc_env.sh on local user23. The scheduler can be restarted:
live children are attached by PID/start-time/output path, never submitted twice.
GPU flock descriptors are inherited by children so a scheduler interruption does
not release a still-running experiment's reservation. Failed runs stay failed;
their files are preserved and no automatic retries are made.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_HOSTNAME = "user23"
GPU_UUIDS = {
    0: "GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248",
    1: "GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a",
    3: "GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456",
}
SEEDS = [1475, 67137, 2024, 3407, 42]
XP_GPU_NAMES = {"TITAN Xp", "NVIDIA TITAN Xp"}
GPU_FIELDS = (
    "index", "uuid", "memory.total", "memory.used", "utilization.gpu", "name", "memory.free",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_source_snapshot(root=ROOT):
    root = Path(root).resolve()
    snapshot = read_json(root / "snapshots/source_snapshot.json")
    files = snapshot["files"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != snapshot["sha256"]:
        raise ValueError("Source snapshot manifest hash is invalid")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source snapshot path escapes experiment root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Executable source changed after freeze: {name}")
    immutable_manifest = root.parent / "provenance/immutable_inputs.json"
    if immutable_manifest.is_file():
        for name, expected in read_json(immutable_manifest).items():
            path = Path(name)
            if not path.is_absolute():
                path = root.parent / path
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Immutable input/tool changed after freeze: {name}")
    return snapshot["sha256"]


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    total_mb: int
    used_mb: int
    utilization: int
    pids: tuple[int, ...] = ()
    name: str = "TITAN Xp"
    free_mb: int | None = None

    def idle_healthy(self, allowed, minimum_free_mb=12000, maximum_utilization=10, allowed_uuids=None):
        uuids = GPU_UUIDS if allowed_uuids is None else {int(index): uuid for index, uuid in allowed_uuids.items()}
        return (
            self.index in allowed and self.index in GPU_UUIDS and not self.pids
            and self.uuid == GPU_UUIDS[self.index] == uuids.get(self.index)
            and self.name.strip() in XP_GPU_NAMES
            and (self.free_mb if self.free_mb is not None else self.total_mb - self.used_mb) >= minimum_free_mb
            and self.utilization <= maximum_utilization
        )


def query_gpus():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    applications = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    pids = {}
    for fields in csv.reader(applications.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 2:
            raise RuntimeError("Unrecognized nvidia-smi compute process row")
        pids.setdefault(fields[0].strip(), []).append(int(fields[1]))
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != len(GPU_FIELDS):
            raise RuntimeError("Unrecognized nvidia-smi GPU health row")
        devices.append(GPU(int(row[0]), row[1].strip(), int(row[2]), int(row[3]),
                           int(row[4]), tuple(pids.get(row[1].strip(), [])), row[5].strip(), int(row[6])))
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return devices


def validate_config(config):
    seeds = config["first_seeds"] + config["additional_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique pre-registered nonnegative integers")
    if seeds != SEEDS:
        raise ValueError(f"XP validation requires the fixed five seeds {SEEDS}")
    if config.get("causal") is not False or config["selection_protocol"] != "validation":
        raise ValueError("This queue requires causal=false and validation selection")
    if config["rng_policy"] != "continuous":
        raise ValueError("Multi-seed queue requires continuous RNG")
    if config.get("protocol_version") != 2 or config.get("batch_policy") != "capacity_safe":
        raise ValueError("V2 multi-seed queue requires capacity_safe batch policy")
    if any(config.get(name) != value for name, value in (
            ("edge_capacity", 1000), ("node_capacity", 5200), ("batch_max_retries", 10000))):
        raise ValueError("V2 graph capacity limits must remain 1000 edges / 5200 nodes / 10000 retries")
    if config["split_seed"] != 1475:
        raise ValueError("Pre-registered IEMOCAP split seed must remain 1475")
    if config["datasets"] != ["IEMOCAP", "MELD"] or config["variants"] != ["baseline", "cmerc"]:
        raise ValueError("Paired matrix must include both datasets and both variants")
    if config["epochs"] != {"IEMOCAP": 80, "MELD": 40}:
        raise ValueError("Formal epoch budget must remain 80/40")
    if config["allowed_gpu_indices"] != [0, 1, 3]:
        raise ValueError("XP GPU whitelist must be exactly host indices 0/1/3")
    if {int(index): uuid for index, uuid in config.get("allowed_gpu_uuids", {}).items()} != GPU_UUIDS:
        raise ValueError("XP GPU UUID whitelist differs from the approved local devices")
    if type(config["minimum_free_gpu_mb"]) not in (int, float) or not 0 < config["minimum_free_gpu_mb"] <= 12180:
        raise ValueError("minimum_free_gpu_mb must be positive and <=12180 MiB")
    environment = config.get("runtime_environment", {})
    if not all(environment.get(key) and Path(environment[key]).is_absolute()
               for key in ("CMERC_ENV_PREFIX", "CMERC_PYTHON")):
        raise ValueError("runtime_environment requires absolute CMERC_ENV_PREFIX and CMERC_PYTHON")
    if any(key.startswith("CUDA_VISIBLE_DEVICES") for key in environment):
        raise ValueError("GPU visibility is owned by the scheduler")


def build_jobs(config, run_root, log_root, result_root, smoke=False):
    jobs = []
    anchors = {("IEMOCAP", 1475), ("MELD", 67137)}
    for stage in (1, 2):
        for seed in config["first_seeds"] + config["additional_seeds"]:
            for dataset in config["datasets"]:
                if (stage == 1) != ((dataset, seed) in anchors):
                    continue
                for variant in config["variants"]:
                    job_id = f"{dataset}_{variant}_validation_seed{seed}"
                    logs = Path(log_root) / variant / dataset / "validation"
                    jobs.append({
                        "id": job_id, "stage": stage, "dataset": dataset, "variant": variant,
                        "seed": seed, "split_seed": config["split_seed"],
                        "batch_policy": config["batch_policy"],
                        "edge_capacity": config["edge_capacity"], "node_capacity": config["node_capacity"],
                        "batch_max_retries": config["batch_max_retries"],
                        "epochs": 1 if smoke else config["epochs"][dataset],
                        "run_dir": str(Path(run_root) / variant / f"{dataset}_validation_seed{seed}"),
                        "result_dir": str(Path(result_root) / variant / dataset / "validation" / f"seed{seed}"),
                        "train_log": str(logs / f"seed{seed}.train.log"),
                        "evaluation_log": str(logs / f"seed{seed}.evaluation.log"),
                        "status": "pending",
                    })
    # Smoke covers paired variants on both real datasets, one epoch each.
    return jobs[:4] if smoke else jobs


def training_command(job):
    return ["bash", str(ROOT / "scripts/with_cmerc_env.sh"), "bash",
            str(ROOT / "scripts/run_training.sh"), job["dataset"], job["variant"], "validation",
            "--seed", str(job["seed"]), "--split-seed", str(job["split_seed"]),
            "--rng-policy", "continuous", "--epochs", str(job["epochs"]),
            "--batch-policy", job["batch_policy"], "--edge-capacity", str(job["edge_capacity"]),
            "--node-capacity", str(job["node_capacity"]), "--batch-max-retries", str(job["batch_max_retries"]),
            "--num-workers", "0", "--require-noncausal", "--output-dir", str(Path(job["run_dir"]).parent)]


def evaluation_command(job, python=None):
    return ["bash", str(ROOT / "scripts/with_cmerc_env.sh"), python or os.environ.get("CMERC_PYTHON", sys.executable), "-u",
            str(ROOT / "scripts/evaluate_checkpoint.py"), "--dataset", job["dataset"],
            "--checkpoint", str(Path(job["run_dir"]) / "best.pt"),
            "--output-dir", job["result_dir"], "--device", "cuda:0",
            "--batch-size", "16", "--num-workers", "0", "--require-noncausal"]


def output_guard(job):
    for name in ("run_dir", "result_dir", "train_log", "evaluation_log"):
        if os.path.lexists(job[name]):
            raise FileExistsError(f"Refusing to overwrite {name}: {job[name]}")
    for name in ("train_log", "evaluation_log"):
        if os.path.lexists(memory_profile_path(job[name])):
            raise FileExistsError(f"Refusing to overwrite memory profile: {memory_profile_path(job[name])}")


def memory_profile_path(log_path):
    return Path(str(log_path) + ".memory.jsonl")


def pid_identity(pid):
    try:
        directory = Path("/proc") / str(pid)
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {
            "start_ticks": stat[19],
            "command": (directory / "cmdline").read_bytes().decode().split("\0")[:-1],
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def command_output_path(command):
    for index, value in enumerate(command):
        if value == "--output-dir" and index + 1 < len(command):
            return str(Path(command[index + 1]).resolve())
        if value.startswith("--output-dir="):
            return str(Path(value.split("=", 1)[1]).resolve())
    return None


def matching_live_pid(job, phase):
    target = str((Path(job["run_dir"]).parent if phase == "training" else Path(job["result_dir"])).resolve())
    script = "train_cm.py" if phase == "training" else "evaluate_checkpoint.py"
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        identity = pid_identity(int(directory.name))
        if identity:
            command = identity["command"]
            direct_script = any(Path(part).name == script for part in command)
            wrapper_matches = False
            if phase == "training":
                wrapper_indices = [index for index, part in enumerate(command) if Path(part).name == "run_training.sh"]
                wrapper_matches = any(command[index + 1:index + 4] == [job["dataset"], job["variant"], "validation"] for index in wrapper_indices)
            if not direct_script and not wrapper_matches:
                continue
            seed_matches = phase != "training" or (
                argument_value(command, "--seed") == str(job["seed"])
                and (wrapper_matches or (
                    argument_value(command, "--Dataset") == job["dataset"]
                    and argument_value(command, "--selection-protocol") == "validation")))
            if seed_matches and command_output_path(command) == target:
                matches.append((int(directory.name), identity))
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate live processes for {job['id']} {phase}: {matches}")
    return matches[0] if matches else None


def argument_value(command, name):
    value = None
    for index, part in enumerate(command):
        if part == name and index + 1 < len(command):
            value = command[index + 1]
        elif part.startswith(name + "="):
            value = part.split("=", 1)[1]
    return value


def assert_finite(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite/missing {description}: {value!r}")


def validate_training(job, inspect_checkpoints=True):
    run_dir = Path(job["run_dir"])
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != job["epochs"] or [record["epoch"] for record in records] != list(range(1, job["epochs"] + 1)):
        raise ValueError(f"Incomplete/sequential epoch budget for {job['id']}")
    for record in records:
        if record.get("causal") is not False or record["selection_protocol"] != "validation" or record["selection_split"] != "validation":
            raise ValueError("Incorrect causal/selection protocol")
        if any(record.get(key) is not None for key in ("test_accuracy", "test_weighted_f1")):
            raise ValueError("Test metrics were inspected during validation training")
        for key in ("classification_loss", "cc_loss", "mscl_loss", "cscl_loss", "total_loss",
                    "train_accuracy", "train_weighted_f1", "validation_accuracy", "validation_weighted_f1",
                    "selection_weighted_f1", "elapsed_seconds", "max_gpu_memory_mb"):
            assert_finite(record.get(key), key)
        for key, upper_bound in (("sampler_retries", job["batch_max_retries"]),
                                 ("max_batch_edges", job["edge_capacity"]),
                                 ("max_batch_nodes", job["node_capacity"])):
            value = record.get(key)
            if type(value) is not int or value < 0 or value > upper_bound:
                raise ValueError(f"Missing/out-of-capacity {key}: {value!r}")
    manifest = read_json(run_dir / "run_manifest.json")
    args = manifest["args"]
    expected_args = {"seed": job["seed"], "split_seed": job["split_seed"], "Dataset": job["dataset"],
                     "rng_policy": "continuous", "rnn_backend": "aten", "selection_protocol": "validation", "causal": False,
                     "base_model": "GRU", "batch_size": 16, "lr": 0.0001, "norm": "BN",
                     "graph_type": "hyper", "graph_construct": "direct", "mm_fusion_mthd": "concat_DHT",
                     "modals": "avl", "num_L": 3, "num_K": 4 if job["dataset"] == "IEMOCAP" else 3,
                     "dropout": 0.5 if job["dataset"] == "IEMOCAP" else 0.4,
                     "epochs": job["epochs"], "calibrate": job["variant"] == "cmerc",
                     "batch_policy": job["batch_policy"], "edge_capacity": job["edge_capacity"],
                     "node_capacity": job["node_capacity"], "batch_max_retries": job["batch_max_retries"],
                     "contrastlearning": job["variant"] == "cmerc", "courselearning": job["variant"] == "cmerc"}
    if job["variant"] == "cmerc":
        expected_args.update({"rank_coff": 0.005 if job["dataset"] == "IEMOCAP" else 0.002,
                              "mscl_coff": 0.05 if job["dataset"] == "IEMOCAP" else 0.15,
                              "cscl_coff": 0.05 if job["dataset"] == "IEMOCAP" else 0.15,
                              "epoch_ratio": 0.15 if job["dataset"] == "IEMOCAP" else 0.4,
                              "scheduler_steps": 1})
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise ValueError(f"Manifest {key} differs: {args.get(key)!r} vs {expected!r}")
    split = manifest["split"]
    groups = [set(split[name]) for name in ("train_ids", "validation_ids", "test_ids")]
    if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Data split is empty or overlaps")
    expected_counts = [108, 12, 31] if job["dataset"] == "IEMOCAP" else [1038, 114, 280]
    if [len(group) for group in groups] != expected_counts:
        raise ValueError("Unexpected canonical split counts")
    if not manifest.get("source") or not manifest.get("data") or not split.get("hash"):
        raise ValueError("Missing source/data/split provenance")
    final_test = read_json(run_dir / "final_test.json")
    for key in ("seed", "split_seed", "rng_policy", "selection_protocol", "causal"):
        if final_test.get(key) != args[key]:
            raise ValueError(f"Final test {key} differs from manifest")
    if final_test.get("split_hash") != split["hash"]:
        raise ValueError("Final test split hash differs from manifest")
    for key in ("test_loss", "test_accuracy", "test_weighted_f1"):
        assert_finite(final_test.get(key), f"final_test.{key}")
    best_record = max(records, key=lambda record: record["selection_weighted_f1"])
    if final_test.get("selected_epoch", final_test.get("best_epoch")) != best_record["epoch"]:
        raise ValueError("Final test did not use the validation-selected epoch")
    if inspect_checkpoints:
        import torch
        for name in ("latest.pt", "best.pt"):
            checkpoint = torch.load(run_dir / name, map_location="cpu")
            for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "args", "provenance"):
                if key not in checkpoint:
                    raise ValueError(f"Incomplete {name}: missing {key}")
            expected_epoch = job["epochs"] - 1 if name == "latest.pt" else best_record["epoch"] - 1
            if checkpoint["epoch"] != expected_epoch or any(checkpoint["args"].get(key) != value for key, value in args.items()):
                raise ValueError(f"Incorrect {name} epoch or full configuration")
            if checkpoint["provenance"] != manifest["provenance"]:
                raise ValueError(f"Incorrect {name} source/data/split provenance")
    return manifest


def validate_evaluation(job):
    result_dir = Path(job["result_dir"])
    metrics = read_json(result_dir / "metrics.json")
    if metrics.get("causal") is not False or metrics["dataset"] != job["dataset"]:
        raise ValueError("Incorrect evaluation dataset/causal protocol")
    if Path(metrics["checkpoint"]).resolve() != (Path(job["run_dir"]) / "best.pt").resolve():
        raise ValueError("Evaluation used a different checkpoint")
    manifest = read_json(Path(job["run_dir"]) / "run_manifest.json")
    final_test = read_json(Path(job["run_dir"]) / "final_test.json")
    if metrics.get("selected_epoch") != final_test["selected_epoch"]:
        raise ValueError("Independent evaluation used a different selected epoch")
    for key in ("seed", "split_seed", "rng_policy", "selection_protocol"):
        if metrics.get(key) != manifest["args"][key]:
            raise ValueError(f"Independent evaluation {key} differs from training")
    if metrics.get("split_hash") != manifest["split"]["hash"] or metrics.get("provenance") != manifest["provenance"]:
        raise ValueError("Independent evaluation source/data/split provenance differs")
    if any(metrics.get("training_args", {}).get(key) != value for key, value in manifest["args"].items()):
        raise ValueError("Independent evaluation full training configuration differs")
    if metrics["num_samples"] != (1623 if job["dataset"] == "IEMOCAP" else 2610):
        raise ValueError("Incomplete test predictions")
    for key in ("accuracy", "weighted_f1", "macro_f1"):
        assert_finite(metrics.get(key), key)
    assert_finite(metrics["cel"].get("total"), "CEL")
    assert_finite(metrics["calibration"].get("ece"), "ECE")
    for name in ("predictions.npz", "predictions.csv"):
        if not (result_dir / name).is_file():
            raise ValueError(f"Missing {name}")
    import numpy as np
    with np.load(result_dir / "predictions.npz", allow_pickle=False) as predictions:
        for name in predictions.files:
            if np.issubdtype(predictions[name].dtype, np.number) and not np.isfinite(predictions[name]).all():
                raise ValueError(f"Nonfinite predictions {name}")
    return metrics


def validate_stage(jobs, inspect_checkpoints=True):
    splits = {}
    paired = {}
    sources = set()
    data_hashes = {}
    for job in jobs:
        if job["status"] != "completed":
            raise ValueError(f"Stage contains incomplete/failed run {job['id']}")
        manifest = validate_training(job, inspect_checkpoints=inspect_checkpoints)
        validate_evaluation(job)
        sources.add(json.dumps(manifest["source"], sort_keys=True))
        for name, digest in manifest["data"].items():
            previous_digest = data_hashes.setdefault(name, digest)
            if previous_digest != digest:
                raise ValueError(f"Input feature/difficulty SHA changed between runs: {name}")
        signature = json.dumps(manifest["split"], sort_keys=True)
        previous = splits.setdefault(job["dataset"], signature)
        if previous != signature:
            raise ValueError("Different training/validation/test IDs across seeds or paired variants")
        paired.setdefault((job["dataset"], job["seed"]), set()).add(job["variant"])
    if any(variants != {"baseline", "cmerc"} for variants in paired.values()):
        raise ValueError("Missing paired Baseline/CMERC")
    if len(sources) != 1:
        raise ValueError("Source files/base commit changed between runs")
    return {"passed_at": utc_now(), "runs": len(jobs), "split_signatures": {
        dataset: hashlib.sha256(signature.encode()).hexdigest() for dataset, signature in splits.items()},
        "criterion": "complete, finite, paired, fixed split, noncausal, validation only; no score threshold"}


class Scheduler:
    def __init__(self, config, jobs, state_path, gpu_lock_root, smoke=False):
        if socket.gethostname() != REQUIRED_HOSTNAME:
            raise ValueError("XP queue must run on local user23; do not migrate to a different server")
        self.config = config
        self.state_path = Path(state_path)
        self.gpu_lock_root = Path(gpu_lock_root)
        self.smoke = smoke
        self.processes = {}
        self.gpu_locks = {}
        self.stop = False
        signature = hashlib.sha256(json.dumps({"config": config, "jobs": jobs}, sort_keys=True).encode()).hexdigest()
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state["matrix_hash"] != signature:
                raise ValueError("Existing queue state belongs to a different preregistered matrix")
            if self.state["server"] != socket.gethostname():
                raise ValueError("Existing queue is assigned to a different server; do not migrate live runs")
        else:
            self.state = {"matrix_hash": signature, "server": socket.gethostname(), "created_at": utc_now(),
                          "config": config, "smoke": smoke, "jobs": jobs, "phase": "first_seeds"}
            self.save()

    def save(self):
        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)

    def summarize(self, seeds, first_stage=False, require_complete=False):
        if self.smoke:
            return
        run_root = Path(self.state["jobs"][0]["run_dir"]).parent.parent
        result_root = Path(self.state["jobs"][0]["result_dir"]).parents[3]
        output = Path(self.config.get("statistics_root", result_root)) / "summary.json"
        report = Path(self.config.get("summary_report", ROOT.parent / "reports/XP_MAINLINE_REPRODUCTION.md"))
        command = [sys.executable, str(ROOT / "scripts/summarize_results.py"), "--multiseed",
                   "--seeds", ",".join(str(seed) for seed in SEEDS),
                   "--run-root", str(run_root), "--result-root", str(result_root),
                   "--output", str(output), "--report", str(report)]
        historical = self.config.get("historical_reference")
        if historical and Path(historical).is_file():
            command += ["--historical", str(historical)]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        summary_record = {"command": command, "updated_at": utc_now(), "exit_code": process.returncode,
                          "stdout": process.stdout[-4000:], "stderr": process.stderr[-4000:],
                          "output": str(output)}
        self.state["first_seed_summary" if first_stage else "last_summary"] = summary_record
        self.save()
        if process.returncode != 0:
            if require_complete:
                raise RuntimeError(f"Summary failed: {process.stderr[-2000:]}")
            return
        if require_complete and read_json(output).get("complete") is not True:
            raise RuntimeError("Generated summary did not certify a complete valid paired matrix")

    def check_disk(self):
        free = shutil.disk_usage(ROOT).free
        if free < self.config["minimum_free_disk_gib"] * 1024 ** 3:
            raise RuntimeError(f"Insufficient disk: {free / 1024 ** 3:.2f} GiB free; >=6 GiB required")

    def acquire_gpu(self, excluded=()):
        for gpu in query_gpus():
            if gpu.index in excluded or not gpu.idle_healthy(
                    self.config["allowed_gpu_indices"], self.config["minimum_free_gpu_mb"],
                    self.config["maximum_idle_utilization"], self.config["allowed_gpu_uuids"]):
                continue
            lock = acquire_lock(self.gpu_lock_root / f"cmerc-{gpu.uuid}.lock")
            if lock is None:
                continue
            # GPU processes and health may change between the query and lock.
            refreshed = next((item for item in query_gpus() if item.uuid == gpu.uuid), None)
            if refreshed and refreshed.index == gpu.index and refreshed.idle_healthy(
                    self.config["allowed_gpu_indices"], self.config["minimum_free_gpu_mb"],
                    self.config["maximum_idle_utilization"], self.config["allowed_gpu_uuids"]):
                return refreshed, lock
            lock.close()
        return None

    def start(self, job, phase, gpu, lock):
        self.check_disk()
        self.state["source_snapshot_sha256"] = verify_source_snapshot()
        if phase == "training":
            output_guard(job)
        else:
            if any(os.path.lexists(path) for path in (
                    job["result_dir"], job["evaluation_log"], memory_profile_path(job["evaluation_log"]))):
                raise FileExistsError(f"Evaluation output already exists: {job['id']}")
        command = training_command(job) if phase == "training" else evaluation_command(job, self.config["runtime_environment"]["CMERC_PYTHON"])
        log_path = Path(job["train_log" if phase == "training" else "evaluation_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(self.config["runtime_environment"])
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        env["CMERC_HOST_GPU_INDEX"] = str(gpu.index)
        env["CMERC_HOST_GPU_UUID"] = gpu.uuid
        env["CMERC_PROFILE_OUTPUT"] = str(memory_profile_path(log_path))
        refreshed = next((device for device in query_gpus() if device.uuid == gpu.uuid), None)
        if refreshed is None or not refreshed.idle_healthy(
                self.config["allowed_gpu_indices"], self.config["minimum_free_gpu_mb"],
                self.config["maximum_idle_utilization"], self.config["allowed_gpu_uuids"]):
            raise RuntimeError("GPU became busy or failed identity/resource checks before launch")
        with log_path.open("x", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        self.processes[job["id"]] = process
        self.gpu_locks[job["id"]] = lock
        identity = pid_identity(process.pid)
        job.update({"status": f"running_{phase}", "phase": phase, "pid": process.pid,
                    "pid_start_ticks": identity["start_ticks"] if identity else None,
                    "command": command, "host_gpu_index": gpu.index, "gpu_uuid": gpu.uuid,
                    f"{phase}_pid": process.pid, f"{phase}_gpu_uuid": gpu.uuid,
                    f"{phase}_host_gpu_index": gpu.index,
                    f"{phase}_gpu_free_mb_at_launch": refreshed.free_mb,
                    "server": socket.gethostname(), f"{phase}_started_at": utc_now()})
        job.setdefault("started_at", job[f"{phase}_started_at"])
        self.save()
        print(f"Started {job['id']} {phase}: host GPU {gpu.index}, PID {process.pid}", flush=True)

    def attach(self, job):
        phase = job.get("phase", "training")
        match = matching_live_pid(job, phase)
        if not match:
            return False
        pid, identity = match
        if job.get("pid_start_ticks") and (job["pid"] != pid or job["pid_start_ticks"] != identity["start_ticks"]):
            raise RuntimeError("Recorded PID identity does not match live process")
        try:
            env = dict(item.split("=", 1) for item in (Path("/proc") / str(pid) / "environ").read_bytes().decode().split("\0") if "=" in item)
            visible = env["CUDA_VISIBLE_DEVICES"]
            devices = query_gpus()
            gpu = next((device for device in devices if device.uuid == visible
                        or (visible.isdigit() and device.index == int(visible))), None)
            if gpu is None:
                raise ValueError("visible GPU is absent from the local inventory")
            index = gpu.index
        except (OSError, KeyError, ValueError) as error:
            raise RuntimeError("Cannot identify attached process's host GPU") from error
        if (index not in self.config["allowed_gpu_indices"] or GPU_UUIDS.get(index) != gpu.uuid
                or self.config["allowed_gpu_uuids"].get(str(index), self.config["allowed_gpu_uuids"].get(index)) != gpu.uuid
                or gpu.name.strip() not in XP_GPU_NAMES):
            raise RuntimeError("Live run is on an unapproved GPU")
        if set(gpu.pids) - {pid}:
            raise RuntimeError("Attached process conflicts with another GPU task")
        if job.get("gpu_uuid") and job["gpu_uuid"] != gpu.uuid:
            raise RuntimeError("Attached GPU UUID differs from the recorded run")
        job.update({"status": f"running_{phase}", "phase": phase, "pid": pid,
                    "pid_start_ticks": identity["start_ticks"], "command": identity["command"],
                    "host_gpu_index": index, "gpu_uuid": gpu.uuid, "server": socket.gethostname(),
                    f"{phase}_pid": pid, f"{phase}_gpu_uuid": gpu.uuid, f"{phase}_host_gpu_index": index,
                    "attached_at": utc_now()})
        self.save()
        print(f"Attached {job['id']} {phase}: host GPU {index}, PID {pid}", flush=True)
        return True

    def release(self, job):
        lock = self.gpu_locks.pop(job["id"], None)
        if lock:
            lock.close()
        self.processes.pop(job["id"], None)

    def fail(self, job, error):
        job.update({"status": "failed", "error": str(error), "ended_at": utc_now()})
        self.release(job)
        self.save()
        print(f"Failed {job['id']}: {error}; artifacts retained, no automatic retry", flush=True)
        self.summarize(self.config["first_seeds"] + self.config["additional_seeds"])

    def wait_for_evaluation_gpu(self, job, attempts=15, poll_seconds=1):
        """Keep the lease while the last training utilization sample expires."""
        uuid, index = job["gpu_uuid"], job["host_gpu_index"]
        for attempt in range(1, attempts + 1):
            device = next((gpu for gpu in query_gpus() if gpu.uuid == uuid), None)
            if device is None or device.index != index or GPU_UUIDS.get(index) != uuid:
                raise RuntimeError("Reserved GPU UUID/index changed before independent evaluation")
            passed = device.idle_healthy(
                self.config["allowed_gpu_indices"], self.config["minimum_free_gpu_mb"],
                self.config["maximum_idle_utilization"], self.config["allowed_gpu_uuids"])
            job["evaluation_idle_wait"] = {"attempts": attempt, "passed": passed,
                                            "checked_at": utc_now(), "last_gpu": device.__dict__}
            self.save()
            if passed:
                return device
            if attempt < attempts:
                time.sleep(poll_seconds)
        raise RuntimeError(f"GPU remained blocked before evaluation after {attempts} checks: "
                           f"{job['evaluation_idle_wait']['last_gpu']}")

    def refresh(self, job):
        process = self.processes.get(job["id"])
        if process is not None:
            exit_code = process.poll()
            if exit_code is None:
                return
        else:
            identity = pid_identity(job["pid"])
            if identity and identity["start_ticks"] == job["pid_start_ticks"]:
                return
            exit_code = None  # Adopted orphan exit code cannot be reconstructed.
        phase = job["phase"]
        job[f"{phase}_ended_at"] = utc_now()
        job[f"{phase}_exit_code"] = exit_code
        job["exit_code"] = exit_code
        if exit_code is None:
            job["adopted_exit_code_unavailable"] = True
        try:
            if exit_code not in (0, None):
                raise RuntimeError(f"{phase} exited {exit_code}")
            if phase == "training":
                validate_training(job)
                # Retain same-GPU reservation for independent evaluation.
                lock = self.gpu_locks.get(job["id"])
                if lock is None:
                    lock = acquire_lock(self.gpu_lock_root / f"cmerc-{job['gpu_uuid']}.lock")
                    if lock is None:
                        raise RuntimeError("Cannot reserve attached run GPU for evaluation")
                    self.gpu_locks[job["id"]] = lock
                gpu = self.wait_for_evaluation_gpu(job)
                self.start(job, "evaluation", gpu, lock)
            else:
                validate_evaluation(job)
                job.update({"status": "completed", "ended_at": utc_now()})
                self.release(job)
                self.save()
                print(f"Completed {job['id']}", flush=True)
                self.summarize(self.config["first_seeds"] + self.config["additional_seeds"])
        except Exception as error:
            self.fail(job, error)

    def run(self):
        for job in self.state["jobs"]:
            if job["status"].startswith("running_"):
                try:
                    if not self.attach(job):
                        self.refresh(job)
                except Exception as error:
                    self.fail(job, error)
        while not self.stop:
            active = [job for job in self.state["jobs"] if job["status"].startswith("running_")]
            for job in active:
                self.refresh(job)
            stage_one = [job for job in self.state["jobs"] if job["stage"] == 1]
            if all(job["status"] in ("completed", "failed") for job in stage_one):
                if any(job["status"] == "failed" for job in stage_one):
                    self.state["phase"] = "first_seed_gate_failed"
                    self.save()
                    return 1
                if not self.smoke and "first_seed_gate" not in self.state:
                    try:
                        gate = validate_stage(stage_one)
                        self.state["first_seed_gate"] = gate
                        self.summarize(SEEDS)
                    except Exception as error:
                        self.state.update({"phase": "first_seed_gate_failed", "gate_error": str(error)})
                        self.save()
                        return 1
                    self.state["phase"] = "additional_seeds"
                    self.save()
            if all(job["status"] in ("completed", "failed") for job in self.state["jobs"]):
                if any(job["status"] == "failed" for job in self.state["jobs"]):
                    self.state["phase"] = "failed"
                    self.save()
                    return 1
                if not self.smoke:
                    try:
                        self.state["final_gate"] = validate_stage(self.state["jobs"])
                        self.summarize(self.config["first_seeds"] + self.config["additional_seeds"], require_complete=True)
                    except Exception as error:
                        self.state.update({"phase": "final_gate_failed", "gate_error": str(error)})
                        self.save()
                        return 1
                else:
                    try:
                        self.state["smoke_gate"] = validate_stage(self.state["jobs"])
                    except Exception as error:
                        self.state.update({"phase": "smoke_gate_failed", "gate_error": str(error)})
                        self.save()
                        return 1
                self.state.update({"phase": "completed", "completed_at": utc_now()})
                self.save()
                return 0
            allowed_stage = 2 if "first_seed_gate" in self.state else 1
            for job in self.state["jobs"]:
                if job["status"] != "pending" or job["stage"] > allowed_stage:
                    continue
                try:
                    # Attach before looking at output files, including the period
                    # before epoch 1/checkpoint files have been emitted.
                    if self.attach(job):
                        continue
                    output_guard(job)
                    excluded = [item["host_gpu_index"] for item in self.state["jobs"]
                                if item["status"].startswith("running_")]
                    allocation = self.acquire_gpu(excluded)
                    if allocation is None:
                        break
                    gpu, lock = allocation
                    try:
                        self.start(job, "training", gpu, lock)
                    except Exception:
                        lock.close()
                        raise
                except Exception as error:
                    self.fail(job, error)
            time.sleep(self.config["poll_seconds"])
        self.state["scheduler_stopped_at"] = utc_now()
        self.save()
        return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/multiseed_validation.json")
    parser.add_argument("--run-root", type=Path, default=ROOT.parent / "runs")
    parser.add_argument("--log-root", type=Path, default=ROOT.parent / "logs")
    parser.add_argument("--result-root", type=Path, default=ROOT.parent / "results")
    parser.add_argument("--state", type=Path, default=ROOT.parent / "state/queue.json")
    parser.add_argument("--report", type=Path, help="default: batch/reports/XP_MAINLINE_REPRODUCTION.md")
    parser.add_argument("--statistics-root", type=Path, help="summary output directory; default: result-root")
    parser.add_argument("--historical", type=Path, help="optional existing historical summary")
    parser.add_argument("--gpu-lock-root", type=Path, default=Path("/tmp/cmerc-gpu-locks"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="four paired dataset/variant runs, one real epoch each, isolated outputs")
    parser.add_argument("--smoke-id", default="scheduler_validation_v2")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    for name, value in (("summary_report", args.report), ("statistics_root", args.statistics_root),
                        ("historical_reference", args.historical)):
        if value is not None:
            config[name] = str(value.resolve())
    validate_config(config)
    if args.smoke:
        if Path(args.smoke_id).name != args.smoke_id or args.smoke_id in (".", ".."):
            raise ValueError("smoke-id must be a single path component")
        args.run_root = args.run_root / "scheduler_smoke" / args.smoke_id
        args.log_root = args.log_root / "scheduler_smoke" / args.smoke_id
        args.result_root = args.result_root / "scheduler_smoke" / args.smoke_id
        args.state = args.state.parent / f"smoke_{args.smoke_id}.json"
    jobs = build_jobs(config, args.run_root.resolve(), args.log_root.resolve(), args.result_root.resolve(), args.smoke)
    if args.dry_run:
        for job in jobs:
            job["training_command"] = training_command(job)
            job["evaluation_command"] = evaluation_command(job, config["runtime_environment"]["CMERC_PYTHON"])
            job["memory_profiles"] = [str(memory_profile_path(job[key])) for key in ("train_log", "evaluation_log")]
        print(json.dumps({"config": config, "jobs": jobs, "gpus": [gpu.__dict__ for gpu in query_gpus()],
                          "free_disk_gib": shutil.disk_usage(ROOT).free / 1024 ** 3}, indent=2, ensure_ascii=False))
        return 0
    queue_lock = acquire_lock(args.state.with_suffix(".lock"))
    if queue_lock is None:
        raise RuntimeError("Another scheduler already owns this queue")
    scheduler = Scheduler(config, jobs, args.state, args.gpu_lock_root, args.smoke)
    scheduler.state["source_snapshot_sha256"] = verify_source_snapshot()
    scheduler.save()
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda *_: setattr(scheduler, "stop", True))
    with queue_lock:
        result = scheduler.run()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
