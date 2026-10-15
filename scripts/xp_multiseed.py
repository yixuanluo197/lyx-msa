#!/usr/bin/env python3
"""Standard-library supervisor for the predeclared local TITAN Xp batch.

Run this supervisor in tmux or under a persistent service. Failed/interrupted
tasks are never automatically resumed: train_cm.py appends its epoch records.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time
from datetime import datetime, timezone


GPU_CANDIDATES = {
    0: "GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248",
    1: "GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a",
    3: "GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456",
}
SEEDS = {"IEMOCAP": [1475, 42, 3407, 2025, 2026], "MELD": [67137, 42, 3407, 2025, 2026]}
PAPER = {"IEMOCAP": 71.98, "MELD": 66.85}
HISTORICAL = {
    "IEMOCAP": {"v100_cmerc_official_test": 70.3753, "official_checkpoint": 71.9838},
    "MELD": {"v100_cmerc_official_test": 66.1767, "official_checkpoint": 66.8484},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_matrix():
    # Both public-seed CMERC anchors and both baseline diagnostics precede
    # the remaining fixed seeds; no result can alter this order.
    pairs = [(dataset, variant, SEEDS[dataset][0])
             for variant in ("cmerc", "baseline") for dataset in SEEDS]
    pairs += [(dataset, "cmerc", seed) for seed in SEEDS["IEMOCAP"][1:] for dataset in SEEDS]
    return [{"id": f"{dataset}_{variant}_official_test_seed{seed}",
             "dataset": dataset, "variant": variant, "seed": seed,
             "protocol": "official_test", "epochs": 80 if dataset == "IEMOCAP" else 40,
             "causal": False} for dataset, variant, seed in pairs]


def expected_config(task):
    iemocap = task["dataset"] == "IEMOCAP"
    config = {
        "Dataset": task["dataset"], "seed": task["seed"], "epochs": task["epochs"],
        "selection_protocol": "official_test", "causal": False, "batch_size": 16,
        "num_workers": 0, "lr": 0.0001, "l2": 0.00003, "base_model": "GRU", "graph_type": "hyper",
        "graph_construct": "direct", "multi_modal": True, "mm_fusion_mthd": "concat_DHT",
        "modals": "avl", "norm": "BN", "num_L": 3,
        "num_K": 4 if iemocap else 3, "dropout": 0.5 if iemocap else 0.4,
        "calibrate": task["variant"] == "cmerc",
        "contrastlearning": task["variant"] == "cmerc",
        "courselearning": task["variant"] == "cmerc",
    }
    if task["variant"] == "cmerc":
        config.update(scheduler_steps=1, rank_coff=0.005 if iemocap else 0.002,
                      mscl_coff=0.05 if iemocap else 0.15,
                      cscl_coff=0.05 if iemocap else 0.15,
                      epoch_ratio=0.15 if iemocap else 0.4)
    return config


def task_paths(root, task):
    run = root / "runs" / task["variant"] / f"{task['dataset']}_official_test_seed{task['seed']}"
    result = root / "results" / task["variant"] / task["dataset"] / "official_test" / f"seed{task['seed']}"
    return run, result


def create_batch(root, code, environment, python_wrapper=None):
    root, code, environment = Path(root).resolve(), Path(code).resolve(), Path(environment).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "batch_manifest.json").exists() or (root / "state.json").exists():
        raise ValueError("batch already exists; refusing to replace its matrix or state")
    if (root / "runs").exists() or (root / "results").exists():
        raise ValueError("pre-existing run/result directories; choose a fresh batch")
    if not (code / "scripts/run_training.sh").is_file():
        raise ValueError("code-root must contain the frozen CMERC checkout")
    files = sorted(set(code.glob("*.py")) | set((code / "scripts").glob("*.py")) |
                   set((code / "scripts").glob("*.sh")) | set((code / "caldiff").glob("*.py")))
    inputs = [path for folder in ("data", "caldiff", "checkpoints")
              for path in sorted((code / folder).glob("*.pkl"))]
    hashes = {str(path.relative_to(code)): sha256(path) for path in files + inputs}
    git = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    diff = subprocess.run(["git", "-C", str(code), "diff", "HEAD", "--"],
                          capture_output=True, text=True)
    (root / "code_diff.patch").write_text(diff.stdout, encoding="utf-8")
    wrapper = Path(python_wrapper).resolve() if python_wrapper else environment / "bin/python"
    observers = [path for path in (wrapper, root / "profile_run.py") if path.is_file()]
    manifest = {"batch_id": root.name, "created_at": now(), "server": "local user23",
                "code_root": str(code), "env_prefix": str(environment),
                "training_python": str(wrapper), "require_preflight": "preflight/status.json",
                "meld_concurrency_evidence": "preflight/meld_concurrency.json",
                "initial_distinct_training_gpus": 3,
                "observer_sha256": {str(path): sha256(path) for path in observers},
                "commit": git.stdout.strip() if git.returncode == 0 else None,
                "file_sha256": hashes, "gpu_candidates": GPU_CANDIDATES,
                "tasks": task_matrix(), "seeds_origin": "first public README seed; four predeclared reproduction seeds",
                "selection_bias": "official_test selects the best checkpoint using test W-F1 each epoch",
                "curriculum": "fixed author difficulty files shared across seeds",
                "historical_reference": HISTORICAL, "paper_weighted_f1": PAPER,
                "threshold_points": 0.5, "sample_std_ddof": 1}
    atomic_json(root / "batch_manifest.json", manifest)
    save_state(root, {"updated_at": now(), "tasks": {
        task["id"]: {"status": "planned", "pid": None, "exitcode": None} for task in manifest["tasks"]}})
    (root / "logs").mkdir(exist_ok=True)
    summarize(root)
    return manifest


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        # A zombie is no longer performing training, even if kill(pid, 0) works.
        stat = Path(f"/proc/{int(pid)}/stat")
        return not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def refuse_duplicate(root, task, state):
    run, result = task_paths(root, task)
    if pid_alive(state.get("pid")):
        raise ValueError(f"live PID {state['pid']} already owns {task['id']}")
    if state["status"] != "planned":
        raise ValueError(f"{task['id']} has status {state['status']}; automatic retry is prohibited")
    if run.exists() or result.exists():
        raise ValueError(f"{task['id']} has existing artifacts; refusing append or overwrite")


def query_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    process_pids = {}
    for row in csv.reader(processes.stdout.splitlines()):
        if row:
            process_pids.setdefault(row[0].strip(), []).append(int(row[1].strip()))
    available = []
    for row in csv.reader(result.stdout.splitlines()):
        index, uuid, name, free = [part.strip() for part in row]
        if GPU_CANDIDATES.get(int(index)) != uuid:
            continue
        if "TITAN Xp" not in name:
            raise ValueError(f"candidate {uuid} is {name}, expected TITAN Xp")
        available.append({"index": int(index), "uuid": uuid, "free_mib": int(free),
                          "process_pids": process_pids.get(uuid, [])})
    return available


def meld_concurrency_policy(root):
    """Enable a second MELD only after measured total throughput improves."""
    policy = {"max_meld_per_gpu": 1, "minimum_free_mib": 3000,
              "reason": "MELD dual-task throughput has not passed the 10% gate"}
    evidence_path = Path(root) / "preflight/meld_concurrency.json"
    profile_path = Path(root) / "preflight/MELD_resource_profile.json"
    if not evidence_path.is_file() or not profile_path.is_file():
        return policy
    evidence, profile = read_json(evidence_path), read_json(profile_path)
    single = evidence.get("single_task_throughput")
    dual = evidence.get("dual_task_total_throughput")
    peak = profile.get("peak_reserved_mb")
    measurements = (single, dual, peak)
    if evidence.get("enabled") is not True or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0 for value in measurements
    ):
        return policy
    improvement = dual / single - 1
    if improvement + 1e-12 < 0.1:
        return policy
    policy.update(max_meld_per_gpu=2, minimum_free_mib=max(2200, math.ceil(peak) + 1024),
                  throughput_improvement_fraction=improvement,
                  evidence_sha256=sha256(evidence_path), profile_sha256=sha256(profile_path),
                  reason="measured dual MELD total throughput improved by at least 10%")
    return policy


def eligible_gpus(task, phase, inventory, children, policy, initial_training_uuids=()):
    """Prefer idle cards; share only with one owned MELD and enough headroom."""
    candidates = []
    for gpu in inventory:
        if GPU_CANDIDATES.get(gpu["index"]) != gpu["uuid"]:
            continue
        occupants = [child for child in children.values() if child[2]["uuid"] == gpu["uuid"]]
        own_pids = {child[0].pid for child in occupants}
        if not set(gpu.get("process_pids", [])).issubset(own_pids):
            continue
        if phase == "training" and len(initial_training_uuids) < 3:
            # Even if an early task finishes quickly, each of the first three
            # training launches must use a different healthy physical card.
            if gpu["uuid"] in initial_training_uuids or occupants:
                continue
        if occupants:
            if (task["dataset"] != "MELD" or policy["max_meld_per_gpu"] < 2
                    or len(occupants) != 1 or occupants[0][1]["dataset"] != "MELD"):
                continue
            minimum_free = policy["minimum_free_mib"]
        else:
            minimum_free = 11000 if task["dataset"] == "IEMOCAP" else 3000
        if gpu["free_mib"] >= minimum_free:
            candidates.append((len(occupants), gpu["index"], gpu))
    return [item[2] for item in sorted(candidates, key=lambda item: item[:2])]


def command_for(manifest, root, task, phase):
    code = Path(manifest["code_root"])
    wrapper = ["bash", str(code / "scripts/with_cmerc_env.sh")]
    if phase == "training":
        return wrapper + ["bash", str(code / "scripts/run_training.sh"), task["dataset"],
                          task["variant"], "official_test", "--seed", str(task["seed"]), "--num-workers", "0"]
    run, result = task_paths(root, task)
    return wrapper + [str(Path(manifest["env_prefix"]) / "bin/python"),
                      str(code / "scripts/evaluate_checkpoint.py"), "--dataset", task["dataset"],
                      "--checkpoint", str(run / "best.pt"), "--output-dir", str(result),
                      "--device", "cuda:0", "--batch-size", "16", "--num-workers", "0"]


def validate_training(root, task, metadata):
    run, _ = task_paths(root, task)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if [record.get("epoch") for record in records] != list(range(1, task["epochs"] + 1)):
        raise ValueError("epoch records are missing, duplicated, or incomplete")
    if any(record.get("causal") is not False or record.get("selection_protocol") != "official_test"
           or record.get("effective_norm") != "BN" for record in records):
        raise ValueError("training records violate noncausal official_test BN protocol")
    for key, expected in expected_config(task).items():
        if metadata.get("args", {}).get(key) != expected:
            raise ValueError(f"checkpoint config {key}: expected {expected!r}, got {metadata.get('args', {}).get(key)!r}")
    if metadata.get("epoch") != task["epochs"] - 1 or metadata.get("causal") is not False:
        raise ValueError("latest checkpoint does not contain the complete noncausal epoch budget")
    best = metadata.get("best_epoch")
    if not isinstance(best, int) or not 0 <= best < task["epochs"]:
        raise ValueError("invalid selected checkpoint epoch")
    for filename in ("best.pt", "latest.pt"):
        if not (run / filename).is_file():
            raise ValueError(f"missing {filename}")
    profiles = [json.loads(line) for line in (run / "memory_profile.jsonl").read_text().splitlines()
                if line.strip()] if (run / "memory_profile.jsonl").is_file() else []
    reserved = max((record.get("peak_reserved_mb", 0) for record in profiles), default=0)
    return {"selected_epoch": best + 1, "training_seconds": sum(r["elapsed_seconds"] for r in records),
            "max_allocated_mib": max(r["max_gpu_memory_mb"] for r in records),
            "max_reserved_mib": reserved or None,
            "config": metadata["args"]}


def checkpoint_metadata(manifest, root, task):
    run, _ = task_paths(root, task)
    # Only load this batch's own trusted checkpoint, in its matching environment.
    script = ("import json,sys,torch; c=torch.load(sys.argv[1],map_location='cpu'); "
              "print(json.dumps({k:c[k] for k in ('epoch','best_epoch','causal','args')}))")
    code = Path(manifest["code_root"])
    environment = dict(os.environ, CMERC_ENV_PREFIX=manifest["env_prefix"], CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(["bash", str(code / "scripts/with_cmerc_env.sh"),
                             str(Path(manifest["env_prefix"]) / "bin/python"), "-c", script,
                             str(run / "latest.pt")], cwd=code, env=environment,
                            capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)
    # The selected best checkpoint must agree with latest's recorded selection.
    best_result = subprocess.run(["bash", str(code / "scripts/with_cmerc_env.sh"),
                                  str(Path(manifest["env_prefix"]) / "bin/python"), "-c", script,
                                  str(run / "best.pt")], cwd=code, env=environment,
                                 capture_output=True, text=True, check=True)
    best = json.loads(best_result.stdout)
    if best["epoch"] != metadata["best_epoch"] or best["args"] != metadata["args"] or best["causal"] is not False:
        raise ValueError("best/latest checkpoint metadata disagree")
    return metadata


def validate_evaluation(root, task):
    run, result = task_paths(root, task)
    metrics = read_json(result / "metrics.json")
    if metrics.get("causal") is not False or metrics.get("dataset") != task["dataset"]:
        raise ValueError("evaluation dataset/causal metadata mismatch")
    if Path(metrics.get("checkpoint", "")).resolve() != (run / "best.pt").resolve():
        raise ValueError("evaluation checkpoint path mismatch")
    for key in ("weighted_f1", "accuracy", "macro_f1"):
        if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]):
            raise ValueError(f"evaluation missing finite {key}")
    cel = metrics.get("cel", {})
    if set(cel.get("by_removal", {})) != {"text", "audio", "visual", "intra_context", "inter_context"}:
        raise ValueError("evaluation must include all five CEL removals")
    if not isinstance(cel.get("total"), (int, float)) or not math.isfinite(cel["total"]):
        raise ValueError("evaluation missing finite CEL")
    for filename in ("predictions.csv", "predictions.npz"):
        if not (result / filename).is_file():
            raise ValueError(f"missing independent evaluation artifact {filename}")
    return metrics


def save_state(root, state):
    state["updated_at"] = now()
    atomic_json(root / "state.json", state)
    fields = ("id", "status", "phase", "pid", "gpu_uuid", "exitcode", "error", "log_path")
    temporary = root / "status.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for identifier, entry in state["tasks"].items():
            writer.writerow({"id": identifier, **entry})
    temporary.replace(root / "status.csv")


def run_batch(root, poll_seconds=20, stagger_seconds=10):
    root = Path(root).resolve()
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, state = read_json(root / "batch_manifest.json"), read_json(root / "state.json")
        code = Path(manifest["code_root"])
        preflight = read_json(root / manifest["require_preflight"])
        if preflight.get("status") != "passed":
            raise ValueError("preflight has not passed; refusing full training")
        if manifest["tasks"] != task_matrix():
            raise ValueError("manifest matrix differs from the predeclared 12-task matrix")
        for relative, expected in manifest["file_sha256"].items():
            if sha256(code / relative) != expected:
                raise ValueError(f"frozen source/input changed: {relative}")
        for observer, expected in manifest.get("observer_sha256", {}).items():
            if sha256(observer) != expected:
                raise ValueError(f"frozen observer changed: {observer}")
        for task in manifest["tasks"]:
            entry = state["tasks"][task["id"]]
            if pid_alive(entry.get("pid")):
                raise ValueError(f"prior live PID {entry['pid']} for {task['id']}; do not start a second supervisor")
            if entry["status"] in ("launching", "running", "evaluating", "interrupted"):
                entry.update(status="failed", error="prior supervisor lost; no automatic append/resume", finished_at=now())
        save_state(root, state)
        children = {}
        policy = meld_concurrency_policy(root)
        state["scheduling_policy"] = policy
        initial_training_uuids = set()
        for task in manifest["tasks"]:
            uuid = state["tasks"][task["id"]].get("training_gpu_uuid")
            if uuid:
                initial_training_uuids.add(uuid)
            if len(initial_training_uuids) == 3:
                break
        save_state(root, state)

        def launch(task, gpu, phase):
            entry = state["tasks"][task["id"]]
            command = command_for(manifest, root, task, phase)
            log_path = root / "logs" / f"{task['id']}_{phase}.log"
            environment = os.environ.copy()
            environment.update(manifest.get("runtime_environment", {}))
            environment.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CMERC_ENV_PREFIX=manifest["env_prefix"],
                               CMERC_PYTHON=manifest["training_python"],
                               CMERC_PROFILE_OUTPUT=str(task_paths(root, task)[0] / "memory_profile.jsonl"),
                               CMERC_OUTPUT_DIR=str(root / "runs"))
            # Persist intent before spawning, so a crash between Popen and PID
            # recording cannot leave this task eligible for a second launch.
            entry.update(status="launching", phase=phase, command=command, command_shell=shlex.join(command),
                         log_path=str(log_path), gpu_uuid=gpu["uuid"], gpu_index=gpu["index"],
                         phase_started_at=now(), exitcode=None)
            save_state(root, state)
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=code, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            entry.update(status="running" if phase == "training" else "evaluating", phase=phase,
                         pid=process.pid, gpu_uuid=gpu["uuid"], gpu_index=gpu["index"], command=command,
                         command_shell=shlex.join(command), log_path=str(log_path), exitcode=None,
                         phase_started_at=now())
            if phase == "training":
                entry["started_at"] = now()
                if len(initial_training_uuids) < 3:
                    initial_training_uuids.add(gpu["uuid"])
            entry[f"{phase}_gpu_uuid"] = gpu["uuid"]
            entry[f"{phase}_command"] = command
            entry[f"{phase}_log_path"] = str(log_path)
            children[task["id"]] = (process, task, gpu, phase)
            save_state(root, state)

        try:
            while True:
                for identifier, (process, task, gpu, phase) in list(children.items()):
                    exitcode = process.poll()
                    if exitcode is None:
                        continue
                    del children[identifier]
                    entry = state["tasks"][identifier]
                    entry.update(exitcode=exitcode, pid=None)
                    entry[f"{phase}_exitcode"] = exitcode
                    try:
                        if exitcode:
                            raise ValueError(f"{phase} process exited {exitcode}; see {entry['log_path']}")
                        if phase == "training":
                            details = validate_training(root, task, checkpoint_metadata(manifest, root, task))
                            if manifest["training_python"] != str(Path(manifest["env_prefix"]) / "bin/python") and details["max_reserved_mib"] is None:
                                raise ValueError("training observer did not record peak reserved memory")
                            entry.update(details)
                            atomic_json(task_paths(root, task)[0] / "effective_config.json", details["config"])
                            entry.update(status="awaiting_evaluation", phase="evaluation")
                        else:
                            entry["metrics"] = validate_evaluation(root, task)
                            entry.update(status="completed", finished_at=now())
                    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
                        entry.update(status="failed", error=str(error), finished_at=now())
                    save_state(root, state)
                    summarize(root)
                pending = [task for task in manifest["tasks"] if state["tasks"][task["id"]]["status"] == "planned"]
                evaluations = [task for task in manifest["tasks"] if state["tasks"][task["id"]]["status"] == "awaiting_evaluation"]
                if not pending and not evaluations and not children:
                    break
                for task in evaluations + pending:
                    phase = "training" if state["tasks"][task["id"]]["status"] == "planned" else "evaluation"
                    candidates = eligible_gpus(task, phase, query_gpus(), children, policy, initial_training_uuids)
                    if not candidates:
                        if len(initial_training_uuids) < 3:
                            break
                        continue
                    entry = state["tasks"][task["id"]]
                    try:
                        if entry["status"] == "planned":
                            refuse_duplicate(root, task, entry)
                            launch(task, candidates[0], "training")
                        else:
                            if task_paths(root, task)[1].exists():
                                raise ValueError("evaluation artifacts already exist; refusing overwrite")
                            launch(task, candidates[0], "evaluation")
                    except (ValueError, OSError) as error:
                        entry.update(status="failed", error=str(error), finished_at=now())
                        save_state(root, state)
                    time.sleep(stagger_seconds)
                time.sleep(poll_seconds)
        except BaseException:
            # Keep detached children alive. Record uncertainty, retain their PID,
            # and prohibit restart until the operator reconciles their artifacts.
            for identifier in children:
                state["tasks"][identifier].update(status="interrupted", error="supervisor interrupted; child may still be alive")
            save_state(root, state)
            summarize(root)
            raise
        summarize(root)
        return 0 if all(entry["status"] == "completed" for entry in state["tasks"].values()) else 1


def summarize(root):
    root = Path(root).resolve()
    # A separate lock allows live summaries while the supervisor owns its lock,
    # without collisions on the shared report temporary files.
    with (root / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _summarize(root)


def _summarize(root):
    root = Path(root).resolve()
    manifest, state = read_json(root / "batch_manifest.json"), read_json(root / "state.json")
    rows = []
    for task in manifest["tasks"]:
        entry = state["tasks"][task["id"]]
        row = {key: task[key] for key in ("id", "dataset", "variant", "seed", "epochs")}
        row.update(status=entry["status"], exitcode=entry.get("exitcode"), error=entry.get("error"),
                   selected_epoch=entry.get("selected_epoch"), training_seconds=entry.get("training_seconds"),
                   max_allocated_mib=entry.get("max_allocated_mib"), max_reserved_mib=entry.get("max_reserved_mib"))
        if entry["status"] == "completed":
            try:
                metrics = validate_evaluation(root, task)
                row.update({key: metrics[key] for key in ("weighted_f1", "accuracy", "macro_f1")})
                row["cel"] = metrics["cel"]["total"]
            except (ValueError, OSError, KeyError) as error:
                row.update(status="failed", error=f"completed evaluation invalid: {error}")
        rows.append(row)
    summary = {"batch_id": manifest["batch_id"], "updated_at": now(), "groups": {},
               "scheduling_policy": state.get("scheduling_policy", {"max_meld_per_gpu": 1}),
               "historical_reference_separate": HISTORICAL, "all_tasks_terminal": all(
                   row["status"] in ("completed", "failed") for row in rows)}
    for dataset in SEEDS:
        selected = [row for row in rows if row["dataset"] == dataset and row["variant"] == "cmerc"]
        scores = [row["weighted_f1"] for row in selected if row["status"] == "completed"]
        group = {"planned_n": 5, "completed_n": len(scores), "seeds": SEEDS[dataset],
                 "paper_weighted_f1": PAPER[dataset], "mean": None, "sample_std": None,
                 "minimum": None, "maximum": None, "difference_points": None,
                 "within_half_point": None}
        if len(scores) == 5:
            mean = statistics.mean(scores)
            group.update(mean=mean, sample_std=statistics.stdev(scores), minimum=min(scores), maximum=max(scores),
                         difference_points=mean - PAPER[dataset], within_half_point=abs(mean - PAPER[dataset]) <= 0.5 + 1e-12)
        summary["groups"][dataset] = group
    summary["both_datasets_within_half_point"] = all(
        group["within_half_point"] is True for group in summary["groups"].values())
    fields = ["id", "dataset", "variant", "seed", "epochs", "status", "exitcode", "error", "weighted_f1",
              "accuracy", "macro_f1", "cel", "selected_epoch", "training_seconds", "max_allocated_mib", "max_reserved_mib"]
    with (root / "results.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "results.csv.tmp").replace(root / "results.csv")
    atomic_json(root / "summary.json", summary)
    lines = [f"# XP 多 seed 批次 {manifest['batch_id']}", "",
             "服务器：本地 user23；设备：TITAN Xp 候选 GPU 0、1、3。", "",
             f"MELD 每卡最多 {summary['scheduling_policy']['max_meld_per_gpu']} 个本批次任务；前三个正式训练使用不同 GPU。",
             "仅当预检实测双任务总吞吐改善至少 10% 且显存有余量时启用第二个 MELD；不与 IEMOCAP 混跑。", "",
             "公开 official_test 协议每 epoch 用测试集 W-F1 选 best checkpoint，存在测试集选模偏差。",
             "首个 seed 来自作者 README，其余四个为预先固定的复现种子；作者完整五 seed 列表未知。",
             "固定作者难度文件，统计仅反映该 curriculum 条件下的训练随机性。",
             "单 seed baseline 仅作训练链诊断，不支持五 seed 配对提升或显著性结论。", "",
             "| 数据集 | 完成数 | XP 五 seed Mean ± 样本 Std（ddof=1） | 最小 / 最大 | 与论文差 | ±0.5 点 |",
             "| --- | ---: | --- | --- | --- | --- |"]
    for dataset, group in summary["groups"].items():
        if group["mean"] is None:
            cells = "未满五次，不统计 | — | — | 未判定"
        else:
            cells = (f"{group['mean']:.4f} ± {group['sample_std']:.4f} | {group['minimum']:.4f} / {group['maximum']:.4f} | "
                     f"{group['difference_points']:+.4f} | {'达到' if group['within_half_point'] else '未达到'}")
        lines.append(f"| {dataset} | {group['completed_n']}/5 | {cells} |")
    lines += ["", "| 数据集 | 论文五次均值 | 官方 checkpoint 历史评测 | V100 单 seed 从头训练 | XP 官方 seed 单次 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in SEEDS:
        anchor = next(row for row in rows if row["dataset"] == dataset and row["variant"] == "cmerc" and row["seed"] == SEEDS[dataset][0])
        xp = f"{anchor['weighted_f1']:.4f}" if anchor["status"] == "completed" else "待完成/失败"
        lines.append(f"| {dataset} | {PAPER[dataset]:.2f} | {HISTORICAL[dataset]['official_checkpoint']:.4f} | {HISTORICAL[dataset]['v100_cmerc_official_test']:.4f} | {xp} |")
    lines += ["", "历史 checkpoint / V100 指标未纳入 XP 均值；论文使用 RTX 4090，本批为 XP，不能将差距直接归因于 GPU。",
              "XP 环境、代码与输入哈希见 batch_manifest.json；独立官方 checkpoint 验证和峰值 reserved 测量见批次审计产物。",
              "逐任务状态、失败原因、配置、选中 epoch、耗时与显存见 state.json、status.csv 和 results.csv。", "",
              "| 任务 | 状态 | 退出码 | 错误 |", "| --- | --- | --- | --- |"]
    for row in rows:
        error = str(row.get("error") or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['id']} | {row['status']} | {row.get('exitcode')} | {error} |")
    lines += ["", ("当前公开代码协议下两个数据集的五 seed 均值数值接近论文。" if summary["both_datasets_within_half_point"]
                   else "尚未满足两个数据集完整五 seed 均值均在论文 ±0.5 点范围内的条件；不追加 seed 或调参。"), "",
              "复用命令：", "", "```bash",
              f"python scripts/xp_multiseed.py run --batch-root {shlex.quote(str(root))}",
              f"python scripts/xp_multiseed.py summarize --batch-root {shlex.quote(str(root))}", "```", ""]
    (root / "REPORT.md.tmp").write_text("\n".join(lines), encoding="utf-8")
    (root / "REPORT.md.tmp").replace(root / "REPORT.md")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--batch-root", type=Path, required=True)
    create.add_argument("--code-root", type=Path, required=True)
    create.add_argument("--env-prefix", type=Path, required=True)
    create.add_argument("--python-wrapper", type=Path, help="training-only observer wrapper; evaluation uses env Python")
    run = commands.add_parser("run")
    run.add_argument("--batch-root", type=Path, required=True)
    run.add_argument("--poll-seconds", type=float, default=20)
    run.add_argument("--stagger-seconds", type=float, default=10)
    report = commands.add_parser("summarize")
    report.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_batch(args.batch_root, args.code_root, args.env_prefix, args.python_wrapper)
    elif args.command == "run":
        if args.poll_seconds < 1 or args.stagger_seconds < 0:
            parser.error("poll-seconds must be >= 1 and stagger-seconds >= 0")
        return run_batch(args.batch_root, args.poll_seconds, args.stagger_seconds)
    else:
        summarize(args.batch_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
