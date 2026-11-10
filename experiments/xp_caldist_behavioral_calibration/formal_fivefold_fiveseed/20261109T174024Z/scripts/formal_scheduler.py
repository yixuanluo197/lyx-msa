#!/usr/bin/env python3
"""Dependency-safe four-GPU scheduler for the sealed Protocol B training plan.

The scheduler deliberately has no Evaluation stage.  A failed task is retried
from scratch in a fresh attempt directory; checkpoint resume is rejected.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Callable, Iterable


GIB = 1024 ** 3
MIN_FREE_BYTES = 50 * GIB
KINDS = ("teacher", "difficulty", "cmerc", "calibration")
DATASETS = ("IEMOCAP", "MELD")
SEEDS = (1475, 67137, 2024, 3407, 42)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    dataset: str
    fold: int
    seed: int | None
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    reused: bool = False
    reused_output: str | None = None


class Plan:
    def __init__(self, tasks: Iterable[Task]):
        rows = list(tasks)
        self.tasks = {task.task_id: task for task in rows}
        if len(self.tasks) != len(rows):
            raise ValueError("task IDs must be unique")
        self._validate()

    def _validate(self) -> None:
        for task in self.tasks.values():
            if task.kind not in KINDS:
                raise ValueError(f"unsupported task kind: {task.kind}")
            if task.dataset not in DATASETS or task.fold not in range(1, 6):
                raise ValueError(f"invalid dataset/fold: {task.task_id}")
            lowered = [token.lower() for token in (task.task_id, *task.command)]
            if any("evaluation" in token for token in lowered):
                raise ValueError("Evaluation surfaces are forbidden in this stage")
            if any(token == "--resume" or token.startswith("--resume=") for token in lowered):
                raise ValueError("resume is forbidden; retries must start from scratch")
            missing = set(task.dependencies) - set(self.tasks)
            if missing:
                raise ValueError(f"unknown dependencies for {task.task_id}: {sorted(missing)}")
        visiting, visited = set(), set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self.tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)


def validate_gpu_uuids(values: Iterable[str]) -> tuple[str, ...]:
    uuids = tuple(values)
    if not uuids or len(uuids) > 4:
        raise ValueError("one to four GPU UUIDs are required")
    if len(set(uuids)) != len(uuids):
        raise ValueError("GPU UUIDs must be unique")
    if any(not value.startswith("GPU-") for value in uuids):
        raise ValueError("bind workers by full NVIDIA GPU UUID")
    return uuids


def require_disk_space(path: Path, minimum: int = MIN_FREE_BYTES) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise RuntimeError(f"at least 50 GiB free disk is required; found {free / GIB:.2f} GiB")
    return free


def validate_task_artifacts(task: Task, output: Path) -> dict:
    """Validate the minimum frozen artifact contract after a zero process exit."""
    try:
        if task.kind == "teacher":
            required = [output / name for name in ("teacher_final.pt", "access_audit.json",
                                                    "teacher_checkpoint_metadata.json")]
            if not all(path.is_file() for path in required):
                raise ValueError("Teacher output is incomplete")
            audit = json.loads((output / "access_audit.json").read_text())
            if not audit.get("exact_final_coverage") or audit.get("rejected_ids"):
                raise ValueError("Teacher access audit failed")
        elif task.kind == "difficulty":
            difficulty = output if output.is_file() else output / "difficulty.json"
            if not difficulty.is_file():
                raise ValueError("Difficulty JSON is missing")
            artifact = json.loads(difficulty.read_text())
            if artifact.get("status") != "formal_fold_local_difficulty" or not artifact.get("dialogues"):
                raise ValueError("Difficulty artifact status/content is invalid")
            if output.is_dir() and not (output / "six_view_probabilities.npz").is_file():
                raise ValueError("six-view probabilities are missing")
        elif task.kind == "cmerc":
            required = [output / name for name in ("best.pt", "latest.pt", "completion.json",
                                                    "first_epoch_gate.json", "effective_config.json")]
            if not all(path.is_file() for path in required):
                raise ValueError("CMERC output is incomplete")
            completion = json.loads((output / "completion.json").read_text())
            if completion.get("status") != "completed" or not completion.get("first_epoch_gate_passed"):
                raise ValueError("CMERC completion/gate failed")
        elif task.kind == "calibration":
            required = [output / name for name in ("parameters.json", "calibration_metrics.json",
                                                    "oof_diagnostic.json", "effective_config.json",
                                                    "provenance.json", "calibration_probabilities.npz")]
            if not all(path.is_file() for path in required):
                raise ValueError("Calibration output is incomplete")
            params = json.loads((output / "parameters.json").read_text())
            if set(params) != {"E0", "E1", "E2", "E3"}:
                raise ValueError("Calibration parameter set is incomplete")
        else:
            raise ValueError(f"unsupported task kind: {task.kind}")
        return {"status": "passed", "output": str(output)}
    except Exception as error:
        return {"status": "failed", "output": str(output), "error": f"{type(error).__name__}: {error}"}


def _python_command(script: Path, *arguments: object) -> tuple[str, ...]:
    return (sys.executable, str(script), *(str(value) for value in arguments))


def build_protocol_b_plan(formal_root: Path, teacher_pilot: Path, cmerc_pilot: Path) -> Plan:
    """Build the frozen 10 Teacher / 10 Difficulty / 50 CMERC / 50 Calibration DAG.

    Commands contain runtime placeholders resolved by :class:`Scheduler`.
    Role artifacts are immutable prerequisites produced during protocol setup.
    """
    formal_root = formal_root.resolve()
    scripts = formal_root / "scripts"
    manifest = formal_root / "configs/approved_split_manifests.json"
    teacher_config = formal_root / "configs/teacher_protocol.json"
    source_manifest = formal_root / "provenance/teacher_source_manifest.json"
    source_root = teacher_pilot.resolve() / "source_snapshot"
    tasks: list[Task] = []
    for dataset in DATASETS:
        for fold in range(1, 6):
            historical_run = cmerc_pilot / "runs" / dataset / "fold1" / "seed1475"
            artifacts = (historical_run / "artifacts" if fold == 1 else
                         formal_root / "runs/shared" / dataset / f"fold{fold}" / "artifacts")
            teacher_artifact = artifacts / ("training_role_artifact.pt" if fold == 1 else
                                             "teacher_training_role_artifact.pt")
            teacher_id = f"teacher:{dataset}:fold{fold}"
            difficulty_id = f"difficulty:{dataset}:fold{fold}"
            reused_fold = fold == 1
            historical_teacher = teacher_pilot / "runs" / dataset / "fold1" / "teacher"
            historical_difficulty = teacher_pilot / "results" / f"{dataset}_fold1_difficulty.json"
            teacher_command = _python_command(
                scripts / "teacher_runner.py", "--artifact", teacher_artifact,
                "--approved-manifest", manifest, "--source-root", source_root,
                "--source-manifest", source_manifest, "--teacher-config", teacher_config,
                "--output-dir", "{attempt_dir}/output", "--device", "cuda:0",
            )
            tasks.append(Task(teacher_id, "teacher", dataset, fold, None, (), teacher_command,
                              reused_fold, str(historical_teacher) if reused_fold else None))
            difficulty_command = _python_command(
                scripts / "generate_fold_local_difficulty.py",
                "--approved-manifest", manifest, "--manifest-sha256", "{sha256:" + str(manifest) + "}",
                "--dataset", dataset, "--fold", f"fold{fold}",
                "--teacher-checkpoint", "{dep:" + teacher_id + "}/teacher_final.pt",
                "--teacher-sha256", "{sha256:dep:" + teacher_id + ":teacher_final.pt}",
                "--training-role-artifact", teacher_artifact,
                "--training-role-artifact-sha256", "{sha256:" + str(teacher_artifact) + "}",
                "--source-root", source_root, "--source-manifest", source_manifest,
                "--source-manifest-sha256", "{sha256:" + str(source_manifest) + "}",
                "--teacher-config", teacher_config, "--teacher-config-sha256", "{sha256:" + str(teacher_config) + "}",
                "--output", "{attempt_dir}/output/difficulty.json",
                "--probabilities-output", "{attempt_dir}/output/six_view_probabilities.npz",
                "--device", "cuda:0", "--batch-size", 1,
            )
            tasks.append(Task(difficulty_id, "difficulty", dataset, fold, None, (teacher_id,),
                              difficulty_command, reused_fold,
                              str(historical_difficulty) if reused_fold else None))
            for seed in SEEDS:
                cmerc_id = f"cmerc:{dataset}:fold{fold}:seed{seed}"
                calibration_id = f"calibration:{dataset}:fold{fold}:seed{seed}"
                reused_model = fold == 1 and seed == 1475
                cmerc_command = _python_command(
                    scripts / "cmerc_runner.py",
                    "--training-artifact", artifacts / "training_role_artifact.pt",
                    "--development-artifact", artifacts / "development_role_artifact.pt",
                    "--approved-manifest", manifest,
                    "--difficulty", "{dep:" + difficulty_id + "}",
                    "--source-root", source_root, "--source-manifest", source_manifest,
                    "--seed", seed, "--output-dir", "{attempt_dir}/output", "--device", "cuda:0",
                )
                tasks.append(Task(cmerc_id, "cmerc", dataset, fold, seed, (difficulty_id,), cmerc_command,
                                  reused_model, str(historical_run / "training") if reused_model else None))
                calibration_command = _python_command(
                    scripts / "calibration_only.py", "--checkpoint", "{dep:" + cmerc_id + "}/best.pt",
                    "--calibration-artifact", artifacts / "calibration_role_artifact.pt",
                    "--manifest", manifest, "--output", "{attempt_dir}/output", "--device", "cuda:0",
                    "--seed", 20261109,
                )
                tasks.append(Task(calibration_id, "calibration", dataset, fold, seed, (cmerc_id,),
                                  calibration_command, reused_model,
                                  str(historical_run / "calibration") if reused_model else None))
    return Plan(tasks)


class Scheduler:
    def __init__(self, plan: Plan, state_dir: Path, gpu_uuids: Iterable[str], *, max_attempts: int = 2,
                 process_runner: Callable = subprocess.run,
                 artifact_validator: Callable = validate_task_artifacts,
                 recover_existing: bool = False):
        self.plan = plan
        self.state_dir = state_dir
        self.gpu_uuids = validate_gpu_uuids(gpu_uuids)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.process_runner = process_runner
        self.artifact_validator = artifact_validator
        self.lock = threading.Lock()
        self.ledger_path = state_dir / "task_ledger.json"
        self.ledger = self._recover_ledger() if recover_existing else self._initial_ledger()

    def _recover_ledger(self) -> dict:
        if not self.ledger_path.is_file():
            raise FileNotFoundError(f"scheduler ledger does not exist: {self.ledger_path}")
        ledger = json.loads(self.ledger_path.read_text())
        if set(ledger.get("tasks", {})) != set(self.plan.tasks):
            raise ValueError("existing ledger task IDs do not match the frozen plan")
        for task_id, row in ledger["tasks"].items():
            if row["status"] == "running" and row.get("attempts"):
                attempt = row["attempts"][-1]
                output = Path(attempt["directory"]) / "output"
                validation = self.artifact_validator(self.plan.tasks[task_id], output)
                if validation["status"] == "passed":
                    attempt.update(process_returncode=0, artifact_validation=validation,
                                   returncode=0, finished_at=utc_now(), recovered_after_parent_stop=True)
                    row["status"] = "completed"
                    continue
            if row["status"] in {"failed", "running"}:
                row["status"] = "pending"
        ledger["recovered_at"] = utc_now()
        return ledger

    def _initial_ledger(self) -> dict:
        tasks = {}
        for task in self.plan.tasks.values():
            tasks[task.task_id] = {
                "kind": task.kind, "dataset": task.dataset, "fold": task.fold, "seed": task.seed,
                "dependencies": list(task.dependencies), "status": "reused" if task.reused else "pending",
                "reused_output": task.reused_output, "attempts": [],
            }
        return {"schema": "formal_scheduler_ledger_v1", "created_at": utc_now(),
                "evaluation_stage_present": False, "tasks": tasks}

    def _write_ledger(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.ledger_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.ledger, indent=2) + "\n")
        temporary.replace(self.ledger_path)

    @staticmethod
    def _safe_id(task_id: str) -> str:
        return task_id.replace(":", "__")

    def _successful_output(self, task_id: str) -> Path:
        row = self.ledger["tasks"][task_id]
        if row["status"] == "reused":
            return Path(row["reused_output"] or "")
        successful = [attempt for attempt in row["attempts"] if attempt["returncode"] == 0]
        if not successful:
            raise RuntimeError(f"dependency has no successful output: {task_id}")
        output = Path(successful[-1]["directory"]) / "output"
        return output / "difficulty.json" if row["kind"] == "difficulty" else output

    def _expand(self, task: Task, attempt_dir: Path) -> list[str]:
        expanded = []
        for raw in task.command:
            token = str(raw).replace("{attempt_dir}", str(attempt_dir))
            if token.startswith("{dep:") and token.endswith("}"):
                token = str(self._successful_output(token[5:-1]))
            elif token.startswith("{sha256:dep:") and token.endswith("}"):
                body = token[len("{sha256:dep:"):-1]
                dependency, relative = body.rsplit(":", 1)
                token = file_sha256(self._successful_output(dependency) / relative)
            elif token.startswith("{sha256:") and token.endswith("}"):
                token = file_sha256(Path(token[len("{sha256:"):-1]))
            else:
                token = re.sub(
                    r"\{dep:([^}]+)\}",
                    lambda match: str(self._successful_output(match.group(1))),
                    token,
                )
            expanded.append(token)
        if any(value == "--resume" or value.startswith("--resume=") for value in expanded):
            raise RuntimeError("scheduler refuses resume commands")
        return expanded

    def _run_one(self, task: Task, gpu_uuid: str) -> bool:
        row = self.ledger["tasks"][task.task_id]
        for attempt_number in range(len(row["attempts"]) + 1, self.max_attempts + 1):
            attempt_dir = self.state_dir / "attempts" / self._safe_id(task.task_id) / f"attempt-{attempt_number:04d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = self._expand(task, attempt_dir)
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            record = {"attempt": attempt_number, "directory": str(attempt_dir), "gpu_uuid": gpu_uuid,
                      "command": command, "started_at": utc_now(), "returncode": None}
            with self.lock:
                row["status"] = "running"
                row["attempts"].append(record)
                self._write_ledger()
            with (attempt_dir / "stdout.log").open("wb") as stdout, (attempt_dir / "stderr.log").open("wb") as stderr:
                result = self.process_runner(command, env=environment, stdout=stdout, stderr=stderr)
            validation = (self.artifact_validator(task, attempt_dir / "output")
                          if result.returncode == 0 else {"status": "not_run_process_failed"})
            effective_returncode = int(result.returncode) if result.returncode else (0 if validation["status"] == "passed" else 90)
            with self.lock:
                record["process_returncode"] = int(result.returncode)
                record["artifact_validation"] = validation
                record["returncode"] = effective_returncode
                record["finished_at"] = utc_now()
                row["status"] = "completed" if effective_returncode == 0 else "failed"
                self._write_ledger()
            if effective_returncode == 0:
                return True
        return False

    def run(self, *, check_disk: bool = True) -> dict:
        if check_disk:
            require_disk_space(self.state_dir)
        for task_id, row in self.ledger["tasks"].items():
            if row["status"] in {"reused", "completed"}:
                output = (Path(row["reused_output"] or "") if row["status"] == "reused"
                          else self._successful_output(task_id))
                validation = self.artifact_validator(self.plan.tasks[task_id], output)
                row["artifact_validation"] = validation
                if validation["status"] != "passed":
                    self._write_ledger()
                    raise RuntimeError(f"existing artifact validation failed: {task_id}: {validation}")
        self._write_ledger()
        unfinished = {task_id for task_id, row in self.ledger["tasks"].items() if row["status"] == "pending"}
        successful = {task_id for task_id, row in self.ledger["tasks"].items()
                      if row["status"] in {"reused", "completed"}}
        free_gpus = list(self.gpu_uuids)
        failed = set()
        running = {}
        with ThreadPoolExecutor(max_workers=len(self.gpu_uuids)) as pool:
            while unfinished or running:
                ready = [self.plan.tasks[task_id] for task_id in sorted(unfinished)
                         if set(self.plan.tasks[task_id].dependencies) <= successful]
                while free_gpus and ready:
                    task, gpu_uuid = ready.pop(0), free_gpus.pop(0)
                    unfinished.remove(task.task_id)
                    future = pool.submit(self._run_one, task, gpu_uuid)
                    running[future] = (task, gpu_uuid)
                if not running:
                    blocked = sorted(unfinished)
                    raise RuntimeError(f"no runnable tasks; failed or blocked dependencies: {blocked}; failed={sorted(failed)}")
                completed, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in completed:
                    task, gpu_uuid = running.pop(future)
                    free_gpus.append(gpu_uuid)
                    if future.result():
                        successful.add(task.task_id)
                    else:
                        failed.add(task.task_id)
        if failed:
            blocked = sorted(task_id for task_id in self.plan.tasks if task_id not in successful)
            raise RuntimeError(f"failed tasks after fresh attempts; blocked dependencies: {blocked}; failed tasks: {sorted(failed)}")
        return self.ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--teacher-pilot", type=Path, required=True)
    parser.add_argument("--cmerc-pilot", type=Path, required=True)
    parser.add_argument("--gpu-uuid", action="append", required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="run tasks; omission only validates and prints the plan")
    parser.add_argument("--recover-ledger", action="store_true",
                        help="preserve completed artifacts and requeue failed/stale tasks from the existing ledger")
    args = parser.parse_args()
    plan = build_protocol_b_plan(args.formal_root, args.teacher_pilot, args.cmerc_pilot)
    uuids = validate_gpu_uuids(args.gpu_uuid)
    summary = {kind: sum(task.kind == kind for task in plan.tasks.values()) for kind in KINDS}
    payload = {"status": "validated_plan", "tasks": summary, "gpu_uuids": list(uuids),
               "reused": sum(task.reused for task in plan.tasks.values()), "evaluation_stage_present": False}
    if not args.execute:
        print(json.dumps(payload, indent=2))
        return 0
    scheduler = Scheduler(plan, args.formal_root / "state/scheduler", uuids,
                          max_attempts=args.max_attempts, recover_existing=args.recover_ledger)
    scheduler.run()
    print(json.dumps({**payload, "status": "completed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
