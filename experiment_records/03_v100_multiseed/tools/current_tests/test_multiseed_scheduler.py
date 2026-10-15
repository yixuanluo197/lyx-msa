"""CPU regressions for GPU allocation, fixed matrix, non-overwrite and recovery."""
import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("multiseed_scheduler", ROOT / "scripts/multiseed_scheduler.py")
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


def config():
    return scheduler.read_json(ROOT / "configs/multiseed_validation.json")


def healthy_gpu(index=1, **kwargs):
    values = dict(index=index, uuid=f"GPU-test-{index}", total_mb=32768, used_mb=0,
                  utilization=0, volatile_ecc="0", aggregate_ecc="0", retired_pending="No", pids=())
    return scheduler.GPU(**{**values, **kwargs})


class AllocationTests(unittest.TestCase):
    def test_known_bad_gpu_is_never_healthy_even_if_whitelisted(self):
        self.assertFalse(healthy_gpu(4).idle_healthy([1, 2, 3, 4, 5]))
        settings = config()
        settings["allowed_gpu_indices"].append(4)
        with self.assertRaisesRegex(ValueError, "GPU 4"):
            scheduler.validate_config(settings)

    def test_busy_ecc_or_retirement_devices_are_not_allocated(self):
        for changes in ({"pids": (99,)}, {"volatile_ecc": "1"}, {"aggregate_ecc": "7"},
                        {"retired_pending": "Yes"}, {"used_mb": 30000}, {"utilization": 99}):
            self.assertFalse(healthy_gpu(**changes).idle_healthy([1, 2, 3, 5]))

    def test_all_busy_devices_yield_no_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = scheduler.Scheduler(config(), [], Path(directory) / "state.json", Path(directory) / "locks")
            with patch.object(scheduler, "query_gpus", return_value=[healthy_gpu(1, pids=(12,)), healthy_gpu(4)]):
                self.assertIsNone(manager.acquire_gpu())

    def test_health_is_rechecked_after_gpu_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = scheduler.Scheduler(config(), [], Path(directory) / "state.json", Path(directory) / "locks")
            with patch.object(scheduler, "query_gpus", side_effect=[[healthy_gpu()], [healthy_gpu(pids=(55,))]]):
                self.assertIsNone(manager.acquire_gpu())

    def test_gpu_lock_prevents_two_simultaneous_allocations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "GPU.lock"
            first = scheduler.acquire_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(scheduler.acquire_lock(path))
            first.close()
            second = scheduler.acquire_lock(path)
            self.assertIsNotNone(second)
            second.close()

    def test_child_keeps_gpu_lock_if_scheduler_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "GPU.lock"
            handle = scheduler.acquire_lock(path)
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], pass_fds=(handle.fileno(),))
            try:
                handle.close()
                self.assertIsNone(scheduler.acquire_lock(path))
            finally:
                process.terminate()
                process.wait(timeout=5)
            recovered = scheduler.acquire_lock(path)
            self.assertIsNotNone(recovered)
            recovered.close()


class MatrixAndGuardTests(unittest.TestCase):
    def test_v2_config_and_dispatch_require_capacity_safe_policy(self):
        settings = config()
        scheduler.validate_config(settings)
        job = scheduler.build_jobs(settings, Path("/tmp/v2runs"), Path("/tmp/v2logs"), Path("/tmp/v2results"))[0]
        command = scheduler.training_command(job)
        self.assertEqual(scheduler.argument_value(command, "--batch-policy"), "capacity_safe")
        self.assertEqual(scheduler.argument_value(command, "--edge-capacity"), "1000")
        self.assertEqual(scheduler.argument_value(command, "--node-capacity"), "5200")
        self.assertEqual(scheduler.argument_value(command, "--batch-max-retries"), "10000")
        settings["batch_policy"] = "legacy"
        with self.assertRaisesRegex(ValueError, "capacity_safe"):
            scheduler.validate_config(settings)
        settings["batch_policy"] = "capacity_safe"
        settings["edge_capacity"] = 2000
        with self.assertRaisesRegex(ValueError, "capacity limits"):
            scheduler.validate_config(settings)

    def test_dispatch_rejects_post_freeze_code_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("original\n")
            files = {"model.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            (root / "snapshots").mkdir()
            snapshot = {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
            (root / "snapshots/source_snapshot.json").write_text(json.dumps(snapshot))
            self.assertEqual(scheduler.verify_source_snapshot(root), snapshot["sha256"])
            source.write_text("modified\n")
            with self.assertRaisesRegex(ValueError, "changed after freeze"):
                scheduler.verify_source_snapshot(root)

    def test_pre_registered_matrix_has_twenty_disjoint_outputs(self):
        jobs = scheduler.build_jobs(config(), Path("/tmp/runs"), Path("/tmp/logs"), Path("/tmp/results"))
        self.assertEqual(len(jobs), 20)
        self.assertEqual(sum(job["stage"] == 1 for job in jobs), 12)
        for key in ("run_dir", "result_dir", "train_log", "evaluation_log"):
            self.assertEqual(len({job[key] for job in jobs}), 20)
        self.assertEqual({job["split_seed"] for job in jobs}, {1475})
        self.assertTrue(all("--causal" not in scheduler.training_command(job) for job in jobs))
        self.assertTrue(all(scheduler.argument_value(scheduler.training_command(job), "--output-dir") == str(Path(job["run_dir"]).parent) for job in jobs))

    def test_smoke_is_four_real_paired_one_epoch_runs(self):
        jobs = scheduler.build_jobs(config(), Path("/tmp/smoke"), Path("/tmp/smokelogs"), Path("/tmp/smokeresults"), smoke=True)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({(job["dataset"], job["variant"]) for job in jobs},
                         {(data, variant) for data in ("IEMOCAP", "MELD") for variant in ("baseline", "cmerc")})
        self.assertEqual({job["epochs"] for job in jobs}, {1})

    def test_every_existing_output_is_protected_even_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in ("run_dir", "result_dir", "train_log", "evaluation_log"):
                paths = {name: str(Path(directory) / f"{name}-{key}") for name in
                         ("run_dir", "result_dir", "train_log", "evaluation_log")}
                Path(paths[key]).touch()
                with self.assertRaises(FileExistsError):
                    scheduler.output_guard(paths)

    def test_queue_cannot_change_matrix_when_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            scheduler.Scheduler(config(), [], path, Path(directory) / "locks")
            modified = config()
            modified["split_seed"] = 42
            with self.assertRaisesRegex(ValueError, "different preregistered"):
                scheduler.Scheduler(modified, [], path, Path(directory) / "locks")

    def test_queue_cannot_attach_another_server_pid_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            scheduler.Scheduler(config(), [], path, Path(directory) / "locks")
            with patch.object(scheduler.socket, "gethostname", return_value="other-server"):
                with self.assertRaisesRegex(ValueError, "different server"):
                    scheduler.Scheduler(config(), [], path, Path(directory) / "locks")

    def test_attach_matches_seed_before_first_epoch_exists(self):
        jobs = scheduler.build_jobs(config(), Path("/runs"), Path("/logs"), Path("/results"))
        job = jobs[0]
        command = scheduler.training_command(job)
        # bash wrapper becomes python/train_cm; duplicate --seed values obey final argparse value.
        command = ["python", "train_cm.py", "--Dataset", "IEMOCAP", "--selection-protocol", "validation"] + command[7:]
        fake_dirs = [Path("/proc/123")]
        with patch.object(Path, "iterdir", return_value=iter(fake_dirs)), patch.object(scheduler, "pid_identity", return_value={"command": command, "start_ticks": "18"}):
            self.assertEqual(scheduler.matching_live_pid(job, "training")[0], 123)
        other = {**job, "seed": 42}
        with patch.object(Path, "iterdir", return_value=iter(fake_dirs)), patch.object(scheduler, "pid_identity", return_value={"command": command, "start_ticks": "18"}):
            self.assertIsNone(scheduler.matching_live_pid(other, "training"))

    def test_attach_recognizes_live_wrapper_before_python_exec(self):
        job = scheduler.build_jobs(config(), Path("/runs"), Path("/logs"), Path("/results"))[0]
        command = scheduler.training_command(job)
        with patch.object(Path, "iterdir", return_value=iter([Path("/proc/123")])), patch.object(scheduler, "pid_identity", return_value={"command": command, "start_ticks": "18"}):
            self.assertEqual(scheduler.matching_live_pid(job, "training")[0], 123)
        with patch.object(Path, "iterdir", return_value=iter([Path("/proc/123")])), patch.object(scheduler, "pid_identity", return_value={"command": command, "start_ticks": "18"}):
            self.assertIsNone(scheduler.matching_live_pid({**job, "seed": 42}, "training"))

    def test_gate_rejects_cross_seed_split_changes_without_using_scores(self):
        jobs = scheduler.build_jobs(config(), Path("/runs"), Path("/logs"), Path("/results"))[:12]
        for job in jobs:
            job["status"] = "completed"
        with patch.object(scheduler, "validate_training", side_effect=lambda job, **_: {"split": {"hash": str(job["seed"])}, "source": {"files": "fixed"}, "data": {"features": "fixed"}}), patch.object(scheduler, "validate_evaluation"):
            with self.assertRaisesRegex(ValueError, "Different training/validation"):
                scheduler.validate_stage(jobs, inspect_checkpoints=False)
        with patch.object(scheduler, "validate_training", return_value={"split": {"hash": "fixed"}, "source": {"files": "fixed"}, "data": {"features": "fixed"}}), patch.object(scheduler, "validate_evaluation"):
            self.assertEqual(scheduler.validate_stage(jobs, inspect_checkpoints=False)["runs"], 12)


class ShellCompatibilityTests(unittest.TestCase):
    def run_script(self, args, **env):
        return subprocess.run(["bash", str(ROOT / "scripts/run_training.sh"), *args], text=True,
                              capture_output=True, env={**os.environ, "CMERC_PYTHON": "/bin/echo", **env})

    def test_legacy_entry_defaults_and_explicit_seed_are_preserved(self):
        response = self.run_script(["IEMOCAP", "baseline", "validation"], CMERC_OUTPUT_DIR="/tmp/example")
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertIn("--output-dir /tmp/example/baseline", response.stdout)
        self.assertIn("--seed 1475 --split-seed 1475 --rng-policy legacy", response.stdout)
        explicit = self.run_script(["IEMOCAP", "baseline", "validation", "--seed", "42"])
        self.assertIn("--seed 42 --split-seed 42", explicit.stdout)
        fixed = self.run_script(["IEMOCAP", "cmerc", "validation", "--seed", "42", "--split-seed", "1475", "--rng-policy", "continuous"])
        self.assertIn("--seed 42 --split-seed 1475 --rng-policy continuous", fixed.stdout)
        self.assertIn("--batch-size 16", fixed.stdout)

    def test_noncausal_entry_rejects_causal_flag(self):
        response = self.run_script(["IEMOCAP", "cmerc", "validation", "--causal"])
        self.assertEqual(response.returncode, 2)

    def test_sequential_queue_logs_are_seed_isolated_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "CMERC_PYTHON": "/bin/echo", "CMERC_OUTPUT_DIR": str(Path(directory) / "runs"),
                   "CMERC_LOG_DIR": str(Path(directory) / "logs")}
            command = ["bash", str(ROOT / "scripts/run_job_queue.sh"), "--seeds", "1475,42", "--split-seed", "1475",
                       "--rng-policy", "continuous", "IEMOCAP", "baseline", "validation"]
            response = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(response.returncode, 0, response.stderr)
            logs = sorted((Path(directory) / "logs").glob("*.log"))
            self.assertEqual(len(logs), 2)
            before = {path: path.read_bytes() for path in logs}
            repeated = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 3)
            self.assertEqual(before, {path: path.read_bytes() for path in logs})


if __name__ == "__main__":
    unittest.main()
