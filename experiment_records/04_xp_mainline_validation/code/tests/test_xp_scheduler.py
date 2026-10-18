"""CPU-only XP adaptation checks; no training or GPU allocation is performed."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("xp_scheduler_under_test", ROOT / "scripts/multiseed_scheduler.py")
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


def config():
    return {
        "first_seeds": [1475, 67137, 2024], "additional_seeds": [3407, 42],
        "split_seed": 1475, "rng_policy": "continuous", "protocol_version": 2,
        "batch_policy": "capacity_safe", "edge_capacity": 1000, "node_capacity": 5200,
        "batch_max_retries": 10000, "selection_protocol": "validation", "causal": False,
        "datasets": ["IEMOCAP", "MELD"], "variants": ["baseline", "cmerc"],
        "epochs": {"IEMOCAP": 80, "MELD": 40}, "allowed_gpu_indices": [0, 1, 3],
        "allowed_gpu_uuids": {str(index): uuid for index, uuid in scheduler.GPU_UUIDS.items()},
        "minimum_free_gpu_mb": 11000, "maximum_idle_utilization": 10,
        "minimum_free_disk_gib": 6, "poll_seconds": 1,
        "runtime_environment": {"CMERC_ENV_PREFIX": "/xp/environment", "CMERC_PYTHON": "/batch/tools/xp_python",
                                "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8"},
    }


def gpu(index=0, **changes):
    return scheduler.GPU(**{**dict(index=index, uuid=scheduler.GPU_UUIDS.get(index, "GPU-unapproved"),
                                   total_mb=12288, used_mb=108, utilization=0, pids=(), name="TITAN Xp"), **changes})


class XPSchedulerTests(unittest.TestCase):
    def setUp(self):
        hostname = patch.object(scheduler.socket, "gethostname", return_value="user23")
        hostname.start()
        self.addCleanup(hostname.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def jobs(self, smoke=False):
        return scheduler.build_jobs(config(), self.directory / "runs", self.directory / "logs", self.directory / "results", smoke)

    def manager(self, jobs=None, settings=None):
        return scheduler.Scheduler(settings or config(), self.jobs() if jobs is None else jobs,
                                   self.directory / "state/queue.json", self.directory / "locks")

    def test_fixed_matrix_and_exact_four_anchors_have_independent_paths(self):
        settings = config()
        scheduler.validate_config(settings)
        jobs = self.jobs()
        self.assertEqual(len(jobs), 20)
        self.assertEqual([(job["dataset"], job["variant"], job["seed"]) for job in jobs[:4]], [
            ("IEMOCAP", "baseline", 1475), ("IEMOCAP", "cmerc", 1475),
            ("MELD", "baseline", 67137), ("MELD", "cmerc", 67137),
        ])
        self.assertEqual([job["stage"] for job in jobs], [1] * 4 + [2] * 16)
        for key in ("run_dir", "result_dir", "train_log", "evaluation_log"):
            self.assertEqual(len({job[key] for job in jobs}), 20)
            self.assertTrue(all(Path(job[key]).is_relative_to(self.directory) for job in jobs))
        for dataset in settings["datasets"]:
            for variant in settings["variants"]:
                self.assertEqual({job["seed"] for job in jobs if job["dataset"] == dataset and job["variant"] == variant}, set(scheduler.SEEDS))
        for job in jobs:
            command = scheduler.training_command(job)
            self.assertEqual(scheduler.argument_value(command, "--batch-policy"), "capacity_safe")
            self.assertEqual(scheduler.argument_value(command, "--edge-capacity"), "1000")
            self.assertEqual(scheduler.argument_value(command, "--node-capacity"), "5200")
            self.assertEqual(scheduler.argument_value(command, "--batch-max-retries"), "10000")
            self.assertNotIn("--causal", command)
        self.assertEqual({job["epochs"] for job in self.jobs(smoke=True)}, {1})

    def test_configuration_rejects_foreign_gpu_uuid_threshold_or_seed(self):
        changes = [
            {"allowed_gpu_indices": [0, 1, 2]},
            {"allowed_gpu_uuids": {"0": "GPU-other", "1": scheduler.GPU_UUIDS[1], "3": scheduler.GPU_UUIDS[3]}},
            {"minimum_free_gpu_mb": 12181}, {"minimum_free_gpu_mb": 0},
            {"first_seeds": [1475, 67137, 2025]},
            {"runtime_environment": {"CMERC_ENV_PREFIX": "relative", "CMERC_PYTHON": "/wrapper"}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                scheduler.validate_config({**config(), **change})
        scheduler.validate_config({**config(), "minimum_free_gpu_mb": 12180})

    def test_wrong_host_cannot_create_or_resume_queue(self):
        with patch.object(scheduler.socket, "gethostname", return_value="biggpu"):
            with self.assertRaisesRegex(ValueError, "local user23"):
                self.manager()
        self.assertFalse((self.directory / "state/queue.json").exists())

    def test_gpu_busy_uuid_mismatch_and_wrong_model_are_not_allocated(self):
        manager = self.manager()
        for device in (gpu(pids=(123,)), gpu(uuid="GPU-wrong"), gpu(name="Tesla V100"), gpu(index=2), gpu(used_mb=2000)):
            with self.subTest(device=device), patch.object(scheduler, "query_gpus", return_value=[device]):
                self.assertIsNone(manager.acquire_gpu())
        with patch.object(scheduler, "query_gpus", side_effect=[[gpu()], [gpu(pids=(999,))]]):
            self.assertIsNone(manager.acquire_gpu())

    def test_query_has_xp_model_and_processes_without_ecc_fields(self):
        run_results = [Mock(stdout=f"0, {scheduler.GPU_UUIDS[0]}, 12288, 108, 0, NVIDIA TITAN Xp, 12076\n"),
                       Mock(stdout=f"{scheduler.GPU_UUIDS[0]}, 123\n")]
        with patch.object(scheduler.subprocess, "run", side_effect=run_results) as run:
            devices = scheduler.query_gpus()
        self.assertEqual(devices[0].pids, (123,))
        self.assertEqual(devices[0].name, "NVIDIA TITAN Xp")
        self.assertEqual(devices[0].free_mb, 12076)
        self.assertIn("name", run.call_args_list[0].args[0][1])
        self.assertNotIn("ecc", run.call_args_list[0].args[0][1])

    def test_actual_free_memory_blocks_gpu_even_when_nominal_difference_is_sufficient(self):
        manager = self.manager()
        device = gpu(total_mb=12288, used_mb=1200, free_mb=10984)
        self.assertGreaterEqual(device.total_mb - device.used_mb, 11000)
        with patch.object(scheduler, "query_gpus", return_value=[device]):
            self.assertIsNone(manager.acquire_gpu())
        with patch.object(scheduler, "query_gpus", return_value=[gpu(free_mb=11000)]):
            allocation = manager.acquire_gpu()
            self.assertIsNotNone(allocation)
            allocation[1].close()

    def test_actual_nvidia_prefixed_and_legacy_xp_names_can_be_allocated(self):
        manager = self.manager()
        for name in ("NVIDIA TITAN Xp", "TITAN Xp"):
            with self.subTest(name=name), patch.object(scheduler, "query_gpus", return_value=[gpu(name=name)]):
                allocated = manager.acquire_gpu()
                self.assertIsNotNone(allocated)
                self.assertEqual(allocated[0].name, name)
                allocated[1].close()

    def test_source_snapshot_also_rechecks_external_symlink_input_hash(self):
        root = self.directory / "batch/code"
        (root / "snapshots").mkdir(parents=True)
        source = root / "model.py"
        source.write_text("frozen source\n")
        files = {"model.py": hashlib.sha256(source.read_bytes()).hexdigest()}
        snapshot = {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        scheduler.atomic_json(root / "snapshots/source_snapshot.json", snapshot)
        external_input = self.directory / "shared_features.pkl"
        external_input.write_bytes(b"frozen features")
        symlink = root / "features.pkl"
        symlink.symlink_to(external_input)
        scheduler.atomic_json(root.parent / "provenance/immutable_inputs.json", {
            str(symlink): hashlib.sha256(external_input.read_bytes()).hexdigest(),
        })
        self.assertEqual(scheduler.verify_source_snapshot(root), snapshot["sha256"])
        external_input.write_bytes(b"unexpected external change")
        with self.assertRaisesRegex(ValueError, "Immutable input/tool changed"):
            scheduler.verify_source_snapshot(root)
        # Additional external references never relax the source-file boundary.
        files = {"../outside.py": "bad-sha"}
        snapshot = {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        scheduler.atomic_json(root / "snapshots/source_snapshot.json", snapshot)
        with self.assertRaisesRegex(ValueError, "path escapes"):
            scheduler.verify_source_snapshot(root)

    def test_lock_and_inherited_child_reservation_survive_parent_close(self):
        path = self.directory / "locks/gpu.lock"
        handle = scheduler.acquire_lock(path)
        self.assertIsNone(scheduler.acquire_lock(path))
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

    def test_every_output_and_memory_sidecar_is_guarded(self):
        job = self.jobs()[0]
        paths = [Path(job[key]) for key in ("run_dir", "result_dir", "train_log", "evaluation_log")]
        paths += [scheduler.memory_profile_path(job[key]) for key in ("train_log", "evaluation_log")]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            with self.assertRaises(FileExistsError):
                scheduler.output_guard(job)
            path.unlink()

    def test_train_and_eval_use_uuid_runtime_wrapper_and_separate_profiles(self):
        manager = self.manager()
        job = manager.state["jobs"][0]
        profile_paths = []
        for phase in ("training", "evaluation"):
            lock = scheduler.acquire_lock(self.directory / f"locks/{phase}.lock")
            process = Mock(pid=123)
            with patch.object(manager, "check_disk"), patch.object(scheduler, "verify_source_snapshot", return_value="frozen"), \
                    patch.object(scheduler, "query_gpus", return_value=[gpu()]), \
                    patch.object(scheduler, "pid_identity", return_value={"start_ticks": "42"}), \
                    patch.object(scheduler.subprocess, "Popen", return_value=process) as popen:
                manager.start(job, phase, gpu(), lock)
            command, environment = popen.call_args.args[0], popen.call_args.kwargs["env"]
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], scheduler.GPU_UUIDS[0])
            self.assertEqual(environment["CMERC_PYTHON"], config()["runtime_environment"]["CMERC_PYTHON"])
            self.assertEqual(environment["CMERC_ENV_PREFIX"], "/xp/environment")
            self.assertEqual(environment["PYTORCH_CUDA_ALLOC_CONF"], "garbage_collection_threshold:0.8")
            profile_paths.append(environment["CMERC_PROFILE_OUTPUT"])
            if phase == "evaluation":
                self.assertEqual(command[2], environment["CMERC_PYTHON"])
            self.assertEqual(popen.call_args.kwargs["pass_fds"], (lock.fileno(),))
            self.assertEqual(job[f"{phase}_pid"], 123)
            self.assertEqual(job[f"{phase}_gpu_uuid"], scheduler.GPU_UUIDS[0])
            self.assertEqual(job[f"{phase}_host_gpu_index"], 0)
            manager.release(job)
        self.assertNotEqual(profile_paths[0], profile_paths[1])

    def test_training_to_eval_waits_for_stale_utilization_while_lease_is_held(self):
        manager = self.manager()
        job = manager.state["jobs"][0]
        job.update(status="running_training", phase="training", pid=123, pid_start_ticks="77",
                   host_gpu_index=0, gpu_uuid=scheduler.GPU_UUIDS[0])
        manager.processes[job["id"]] = Mock(poll=Mock(return_value=0))
        lease_path = self.directory / "locks/transition.lock"
        lease = scheduler.acquire_lock(lease_path)
        manager.gpu_locks[job["id"]] = lease
        def lease_stays_held(_):
            self.assertIsNone(scheduler.acquire_lock(lease_path))
        with patch.object(scheduler, "validate_training"), \
                patch.object(scheduler, "query_gpus", side_effect=[[gpu(utilization=80)], [gpu(utilization=0)]]), \
                patch.object(scheduler.time, "sleep", side_effect=lease_stays_held) as sleep, \
                patch.object(manager, "start") as start, patch.object(manager, "summarize"):
            manager.refresh(job)
        start.assert_called_once_with(job, "evaluation", gpu(utilization=0), lease)
        sleep.assert_called_once_with(1)
        self.assertEqual(job["training_exit_code"], 0)
        self.assertEqual(job["evaluation_idle_wait"]["attempts"], 2)
        self.assertTrue(job["evaluation_idle_wait"]["passed"])
        self.assertNotEqual(job["status"], "failed")
        manager.release(job)

    def test_eval_idle_wait_timeout_preserves_blocking_process_reason(self):
        manager = self.manager()
        job = manager.state["jobs"][0]
        job.update(host_gpu_index=0, gpu_uuid=scheduler.GPU_UUIDS[0])
        with patch.object(scheduler, "query_gpus", return_value=[gpu(pids=(999,))]), \
                patch.object(scheduler.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "remained blocked.*999"):
                manager.wait_for_evaluation_gpu(job, attempts=2)
        sleep.assert_called_once_with(1)
        self.assertFalse(job["evaluation_idle_wait"]["passed"])
        self.assertEqual(job["evaluation_idle_wait"]["last_gpu"]["pids"], (999,))

    def test_attach_accepts_uuid_and_rejects_foreign_gpu_process(self):
        manager = self.manager()
        job = manager.state["jobs"][0]
        identity = {"start_ticks": "777", "command": scheduler.training_command(job)}
        environment = f"CUDA_VISIBLE_DEVICES={scheduler.GPU_UUIDS[0]}\0".encode()
        with patch.object(scheduler, "matching_live_pid", return_value=(123, identity)), \
                patch.object(Path, "read_bytes", return_value=environment), \
                patch.object(scheduler, "query_gpus", return_value=[gpu(pids=(123,), name="NVIDIA TITAN Xp")]):
            self.assertTrue(manager.attach(job))
        self.assertEqual(job["gpu_uuid"], scheduler.GPU_UUIDS[0])
        self.assertEqual(job["pid_start_ticks"], "777")
        with patch.object(scheduler, "matching_live_pid", return_value=(123, identity)), \
                patch.object(Path, "read_bytes", return_value=environment), \
                patch.object(scheduler, "query_gpus", return_value=[gpu(pids=(123, 999))]):
            with self.assertRaisesRegex(RuntimeError, "another GPU task"):
                manager.attach(job)

    def test_failure_keeps_artifacts_and_exit_code_without_retry(self):
        manager = self.manager()
        job = manager.state["jobs"][0]
        marker = Path(job["run_dir"]) / "preserved.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("keep")
        job.update(status="running_training", phase="training", pid=123, pid_start_ticks="77")
        manager.processes[job["id"]] = Mock(poll=Mock(return_value=7))
        with patch.object(manager, "summarize"), contextlib.redirect_stdout(io.StringIO()):
            manager.refresh(job)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["training_exit_code"], 7)
        self.assertEqual(marker.read_text(), "keep")
        for anchor in manager.state["jobs"][:4]:
            anchor["status"] = "failed"
        with patch.object(manager, "start") as start:
            self.assertEqual(manager.run(), 1)
            start.assert_not_called()

    def test_four_anchor_gate_validates_stage_without_partial_summary_completeness(self):
        manager = self.manager()
        for job in manager.state["jobs"][:4]:
            job["status"] = "completed"
        def stop_after_iteration(_):
            manager.stop = True
        with patch.object(scheduler, "validate_stage", return_value={"runs": 4}) as gate, \
                patch.object(manager, "summarize") as summarize, patch.object(manager, "acquire_gpu", return_value=None), \
                patch.object(scheduler, "matching_live_pid", return_value=None), patch.object(scheduler.time, "sleep", side_effect=stop_after_iteration):
            self.assertEqual(manager.run(), 130)
        self.assertEqual(len(gate.call_args.args[0]), 4)
        self.assertEqual(manager.state["first_seed_gate"], {"runs": 4})
        summarize.assert_called_once_with(scheduler.SEEDS)

    def test_summary_uses_full_five_seeds_explicit_batch_paths_and_optional_history(self):
        settings = config()
        settings.update(statistics_root=str(self.directory / "statistics"), summary_report=str(self.directory / "reports/XP_MAINLINE_REPRODUCTION.md"),
                        historical_reference=str(self.directory / "missing_history.json"))
        manager = self.manager(settings=settings)
        response = Mock(returncode=0, stdout="", stderr="")
        with patch.object(scheduler.subprocess, "run", return_value=response) as run:
            manager.summarize([1475], first_stage=True)
        command = run.call_args.args[0]
        self.assertEqual(scheduler.argument_value(command, "--seeds"), "1475,67137,2024,3407,42")
        self.assertEqual(scheduler.argument_value(command, "--output"), str(self.directory / "statistics/summary.json"))
        self.assertEqual(scheduler.argument_value(command, "--report"), settings["summary_report"])
        self.assertNotIn("--historical", command)

    def test_dry_run_has_twenty_jobs_custom_paths_and_no_gpu_process(self):
        configuration = self.directory / "config.json"
        configuration.write_text(json.dumps(config()))
        output = io.StringIO()
        with patch.object(scheduler, "query_gpus", return_value=[gpu()]), \
                patch.object(scheduler.subprocess, "Popen") as popen, contextlib.redirect_stdout(output):
            exit_code = scheduler.main(["--config", str(configuration), "--dry-run", "--run-root", str(self.directory / "custom_runs"),
                                        "--result-root", str(self.directory / "custom_results"), "--log-root", str(self.directory / "custom_logs")])
        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(len(result["jobs"]), 20)
        self.assertTrue(all(job["run_dir"].startswith(str(self.directory / "custom_runs")) for job in result["jobs"]))
        self.assertEqual(result["jobs"][0]["evaluation_command"][2], "/batch/tools/xp_python")
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
