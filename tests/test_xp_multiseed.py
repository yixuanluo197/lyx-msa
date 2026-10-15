"""CPU-only checks for batch identity, restart safety and honest statistics."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.xp_multiseed import (
    GPU_CANDIDATES, SEEDS, atomic_json, checkpoint_metadata, command_for, create_batch,
    eligible_gpus, expected_config, meld_concurrency_policy, query_gpus, refuse_duplicate, summarize, task_matrix,
    task_paths, validate_training, run_batch,
)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "batch"
        self.code = self.base / "code"
        (self.code / "scripts").mkdir(parents=True)
        (self.code / "scripts/run_training.sh").write_text("#!/bin/bash\n")
        self.manifest = create_batch(self.root, self.code, self.base / "env")

    def state(self):
        return json.loads((self.root / "state.json").read_text())

    def test_matrix_has_fixed_twelve_unique_tasks_and_anchors_first(self):
        matrix = task_matrix()
        self.assertEqual(len(matrix), 12)
        self.assertEqual(len({task["id"] for task in matrix}), 12)
        self.assertEqual([(task["dataset"], task["variant"], task["seed"]) for task in matrix[:4]], [
            ("IEMOCAP", "cmerc", 1475), ("MELD", "cmerc", 67137),
            ("IEMOCAP", "baseline", 1475), ("MELD", "baseline", 67137),
        ])
        for dataset, seeds in SEEDS.items():
            group = [task for task in matrix if task["dataset"] == dataset and task["variant"] == "cmerc"]
            self.assertEqual([task["seed"] for task in group], seeds)
            self.assertTrue(all(task["causal"] is False and task["protocol"] == "official_test" for task in group))

    def test_create_cannot_replace_existing_batch(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_batch(self.root, self.code, self.base / "other_env")

    def test_duplicate_guard_rejects_live_failed_or_existing_run(self):
        task = task_matrix()[0]
        with patch("scripts.xp_multiseed.pid_alive", return_value=True):
            with self.assertRaisesRegex(ValueError, "live PID"):
                refuse_duplicate(self.root, task, {"status": "planned", "pid": 123})
        with patch("scripts.xp_multiseed.pid_alive", return_value=False):
            with self.assertRaisesRegex(ValueError, "automatic retry"):
                refuse_duplicate(self.root, task, {"status": "failed", "pid": None})
            run, _ = task_paths(self.root, task)
            run.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "existing artifacts"):
                refuse_duplicate(self.root, task, {"status": "planned", "pid": None})

    def test_command_contains_seed_and_separate_evaluation_path(self):
        task = next(task for task in task_matrix() if task["dataset"] == "MELD" and task["seed"] == 42)
        command = command_for(self.manifest, self.root, task, "training")
        self.assertEqual(command[-4:], ["--seed", "42", "--num-workers", "0"])
        self.assertNotIn("--epochs", command)
        self.assertNotIn("--causal", command)
        evaluation = command_for(self.manifest, self.root, task, "evaluation")
        self.assertIn(str(self.root / "results/cmerc/MELD/official_test/seed42"), evaluation)

    def test_missing_duplicate_epochs_and_causal_config_fail(self):
        task = task_matrix()[0]
        run, _ = task_paths(self.root, task)
        run.mkdir(parents=True)
        records = [{"epoch": index, "causal": False, "effective_norm": "BN",
                    "selection_protocol": "official_test", "elapsed_seconds": 3,
                    "max_gpu_memory_mb": 9000} for index in range(1, 81)]
        metadata = {"args": expected_config(task), "epoch": 79, "best_epoch": 4, "causal": False}
        with self.assertRaises(FileNotFoundError):
            validate_training(self.root, task, metadata)
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records[:-1]))
        with self.assertRaisesRegex(ValueError, "missing, duplicated"):
            validate_training(self.root, task, metadata)
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records + [records[-1]]))
        with self.assertRaisesRegex(ValueError, "missing, duplicated"):
            validate_training(self.root, task, metadata)
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
        metadata["args"]["causal"] = True
        with self.assertRaisesRegex(ValueError, "config causal"):
            validate_training(self.root, task, metadata)
        metadata["args"]["causal"] = False
        for filename in ("best.pt", "latest.pt"):
            (run / filename).touch()
        (run / "memory_profile.jsonl").write_text(json.dumps({"peak_reserved_mb": 10300}) + "\n")
        self.assertEqual(validate_training(self.root, task, metadata)["max_reserved_mib"], 10300)

    def complete(self, task, score, state):
        run, result = task_paths(self.root, task)
        result.mkdir(parents=True)
        atomic_json(result / "metrics.json", {
            "dataset": task["dataset"], "causal": False,
            "checkpoint": str(run / "best.pt"), "weighted_f1": score,
            "accuracy": score + 1, "macro_f1": score - 1,
            "cel": {"total": 1000, "by_removal": {name: 200 for name in (
                "text", "audio", "visual", "intra_context", "inter_context")}},
        })
        for filename in ("predictions.npz", "predictions.csv"):
            (result / filename).touch()
        state["tasks"][task["id"]].update(status="completed", exitcode=0)

    def test_partial_or_missing_evaluation_cannot_be_reported_as_five_seed_mean(self):
        state = self.state()
        task = task_matrix()[0]
        self.complete(task, 71.9, state)
        atomic_json(self.root / "state.json", state)
        report = summarize(self.root)
        self.assertEqual(report["groups"]["IEMOCAP"]["completed_n"], 1)
        self.assertIsNone(report["groups"]["IEMOCAP"]["mean"])
        self.assertFalse(report["both_datasets_within_half_point"])
        (task_paths(self.root, task)[1] / "predictions.npz").unlink()
        report = summarize(self.root)
        self.assertEqual(report["groups"]["IEMOCAP"]["completed_n"], 0)
        self.assertIn("missing independent evaluation artifact", (self.root / "results.csv").read_text())

    def test_sample_std_uses_ddof_one_and_failed_runs_are_retained(self):
        state = self.state()
        tasks = [task for task in task_matrix() if task["dataset"] == "IEMOCAP" and task["variant"] == "cmerc"]
        for task, score in zip(tasks, [70, 71, 72, 73, 74]):
            self.complete(task, score, state)
        baseline = task_matrix()[2]
        state["tasks"][baseline["id"]].update(status="failed", exitcode=137, error="OOM")
        atomic_json(self.root / "state.json", state)
        report = summarize(self.root)
        group = report["groups"]["IEMOCAP"]
        self.assertEqual(group["mean"], 72)
        self.assertAlmostEqual(group["sample_std"], math.sqrt(2.5))
        self.assertEqual(group["minimum"], 70)
        self.assertEqual(group["maximum"], 74)
        self.assertTrue(group["within_half_point"])
        self.assertFalse(report["both_datasets_within_half_point"])
        self.assertIn("OOM", (self.root / "REPORT.md").read_text())
        self.assertEqual(len((self.root / "results.csv").read_text().splitlines()), 13)

    def test_gpu_inventory_retains_pid_ownership_and_filters_wrong_uuid(self):
        inventory = "\n".join(f"{index}, {uuid}, TITAN Xp, 12000" for index, uuid in GPU_CANDIDATES.items())
        inventory += "\n2, GPU-other-user, TITAN Xp, 12000\n"
        with patch("scripts.xp_multiseed.subprocess.run") as run:
            run.side_effect = [type("Result", (), {"stdout": inventory})(),
                               type("Result", (), {"stdout": f"{GPU_CANDIDATES[1]}, 123, 2000\n"})()]
            candidates = query_gpus()
        self.assertEqual([gpu["index"] for gpu in candidates], [0, 1, 3])
        self.assertEqual(candidates[1]["process_pids"], [123])
        eligible = eligible_gpus(task_matrix()[0], "training", candidates, {}, {"max_meld_per_gpu": 1})
        self.assertEqual([gpu["index"] for gpu in eligible], [0, 3])

    def test_preflight_missing_or_failed_prevents_any_gpu_launch(self):
        with patch("scripts.xp_multiseed.query_gpus") as query:
            with self.assertRaises(FileNotFoundError):
                run_batch(self.root)
            (self.root / "preflight").mkdir()
            atomic_json(self.root / "preflight/status.json", {"status": "failed"})
            with self.assertRaisesRegex(ValueError, "preflight has not passed"):
                run_batch(self.root)
            query.assert_not_called()

    def test_checkpoint_metadata_uses_matching_environment_and_no_gpu(self):
        task = task_matrix()[0]
        args = expected_config(task)
        payloads = [{"epoch": 79, "best_epoch": 4, "causal": False, "args": args},
                    {"epoch": 4, "best_epoch": 4, "causal": False, "args": args}]
        with patch("scripts.xp_multiseed.subprocess.run") as run:
            run.side_effect = [type("Result", (), {"stdout": json.dumps(payload)})() for payload in payloads]
            metadata = checkpoint_metadata(self.manifest, self.root, task)
        self.assertEqual(metadata["epoch"], 79)
        for invocation in run.call_args_list:
            self.assertEqual(invocation.kwargs["env"]["CMERC_ENV_PREFIX"], self.manifest["env_prefix"])
            self.assertEqual(invocation.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_meld_concurrency_requires_measured_ten_percent_throughput_gain(self):
        self.assertEqual(meld_concurrency_policy(self.root)["max_meld_per_gpu"], 1)
        preflight = self.root / "preflight"
        preflight.mkdir()
        atomic_json(preflight / "MELD_resource_profile.json", {"peak_reserved_mb": 1500})
        evidence = {"enabled": True, "single_task_throughput": 10, "dual_task_total_throughput": 10.9}
        atomic_json(preflight / "meld_concurrency.json", evidence)
        self.assertEqual(meld_concurrency_policy(self.root)["max_meld_per_gpu"], 1)
        evidence["dual_task_total_throughput"] = 11
        atomic_json(preflight / "meld_concurrency.json", evidence)
        policy = meld_concurrency_policy(self.root)
        self.assertEqual(policy["max_meld_per_gpu"], 2)
        self.assertEqual(policy["minimum_free_mib"], 2524)
        evidence["enabled"] = False
        atomic_json(preflight / "meld_concurrency.json", evidence)
        self.assertEqual(meld_concurrency_policy(self.root)["max_meld_per_gpu"], 1)
        evidence = {"enabled": True, "improvement_percent": 50}
        atomic_json(preflight / "meld_concurrency.json", evidence)
        self.assertEqual(meld_concurrency_policy(self.root)["max_meld_per_gpu"], 1)

    def test_meld_dual_candidate_rejects_foreign_pid_iemocap_and_third_task(self):
        task = next(task for task in task_matrix() if task["dataset"] == "MELD")
        gpu = {"index": 1, "uuid": GPU_CANDIDATES[1], "free_mib": 6000, "process_pids": [123]}
        process = type("Process", (), {"pid": 123})()
        children = {"first": (process, task, gpu, "training")}
        policy = {"max_meld_per_gpu": 2, "minimum_free_mib": 2200}
        startup_done = set(GPU_CANDIDATES.values())
        self.assertEqual(eligible_gpus(task, "training", [gpu], children, policy, startup_done), [gpu])
        foreign = {**gpu, "process_pids": [123, 999]}
        self.assertEqual(eligible_gpus(task, "training", [foreign], children, policy, startup_done), [])
        iemocap = task_matrix()[0]
        iemocap_child = {"first": (process, iemocap, gpu, "training")}
        self.assertEqual(eligible_gpus(task, "training", [gpu], iemocap_child, policy, startup_done), [])
        self.assertEqual(eligible_gpus(iemocap, "training", [gpu], children, policy, startup_done), [])
        children["second"] = (type("Process", (), {"pid": 124})(), task, gpu, "evaluation")
        self.assertEqual(eligible_gpus(task, "training", [gpu], children, policy, startup_done), [])
        del children["second"]
        too_full = {**gpu, "free_mib": 2199}
        self.assertEqual(eligible_gpus(task, "training", [too_full], children, policy, startup_done), [])
        self.assertEqual(eligible_gpus(task, "training", [gpu], children,
                                       {"max_meld_per_gpu": 1}, startup_done), [])

    def test_initial_three_training_launches_use_distinct_cards_and_idle_cards_are_preferred(self):
        task = next(task for task in task_matrix() if task["dataset"] == "MELD")
        owned = {"index": 0, "uuid": GPU_CANDIDATES[0], "free_mib": 6000, "process_pids": [123]}
        idle = {"index": 1, "uuid": GPU_CANDIDATES[1], "free_mib": 12000, "process_pids": []}
        children = {"first": (type("Process", (), {"pid": 123})(), task, owned, "training")}
        policy = {"max_meld_per_gpu": 2, "minimum_free_mib": 2200}
        initial = {owned["uuid"]}
        self.assertEqual(eligible_gpus(task, "training", [owned, idle], children, policy, initial), [idle])
        # A finished first task does not permit reusing its card for launch two.
        finished_card = {**owned, "process_pids": []}
        self.assertEqual(eligible_gpus(task, "training", [finished_card, idle], {}, policy, initial), [idle])
        self.assertEqual(eligible_gpus(task, "training", [owned, idle], children, policy,
                                       set(GPU_CANDIDATES.values())), [idle, owned])


if __name__ == "__main__":
    unittest.main()
