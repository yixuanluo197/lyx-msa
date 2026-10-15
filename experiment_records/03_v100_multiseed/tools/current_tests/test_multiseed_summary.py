"""CPU-only summary checks; synthetic metrics are never formal experiments."""

import json
import hashlib
import math
from pathlib import Path
import tempfile
import os
import subprocess
import contextlib
import io
import sys
import types
import xml.etree.ElementTree as ET
from unittest import mock
import unittest

from scripts.summarize_results import collect_multiseed, save_multiseed
from scripts import summarize_results
from scripts import evaluate_checkpoint


class MultiSeedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root, self.result_root = self.root / "runs", self.root / "results"

    def tearDown(self):
        self.temporary.cleanup()

    def make_run(self, seed, variant="baseline", dataset="IEMOCAP", score=60.0,
                 split_hash="fixed", causal=False, epochs=None):
        run = self.run_root / variant / f"{dataset}_validation_seed{seed}"
        run.mkdir(parents=True)
        metadata = {
            "args": {"seed": seed, "Dataset": dataset, "selection_protocol": "validation",
                     "split_seed": 1475, "rng_policy": "continuous", "rnn_backend": "aten", "causal": causal,
                     "batch_policy": "capacity_safe", "edge_capacity": 1000, "node_capacity": 5200, "batch_max_retries": 10000},
            "provenance": {"split": {"train_ids": ["train"], "validation_ids": ["valid"],
                                     "test_ids": ["test"]},
                           "source": {"base_commit": "fixture", "files": {"train_cm.py": "fixture"}},
                           "data": {"features.pkl": "fixture"}},
        }
        if split_hash != "fixed":
            metadata["provenance"]["split"]["train_ids"] = [split_hash]
        split = metadata["provenance"]["split"]
        split["hash"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
        (run / "run_manifest.json").write_text(json.dumps(metadata))
        budget = 80 if dataset == "IEMOCAP" else 40
        records = [{"epoch": epoch, "selection_protocol": "validation", "causal": causal,
                    "selection_weighted_f1": 50.0 + epoch / 100,
                    "train_weighted_f1": 70.0, "validation_weighted_f1": 50.0,
                    "test_accuracy": None, "test_weighted_f1": None, "total_loss": 1.0,
                    "train_loss": 1.0 - epoch / 200, "validation_loss": .8 + epoch / 500,
                    "sampler_retries": 1, "max_batch_edges": 20, "max_batch_nodes": 60, "max_gpu_memory_mb": 256.0,
                    "elapsed_seconds": 2.0} for epoch in range(1, (epochs or budget) + 1)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records))
        for name in ("best.pt", "latest.pt"):
            (run / name).touch()
        (run / "final_test.json").write_text("{}")
        result = self.result_root / variant / dataset / "validation" / f"seed{seed}"
        result.mkdir(parents=True)
        metrics = {"causal": causal, "effective_norm": "BN", "checkpoint": str(run / "best.pt"),
                   "seed": seed, "split_hash": split["hash"], "selected_epoch": budget,
                   "rnn_backend": "aten",
                   "batch_policy": "capacity_safe", "edge_capacity": 1000, "node_capacity": 5200, "batch_max_retries": 10000,
                   "provenance": metadata["provenance"],
                   "accuracy": score, "weighted_f1": score, "macro_f1": score - 1,
                   "cel": {"total": 1000.0 - score}, "calibration": {"ece": 0.1}}
        (result / "metrics.json").write_text(json.dumps(metrics))
        return run, result

    def test_all_planned_seeds_retained_and_sample_std(self):
        for seed, score in ((1475, 60.0), (67137, 62.0), (2024, 64.0)):
            self.make_run(seed, score=score)
            self.make_run(seed, "cmerc", score=score + 1)
        payload = collect_multiseed(self.run_root, self.result_root, [1475, 67137, 2024])
        group = payload["groups"][0]
        self.assertEqual(payload["planned_runs"], 12)
        self.assertEqual(payload["valid_runs"], 6)
        self.assertEqual(group["metrics"]["weighted_f1"]["mean"], 62.0)
        self.assertEqual(group["metrics"]["weighted_f1"]["std"], 2.0)
        self.assertEqual(len(payload["pairs"]), 3)
        self.assertTrue(all(pair["weighted_f1_gain"] == 1 for pair in payload["pairs"]))
        self.assertFalse(payload["complete"])

    def test_split_conflict_invalidates_dataset_not_selectively_drops_seed(self):
        self.make_run(1475)
        self.make_run(1475, "cmerc", split_hash="other")
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["valid_runs"], 0)
        self.assertTrue(all(row["status"] == "failed" for row in payload["registry"] if row["dataset"] == "IEMOCAP"))

    def test_causal_and_partial_training_are_excluded(self):
        self.make_run(1475, causal=True)
        self.make_run(1475, "cmerc", epochs=3)
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["registry"][0]["status"], "failed")
        self.assertEqual(payload["registry"][1]["status"], "pending")
        self.assertEqual(payload["valid_runs"], 0)

    def test_nonfinite_evaluation_and_epoch_test_score_rejected(self):
        _, result = self.make_run(1475, score=math.nan)
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["registry"][0]["status"], "failed")
        run, _ = self.make_run(1475, "cmerc")
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        records[0]["test_weighted_f1"] = 90
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records))
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["registry"][1]["status"], "failed")

    def test_unplanned_seed_not_filtered_in_by_good_score(self):
        self.make_run(999, score=99)
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["unplanned_seeds"], [999])
        self.assertEqual(payload["valid_runs"], 0)

    def test_output_csv_report_no_seed_collision(self):
        self.make_run(1475, score=60)
        self.make_run(67137, score=70)
        payload = collect_multiseed(self.run_root, self.result_root, [1475, 67137])
        output, report = self.root / "out" / "summary.json", self.root / "report.md"
        save_multiseed(payload, output, report, self.root / "absent.json", plots=False)
        self.assertEqual(len(json.loads(output.read_text())["runs"]), 2)
        self.assertIn("部分完成", report.read_text())
        self.assertTrue((output.parent / "registry.csv").is_file())
        self.assertEqual(len((output.parent / "per_seed.csv").read_text().splitlines()), 3)

    def test_v2_sampling_metadata_and_budget_are_verified(self):
        run, result = self.make_run(1475)
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["runs"][0]["sampler_retries_total"], 80)
        self.assertEqual(payload["runs"][0]["max_batch_edges"], 20)
        self.assertEqual(payload["runs"][0]["max_gpu_memory_mb"], 256.0)
        evaluation = json.loads((result / "metrics.json").read_text())
        evaluation["batch_policy"] = "legacy"
        (result / "metrics.json").write_text(json.dumps(evaluation))
        self.assertEqual(collect_multiseed(self.run_root, self.result_root, [1475])["registry"][0]["status"], "failed")
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        records[0]["max_batch_edges"] = 1049
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records))
        entry = collect_multiseed(self.run_root, self.result_root, [1475])["registry"][0]
        self.assertEqual(entry["status"], "failed")
        self.assertIn("capacity", entry["reason"])

    def test_v1_complete_result_cannot_be_mixed_with_v2(self):
        run, _ = self.make_run(1475)
        metadata = json.loads((run / "run_manifest.json").read_text())
        metadata["args"]["batch_policy"] = "legacy"
        (run / "run_manifest.json").write_text(json.dumps(metadata))
        self.make_run(1475, "cmerc")
        payload = collect_multiseed(self.run_root, self.result_root, [1475])
        self.assertEqual(payload["registry"][0]["status"], "failed")
        self.assertEqual(len(payload["pairs"]), 0)
        self.assertEqual(payload["valid_runs"], 1)

    def test_official_test_cannot_be_primary_stats(self):
        with self.assertRaises(ValueError):
            collect_multiseed(self.run_root, self.result_root, [1475], "official_test")

    def test_evaluator_requires_noncausal_flag_is_supported_and_conflict_rejected(self):
        argv = ["--dataset", "IEMOCAP", "--checkpoint", "best.pt", "--output-dir", "out", "--require-noncausal"]
        self.assertTrue(evaluate_checkpoint.parse_args(argv).require_noncausal)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_checkpoint.parse_args(argv + ["--causal"])

    def test_evaluator_matches_checkpoint_recurrent_backend_without_changing_legacy(self):
        models, calls = [object(), object(), object()], []
        helper = types.SimpleNamespace(configure_recurrent_backend=lambda model, backend: calls.append((model, backend)))
        with mock.patch.dict(sys.modules, {"train_cm": helper}):
            self.assertEqual(evaluate_checkpoint.configure_evaluation_backend(models, {"args": {"rnn_backend": "aten"}}), "aten")
            self.assertEqual(calls, [(model, "aten") for model in models])
            calls.clear()
            self.assertEqual(evaluate_checkpoint.configure_evaluation_backend(models, {"weight": 1}), "cudnn")
            self.assertEqual(calls, [])

    def test_checkpoint_metadata_exports_identity_and_one_based_epoch(self):
        checkpoint = {"args": {"seed": 2024, "rnn_backend": "aten", "causal": False,
                               "batch_policy": "capacity_safe", "edge_capacity": 1000,
                               "node_capacity": 5200, "batch_max_retries": 10000}, "epoch": 3,
                      "provenance": {"split": {"hash": "fixed"}, "source": {"commit": "fixture"}}}
        metadata = evaluate_checkpoint.checkpoint_provenance(checkpoint)
        self.assertEqual(metadata["selected_epoch"], 4)
        self.assertEqual(metadata["seed"], 2024)
        self.assertEqual(metadata["rnn_backend"], "aten")
        self.assertEqual(metadata["batch_policy"], "capacity_safe")
        self.assertEqual(metadata["edge_capacity"], 1000)
        self.assertEqual(metadata["split_hash"], "fixed")
        self.assertEqual(evaluate_checkpoint.checkpoint_provenance({"layer.weight": 0}), {})

    def test_summary_cli_without_matplotlib_writes_real_series_svg_and_report(self):
        for seed in (1475, 67137):
            self.make_run(seed)
            self.make_run(seed, "cmerc")
        output, report = self.root / "out" / "summary.json", self.root / "report.md"
        argv = ["summarize_results.py", "--multiseed", "--run-root", str(self.run_root),
                "--result-root", str(self.result_root), "--output", str(output),
                "--report", str(report), "--seeds", "1475,67137"]
        with mock.patch.dict(sys.modules, {"matplotlib": None}), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summarize_results.main(), 0)
        payload = json.loads(output.read_text())
        self.assertEqual(payload["valid_runs"], 4)
        self.assertTrue(report.exists())
        self.assertIn("部分完成", report.read_text())
        self.assertTrue((output.parent / "per_seed.csv").is_file())
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        root = ET.parse(output.parent / "figures" / "IEMOCAP_convergence.svg").getroot()
        curves = root.findall("svg:polyline", namespace)
        self.assertEqual(len(curves), 18)  # 10 actual loss series + 8 actual W-F1 series.
        self.assertTrue(all(int(curve.attrib["data-point-count"]) == 80 for curve in curves))
        labels = [element.text for element in root.findall("svg:text", namespace)]
        self.assertIn("seed1475 train CE", labels)
        self.assertIn("seed67137 valid CE", labels)
        self.assertIn("seed1475 train total", labels)
        self.assertIn("seed1475 valid W-F1", labels)
        styles = {curve.attrib["data-series"]: curve.attrib for curve in curves}
        self.assertEqual(styles["seed1475 train total"]["stroke-dasharray"], "2,3")
        self.assertEqual(styles["seed1475 valid CE"]["stroke-dasharray"], "6,3")
        selected = ET.parse(output.parent / "figures" / "selected_epochs.svg").getroot()
        self.assertEqual(len(selected.findall("svg:circle", namespace)), 4)
        meld = ET.parse(output.parent / "figures" / "MELD_convergence.svg").getroot()
        self.assertEqual(len(meld.findall("svg:polyline", namespace)), 0)
        self.assertIn("No completed runs", [element.text for element in meld.findall("svg:text", namespace)])

    def test_shell_seed_output_paths_are_isolated_and_logs_never_overwrite(self):
        self.make_run(1475)
        self.make_run(2024)
        # /bin/echo records invocation only: this is a routing test, not inference.
        log_root = self.root / "eval_logs"
        script = Path(__file__).parents[1] / "scripts" / "evaluate_training_matrix.sh"
        env = {**os.environ, "CMERC_RUN_DIR": str(self.run_root),
               "CMERC_RESULT_DIR": str(self.root / "eval_output"),
               "CMERC_EVAL_LOG_DIR": str(log_root), "CMERC_PYTHON": "/bin/echo"}
        args = ["bash", str(script), "--seeds", "1475,2024", "IEMOCAP", "baseline", "validation"]
        first = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        for seed in (1475, 2024):
            log = log_root / f"IEMOCAP_baseline_validation_seed{seed}.log"
            contents = log.read_text()
            self.assertIn(f"IEMOCAP_validation_seed{seed}/best.pt", contents)
            self.assertIn(f"baseline/IEMOCAP/validation/seed{seed}", contents)
        saved = (log_root / "IEMOCAP_baseline_validation_seed1475.log").read_bytes()
        second = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(second.returncode, 3)
        self.assertEqual(saved, (log_root / "IEMOCAP_baseline_validation_seed1475.log").read_bytes())


if __name__ == "__main__":
    unittest.main()
