#!/usr/bin/env python3
"""Independent consistency checks for the two completed Teacher pilots."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"IEMOCAP": {"epochs": 80, "classes": 6}, "MELD": {"epochs": 40, "classes": 7}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    manifest_path = ROOT / "configs/approved_split_manifests.reference.json"
    manifest = json.loads(manifest_path.read_text())
    generator = import_script(ROOT / "scripts/generate_fold_local_difficulty.py", "pilot_generator_verify")
    report = {"status": "passed", "manifest_sha256": sha256(manifest_path), "datasets": {}}
    for dataset, expected in DATASETS.items():
        run = ROOT / "runs" / dataset / "fold1"
        teacher = run / "teacher"
        checkpoint_path = teacher / "teacher_final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = json.loads((teacher / "teacher_checkpoint_metadata.json").read_text())
        access = json.loads((teacher / "access_audit.json").read_text())
        epochs = [json.loads(line) for line in (teacher / "epochs.jsonl").read_text().splitlines()]
        training_ids = manifest["protocols"][dataset]["folds"]["fold1"]["roles"]["training"]["dialogue_ids"]
        other_ids = set()
        for role in ("development", "calibration", "evaluation"):
            other_ids.update(manifest["protocols"][dataset]["folds"]["fold1"]["roles"][role]["dialogue_ids"])
        assert len(epochs) == expected["epochs"] == checkpoint["epoch"]
        assert metadata["checkpoint_sha256"] == sha256(checkpoint_path)
        assert access["exact_final_coverage"] and access["rejected_ids"] == []
        assert access["requested_ids"] == training_ids and set(access["accessed_ids"]) == set(training_ids)
        assert set(access["accessed_ids"]).isdisjoint(other_ids)
        assert all(row["exact_training_coverage"] and row["max_batch_edges"] <= 1000
                   and row["max_batch_nodes"] <= 5200 for row in epochs)
        assert checkpoint["selection"] == "fixed_final_epoch" and checkpoint["access_roles"] == ["training"]

        difficulty_path = ROOT / "results" / f"{dataset}_fold1_difficulty.json"
        difficulty = json.loads(difficulty_path.read_text())
        rows = difficulty["dialogues"]
        assert [row["dialogue_id"] for row in rows] == training_ids
        assert {row["dialogue_id"] for row in rows}.isdisjoint(other_ids)
        numeric = ("speaker_term", "text_gap", "audio_gap", "visual_gap", "intra_gap",
                   "inter_gap", "total_gap", "final_difficulty", "difficulty_total_length")
        assert all(all(math.isfinite(float(row[key])) for key in numeric) for row in rows)
        for row in rows:
            expected_difficulty = (row["total_gap"] + row["speaker_term"]) / (row["valid_label_count"] + row["speaker_term"])
            assert math.isclose(row["final_difficulty"], expected_difficulty, rel_tol=1e-12, abs_tol=1e-12)

        probability_path = ROOT / "results" / f"{dataset}_fold1_six_view_probabilities.npz"
        archive = np.load(probability_path, allow_pickle=False)
        assert archive["dialogue_ids"].tolist() == training_ids
        assert len(archive["offsets"]) == len(training_ids) + 1
        assert archive["offsets"][0] == 0 and np.all(np.diff(archive["offsets"]) > 0)
        total = int(archive["offsets"][-1])
        assert archive["labels"].shape == archive["supervision_mask"].shape == (total,)
        for view in ("full", "text", "audio", "visual", "intra", "inter"):
            probabilities = archive[f"{view}_probabilities"]
            assert probabilities.shape == (total, expected["classes"])
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1.0, atol=1e-6, rtol=1e-5)
        assert difficulty["probabilities_sha256"] == sha256(probability_path)

        values = np.asarray([row["final_difficulty"] for row in rows])
        report["datasets"][dataset] = {
            "training_dialogues": len(training_ids), "training_utterances": total,
            "unknown_masked": int((~archive["supervision_mask"]).sum()),
            "epochs": len(epochs), "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_tensor_count": len(checkpoint["model_state_dict"]),
            "difficulty_sha256": sha256(difficulty_path),
            "probabilities_sha256": sha256(probability_path),
            "difficulty_min": float(values.min()), "difficulty_mean": float(values.mean()),
            "difficulty_max": float(values.max()),
            "max_sampler_retries": max(row["sampler_retries"] for row in epochs),
            "max_batch_edges": max(row["max_batch_edges"] for row in epochs),
            "max_batch_nodes": max(row["max_batch_nodes"] for row in epochs),
            "roles_excluded": ["development", "calibration", "evaluation"],
        }
    output = ROOT / "results/final_artifact_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
