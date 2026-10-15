import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from unittest.mock import patch

from dataloader import resolve_meld_split, select_curriculum_keys, split_iemocap_dialogues
from train_cm import (
    choose_selection_metrics,
    combine_training_losses,
    get_IEMOCAP_loaders,
    get_MELD_loaders,
    load_training_checkpoint,
    save_training_checkpoint,
    write_jsonl_record,
)


class DialogueSplitTests(unittest.TestCase):
    def test_iemocap_validation_split_is_seeded_disjoint_and_ten_percent(self):
        keys = [f"dialogue-{index}" for index in range(20)]

        train_keys, valid_keys = split_iemocap_dialogues(keys, seed=1475, valid_fraction=0.1)
        repeated_train, repeated_valid = split_iemocap_dialogues(
            keys, seed=1475, valid_fraction=0.1
        )

        self.assertEqual(len(valid_keys), 2)
        self.assertEqual((train_keys, valid_keys), (repeated_train, repeated_valid))
        self.assertFalse(set(train_keys) & set(valid_keys))
        self.assertEqual(set(train_keys) | set(valid_keys), set(keys))

    def test_iemocap_validation_split_changes_with_seed(self):
        keys = [f"dialogue-{index}" for index in range(30)]
        _, first_valid = split_iemocap_dialogues(keys, seed=1, valid_fraction=0.1)
        _, second_valid = split_iemocap_dialogues(keys, seed=2, valid_fraction=0.1)
        self.assertNotEqual(first_valid, second_valid)

    def test_meld_validation_uses_official_ids_and_excludes_them_from_train(self):
        train_ids = ["train-1", "valid-1", "train-2", "valid-2"]
        valid_ids = ["valid-1", "valid-2"]
        test_ids = ["test-1"]

        self.assertEqual(
            resolve_meld_split(train_ids, valid_ids, test_ids, "train"),
            ["train-1", "train-2"],
        )
        self.assertEqual(
            resolve_meld_split(train_ids, valid_ids, test_ids, "validation"),
            valid_ids,
        )
        self.assertEqual(
            resolve_meld_split(train_ids, valid_ids, test_ids, "test"), test_ids
        )

    def test_curriculum_ratio_is_applied_after_validation_exclusion(self):
        eligible_keys = ["train-1", "train-2", "train-3", "train-4"]
        difficulty = {
            "held-out": 0.0,
            "train-4": 0.1,
            "train-2": 0.2,
            "train-1": 0.3,
            "train-3": 0.4,
        }
        self.assertEqual(
            select_curriculum_keys(eligible_keys, difficulty, ratio=0.5),
            ["train-2", "train-4"],
        )


class _FakeIemocapDataset(Dataset):
    all_train_keys = [f"dialogue-{index}" for index in range(20)]

    def __init__(self, train=True, epoch_ratio=-1, keys=None):
        self.keys = list(keys) if keys is not None else (
            list(self.all_train_keys) if train else ["test-dialogue"]
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.keys[index]

    @staticmethod
    def collate_fn(data):
        return data


class _FakeMeldDataset(Dataset):
    split_keys = {
        "train": ["train-1", "train-2"],
        "validation": ["valid-1"],
        "test": ["test-1"],
    }

    def __init__(self, path, train=True, epoch_ratio=-1, split=None, keys=None):
        if split is not None:
            self.keys = list(self.split_keys[split])
        elif keys is not None:
            self.keys = list(keys)
        else:
            self.keys = ["legacy-train"] if train else ["legacy-test"]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.keys[index]

    @staticmethod
    def collate_fn(data):
        return data


class LoaderProtocolTests(unittest.TestCase):
    @patch("train_cm.IEMOCAPDataset", _FakeIemocapDataset)
    def test_validation_protocol_excludes_iemocap_holdout_from_training(self):
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
            batch_size=4,
            num_workers=0,
            selection_protocol="validation",
            seed=1475,
        )
        self.assertEqual(len(train_loader.dataset.keys), 18)
        self.assertEqual(len(valid_loader.dataset.keys), 2)
        self.assertFalse(set(train_loader.dataset.keys) & set(valid_loader.dataset.keys))
        self.assertEqual(test_loader.dataset.keys, ["test-dialogue"])

    @patch("train_cm.MELDDataset", _FakeMeldDataset)
    def test_validation_protocol_uses_official_meld_splits(self):
        train_loader, valid_loader, test_loader = get_MELD_loaders(
            batch_size=4, num_workers=0, selection_protocol="validation"
        )
        self.assertEqual(train_loader.dataset.keys, ["train-1", "train-2"])
        self.assertEqual(valid_loader.dataset.keys, ["valid-1"])
        self.assertEqual(test_loader.dataset.keys, ["test-1"])

    @patch("train_cm.IEMOCAPDataset", _FakeIemocapDataset)
    def test_default_loader_protocol_preserves_official_full_training_set(self):
        train_loader, _, _ = get_IEMOCAP_loaders(batch_size=4, num_workers=0)
        self.assertEqual(train_loader.dataset.keys, _FakeIemocapDataset.all_train_keys)


class SelectionMetricTests(unittest.TestCase):
    def test_official_protocol_selects_on_test_metrics(self):
        split, loss, f1 = choose_selection_metrics(
            "official_test", valid_loss=1.0, valid_f1=10.0, test_loss=2.0, test_f1=20.0
        )
        self.assertEqual((split, loss, f1), ("test", 2.0, 20.0))

    def test_validation_protocol_selects_only_on_validation_metrics(self):
        split, loss, f1 = choose_selection_metrics(
            "validation", valid_loss=1.0, valid_f1=10.0, test_loss=2.0, test_f1=99.0
        )
        self.assertEqual((split, loss, f1), ("validation", 1.0, 10.0))

    def test_validation_selection_ignores_missing_test_metrics(self):
        split, loss, f1 = choose_selection_metrics(
            "validation",
            valid_loss=1.0,
            valid_f1=10.0,
            test_loss=float("nan"),
            test_f1=float("nan"),
        )
        self.assertEqual((split, loss, f1), ("validation", 1.0, 10.0))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_model_optimizer_epoch_best_and_rng(self):
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            saved_parameters = [parameter.detach().clone() for parameter in model.parameters()]
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=3,
                args={"Dataset": "IEMOCAP", "selection_protocol": "validation"},
                best_metric=71.2,
                best_epoch=2,
            )
            expected_rng_values = (random.random(), np.random.rand(), torch.rand(1).item())

            for parameter in model.parameters():
                parameter.data.zero_()
            random.seed(100)
            np.random.seed(100)
            torch.manual_seed(100)

            metadata = load_training_checkpoint(path, model, optimizer, map_location="cpu")
            actual_rng_values = (random.random(), np.random.rand(), torch.rand(1).item())

            self.assertEqual(metadata["start_epoch"], 4)
            self.assertEqual(metadata["best_metric"], 71.2)
            self.assertEqual(metadata["best_epoch"], 2)
            self.assertEqual(metadata["args"]["selection_protocol"], "validation")
            for actual, expected in zip(model.parameters(), saved_parameters):
                torch.testing.assert_close(actual, expected)
            np.testing.assert_allclose(actual_rng_values, expected_rng_values)


class JsonlTests(unittest.TestCase):
    def test_jsonl_writer_persists_required_epoch_fields(self):
        record = {
            "epoch": 1,
            "classification_loss": 1.0,
            "cc_loss": 2.0,
            "mscl_loss": 3.0,
            "cscl_loss": 4.0,
            "total_loss": 5.0,
            "train_accuracy": 50.0,
            "train_weighted_f1": 49.0,
            "elapsed_seconds": 1.5,
            "sample_ratio": 0.15,
            "max_gpu_memory_mb": 100.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            write_jsonl_record(path, record)
            loaded = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(loaded, record)

    def test_jsonl_writer_serializes_missing_official_validation_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            write_jsonl_record(path, {"validation_weighted_f1": float("nan")})
            loaded = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertIsNone(loaded["validation_weighted_f1"])


class LossAccountingTests(unittest.TestCase):
    def test_total_loss_uses_official_coefficients_and_reports_raw_components(self):
        classification = torch.tensor(1.0)
        total, components = combine_training_losses(
            classification,
            cc_loss=torch.tensor(2.0),
            mscl_loss=torch.tensor(3.0),
            cscl_loss=torch.tensor(4.0),
            calibrate=True,
            contrastlearning=True,
            rank_coff=0.005,
            mscl_coff=0.05,
            cscl_coff=0.05,
        )
        torch.testing.assert_close(total, torch.tensor(1.36))
        self.assertEqual(
            components,
            {
                "classification_loss": 1.0,
                "cc_loss": 2.0,
                "mscl_loss": 3.0,
                "cscl_loss": 4.0,
                "total_loss": total.item(),
            },
        )

    def test_disabled_calibration_terms_do_not_change_total(self):
        classification = torch.tensor(1.0)
        total, _ = combine_training_losses(
            classification,
            cc_loss=torch.tensor(100.0),
            mscl_loss=torch.tensor(100.0),
            cscl_loss=torch.tensor(100.0),
            calibrate=False,
            contrastlearning=False,
            rank_coff=1.0,
            mscl_coff=1.0,
            cscl_coff=1.0,
        )
        torch.testing.assert_close(total, classification)


if __name__ == "__main__":
    unittest.main()
