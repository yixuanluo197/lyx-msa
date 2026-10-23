#!/usr/bin/env python3
"""Independently verify CSVs against raw test labels/text and saved full outputs."""
import csv
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["dialogue_id"], int(row["utterance_index"])


def main():
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["split"] == "test"
    verified = {"datasets": {}, "models": {}, "comparisons": {}}
    records = {}
    for dataset, expected_count in [("IEMOCAP", 1623), ("MELD", 2610)]:
        raw_name = "IEMOCAP_features.pkl" if dataset == "IEMOCAP" else "MELD_features_raw1.pkl"
        raw_path = Path(manifest["batch_root"]) / "code/data" / raw_name
        with raw_path.open("rb") as stream:
            raw = pickle.load(stream, encoding="latin1")
        raw_ids = {str(vid): vid for vid in raw[8]}
        verified["datasets"][dataset] = {"raw_test_dialogues": len(raw_ids),
                                         "expected_test_utterances": expected_count}
        for model in ("baseline", "cmerc"):
            identity = f"{dataset}/{model}"
            meta = manifest["models"][identity]
            base = ROOT / dataset / model
            rows = read_csv(base / "all_predictions.csv")
            assert len(rows) == expected_count
            assert len(set(map(key, rows))) == expected_count
            names = meta["class_names"]
            errors = []
            control_text_count = 0
            for row in rows:
                vid = raw_ids[row["dialogue_id"]]
                index = int(row["utterance_index"])
                truth, prediction = int(row["true_label"]), int(row["predicted_label"])
                assert truth == raw[2][vid][index]
                assert row["text"] == raw[6][vid][index]
                assert row["source_utterance_id"] == str(raw[0][vid][index])
                probabilities = np.array([float(row[f"probability_{name}"]) for name in names])
                assert np.isfinite(probabilities).all() and np.isclose(probabilities.sum(), 1, atol=1e-5)
                assert prediction == int(probabilities.argmax())
                assert float(row["max_probability"]) == probabilities[prediction]
                assert float(row["true_class_probability"]) == probabilities[truth]
                assert row["is_error"] in ("True", "False")
                assert (row["is_error"] == "True") == (truth != prediction)
                if truth != prediction:
                    errors.append(row)
                control_text_count += any(127 <= ord(char) <= 159 for char in row["text"])
            errors.sort(key=lambda row: (-float(row["max_probability"]), *key(row)))
            assert read_csv(base / "errors_all.csv") == errors
            assert len(errors) == meta["statistics"]["error_count"]
            thresholds = {}
            for threshold in (0.8, 0.9):
                subset = [r for r in errors if float(r["max_probability"]) >= threshold]
                assert read_csv(base / f"errors_ge_{str(threshold).replace('.', 'p')}.csv") == subset
                high = [r for r in rows if float(r["max_probability"]) >= threshold]
                item = meta["statistics"]["thresholds"][str(threshold)]
                assert item["high_confidence_count"] == len(high)
                assert item["high_confidence_error_count"] == len(subset)
                assert item["error_rate_among_high_confidence"] == (len(subset) / len(high) if high else None)
                thresholds[str(threshold)] = {"high_predictions": len(high), "high_errors": len(subset)}
            expected_keys = [(vid, index) for batch in meta["alignment"]["batches"]
                             for vid, length in zip(batch["dialogue_ids"], batch["lengths"])
                             for index in range(length)]
            assert list(map(key, rows)) == expected_keys
            assert meta["alignment"]["modality_removal_calls"] == 0
            assert meta["training_args"]["selection_protocol"] == "validation"
            assert meta["training_args"]["causal"] is False
            flags = [meta["training_args"][name] for name in
                     ("calibrate", "contrastlearning", "courselearning")]
            assert flags == [model == "cmerc"] * 3
            with np.load(meta["existing_prediction_comparison"]["path"], allow_pickle=False) as old:
                assert np.array_equal([int(r["true_label"]) for r in rows], old["labels"])
                assert np.array_equal([int(r["predicted_label"]) for r in rows], old["predictions"])
                assert np.array_equal([r["dialogue_id"] for r in rows], old["dialogue_ids"].astype(str))
                assert np.array_equal([int(r["utterance_index"]) for r in rows], old["utterance_indices"])
                probs = np.array([[float(r[f"probability_{n}"]) for n in names] for r in rows])
                assert np.allclose(probs, np.exp(old["log_probabilities"].astype(np.float64)), rtol=2e-5, atol=1e-6)
            verified["models"][identity] = {"utterances": len(rows), "errors": len(errors),
                                            "thresholds": thresholds,
                                            "raw_label_text_id_roundtrip": "passed",
                                            "raw_text_rows_with_7f_9f_controls_preserved": control_text_count}
            records[identity] = rows
        before = {key(r): r for r in records[f"{dataset}/baseline"]}
        after = {key(r): r for r in records[f"{dataset}/cmerc"]}
        joined = read_csv(ROOT / dataset / "baseline_vs_cmerc_all.csv")
        assert list(map(key, joined)) == list(before)
        assert set(before) == set(after)
        transitions = Counter()
        for row in joined:
            b, c = before[key(row)], after[key(row)]
            for field in ("text", "true_label", "true_emotion", "source_utterance_id"):
                assert row[field] == b[field] == c[field]
            for role, original in (("baseline", b), ("cmerc", c)):
                for field in ("predicted_label", "predicted_emotion", "max_probability", "true_class_probability", "is_error"):
                    assert row[f"{role}_{field}"] == original[field]
            transitions[row["error_transition"]] += 1
        wrong_union = [r for r in joined if r["baseline_is_error"] == "True" or r["cmerc_is_error"] == "True"]
        assert read_csv(ROOT / dataset / "baseline_vs_cmerc_errors_union.csv") == wrong_union
        verified["comparisons"][dataset] = {"matched_utterances": len(joined), "wrong_union": len(wrong_union),
                                            "transitions": dict(transitions)}
    for path, info in manifest["input_files_before"].items():
        assert digest(path) == info["sha256"] == manifest["input_files_after_sha256"][path]
    for path, expected in manifest["code_sha256"].items():
        assert digest(path) == expected
    verified.update({"verified_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
                     "inference_rerun": False, "original_input_and_checkpoint_hashes": "unchanged",
                     "csv_sha256": {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob("*.csv")}})
    (ROOT / "verification.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verified.items() if k != "csv_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
