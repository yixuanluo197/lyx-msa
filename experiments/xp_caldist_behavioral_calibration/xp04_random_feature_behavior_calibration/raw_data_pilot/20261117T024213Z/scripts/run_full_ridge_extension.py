"""One explicitly amended Train-only full-rank Ridge selection extension.

This executable reuses the original frontend and original model-selection
folds. It never reads Validation/Test features, performs CMERC inference, or
overwrites an earlier converter. The added menu was not registered at outset.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "8"

import argparse
import copy
import hashlib
import json
import pickle
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
from converter_core import (fit_converter, full_ridge_path, low_rank_path,
                            predict_converter, stateless_transform)
from run_converter_cv import (add_fold_record, array_load, check_oof_gate,
                              finish_record, now, select_record, sha,
                              speakers_from_manifest, write_cases, write_json)


def load_original_folds(out, dataset, rows, groups, old_report):
    base = out / "local_only" / dataset
    definition = json.loads((base / "models_cv_folds.json").read_text())
    actual_hash = hashlib.sha256(json.dumps(definition["folds"], sort_keys=True).encode()).hexdigest()
    if actual_hash != old_report["cvfoldhash"] or actual_hash != definition["sha256"]:
        raise RuntimeError("original model CV fold hash mismatch")
    positions = {int(row): j for j, row in enumerate(rows)}
    folds = []
    held_count = np.zeros(len(rows), dtype=int)
    for fold in definition["folds"]:
        fit = np.array([positions[int(row)] for row in fold["fit_train_row_indices"]], dtype=int)
        held = np.array([positions[int(row)] for row in fold["held_train_row_indices"]], dtype=int)
        if set(groups[fit]) & set(groups[held]):
            raise RuntimeError("a Dialogue crosses original fold Train/held boundaries")
        if set(fit) & set(held) or len(fit) + len(held) != len(rows):
            raise RuntimeError("original fold coverage mismatch")
        held_count[held] += 1
        folds.append((fit, held))
    if not np.all(held_count == 1):
        raise RuntimeError("OOF rows are not covered exactly once")
    return folds, actual_hash


def run(args):
    out = Path(args.out).resolve()
    base, provenance = out / "local_only" / args.dataset, out / "provenance"
    protocol_path = out / "configs/reconstruction_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["version"] < 2:
        raise RuntimeError("full-rank extension requires the explicit amended protocol")
    alphas = protocol["linear_methods"]["full_rank_ridge_alphas"]
    if list(alphas) != [.01, 1., 100.]:
        raise RuntimeError("unexpected full-rank Ridge grid")
    old_path = provenance / f"{args.dataset}_model_selection.json"
    old = json.loads(old_path.read_text())
    if not old["no_valuse"] or not old["no_testuse"] or old["status"] != "MODEL_LOCKED":
        raise RuntimeError("original stage is not an eligible Train-only selection")
    selected_frontend = {key: old["selected"][key] for key in ("candidate", "transform")}
    target_path = base / "target_train.npy"
    group_path = base / "groups_train.npy"
    candidate_path = base / "candidates" / selected_frontend["candidate"] / "train.npy"
    manifest_path = base / "manifest_train.csv"
    rows_path = base / "models_used_train_indices.npy"
    Yall, Xall, Gall = array_load(target_path), array_load(candidate_path), array_load(group_path)
    rows = np.asarray(array_load(rows_path), dtype=int)
    expected = protocol["datasets"][args.dataset]["expected_train"]
    if len(Yall) != expected or len(Xall) != expected or len(Gall) != expected:
        raise RuntimeError("effective Train size mismatch")
    if len(rows) != old["included_train_rows"] or len(np.unique(rows)) != len(rows):
        raise RuntimeError("original model stage used row mismatch")
    speakers_all, speaker_definition = speakers_from_manifest(manifest_path, args.dataset)
    X = stateless_transform(np.asarray(Xall[rows]), selected_frontend["transform"])
    Y, groups, speakers = np.asarray(Yall[rows]), np.asarray(Gall[rows]), speakers_all[rows]
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise RuntimeError("original successful Train rows now contain nonfinite values")
    folds, fold_hash = load_original_folds(out, args.dataset, rows, groups, old)
    extension_dir = base / "full_ridge_extension"
    extension_dir.mkdir(exist_ok=True)
    final_dir = base / "final"
    if final_dir.exists():
        raise RuntimeError("refusing to overwrite an existing final selection")
    input_paths = [protocol_path, old_path, target_path, group_path, candidate_path,
                   manifest_path, rows_path, Path(__file__),
                   Path(__file__).with_name("converter_core.py"),
                   Path(__file__).with_name("run_converter_cv.py")]
    inputs = {str(path): sha(path) for path in input_paths}
    for path in (target_path, group_path, candidate_path, manifest_path):
        original_hash = old["input_hashes"].get(str(path))
        if original_hash is None or inputs[str(path)] != original_hash:
            raise RuntimeError(f"Train input differs from original model stage: {path}")
    complexity_start = max(record["complexity"] for record in old["records"]) + 1
    records = [{**selected_frontend, "method": "full_ridge", "alpha": float(alpha),
                "complexity": complexity_start + j, "folds": [], "projection": "none",
                "protocol_amendment": "Train-only supplemental menu; not preregistered at initial protocol v1"}
               for j, alpha in enumerate(alphas)]
    oofs = [np.empty(Y.shape, dtype=np.float32) for _ in alphas]
    fold_scalers = []
    report_common = {"dataset": args.dataset, "stage": "full_rank_ridge_extension",
                     "created_at": now(), "pid": os.getpid(), "command": [sys.executable] + sys.argv,
                     "protocol_sha256": sha(protocol_path), "protocol_version": protocol["version"],
                     "baseline_selection_protocol_sha256": old["protocol_sha256"],
                     "baseline_selection_sha256": sha(old_path), "input_hashes": inputs,
                     "no_valuse": True, "no_testuse": True, "selection_only_train": True,
                     "amendment_reason": "Train-stage method coverage review requested full-rank regularized multioutput regression before any new Validation access; initial low-rank-only menu did not test that family",
                     "menu_was_fully_preregistered_initially": False,
                     "cvfoldhash": fold_hash, "same_folds_as_original_model_selection": True,
                     "included_train_rows": len(rows), "included_train_groups": len(np.unique(groups)),
                     "excluded_train_indices": sorted(set(range(expected)) - set(rows.tolist())),
                     "frontend_locked": selected_frontend, "speaker_key_definition": speaker_definition,
                     "blas_threads": 8, "no_svd_executed_for_full_ridge_candidates": True,
                     "scores_are_selection_not_unbiased_estimates": True}
    for fold, (fit, held) in enumerate(folds):
        print(json.dumps({"event": "FULL_RIDGE_FOLD_START", "fold": fold,
                          "fit_rows": len(fit), "held_rows": len(held), "time": now()}), flush=True)
        scale = np.maximum(np.std(Y[fit], axis=0, dtype=np.float64), protocol["feature_gate"]["scale_floor"])
        models = full_ridge_path(X[fit], Y[fit], alphas=alphas, groups=groups[fit])
        scalers_path = extension_dir / f"fold_{fold}_scalers.npz"
        np.savez(scalers_path, **{key: models[0][key] for key in
                                 ("x_mean", "x_std", "x_scale", "y_mean", "y_std", "y_scale")},
                 metric_target_scale=scale, fit_train_row_indices=rows[fit])
        fold_scalers.append({"fold": fold, "path": str(scalers_path), "sha256": sha(scalers_path)})
        for record, output, model in zip(records, oofs, models):
            prediction = predict_converter(model, X[held])
            output[held] = prediction
            add_fold_record(record, fold, Y[held], prediction, groups[held], scale, model, speakers[held])
            print(json.dumps({"event": "FULL_RIDGE_FOLD_COMPLETE", "fold": fold, "alpha": record["alpha"],
                              "dialogue_equal_nrmse": record["folds"][-1]["dialogue_equal_nrmse"],
                              "max_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                              "time": now()}), flush=True)
        del models
        write_json(provenance / f"{args.dataset}_full_ridge_progress.json",
                   {**report_common, "records": records, "fold_scalers": fold_scalers, "status": "RUNNING"})
    for record, output in zip(records, oofs):
        finish_record(record, Y, output, groups)
        record["train_oof_gate"] = check_oof_gate(record, Y, output, protocol, len(rows) == expected)
    baseline_records = copy.deepcopy(old["records"])
    combined = baseline_records + records
    chosen, best = select_record(combined)
    if chosen in baseline_records:
        # Baseline stage has already applied the identical complexity/tie rule;
        # adding more complex candidates cannot pick a different baseline winner.
        if any(chosen.get(key) != value for key, value in old["selected"].items()):
            raise RuntimeError("baseline winner changed unexpectedly under unchanged score rule")
        prediction = np.asarray(array_load(base / "OOF_selected.npy"))
    else:
        prediction = oofs[records.index(chosen)]
    print(json.dumps({"event": "FINAL_TRAIN_SELECTION", "selected": {
        key: chosen[key] for key in ("candidate", "transform", "method", "rank", "alpha") if key in chosen},
        "dialogue_equal_nrmse": chosen["score_dialogue_equal_nrmse"], "time": now()}), flush=True)
    if chosen["method"] in ("pcr", "ridge"):
        linear = protocol["linear_methods"]
        candidates = low_rank_path(X, Y, ranks=linear["pcr_ranks"] + [linear["ridge_projection_rank"]],
                                   alphas=[0.] + linear["ridge_alphas"], groups=groups)
        model = next(model for model in candidates if model["kind"] == chosen["method"] and
                     model["requested_rank"] == chosen["rank"] and model["alpha"] == chosen["alpha"])
    else:
        model = fit_converter(X, Y, chosen["method"], groups=groups, speakers=speakers,
                              alpha=chosen.get("alpha", 1.))
    model.update(frontend_candidate=selected_frontend["candidate"],
                 input_transform=selected_frontend["transform"],
                 protocol_sha256=sha(protocol_path), protocol_version=protocol["version"],
                 model_selection_cvfoldhash=fold_hash, fit_scope="XP04 effective Train successful pairs only",
                 validation_used=False, speaker_key_definition=speaker_definition,
                 selection_protocol="v2 including explicit Train-only full-rank ridge amendment",
                 model_is_original_author_converter=False)
    final_dir.mkdir(mode=0o700)
    model_path = final_dir / "converter.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=5)
    np.save(final_dir / "train_scale.npy", np.maximum(np.std(Y, axis=0, dtype=np.float64),
                                                     protocol["feature_gate"]["scale_floor"]))
    np.save(final_dir / "OOF_selected.npy", prediction)
    report = {**report_common, "status": "MODEL_LOCKED_FINAL_TRAIN_ONLY", "records": combined,
              "selected": {key: chosen[key] for key in ("candidate", "transform", "method", "rank", "alpha") if key in chosen},
              "selected_score": chosen["score_dialogue_equal_nrmse"], "best_numeric_score": best,
              "train_oof_gate": chosen["train_oof_gate"], "fold_scalers": fold_scalers,
              "training_feature_shape": list(X.shape), "training_target_shape": list(Y.shape),
              "model_path": str(model_path), "model_sha256": sha(model_path),
              "train_scale_path": str(final_dir / "train_scale.npy"),
              "train_scale_sha256": sha(final_dir / "train_scale.npy"),
              "OOF_selected_sha256": sha(final_dir / "OOF_selected.npy"), "locked_at": now(),
              "max_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
              "original_model_sha256_unchanged": sha(base / "converter.pkl") == old["model_sha256"],
              "final_fit_uses_current_protocol_sha256": model["protocol_sha256"] == sha(protocol_path)}
    write_json(provenance / f"{args.dataset}_model_selection_final.json", report)
    write_cases(final_dir / "case_summary.csv", combined)
    print(json.dumps({"event": "MODEL_LOCKED_FINAL", "dataset": args.dataset, "selected": report["selected"],
                      "score": report["selected_score"], "max_rss_GiB": report["max_rss_GiB"],
                      "oof_gate": report["train_oof_gate"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MELD", "IEMOCAP"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    log_path = out / "provenance" / f"{args.dataset}_full_ridge_extension.log"
    if log_path.exists():
        raise RuntimeError("refusing to overwrite extension log")
    class Tee:
        def __init__(self, original, stream):
            self.original, self.stream = original, stream
        def write(self, text):
            self.original.write(text)
            self.stream.write(text)
        def flush(self):
            self.original.flush()
            if not self.stream.closed:
                self.stream.flush()
    original_stdout = sys.stdout
    with log_path.open("x") as log:
        try:
            sys.stdout = Tee(original_stdout, log)
            print(json.dumps({"event": "START", "pid": os.getpid(), "command": [sys.executable] + sys.argv,
                              "log_path": str(log_path), "time": now()}), flush=True)
            tests = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_converter_core.py"],
                                   cwd=Path(__file__).parent, capture_output=True, text=True)
            write_json(out / "provenance" / f"{args.dataset}_full_ridge_synthetic_tests.json",
                       {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr,
                        "fresh_run_at": now(), "test_file_sha256": sha(Path(__file__).with_name("test_converter_core.py"))})
            if tests.returncode:
                raise RuntimeError("synthetic tests failed")
            run(args)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
