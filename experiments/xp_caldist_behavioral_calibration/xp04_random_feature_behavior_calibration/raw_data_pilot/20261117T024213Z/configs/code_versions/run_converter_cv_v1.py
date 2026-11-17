"""Frozen-protocol Train-only converter/frontend GroupKFold selection runner."""
from __future__ import annotations

# These must precede NumPy/SciPy/sklearn imports in this executable.
import os
for _thread_key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_key] = "8"

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from converter_core import (compute_metrics, fit_converter, low_rank_path,
                            predict_converter, speaker_fallback_mask,
                            stateless_transform)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_load(path):
    # Defense in depth: no accidental generic split discovery may read Val/Test.
    if "validation" in path.name.lower() or "test" in path.name.lower():
        raise RuntimeError(f"forbidden selection-stage input: {path.name}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def speakers_from_manifest(path, dataset):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row.get("split", "train").lower() != "train" for row in rows):
        raise RuntimeError("manifest contains non-Train rows")
    if dataset == "IEMOCAP":
        speakers = [row["raw_utterance_id"][:5] +
                    row["raw_utterance_id"].rsplit("_", 1)[1][0] for row in rows]
        definition = "raw_utterance_id[:5] + raw_utterance_id.rsplit('_', 1)[1][0]; global Session+gender person, not dialogue-relative speaker_id slot"
    else:
        speakers = [row.get("speaker_id", "") for row in rows]
        definition = "not used: MELD unequal input/output width excludes speaker_diagonal_affine"
    return np.asarray(speakers), definition


def affine_from_column_match(exact):
    """Reuse the fit-fold Pearson search; OLS is fit from the same fold moments.

    Exact matching can select a later identical-valued source column rather
    than the first tied correlated column. Predictions are equivalent under
    OLS; the exact source is explicitly preferred as a deterministic tie rule.
    """
    model = dict(exact)
    index = exact["selected_indices"]
    valid = (index >= 0) & (exact["y_std"] > 0)
    slope = np.zeros(len(index), dtype=float)
    slope[valid] = np.divide(exact["signed_training_correlations"][valid] * exact["y_std"][valid],
                             exact["x_std"][index[valid]], out=np.zeros(valid.sum()),
                             where=exact["x_std"][index[valid]] > 0)
    model.update(kind="best_column_affine", slope=slope,
                 intercept=exact["y_mean"] - slope * exact["x_mean"][np.maximum(index, 0)],
                 matching_rule="fit-fold max absolute Pearson plus OLS; exactly identical source preferred on correlated ties; shared search with best_column_exact")
    return model


def method_list(protocol, input_width, target_width):
    methods = protocol["simple_methods_complexity_order"]
    return [method for method in methods if input_width == target_width or
            method in ("best_column_exact", "best_column_affine")]


def simple_models(X, Y, groups, speakers, methods):
    cached = None
    for method in methods:
        if method == "best_column_affine" and cached is not None:
            model = affine_from_column_match(cached)
        else:
            model = fit_converter(X, Y, method, groups=groups, speakers=speakers)
        if method == "best_column_exact":
            cached = model
        yield method, model


def add_fold_record(record, fold, y, prediction, group, scale, model, speaker):
    summary = compute_metrics(y, prediction, groups=group, target_scale=scale)
    record["folds"].append({"fold": fold, **summary,
                            "n_fit_rows": model["n_fit_rows"],
                            "n_fit_groups": model.get("n_fit_groups"),
                            "exact_train_column_count": model.get("exact_train_column_count"),
                            "unseen_speaker_fallback_rows": int(speaker_fallback_mask(model, speaker).sum())})


def finish_record(record, Y, oof, groups):
    record["oof_summary"] = compute_metrics(Y, oof, groups=groups)
    folds = record["folds"]
    total_groups = sum(fold["dialogue_count"] for fold in folds)
    record["score_dialogue_equal_nrmse"] = sum(fold["dialogue_equal_nrmse"] *
                                               fold["dialogue_count"] for fold in folds) / total_groups
    count = sum(fold["jointly_finite_elements"] for fold in folds)
    record["oof_global_train_scale_nrmse"] = float(np.sqrt(sum(
        fold["global_train_scale_nrmse"] ** 2 * fold["jointly_finite_elements"]
        for fold in folds) / count))
    record["oof_summary"]["global_train_scale_nrmse"] = record["oof_global_train_scale_nrmse"]
    record["oof_summary"]["dialogue_equal_nrmse"] = record["score_dialogue_equal_nrmse"]
    record["normalization"] = "each held-out fold uses only its own fit-fold target std floored at protocol scale_floor"


def select_record(records):
    best = min(record["score_dialogue_equal_nrmse"] for record in records)
    eligible = [record for record in records if record["score_dialogue_equal_nrmse"] <= best * 1.01 + 1e-15]
    chosen = min(eligible, key=lambda record: (record["complexity"], record["candidate"],
                                              record["transform"], record["method"],
                                              record.get("rank", 0), record.get("alpha", 0)))
    return chosen, best


def check_oof_gate(record, Y, prediction, protocol, full_coverage):
    gate = protocol["feature_gate"]
    summary = record["oof_summary"]
    corr = summary["column_correlation_quantiles"]["p05"]
    checks = {
        "full_train_coverage": bool(full_coverage),
        "finite": bool(np.isfinite(prediction).all() and np.isfinite(Y).all()),
        "allclose": bool(np.allclose(prediction, Y, **gate["all_elements_allclose"])),
        "MAE": bool(summary["mean_absolute_error"] <= gate["MAE_max"]),
        "train_scaled_RMSE": bool(record["oof_global_train_scale_nrmse"] <= gate["train_scaled_RMSE_max"]),
        "mean_cosine": bool(summary["mean_cosine_similarity"] is not None and summary["mean_cosine_similarity"] >= gate["mean_nonzero_row_cosine_min"]),
        "min_cosine": bool(summary["min_cosine_similarity"] is not None and summary["min_cosine_similarity"] >= gate["min_nonzero_row_cosine_min"]),
        "correlation_p05": bool(corr is not None and corr >= gate["nonconstant_dimension_correlation_p05_min"]),
    }
    return {"pass": all(checks.values()), "checks": checks,
            "scope": "Train OOF stop-complexity diagnostic; not independent Validation acceptance"}


def evaluate_simple(X, Y, groups, speakers, folds, protocol, candidate, transform, keep_all=False):
    methods = method_list(protocol, X.shape[1], Y.shape[1])
    records = [{"candidate": candidate, "transform": transform, "method": method,
                "complexity": protocol["simple_methods_complexity_order"].index(method),
                "folds": []} for method in methods]
    oofs = {method: np.empty(Y.shape, dtype=np.float32) for method in methods}
    for fold, (fit, held) in enumerate(folds):
        scale = np.maximum(np.std(Y[fit], axis=0, dtype=np.float64), protocol["feature_gate"]["scale_floor"])
        for method, model in simple_models(X[fit], Y[fit], groups[fit], speakers[fit], methods):
            prediction = predict_converter(model, X[held], speakers=speakers[held])
            oofs[method][held] = prediction
            record = next(record for record in records if record["method"] == method)
            add_fold_record(record, fold, Y[held], prediction, groups[held], scale, model, speakers[held])
            print(json.dumps({"event": "fold_complete", "candidate": candidate, "transform": transform,
                              "method": method, "fold": fold,
                              "dialogue_equal_nrmse": record["folds"][-1]["dialogue_equal_nrmse"]}), flush=True)
            del model, prediction
    for record in records:
        finish_record(record, Y, oofs[record["method"]], groups)
    chosen, _ = select_record(records)
    return records, (oofs if keep_all else {chosen["method"]: oofs[chosen["method"]]})


def evaluate_linear(X, Y, groups, speakers, folds, protocol, candidate, transform):
    linear = protocol["linear_methods"]
    specs = [("pcr", rank, 0.0) for rank in linear["pcr_ranks"]] + [
        ("ridge", linear["ridge_projection_rank"], alpha) for alpha in linear["ridge_alphas"]]
    complexity_start = len(protocol["simple_methods_complexity_order"])
    records = [{"candidate": candidate, "transform": transform, "method": kind,
                "rank": rank, "alpha": alpha, "complexity": complexity_start + j,
                "folds": []} for j, (kind, rank, alpha) in enumerate(specs)]
    oofs = [np.empty(Y.shape, dtype=np.float32) for _ in specs]
    for fold, (fit, held) in enumerate(folds):
        scale = np.maximum(np.std(Y[fit], axis=0, dtype=np.float64), protocol["feature_gate"]["scale_floor"])
        models = low_rank_path(X[fit], Y[fit], ranks=linear["pcr_ranks"] + [linear["ridge_projection_rank"]],
                               alphas=[0.0] + linear["ridge_alphas"], groups=groups[fit])
        for record, output, (kind, rank, alpha) in zip(records, oofs, specs):
            model = next(model for model in models if model["kind"] == kind and
                         model["requested_rank"] == rank and model["alpha"] == alpha)
            prediction = predict_converter(model, X[held])
            output[held] = prediction
            add_fold_record(record, fold, Y[held], prediction, groups[held], scale, model, speakers[held])
            print(json.dumps({"event": "linear_fold_complete", "method": kind, "rank": rank,
                              "alpha": alpha, "fold": fold,
                              "dialogue_equal_nrmse": record["folds"][-1]["dialogue_equal_nrmse"]}), flush=True)
        del models
    for record, output in zip(records, oofs):
        finish_record(record, Y, output, groups)
    return records, oofs


def write_cases(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate", "transform", "method", "rank", "alpha",
                                                     "complexity", "score_dialogue_equal_nrmse",
                                                     "oof_global_train_scale_nrmse", "oof_MAE", "oof_RMSE"])
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record.get(key, "") for key in writer.fieldnames if not key.startswith("oof_M") and key != "oof_RMSE"},
                             "oof_MAE": record["oof_summary"]["mean_absolute_error"],
                             "oof_RMSE": record["oof_summary"]["rmse"]})


def run(args):
    out = Path(args.out).resolve()
    protocol_path = out / "configs/reconstruction_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    base = out / "local_only" / args.dataset
    provenance = out / "provenance"
    provenance.mkdir(exist_ok=True)
    target_path, group_path = base / "target_train.npy", base / "groups_train.npy"
    manifest_path = base / "manifest_train.csv"
    target_all, groups_all = array_load(target_path), array_load(group_path)
    speakers_all, speaker_definition = speakers_from_manifest(manifest_path, args.dataset)
    expected = protocol["datasets"][args.dataset]["expected_train"]
    if len(target_all) != expected or len(groups_all) != expected or len(speakers_all) != expected:
        raise RuntimeError("effective Train row counts differ from frozen protocol")
    candidates_dir = base / "candidates"
    input_hashes = {str(path): sha(path) for path in (protocol_path, target_path, group_path, manifest_path,
                                                      Path(__file__), Path(__file__).with_name("converter_core.py"))}
    if args.stage == "frontend":
        if args.dataset == "MELD":
            indices_path = candidates_dir / "subset_indices.npy"
            indices = np.asarray(array_load(indices_path), dtype=int)
            if len(indices) != protocol["datasets"][args.dataset]["frontend_probe_rows"] or len(set(indices)) != len(indices):
                raise RuntimeError("MELD probe rows mismatch")
            candidate_paths = sorted(candidates_dir.glob("*/probe.npy"))
            if len(candidate_paths) != 8:
                raise RuntimeError("MELD requires all eight frozen frontend probes")
            input_hashes[str(indices_path)] = sha(indices_path)
        else:
            indices = np.arange(expected)
            candidate_paths = [candidates_dir / candidate / "train.npy" for candidate in
                               ("is10", "is10_compat", "emobase2010")]
    else:
        frontend_path = provenance / f"{args.dataset}_frontend_cv.json"
        selected = json.loads(frontend_path.read_text())["selected"]
        candidate_paths = [candidates_dir / selected["candidate"] / "train.npy"]
        indices = np.arange(expected)
        input_hashes[str(frontend_path)] = sha(frontend_path)
    common = np.isfinite(target_all[indices]).all(axis=1)
    for path in candidate_paths:
        candidate = array_load(path)
        if candidate.shape != (len(indices), protocol["datasets"][args.dataset]["input_dimensions"]):
            raise RuntimeError(f"candidate shape mismatch: {path}")
        common &= np.isfinite(candidate).all(axis=1)
        input_hashes[str(path)] = sha(path)
        valid_path = path.with_name("train_valid.npy")
        if path.name == "train.npy" and valid_path.exists():
            valid = np.asarray(array_load(valid_path), dtype=bool)
            if valid.shape != (len(indices),):
                raise RuntimeError("train_valid mask shape mismatch")
            common &= valid
            input_hashes[str(valid_path)] = sha(valid_path)
    rows = indices[common]
    Y = np.asarray(target_all[rows])
    groups, speakers = np.asarray(groups_all[rows]), speakers_all[rows]
    if len(np.unique(groups)) < protocol["cv"]["folds"]:
        raise RuntimeError("insufficient successful Train dialogues")
    folds = list(GroupKFold(n_splits=protocol["cv"]["folds"]).split(np.zeros(len(rows)), groups=groups))
    fold_definition = [{"fold": j, "fit_train_row_indices": rows[fit].tolist(),
                        "held_train_row_indices": rows[held].tolist()} for j, (fit, held) in enumerate(folds)]
    fold_hash = hashlib.sha256(json.dumps(fold_definition, sort_keys=True).encode()).hexdigest()
    fold_path = base / f"{args.stage}_cv_folds.json"
    write_json(fold_path, {"folds": fold_definition, "sha256": fold_hash})
    np.save(base / f"{args.stage}_used_train_indices.npy", rows)
    common_report = {"dataset": args.dataset, "stage": args.stage, "protocol_sha256": sha(protocol_path),
                     "pid": os.getpid(), "command": [sys.executable] + sys.argv,
                     "created_at": now(), "no_valuse": True, "no_testuse": True,
                     "selection_only_train": True, "manifest_total_train_rows": expected,
                     "considered_train_rows": len(indices), "included_train_rows": len(rows),
                     "included_train_groups": len(np.unique(groups)),
                     "excluded_considered_train_indices": indices[~common].tolist(),
                     "non_probe_train_rows_not_considered": expected - len(indices),
                     "all_manifest_rows_retained": True,
                     "speaker_key_definition": speaker_definition, "cvfoldhash": fold_hash,
                     "cvfoldfile": str(fold_path), "input_hashes": input_hashes,
                     "blas_threads": 8, "projection_seed": 424242,
                     "scores_are_selection_not_unbiased_estimates": True}
    records = []
    outputs_by_case = {}
    if args.stage == "frontend":
        for path in candidate_paths:
            candidate = path.parent.name
            source = np.asarray(array_load(path)[common])
            for transform in protocol["stateless_input_transforms"]:
                X = stateless_transform(source, transform)
                current_records, outputs = evaluate_simple(X, Y, groups, speakers, folds, protocol, candidate, transform)
                records.extend(current_records)
                local_chosen, _ = select_record(current_records)
                outputs_by_case[(candidate, transform, local_chosen["method"])] = outputs[local_chosen["method"]]
                # Checkpoint completed cases so interrupted runs preserve evidence.
                write_json(provenance / f"{args.dataset}_frontend_cv_progress.json",
                           {**common_report, "records": records, "status": "RUNNING"})
        chosen, best = select_record(records)
        key = (chosen["candidate"], chosen["transform"], chosen["method"])
        if key not in outputs_by_case:
            source_path = next(path for path in candidate_paths if path.parent.name == chosen["candidate"])
            X = stateless_transform(np.asarray(array_load(source_path)[common]), chosen["transform"])
            _, repeated_outputs = evaluate_simple(X, Y, groups, speakers, folds, protocol,
                                                  chosen["candidate"], chosen["transform"], keep_all=True)
            prediction = repeated_outputs[chosen["method"]]
        else:
            prediction = outputs_by_case[key]
        np.save(base / "OOF_frontend_selected.npy", prediction)
        report = {**common_report, "records": records,
                  "selected": {key: chosen[key] for key in ("candidate", "transform", "method")},
                  "selected_score": chosen["score_dialogue_equal_nrmse"], "best_numeric_score": best,
                  "oof_path": str(base / "OOF_frontend_selected.npy"), "status": "COMPLETE"}
        write_json(provenance / f"{args.dataset}_frontend_cv.json", report)
        write_cases(base / "frontend_case_summary.csv", records)
        print(json.dumps({"event": "FRONTEND_SELECTED", "dataset": args.dataset,
                          "selected": report["selected"], "score": report["selected_score"]}), flush=True)
        return report
    candidate, transform = selected["candidate"], selected["transform"]
    X = stateless_transform(np.asarray(array_load(candidate_paths[0])[common]), transform)
    simple_records, simple_outputs = evaluate_simple(X, Y, groups, speakers, folds, protocol, candidate, transform, keep_all=True)
    records.extend(simple_records)
    passing_simple = []
    for record in simple_records:
        record["train_oof_gate"] = check_oof_gate(record, Y, simple_outputs[record["method"]], protocol, len(rows) == expected)
        if record["train_oof_gate"]["pass"]:
            passing_simple.append(record)
    if passing_simple:
        chosen = min(passing_simple, key=lambda item: item["complexity"])
        prediction = simple_outputs[chosen["method"]]
        skipped_linear = True
        best = min(record["score_dialogue_equal_nrmse"] for record in records)
    else:
        linear_records, linear_outputs = evaluate_linear(X, Y, groups, speakers, folds, protocol, candidate, transform)
        for record, output in zip(linear_records, linear_outputs):
            record["train_oof_gate"] = check_oof_gate(record, Y, output, protocol, len(rows) == expected)
        records.extend(linear_records)
        chosen, best = select_record(records)
        prediction = (simple_outputs[chosen["method"]] if chosen in simple_records else
                      linear_outputs[linear_records.index(chosen)])
        skipped_linear = False
    if chosen["method"] in ("pcr", "ridge"):
        linear = protocol["linear_methods"]
        full_models = low_rank_path(X, Y, ranks=linear["pcr_ranks"] + [linear["ridge_projection_rank"]],
                                    alphas=[0.0] + linear["ridge_alphas"], groups=groups)
        model = next(model for model in full_models if model["kind"] == chosen["method"] and
                     model["requested_rank"] == chosen["rank"] and model["alpha"] == chosen["alpha"])
    else:
        model = fit_converter(X, Y, chosen["method"], groups=groups, speakers=speakers)
    model.update(frontend_candidate=candidate, input_transform=transform,
                 protocol_sha256=sha(protocol_path), model_selection_cvfoldhash=fold_hash,
                 fit_scope="XP04 effective Train successful pairs only", validation_used=False,
                 speaker_key_definition=speaker_definition)
    model_path = base / "converter.pkl"
    if model_path.exists():
        raise RuntimeError("refusing to overwrite fitted converter")
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=5)
    scale = np.maximum(np.std(Y, axis=0, dtype=np.float64), protocol["feature_gate"]["scale_floor"])
    np.save(base / "train_scale.npy", scale)
    np.save(base / "OOF_selected.npy", prediction)
    report = {**common_report, "records": records, "status": "MODEL_LOCKED",
              "selected": {key: chosen[key] for key in ("candidate", "transform", "method", "rank", "alpha") if key in chosen},
              "selected_score": chosen["score_dialogue_equal_nrmse"], "best_numeric_score": best,
              "train_oof_gate": chosen["train_oof_gate"], "linear_models_skipped_simple_gate": skipped_linear,
              "model_path": str(model_path), "model_sha256": sha(model_path),
              "train_scale_sha256": sha(base / "train_scale.npy"),
              "OOF_selected_sha256": sha(base / "OOF_selected.npy"),
              "training_feature_shape": list(X.shape), "training_target_shape": list(Y.shape),
              "model_is_original_author_converter": False, "locked_at": now()}
    write_json(provenance / f"{args.dataset}_model_selection.json", report)
    write_cases(base / "model_case_summary.csv", records)
    print(json.dumps({"event": "MODEL_LOCKED", "dataset": args.dataset,
                      "selected": report["selected"], "score": report["selected_score"],
                      "oof_gate": report["train_oof_gate"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MELD", "IEMOCAP"), required=True)
    parser.add_argument("--stage", choices=("frontend", "models"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    log_path = out / "provenance" / f"{args.dataset}_{args.stage}_cv.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path = log_path.with_name(log_path.stem + "_" + datetime.now(timezone.utc).strftime("%H%M%S") + ".log")
    # Parent shell can redirect stdout; an internal tee also preserves the log.
    class Tee:
        def __init__(self, original, stream):
            self.original, self.stream = original, stream
        def write(self, value):
            self.original.write(value)
            self.stream.write(value)
        def flush(self):
            self.original.flush()
            self.stream.flush()
    with log_path.open("x") as log:
        sys.stdout = Tee(sys.stdout, log)
        print(json.dumps({"event": "START", "pid": os.getpid(), "command": [sys.executable] + sys.argv,
                          "log_path": str(log_path), "time": now()}), flush=True)
        tests = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_converter_core.py"],
                               cwd=Path(__file__).parent, capture_output=True, text=True)
        evidence = {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr,
                    "fresh_run_at": now(), "test_file_sha256": sha(Path(__file__).with_name("test_converter_core.py"))}
        write_json(out / "provenance" / f"{args.dataset}_{args.stage}_synthetic_tests.json", evidence)
        if tests.returncode:
            raise RuntimeError("synthetic converter tests failed")
        run(args)


if __name__ == "__main__":
    main()
