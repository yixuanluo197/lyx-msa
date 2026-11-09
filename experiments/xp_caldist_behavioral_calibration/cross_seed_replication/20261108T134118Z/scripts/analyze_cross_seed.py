#!/usr/bin/env python
"""Analyze cross-seed sensitivity and fixed-OOF correctness discrimination."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from cross_seed_core import behavior, contract, js_divergence


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty output table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def scalar(payload, key):
    return np.asarray(payload[key]).item()


def validate(base, condition, spec):
    for key in ("labels", "dialogue_ids", "utterance_indices", "speaker_ids",
                "batch_indices", "dialogue_lengths"):
        if not np.array_equal(base[key], condition[key]):
            raise ValueError(f"metadata mismatch in {spec['id']}: {key}")
    p, q = base["probabilities"], condition["probabilities"]
    if q.shape != (len(p), spec["repeats"], p.shape[1]):
        raise ValueError(f"probability shape mismatch: {spec['id']}")
    if not np.isfinite(q).all() or not np.allclose(q.sum(-1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError(f"invalid probabilities: {spec['id']}")
    if not np.array_equal(condition["original_probabilities"], p):
        raise ValueError(f"original probability mismatch: {spec['id']}")
    missing = condition["visual_missing"]
    active = condition["active"]
    expected = ~missing if spec["target"] == "V" else np.ones(len(p), bool)
    if not np.array_equal(active, expected):
        raise ValueError(f"active mask mismatch: {spec['id']}")
    if not np.array_equal(q[~active], np.repeat(p[~active, None], spec["repeats"], 1)):
        raise ValueError(f"inactive visual prediction changed: {spec['id']}")


def condition_path(root, config, dataset, seed, spec, phase):
    anchor = config["anchors_reused_without_rerun"].get(dataset)
    if phase == "main" and seed == anchor:
        path = Path(config["historical_evidence_sensitivity"]) / f"runs/{dataset}/seed{seed}/conditions/{spec['id']}.npz"
    else:
        path = root / f"runs/{dataset}/seed{seed}/conditions/{spec['id']}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def baseline_path(root, config, dataset, seed):
    path = root / f"runs/{dataset}/seed{seed}/baseline_validation.npz"
    if not path.exists():
        path = Path(config["historical_pilot"]) / f"runs/{dataset}/seed{seed}/validation_predictions.npz"
    return path


def condition_detail(base, payload):
    p, q, labels = base["probabilities"], payload["probabilities"], base["labels"].astype(int)
    op, pp = p.argmax(-1), q.argmax(-1)
    oc, pc = p.max(-1), q.max(-1)
    original_correct = op == labels
    perturbed_correct = pp == labels[:, None]
    values = behavior(p, q)
    values.update(original_prediction=op, perturbed_prediction=pp,
                  original_confidence=oc, perturbed_confidence=pc,
                  original_true_probability=p[np.arange(len(p)), labels],
                  perturbed_true_probability=q[np.arange(len(p))[:, None], np.arange(q.shape[1])[None, :], labels[:, None]],
                  flip=pp != op[:, None], confidence_change=pc-oc[:, None],
                  true_probability_change=q[np.arange(len(p))[:, None], np.arange(q.shape[1])[None, :], labels[:, None]]-p[np.arange(len(p)), labels][:, None],
                  repeat_js=js_divergence(p[:, None, :], q),
                  correct_to_wrong=original_correct[:, None] & ~perturbed_correct,
                  wrong_to_correct=~original_correct[:, None] & perturbed_correct)
    return values


def summarize(dataset, seed, spec, base, payload, detail, subgroup="all", selected=None):
    selected = np.ones(len(base["labels"]), bool) if selected is None else np.asarray(selected, bool)
    active = payload["active"] & selected
    p, q, y = base["probabilities"], payload["probabilities"], base["labels"].astype(int)
    original = p.argmax(-1) == y
    changed = q.argmax(-1) == y[:, None]
    if not selected.any():
        values = {name: np.nan for name in ("original_accuracy", "perturbed_accuracy", "delta_accuracy",
                                             "flip_rate", "prediction_instability_mu", "absolute_confidence_change",
                                             "confidence_mad", "js_divergence", "true_probability_change",
                                             "maximum_confidence_change", "correct_to_wrong_rate", "wrong_to_correct_rate")}
    else:
        values = {
            "original_accuracy": original[selected].mean(),
            "perturbed_accuracy": changed[selected].mean(),
            "delta_accuracy": changed[selected].mean()-original[selected].mean(),
            "flip_rate": detail["flip"][selected].mean(),
            "prediction_instability_mu": detail["prediction_instability_mu"][selected].mean(),
            "absolute_confidence_change": detail["absolute_confidence_change"][selected].mean(),
            "confidence_mad": detail["confidence_mad"][selected].mean(),
            "js_divergence": detail["js_divergence"][selected].mean(),
            "true_probability_change": detail["true_probability_change"][selected].mean(),
            "maximum_confidence_change": detail["confidence_change"][selected].mean(),
            "correct_to_wrong_rate": detail["correct_to_wrong"][selected].mean(),
            "wrong_to_correct_rate": detail["wrong_to_correct"][selected].mean(),
        }
    return {"dataset": dataset, "seed": seed, "reference": "donor" if spec["kind"] == "random_training_donor" else "train_mean",
            "condition_id": spec["id"], "target": spec["target"], "strength": spec["strength"],
            "repeats": spec["repeats"], "subgroup": subgroup,
            "n_utterances": int(selected.sum()), "active_utterances": int(active.sum()),
            **{key: float(value) for key, value in values.items()}}


def probability_features(probabilities):
    p = np.asarray(probabilities, float)
    ordered = np.sort(p, axis=1)
    entropy = -np.where(p > 0, p*np.log(p), 0).sum(1)
    return np.column_stack([p.max(1), entropy, ordered[:, -1]-ordered[:, -2]])


def logo_scores(features, labels, dialogues, c=1.0):
    features, labels, dialogues = np.asarray(features, float), np.asarray(labels, int), np.asarray(dialogues)
    scores = np.full(len(labels), np.nan)
    for held_out in np.unique(dialogues):
        test = dialogues == held_out; train = ~test
        scaler = StandardScaler().fit(features[train])
        x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])
        if len(np.unique(labels[train])) < 2:
            scores[test] = np.clip(labels[train].mean(), 1e-7, 1-1e-7)
        else:
            model = LogisticRegression(C=c, penalty="l2", solver="liblinear", random_state=0,
                                       max_iter=1000).fit(x_train, labels[train])
            scores[test] = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    if not np.isfinite(scores).all():
        raise RuntimeError("incomplete LODO scores")
    return np.clip(scores, 1e-7, 1-1e-7)


def binary_metrics(labels, scores):
    y, s = np.asarray(labels, int), np.clip(np.asarray(scores, float), 1e-7, 1-1e-7)
    return {
        "auroc_success": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        "auprc_success": average_precision_score(y, s) if y.sum() else np.nan,
        "binary_brier": np.mean((s-y)**2),
        "binary_nll": -np.mean(y*np.log(s)+(1-y)*np.log(1-s)),
    }


def bootstrap_differences(labels, dialogues, baseline, candidate, repeats, seed):
    unique = np.unique(dialogues)
    rng = np.random.default_rng(seed)
    metrics = tuple(binary_metrics(labels, baseline))
    draws = {metric: [] for metric in metrics}
    for _ in range(repeats):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(dialogues == dialogue) for dialogue in chosen])
        before, after = binary_metrics(labels[indices], baseline[indices]), binary_metrics(labels[indices], candidate[indices])
        for metric in metrics:
            value = after[metric]-before[metric]
            if np.isfinite(value): draws[metric].append(value)
    return {metric: (float(np.quantile(values, .025)), float(np.quantile(values, .975)), len(values))
            if values else (np.nan, np.nan, 0) for metric, values in draws.items()}


def run(root):
    config = json.loads((root / "configs/preregistered_config.json").read_text())
    if config["inference"]["split"] != "validation":
        raise ValueError("analysis is Validation-only")
    main_specs, donor_specs = contract(config)
    sample_rows, summaries, donor_rows = [], [], []
    bases, sensitivities, donor_sensitivities = {}, {}, {}
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            base = read_npz(baseline_path(root, config, dataset, seed))
            expected = config["validation_sizes"][dataset]
            if len(base["labels"]) != expected["utterances"] or len(np.unique(base["dialogue_ids"])) != expected["dialogues"]:
                raise ValueError("validation size mismatch")
            key = (dataset, seed); bases[key] = base; sensitivities[key] = {}; donor_sensitivities[key] = {}
            for phase, specs in (("main", main_specs), ("donor", donor_specs)):
                for spec in specs:
                    payload = read_npz(condition_path(root, config, dataset, seed, spec, phase))
                    validate(base, payload, spec)
                    detail = condition_detail(base, payload)
                    target_store = sensitivities if phase == "main" else donor_sensitivities
                    target_store[key][spec["id"]] = detail
                    summaries.append(summarize(dataset, seed, spec, base, payload, detail))
                    missing = payload["visual_missing"]
                    if spec["target"] == "V":
                        summaries.append(summarize(dataset, seed, spec, base, payload, detail,
                                                   "visual_present", ~missing))
                        summaries.append(summarize(dataset, seed, spec, base, payload, detail,
                                                   "visual_missing", missing))
                    if phase == "donor":
                        donor_missing = payload["donor_modality_missing"]
                        if spec["target"] == "V":
                            # Donor strata are repeat-level; retain a separate compact table.
                            for name, mask in (("all", np.ones_like(donor_missing, bool)),
                                               ("donor_visual_present", ~donor_missing),
                                               ("donor_visual_missing", donor_missing)):
                                selected_values = mask & payload["active"][:, None]
                                donor_rows.append({"dataset": dataset, "seed": seed, "target": "V", "subgroup": name,
                                                   "n_target_repeat_observations": int(selected_values.sum()),
                                                   "flip_rate": float(detail["flip"][selected_values].mean()) if selected_values.any() else np.nan,
                                                   "confidence_change": float(detail["confidence_change"][selected_values].mean()) if selected_values.any() else np.nan,
                                                   "js_divergence": float(detail["repeat_js"][selected_values].mean()) if selected_values.any() else np.nan})
                    for index in range(len(base["labels"])):
                        for repeat in range(spec["repeats"]):
                            row = {"dataset": dataset, "seed": seed, "reference": phase,
                                   "condition_id": spec["id"], "target": spec["target"],
                                   "strength": spec["strength"], "repeat": repeat,
                                   "dialogue_id": str(base["dialogue_ids"][index]),
                                   "utterance_index": int(base["utterance_indices"][index]),
                                   "label": int(base["labels"][index]), "active": bool(payload["active"][index]),
                                   "visual_missing": bool(missing[index]),
                                   "original_prediction": int(detail["original_prediction"][index]),
                                   "perturbed_prediction": int(detail["perturbed_prediction"][index, repeat]),
                                   "original_confidence": float(detail["original_confidence"][index]),
                                   "perturbed_confidence": float(detail["perturbed_confidence"][index, repeat]),
                                   "flip": int(detail["flip"][index, repeat]),
                                   "prediction_instability_mu": float(detail["flip"][index, repeat]*detail["perturbed_confidence"][index, repeat]),
                                   "confidence_change": float(detail["confidence_change"][index, repeat]),
                                   "absolute_confidence_change": float(abs(detail["confidence_change"][index, repeat])),
                                   "js_divergence": float(detail["repeat_js"][index, repeat]),
                                   "true_probability_change": float(detail["true_probability_change"][index, repeat]),
                                   "correct_to_wrong": int(detail["correct_to_wrong"][index, repeat]),
                                   "wrong_to_correct": int(detail["wrong_to_correct"][index, repeat])}
                            if phase == "donor":
                                row.update(donor_flat_index=int(payload["donor_flat_indices"][index, repeat]),
                                           donor_dialogue_id=str(payload["donor_dialogue_ids"][index, repeat]),
                                           donor_utterance_index=int(payload["donor_utterance_indices"][index, repeat]),
                                           donor_modality_missing=bool(payload["donor_modality_missing"][index, repeat]))
                            sample_rows.append(row)

    oof_rows, metric_rows, bootstrap_rows = [], [], []
    primary = float(config["correctness_prediction"]["primary_strength"])
    c_value = 1.0
    for (dataset, seed), base in bases.items():
        p = base["probabilities"]
        y = (p.argmax(1) == base["labels"]).astype(int)
        probability = probability_features(p)
        sens = {}
        for target, label in (("T-All", "text"), ("A", "audio"), ("V", "visual")):
            spec = next(item for item in main_specs if item["target"] == target and item["strength"] == primary)
            detail = sensitivities[(dataset, seed)][spec["id"]]
            sens[label] = np.column_stack([detail["flip_rate"], detail["absolute_confidence_change"], detail["js_divergence"]])
        donor_sens = {}
        for target, label in (("T-All", "text"), ("A", "audio"), ("V", "visual")):
            spec = next(item for item in donor_specs if item["target"] == target)
            detail = donor_sensitivities[(dataset, seed)][spec["id"]]
            donor_sens[label] = np.column_stack([detail["flip_rate"], detail["absolute_confidence_change"], detail["js_divergence"]])
        definitions = {
            "confidence_only": probability[:, :1],
            "strict_probability_baseline": probability,
        }
        for prefix, base_features in (("confidence", probability[:, :1]), ("strict", probability)):
            for label in ("text", "audio", "visual"):
                definitions[f"{prefix}_plus_{label}"] = np.column_stack([base_features, sens[label]])
            definitions[f"{prefix}_plus_text_audio_visual"] = np.column_stack([base_features, sens["text"], sens["audio"], sens["visual"]])
            for label in ("text", "audio", "visual"):
                definitions[f"{prefix}_plus_donor_{label}"] = np.column_stack([base_features, donor_sens[label]])
            definitions[f"{prefix}_plus_donor_text_audio_visual"] = np.column_stack([base_features, donor_sens["text"], donor_sens["audio"], donor_sens["visual"]])
        scores = {name: logo_scores(features, y, base["dialogue_ids"], c_value)
                  for name, features in definitions.items()}
        for name, values in scores.items():
            metrics = binary_metrics(y, values)
            metric_rows.append({"dataset": dataset, "seed": seed, "model": name,
                                "n_utterances": len(y), "n_dialogues": len(np.unique(base["dialogue_ids"])), **metrics})
            for index, score in enumerate(values):
                oof_rows.append({"dataset": dataset, "seed": seed, "model": name,
                                 "dialogue_id": str(base["dialogue_ids"][index]),
                                 "utterance_index": int(base["utterance_indices"][index]),
                                 "correctness": int(y[index]), "oof_success_probability": float(score)})
            if name not in ("confidence_only", "strict_probability_baseline"):
                comparator = "strict_probability_baseline" if name.startswith("strict_") else "confidence_only"
                intervals = bootstrap_differences(y, base["dialogue_ids"], scores[comparator], values,
                                                  config["statistics"]["bootstrap_repeats"],
                                                  config["statistics"]["bootstrap_seed"])
                before, after = binary_metrics(y, scores[comparator]), metrics
                for metric, (low, high, valid) in intervals.items():
                    bootstrap_rows.append({"dataset": dataset, "seed": seed, "model": name,
                                           "comparator": comparator, "metric": metric,
                                           "comparator_value": before[metric], "model_value": after[metric],
                                           "difference": after[metric]-before[metric], "ci_low": low,
                                           "ci_high": high, "valid_bootstraps": valid,
                                           "bootstrap_repeats": config["statistics"]["bootstrap_repeats"],
                                           "models_refit_inside_bootstrap": False})

    # Cross-seed aggregation never pools duplicated utterances.
    aggregates = []
    for dataset in config["datasets"]:
        identities = sorted({(row["reference"], row["condition_id"], row["target"], row["subgroup"])
                             for row in summaries if row["dataset"] == dataset})
        for reference, condition, target, subgroup in identities:
            selected = [row for row in summaries if row["dataset"] == dataset and row["reference"] == reference
                        and row["condition_id"] == condition and row["subgroup"] == subgroup]
            if len(selected) != len(config["seeds"]):
                raise ValueError("cross-seed summary does not contain five independent model seeds")
            for metric in ("delta_accuracy", "flip_rate", "true_probability_change", "js_divergence"):
                values = np.asarray([row[metric] for row in selected], float)
                aggregates.append({"dataset": dataset, "reference": reference, "condition_id": condition,
                                   "target": target, "subgroup": subgroup, "metric": metric,
                                   "seed_mean": float(values.mean()), "seed_sample_sd": float(values.std(ddof=1)),
                                   "positive_seeds": int((values > 0).sum()), "negative_seeds": int((values < 0).sum()),
                                   "zero_seeds": int((values == 0).sum()), "n_model_seeds": len(values)})

    results = root / "results"
    write_csv(results / "per_utterance_sensitivity.csv", sample_rows)
    write_csv(results / "per_seed_sensitivity.csv", summaries)
    write_csv(results / "donor_reference_results.csv", donor_rows)
    write_csv(results / "cross_seed_results.csv", aggregates)
    write_csv(results / "correctness_oof_predictions.csv", oof_rows)
    write_csv(results / "correctness_oof_metrics.csv", metric_rows)
    write_csv(results / "paired_bootstrap_results.csv", bootstrap_rows)
    print(json.dumps({"per_utterance_rows": len(sample_rows), "summary_rows": len(summaries),
                      "oof_rows": len(oof_rows), "bootstrap_rows": len(bootstrap_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); run(args.root.resolve())
