#!/usr/bin/env python3
"""Plot completed round-two paired runs using training and validation only.

No model import/inference, test-file access, seed selection, smoothing, or error
bars. Actual auxiliary-loss coefficients come from the verified run artifact.
Existing figure directories are never overwritten. Outputs are public-safe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_EPOCHS = 80
COLORS = {"baseline": "#0072B2", "cmerc": "#D55E00"}
MEASURE_COLORS = ("#0072B2", "#D55E00", "#009E73")
POSITION_RULE = {
    "relative_position": "(zero_based_utterance_index + 0.5) / dialogue_length",
    "early": "[0, 1/3)", "middle": "[1/3, 2/3)", "late": "[2/3, 1]",
    "length_one": "middle",
}
LOSS_KEYS = ("classification_loss", "cc_loss", "mscl_loss", "cscl_loss")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "missing or nonfinite numeric figure input")
    return float(value)


def read_inputs(batch):
    """Admission checks precede creating any output or Matplotlib figure."""
    verification_path = batch / "CMERC_VERIFICATION.json"
    verified = json.loads(verification_path.read_text())
    require(verified.get("status") == "passed", "CMERC verification has not passed")
    require(verified.get("completed_epochs") == EXPECTED_EPOCHS,
            "CMERC verification does not describe a completed 80-epoch run")
    require(verified.get("paired_initialization_verified") is True,
            "actual paired initialization has not been verified")
    coefficients = verified.get("gradient_coefficients", {})
    require(coefficients == dict(classification_loss=1., cc_loss=.005,
            mscl_loss=.05, cscl_loss=.05), "unexpected frozen loss coefficients")
    comparison_path = batch / "comparison_validation/comparison.json"
    comparison = json.loads(comparison_path.read_text())
    require(comparison.get("dataset") == "IEMOCAP" and comparison.get("split") == "validation"
            and comparison.get("split_seed") == 1475,
            "figures require the frozen IEMOCAP validation split")
    require(comparison.get("replacement_policy") == "prefix_mean_v1",
            "replacement policy differs from the frozen experiment")
    require(comparison.get("position_rule") == POSITION_RULE,
            "relative-position thirds differ from the preregistered definition")
    records, paths = {}, {"verification": verification_path, "validation_comparison": comparison_path}
    for variant in ("baseline", "cmerc"):
        candidates = list((batch / "runs" / variant).glob("*/metrics.jsonl"))
        require(len(candidates) == 1, "expected exactly one metrics file for " + variant)
        path = candidates[0]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require([row.get("epoch") for row in rows] == list(range(1, EXPECTED_EPOCHS + 1)),
                variant + " must contain exactly 80 contiguous completed epochs")
        for row in rows:
            require(row.get("selection_split") == "validation"
                    and row.get("selection_protocol") == "validation" and row.get("causal") is True,
                    "training records do not describe validation-selected causal runs")
            for key in ("train_weighted_f1", "validation_weighted_f1", *LOSS_KEYS):
                number(row.get(key))
            require(all(0 <= row[key] <= 100 for key in ("train_weighted_f1", "validation_weighted_f1")),
                    "W-F1 must be in percentage units")
            require(all(row[key] >= 0 for key in LOSS_KEYS), "negative loss term in frozen nonnegative losses")
            if variant == "baseline":
                require(all(row[key] == 0 for key in LOSS_KEYS[1:]),
                        "baseline contains an active auxiliary-loss component")
        # The analysis helper hashes the full epoch file. If a comparison carries
        # a training summary it must refer to these exact, unedited histories.
        summary = comparison.get("training", {}).get(variant)
        if summary is not None:
            require(summary.get("completed_epochs") == EXPECTED_EPOCHS and
                    summary.get("metrics_sha256") == sha256(path),
                    "validation comparison is not bound to the plotted training history")
        records[variant], paths[variant + "_metrics"] = rows, path
    for segment in ("early", "middle", "late"):
        item = comparison.get("position_thirds", {}).get(segment, {})
        require(isinstance(item.get("num_samples"), int) and item["num_samples"] > 0,
                "missing or empty preregistered position segment")
        for key in ("accuracy", "weighted_f1", "macro_f1", "ece", "mce", "rmsce"):
            values = item.get("metrics", {}).get(key, {})
            for field in ("baseline", "cmerc", "delta"):
                number(values.get(field))
            require(math.isclose(values["delta"], values["cmerc"] - values["baseline"],
                                 rel_tol=1e-9, abs_tol=1e-9), "inconsistent paired position difference")
    return verified, comparison, records, paths


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.3, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})


def finish(fig, output, name):
    for suffix in ("svg", "pdf"):
        fig.savefig(output / (name + "." + suffix))
    # A standard Matplotlib raster preview makes layout inspection independent
    # of an external SVG/PDF renderer; only the selected Python backend is used.
    fig.savefig(output / (name + ".png"), dpi=180)
    plt.close(fig)


def epoch_axes(ax, label):
    ax.set(xlabel="Epoch", ylabel=label, xlim=(0, EXPECTED_EPOCHS + 2))
    ax.set_xticks([1, 20, 40, 60, 80])
    ax.grid(axis="y", color="#DDDDDD", linewidth=.5, zorder=0)


def plot_convergence(output, records):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for ax, split, letter in zip(axes, ("train", "validation"), ("a", "b")):
        for variant, rows in records.items():
            epochs = [row["epoch"] for row in rows]
            values = [row[split + "_weighted_f1"] for row in rows]
            ax.plot(epochs, values, color=COLORS[variant],
                    linestyle="-" if variant == "baseline" else "--",
                    label="Baseline" if variant == "baseline" else "CMERC")
            if split == "validation":
                best_index = int(np.argmax(values))
                ax.plot(epochs[best_index], values[best_index], "o", markersize=3,
                        color=COLORS[variant], clip_on=False)
        epoch_axes(ax, "Weighted F1 (%)")
        ax.set(ylim=(0, 100), title=f"{letter}  {split.title()}")
    axes[0].legend(loc="lower right")
    fig.suptitle("IEMOCAP causal pair · single training seed 1937", fontsize=10)
    finish(fig, output, "convergence")


def plot_losses(output, records, coefficients):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), layout="constrained")
    epochs = np.arange(1, EXPECTED_EPOCHS + 1)
    for variant, rows in records.items():
        axes[0].plot(epochs, [row["classification_loss"] for row in rows], color=COLORS[variant],
                     linestyle="-" if variant == "baseline" else "--",
                     label="Baseline" if variant == "baseline" else "CMERC")
    for ax, key, color, linestyle in ((axes[1], "cc_loss", "#D55E00", "-"),
            (axes[2], "mscl_loss", "#0072B2", "-"), (axes[2], "cscl_loss", "#009E73", "--")):
        label = {"cc_loss": "CC", "mscl_loss": "MSCL", "cscl_loss": "CSCL"}[key]
        ax.plot(epochs, [coefficients[key] * row[key] for row in records["cmerc"]],
                color=color, linestyle=linestyle, label=f"{coefficients[key]:g} × {label}")
    for ax, title in zip(axes, ("a  Classification", "b  CMERC: weighted CC", "c  CMERC: weighted HSCL")):
        epoch_axes(ax, "Loss contribution")
        ax.set(title=title, ylim=(0, max(.001, ax.dataLim.ymax * 1.08)))
        ax.legend(loc="upper right")
    fig.suptitle("Observed epoch means · single training seed 1937", fontsize=10)
    finish(fig, output, "weighted_losses")


def plot_position_differences(output, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    segments = ("early", "middle", "late")
    labels = [f"{name.title()}\n(n={comparison['position_thirds'][name]['num_samples']})" for name in segments]
    groups = (("accuracy", "weighted_f1", "macro_f1"), ("ece", "mce", "rmsce"))
    display = {"accuracy": "Accuracy", "weighted_f1": "Weighted F1", "macro_f1": "Macro F1",
               "ece": "ECE", "mce": "MCE", "rmsce": "RMSCE"}
    for ax, keys, title in zip(axes, groups,
            ("a  Classification: higher is better", "b  Calibration error: lower is better")):
        all_values = []
        for offset, key, color, marker in zip((-.15, 0., .15), keys, MEASURE_COLORS, ("o", "s", "^")):
            values = [comparison["position_thirds"][name]["metrics"][key]["delta"] for name in segments]
            all_values.extend(values)
            ax.scatter(np.arange(3) + offset, values, s=23, color=color, marker=marker,
                       label=display[key], zorder=3)
        limit = max(1., max(abs(value) for value in all_values) * 1.3)
        ax.axhline(0, color="#555555", linewidth=.7)
        ax.set(title=title, xticks=np.arange(3), xticklabels=labels, ylim=(-limit, limit),
               xlim=(-.45, 2.45), ylabel="CMERC − Baseline (percentage points)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.21), ncols=3)
    fig.suptitle("Validation: preregistered dialogue thirds · single paired seed", fontsize=10)
    finish(fig, output, "validation_position_differences")


def source_data(output, records, comparison, coefficients):
    with (output / "epoch_source.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "epoch", "train_weighted_f1", "validation_weighted_f1",
                         "classification", "weighted_cc", "weighted_mscl", "weighted_cscl"])
        for variant, rows in records.items():
            for row in rows:
                writer.writerow([variant, row["epoch"], row["train_weighted_f1"], row["validation_weighted_f1"],
                    row["classification_loss"], *[row[key] * coefficients[key] for key in LOSS_KEYS[1:]]])
    with (output / "validation_position_source.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position_third", "num_samples", "metric", "baseline", "cmerc", "delta"])
        for name in ("early", "middle", "late"):
            segment = comparison["position_thirds"][name]
            for key in ("accuracy", "weighted_f1", "macro_f1", "ece", "mce", "rmsce"):
                values = segment["metrics"][key]
                writer.writerow([name, segment["num_samples"], key, values["baseline"], values["cmerc"], values["delta"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    batch = args.batch_dir.resolve()
    verified, comparison, records, paths = read_inputs(batch)
    output = batch / "figures"
    output.mkdir(exist_ok=False)
    style()
    plot_convergence(output, records)
    plot_losses(output, records, verified["gradient_coefficients"])
    plot_position_differences(output, comparison)
    source_data(output, records, comparison, verified["gradient_coefficients"])
    contract = {"schema_version": 1, "backend": "Python/matplotlib", "archetype": "quantitative grid",
        "claim": "Describe the observed single-seed training trajectories and preregistered validation differences without inferring statistical significance.",
        "seed_count": 1, "training_seed": 1937, "split_seed": 1475,
        "error_bars": "none; one paired seed, no uncertainty estimate",
        "selection": "validation W-F1; earliest epoch on ties", "test_data_read": False,
        "smoothing": "none", "position_rule": POSITION_RULE,
        "weighted_losses": verified["gradient_coefficients"],
        "loss_semantics": "epoch means multiplied by actual coefficients; magnitudes are not gradient contributions",
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "figures": {name: {"width_inches": 7.2, "formats": ["svg", "pdf", "png"]}
                    for name in ("convergence", "weighted_losses", "validation_position_differences")}}
    (output / "FIGURE_MANIFEST.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    (output / "FIGURE_NOTES.md").write_text(
        "# Causal IEMOCAP single-seed figures\n\n"
        "One paired training seed (1937), split seed 1475; no error bars, significance tests, smoothing, or test-derived figure selection.\n\n"
        "Convergence shows all 80 observed epochs. Validation dots identify each earliest maximal W-F1 epoch. "
        "Loss panels show recorded epoch means and actual coefficient-weighted CC/MSCL/CSCL; their scales differ and their magnitudes do not quantify gradient contributions. "
        "Position panels show all three preregistered validation thirds with fixed symmetric axes around zero. Classification improvement is positive; calibration improvement is negative.\n\n"
        "SVG text is editable; PDF uses embedded TrueType fonts. Each figure is 7.2 inches wide. "
        "PNG files are Matplotlib previews. Companion CSV files contain all plotted values without dialogue IDs.\n")
    print(json.dumps({"status": "completed", "figure_count": 3, "seed_count": 1, "test_data_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
