#!/usr/bin/env python
"""Create the six preregistered figure groups from saved real Validation CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COLORS = {"T-All": "#4477AA", "A": "#EE6677", "V": "#228833"}
TARGETS = ("T-All", "A", "V")


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": "white"})
    result = root / "results"; output = root / "figures"; output.mkdir(parents=True, exist_ok=True)
    summary = rows(result / "per_seed_sensitivity.csv")
    samples = rows(result / "per_utterance_sensitivity.csv")
    oof = rows(result / "correctness_oof_metrics.csv")
    manifest = []

    def export(fig, name, sources):
        files = []
        for extension in ("svg", "pdf", "png"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            if path.stat().st_size == 0: raise ValueError(f"empty figure: {path}")
            files.append(str(path.relative_to(root)))
        manifest.append({"name": name, "sources": sources, "outputs": files})
        plt.close(fig)

    # Five model seeds, one line per seed; never pool utterances across seeds.
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), sharey=True)
    for ax, dataset in zip(axes, ("IEMOCAP", "MELD")):
        selected = [r for r in summary if r["dataset"] == dataset and r["reference"] == "train_mean"
                    and r["subgroup"] == "all" and float(r["strength"]) == 1 and r["target"] in TARGETS]
        for seed in sorted({int(r["seed"]) for r in selected}):
            values = [float(next(r for r in selected if int(r["seed"]) == seed and r["target"] == target)["flip_rate"])
                      for target in TARGETS]
            ax.plot(TARGETS, values, marker="o", alpha=.7, label=str(seed))
        ax.set_title(dataset); ax.set_ylabel("Prediction flip rate" if dataset == "IEMOCAP" else "")
        ax.legend(title="Model seed", fontsize=6, ncol=2)
    axes[0].set_ylim(0, .65)
    fig.suptitle("Cross-seed modality dependence at complete train-mean contraction")
    export(fig, "cross_seed_modality_dependence", ["results/per_seed_sensitivity.csv"])

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    width = .12; x = np.arange(len(TARGETS))
    for dataset_index, dataset in enumerate(("IEMOCAP", "MELD")):
        for rho_index, rho in enumerate((.5, 1.0)):
            means = []
            for target in TARGETS:
                values = [float(r["flip_rate"]) for r in summary if r["dataset"] == dataset
                          and r["reference"] == "train_mean" and r["subgroup"] == "all"
                          and r["target"] == target and float(r["strength"]) == rho]
                means.append(np.mean(values))
            offset = (dataset_index*2 + rho_index - 1.5)*width
            ax.bar(x+offset, means, width, label=f"{dataset} rho={rho:g}", alpha=.75+.1*rho_index)
    ax.set(xticks=x, xticklabels=TARGETS, ylabel="Mean flip rate across five model seeds",
           title="Dataset and modality sensitivity comparison")
    ax.legend(fontsize=7)
    export(fig, "dataset_modality_comparison", ["results/per_seed_sensitivity.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, dataset in zip(axes, ("IEMOCAP", "MELD")):
        x = np.arange(3); width = .34
        for offset, reference, label in ((-.17, "train_mean", "Train mean"), (.17, "donor", "Random donor")):
            values = []
            for target in TARGETS:
                chosen = [float(r["flip_rate"]) for r in summary if r["dataset"] == dataset
                          and r["reference"] == reference and r["subgroup"] == "all"
                          and r["target"] == target and float(r["strength"]) == 1]
                values.append(np.mean(chosen))
            ax.bar(x+offset, values, width, label=label)
        ax.set(xticks=x, xticklabels=TARGETS, title=dataset,
               ylabel="Mean flip rate" if dataset == "IEMOCAP" else "")
        ax.legend(fontsize=7)
    fig.suptitle("Reference sensitivity: training mean versus random training donors")
    export(fig, "mean_vs_donor_reference", ["results/per_seed_sensitivity.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), sharey=True)
    for ax, dataset in zip(axes, ("IEMOCAP", "MELD")):
        data = []
        for target in TARGETS:
            data.append([float(r["flip"]) for r in samples if r["dataset"] == dataset and r["reference"] == "main"
                         and r["target"] == target and float(r["strength"]) == .5])
        ax.violinplot(data, showmeans=True, showextrema=False)
        ax.set(xticks=np.arange(1, 4), xticklabels=TARGETS, title=dataset,
               ylabel="Per-intervention prediction flip" if dataset == "IEMOCAP" else "")
    fig.suptitle("Medium-strength prediction flip distributions (all seeds shown, not inferentially pooled)")
    export(fig, "modality_flip_distributions", ["results/per_utterance_sensitivity.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), sharey=True)
    for ax, dataset in zip(axes, ("IEMOCAP", "MELD")):
        for target in TARGETS:
            selected = [r for r in samples if r["dataset"] == dataset and r["reference"] == "main"
                        and r["target"] == target and float(r["strength"]) == .5]
            correct = [float(r["js_divergence"]) for r in selected if int(r["original_prediction"]) == int(r["label"])]
            wrong = [float(r["js_divergence"]) for r in selected if int(r["original_prediction"]) != int(r["label"])]
            ax.plot([0, 1], [np.mean(correct), np.mean(wrong)], marker="o", color=COLORS[target], label=target)
        ax.set(xticks=[0, 1], xticklabels=["Original correct", "Original wrong"], title=dataset,
               ylabel="Mean JS divergence" if dataset == "IEMOCAP" else "")
        ax.legend(fontsize=7)
    fig.subplots_adjust(wspace=.28)
    fig.suptitle("Behavior response by original correctness (descriptive)")
    export(fig, "correct_wrong_behavior", ["results/per_utterance_sensitivity.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), sharey=True)
    models = ("confidence_only", "confidence_plus_text", "confidence_plus_audio",
              "confidence_plus_visual", "confidence_plus_text_audio_visual")
    labels = ("Conf", "+T", "+A", "+V", "+TAV")
    for ax, dataset in zip(axes, ("IEMOCAP", "MELD")):
        values = [[float(r["auroc_success"]) for r in oof if r["dataset"] == dataset and r["model"] == model]
                  for model in models]
        means = [np.mean(value) for value in values]
        errors = [np.std(value, ddof=1) for value in values]
        ax.bar(np.arange(len(models)), means, yerr=errors, color=["#999999"]+["#4477AA"]*4, capsize=2)
        ax.set(xticks=np.arange(len(models)), xticklabels=labels, title=dataset,
               ylabel="LODO AUROC success" if dataset == "IEMOCAP" else "", ylim=(0, 1))
    fig.suptitle("Confidence-only versus confidence plus mean-reference sensitivity")
    export(fig, "oof_correctness_comparison", ["results/correctness_oof_metrics.csv"])

    (output / "figure_manifest.json").write_text(json.dumps({"real_validation_results_only": True,
                                                               "figure_groups": manifest}, indent=2) + "\n")
    print(json.dumps({"figure_groups": len(manifest), "formats": ["svg", "pdf", "png"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args(); main(arguments.root.resolve())
