"""Render existing diagnostic measurements; never load corpus or model assets."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    columns = ["last_mse", "second_last_mse", "third_last_mse", "fourth_last_mse", "mse"]
    for ax, dataset in zip(axes, ["IEMOCAP", "MELD"]):
        with (ROOT / "local_only" / "overfit" / dataset / "evaluation_curve.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["step"]) for row in rows] == [0, 25, 50, 100, 200, 300, 400, 500]
        steps = [int(row["step"]) for row in rows]
        for column, label in zip(columns, ["Last", "Second last", "Third last", "Fourth last", "Four-layer mean"]):
            ax.plot(steps, [float(row[column]) for row in rows], marker="o",
                    linewidth=2.2 if column == "mse" else 1.2, label=label)
        ax.set(xlabel="Optimizer step", ylabel="FP32 eval MSE (log scale)",
               title=f"{dataset}: fixed 128 Train Fit texts", yscale="log")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"small_sample_overfit.{extension}", dpi=180)
    plt.close(fig)

    with (ROOT / "results" / "train_dev_gap_all.csv").open() as handle:
        gap_rows = list(csv.DictReader(handle))
    # This figure uses the already evaluated CSV only; no extra model pass.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    for ax, dataset in zip(axes, ["IEMOCAP", "MELD"]):
        selected = [r for r in gap_rows if r["dataset"] == dataset]
        for offset, subset in [(-0.18, "fit"), (0.18, "inner_dev")]:
            values = []
            for model in ["public_initial", "best_epoch8"]:
                row = next(r for r in selected if r["model"] == model and r["subset"] == subset)
                values.append(float(row["overall_mse"]))
            ax.bar([offset, 1 + offset], values, width=0.34, label=subset)
        ax.set(xticks=[0, 1], xticklabels=["Public initial", "Epoch 8 student"],
               ylabel="Four-layer FP32 eval MSE", yscale="log", title=dataset)
        ax.legend()
        ax.grid(axis="y", alpha=0.22)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"train_dev_gap.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
