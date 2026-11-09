#!/usr/bin/env python
"""Render publication-ready figures solely from saved experiment CSV files."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

def rows(name):
    with (ROOT / "results" / name).open(newline="") as f: return list(csv.DictReader(f))

def save(fig, name):
    for suffix in ("png", "pdf", "svg"): fig.savefig(FIG / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
test = rows("per_seed_test_metrics.csv"); bins = rows("reliability_bins.csv"); curves = rows("validation_nll_temperature_curves.csv")
manifest = []

for dataset in ("IEMOCAP", "MELD"):
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    ax.plot([0,1], [0,1], "--", color="0.5", lw=1, label="Perfect calibration")
    for method, label, color in (("vanilla", "Vanilla CMERC", "#4C78A8"), ("global_ts", "Global TS", "#E45756")):
        subset = [r for r in bins if r["dataset"] == dataset and r["method"] == method and int(r["count"]) > 0]
        by_bin = {}
        for r in subset: by_bin.setdefault(int(r["bin"]), []).append(r)
        x, y = [], []
        for b in sorted(by_bin):
            group = by_bin[b]; counts=np.array([float(r["count"]) for r in group])
            x.append(np.average([float(r["mean_confidence"]) for r in group], weights=counts))
            y.append(np.average([float(r["accuracy"]) for r in group], weights=counts))
        ax.plot(x, y, marker="o", lw=1.8, label=label, color=color)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean confidence", ylabel="Accuracy", title=f"{dataset}: 15-bin reliability (five-seed summary)")
    ax.legend(frameon=False); ax.grid(alpha=.2); save(fig, f"reliability_{dataset}")
    manifest.append(f"reliability_{dataset}")

seeds=[1475,67137,2024,3407,42]
for metric, ylabel in (("ece15","ECE (15 bins)"),("brier","Multiclass Brier score"),("nll","NLL (nats)")):
    fig, axes=plt.subplots(1,2,figsize=(10,4),sharey=False)
    for ax,dataset in zip(axes,("IEMOCAP","MELD")):
        group=[next(r for r in test if r["dataset"]==dataset and int(r["seed"])==s) for s in seeds]
        x=np.arange(5); w=.36
        ax.bar(x-w/2,[float(r[f"{metric}_before"]) for r in group],w,label="Vanilla",color="#4C78A8")
        ax.bar(x+w/2,[float(r[f"{metric}_after"]) for r in group],w,label="Global TS",color="#E45756")
        ax.set_xticks(x, [str(s) for s in seeds], rotation=25); ax.set_title(dataset); ax.set_xlabel("Model seed"); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel(ylabel); axes[1].legend(frameon=False); fig.suptitle(f"Five-seed {ylabel} comparison")
    save(fig,f"five_seed_{metric}"); manifest.append(f"five_seed_{metric}")

fig,ax=plt.subplots(figsize=(7,4.5))
x=np.arange(5); w=.36
for j,(dataset,color) in enumerate((("IEMOCAP","#4C78A8"),("MELD","#E45756"))):
    group=[next(r for r in test if r["dataset"]==dataset and int(r["seed"])==s) for s in seeds]
    ax.bar(x+(j-.5)*w,[float(r["temperature"]) for r in group],w,label=dataset,color=color)
ax.axhline(1,color="black",ls="--",lw=1,label="No scaling (T=1)"); ax.set_xticks(x,[str(s) for s in seeds]); ax.set(xlabel="Model seed",ylabel="Fitted temperature",title="Validation-fitted temperatures"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
save(fig,"temperature_distribution"); manifest.append("temperature_distribution")

fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,dataset in zip(axes,("IEMOCAP","MELD")):
    for seed in seeds:
        group=[r for r in curves if r["dataset"]==dataset and int(r["seed"])==seed and r["point_type"]=="grid"]
        ax.plot([float(r["temperature"]) for r in group],[float(r["validation_nll"]) for r in group],lw=1,label=str(seed))
    ax.set_xscale("log"); ax.set(xlabel="Temperature (log scale)",ylabel="Validation NLL",title=dataset); ax.axvline(1,color="0.5",ls="--",lw=1); ax.grid(alpha=.2); ax.legend(title="Seed",frameon=False,ncol=2)
fig.suptitle("Validation NLL temperature curves"); save(fig,"validation_nll_curves"); manifest.append("validation_nll_curves")

(FIG/"figure_manifest.json").write_text(json.dumps({"status":"complete","data_source":"saved real Validation/Test outputs","figure_stems":manifest,"formats":["png","pdf","svg"],"empty_reliability_bins":"omitted"},indent=2)+"\n")
print(json.dumps({"figures":manifest,"files":len(manifest)*3},indent=2))
