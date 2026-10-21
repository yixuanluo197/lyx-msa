"""Diagnostic single-panel curves from exported CSV; CPU only, no training."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT/"results"/name).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete",action="store_true")
    args = parser.parse_args()
    summary = json.loads((ROOT/"results/metrics_summary.json").read_text())
    if not args.allow_incomplete:
        assert summary["formal_runs_completed"]==4
    out = ROOT/"results/figures"
    out.mkdir(exist_ok=True)
    curves = load("epoch_curves.csv")
    old = load("xp04_epoch_curves.csv")
    losses = load("epoch_loss_curves.csv")
    files = []
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.fonttype":"none","figure.dpi":120})
    for row in summary["runs"]:
        ident = row["run_id"]
        actual = [r for r in curves if r["run_id"]==ident]
        reference = [r for r in old if r["run_id"]==ident]
        fig,ax = plt.subplots(figsize=(7.5,4.6))
        for split,label,color,style in (("train","README train","#D55E00",":"),
                                       ("test","README test (epoch selection)","#0072B2","-")):
            selected = [r for r in actual if r["split"]==split and r.get("wf1_percent")]
            ax.plot([int(r["epoch"]) for r in selected],[float(r["wf1_percent"]) for r in selected],label=label,color=color,linestyle=style,lw=1.5)
        for key,label,color,style in (("train_weighted_f1","XP04 train","#E69F00","-."),
                                     ("validation_weighted_f1","XP04 validation (epoch selection)","#009E73","--")):
            selected = [r for r in reference if r.get(key)]
            ax.plot([int(r["epoch"]) for r in selected],[float(r[key]) for r in selected],label=label,color=color,linestyle=style,lw=1.4)
        if row["best_epoch"] is not None:
            best = next(r for r in actual if r["split"]=="test" and int(r["epoch"])==row["best_epoch"])
            ax.scatter([row["best_epoch"]],[float(best["wf1_percent"])],c="#0072B2",s=28,zorder=4)
        ax.set(xlabel="Epoch",ylabel="Weighted F1 (%)",xlim=(1,row["epochs_planned"]),ylim=(0,100),
               title=f"{row['dataset']} / {row['model']} / seed {row['seed']}")
        ax.grid(alpha=.18)
        ax.legend(loc="lower right",fontsize=8,frameon=False)
        fig.text(.5,.025,"Different training pools and selection protocols; descriptive comparison only.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.055,1,1))
        for ext in ("png","svg"):
            path = out/(ident+"_wf1."+ext)
            fig.savefig(path,dpi=170)
            files.append(str(path.relative_to(ROOT)))
        plt.close(fig)

        selected = [r for r in losses if r["run_id"]==ident and r["split"]=="train"]
        cfg = json.loads((ROOT/"checkpoints"/ident/"effective_args.json").read_text())
        components = [("classification_loss_batch_mean",1,"Classification","#0072B2")]
        if row["model"]=="cmerc":
            components += [("rank_loss_batch_mean",cfg["rank_coff"],"Weighted CC","#D55E00"),
                           ("Mscl_tav_batch_mean",cfg["mscl_coff"],"Weighted MSCL","#009E73"),
                           ("Cscl_c_batch_mean",cfg["cscl_coff"],"Weighted CSCL","#CC79A7"),
                           ("total_loss_batch_mean",1,"Total","#333333")]
        fig,ax = plt.subplots(figsize=(7.5,4.6))
        for key,coefficient,label,color in components:
            values = [r for r in selected if r.get(key)]
            ax.plot([int(r["epoch"]) for r in values],[float(r[key])*coefficient for r in values],label=label,color=color,lw=1.5)
        ax.set(xlabel="Epoch",ylabel="Training loss (batch mean)",xlim=(1,row["epochs_planned"]),
               title=f"{row['dataset']} / {row['model']} / seed {row['seed']}")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8,frameon=False,ncol=2)
        fig.text(.5,.025,"Regularizers are multiplied by the fixed README coefficients; no loss formula change.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.055,1,1))
        for ext in ("png","svg"):
            path = out/(ident+"_loss."+ext)
            fig.savefig(path,dpi=170)
            files.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    manifest = {"purpose":"diagnostic single-panel curves, not a publication style workflow",
        "python":sys.version,"executable":sys.executable,"matplotlib":matplotlib.__version__,
        "host":platform.node(),"formal_training_complete":summary["formal_runs_completed"]==4,
        "inputs":{name:hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest() for name in
            ("epoch_curves.csv","epoch_loss_curves.csv","xp04_epoch_curves.csv","metrics_summary.json")},
        "files":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        "scientific_caveat":"README test-selected full-train protocol and XP04 validation-selected held-out protocol differ"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"figures":len(files),"formal_training_complete":manifest["formal_training_complete"]}))


if __name__ == "__main__":
    main()
