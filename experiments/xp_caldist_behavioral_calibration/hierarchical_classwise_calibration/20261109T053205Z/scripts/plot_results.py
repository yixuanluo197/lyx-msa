#!/usr/bin/env python3
"""Generate preregistered figures from real Validation OOF results."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent; R=ROOT/"results"; F=ROOT/"figures"; F.mkdir(exist_ok=True)
def read(name):
    with (R/name).open(newline="") as f: return list(csv.DictReader(f))
def save(fig,name):
    for ext in ("png","pdf","svg"): fig.savefig(F/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
metrics=read("oof_metrics.csv"); temps=read("classwise_temperatures.csv"); bins=read("reliability_bins.csv")
methods=["E0","E1","E2","E3"]; colors=["#777777","#4C78A8","#F58518","#54A24B"]; figures=[]

# OOF method comparison.
fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
for row,ds in enumerate(("IEMOCAP","MELD")):
  for col,metric in enumerate(("nll","brier","ece15")):
    ax=axes[row,col]; means=[]; stds=[]
    for method in methods:
      vals=[float(r[metric]) for r in metrics if r["dataset"]==ds and r["method"]==method]
      means.append(np.mean(vals)); stds.append(np.std(vals,ddof=1))
    ax.bar(methods,means,yerr=stds,color=colors,capsize=3); ax.set(title=f"{ds}: {metric.upper()}",ylabel="Five-seed mean ± SD"); ax.grid(axis="y",alpha=.2)
save(fig,"oof_method_comparison"); figures.append("oof_method_comparison")

# Class temperature distributions.
fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
for row,ds in enumerate(("IEMOCAP","MELD")):
  names=[]
  for r in temps:
    if r["dataset"]==ds and r["class_name"] not in names: names.append(r["class_name"])
  for col,method in enumerate(("E2","E3")):
    vals=[[float(r["class_temperature"]) for r in temps if r["dataset"]==ds and r["method"]==method and r["class_name"]==name] for name in names]
    axes[row,col].boxplot(vals,tick_labels=names,showfliers=False); axes[row,col].axhline(1,color="0.4",ls="--",lw=1)
    axes[row,col].tick_params(axis="x",rotation=30); axes[row,col].set(title=f"{ds} {method}",ylabel="Outer-fold class temperature")
save(fig,"class_temperature_distributions"); figures.append("class_temperature_distributions")

# Five-seed stability heatmaps for hierarchical temperatures.
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
seeds=[1475,67137,2024,3407,42]
for ax,ds in zip(axes,("IEMOCAP","MELD")):
  names=[]
  for r in temps:
    if r["dataset"]==ds and r["class_name"] not in names: names.append(r["class_name"])
  matrix=np.array([[np.mean([float(r["class_temperature"]) for r in temps if r["dataset"]==ds and r["method"]=="E3" and int(r["seed"])==seed and r["class_name"]==name]) for name in names] for seed in seeds])
  im=ax.imshow(matrix,aspect="auto",cmap="viridis"); ax.set_xticks(range(len(names)),names,rotation=35,ha="right"); ax.set_yticks(range(len(seeds)),seeds); ax.set(title=f"{ds}: mean E3 temperature",xlabel="Original predicted class",ylabel="Model seed")
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if matrix[i,j]>(matrix.min()+matrix.max())/2 else "black")
  fig.colorbar(im,ax=ax,shrink=.8)
save(fig,"five_seed_class_temperature_stability"); figures.append("five_seed_class_temperature_stability")

# Support versus shrinkage.
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
for ax,ds in zip(axes,("IEMOCAP","MELD")):
  for method,color in (("E2","#F58518"),("E3","#54A24B")):
    rr=[r for r in temps if r["dataset"]==ds and r["method"]==method]
    ax.scatter([int(r["outer_train_support"]) for r in rr],[float(r["absolute_log_offset"]) for r in rr],s=9,alpha=.25,label=method,color=color)
  ax.set(xscale="log",title=ds,xlabel="Outer-training predicted-class support (log)",ylabel="|log T_k - log T_global|"); ax.legend(); ax.grid(alpha=.2)
save(fig,"support_vs_temperature_offset"); figures.append("support_vs_temperature_offset")

# Reliability diagrams, five-seed pooled bin summaries (fixed OOF predictions, not pooled observations).
fig,axes=plt.subplots(2,4,figsize=(14,7),layout="constrained")
for row,ds in enumerate(("IEMOCAP","MELD")):
  for col,method in enumerate(methods):
    ax=axes[row,col]; rr=[r for r in bins if r["dataset"]==ds and r["method"]==method and int(r["support"])>0]
    grouped={}
    for r in rr: grouped.setdefault(int(r["bin_index"]),[]).append(r)
    xs=[]; ys=[]
    for _,items in sorted(grouped.items()):
      weights=np.array([int(x["support"]) for x in items]); xs.append(np.average([float(x["mean_confidence"]) for x in items],weights=weights)); ys.append(np.average([float(x["accuracy"]) for x in items],weights=weights))
    ax.plot([0,1],[0,1],"--",color="0.5",lw=1); ax.plot(xs,ys,"o-",color=colors[col],ms=3); ax.set(xlim=(0,1),ylim=(0,1),title=f"{ds} {method}",xlabel="Confidence",ylabel="Accuracy"); ax.grid(alpha=.15)
save(fig,"class_calibration_reliability_diagrams"); figures.append("class_calibration_reliability_diagrams")

(F/"figure_manifest.json").write_text(json.dumps({"source":"Validation-only nested OOF CSV results","figures":figures,"formats":["png","pdf","svg"]},indent=2)+"\n")
print(json.dumps({"figures":figures},indent=2))
