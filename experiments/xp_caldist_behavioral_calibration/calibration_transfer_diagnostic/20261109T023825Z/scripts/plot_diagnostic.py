#!/usr/bin/env python
"""Plots from real Validation-only diagnostic CSVs."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent; R=ROOT/"results"; F=ROOT/"figures"; F.mkdir(exist_ok=True)
def read(name):
    with (R/name).open(newline="") as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ("png","pdf","svg"):fig.savefig(F/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
seeds=[1475,67137,2024,3407,42]; lodo=read("lodo_temperatures.csv"); dialogs=read("per_dialogue_metrics.csv"); classes=read("classwise_calibration.csv"); sizes=read("temperature_sample_size.csv"); reg=read("regularization_oof_comparison.csv")
figs=[]

fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,ds in zip(axes,("IEMOCAP","MELD")):
    vals=[[float(r["lodo_temperature"]) for r in lodo if r["dataset"]==ds and int(r["seed"])==s] for s in seeds]
    ax.boxplot(vals,tick_labels=list(map(str,seeds)),showmeans=True); ax.axhline(1,color="0.4",ls="--",lw=1)
    ax.set(title=ds,xlabel="Model seed",ylabel="LODO-fitted temperature"); ax.grid(axis="y",alpha=.2)
fig.suptitle("Validation LODO temperature stability"); save(fig,"lodo_temperature_distribution"); figs.append("lodo_temperature_distribution")

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,ds in zip(axes,("IEMOCAP","MELD")):
    rr=[r for r in dialogs if r["dataset"]==ds and r["method"]=="vanilla"]
    by={}
    for r in rr:by.setdefault(r["dialogue_id"],[]).append(float(r["signed_gap"]))
    ranked=sorted(((np.mean(v),k) for k,v in by.items()))
    if len(ranked)>30: ranked=ranked[:15]+ranked[-15:]
    ax.barh(np.arange(len(ranked)),[x[0] for x in ranked],color=["#4C78A8" if x[0]<0 else "#E45756" for x in ranked])
    ax.set_yticks(np.arange(len(ranked)),[x[1] for x in ranked],fontsize=6); ax.axvline(0,color="black",lw=.8); ax.set(title=ds,xlabel="Mean signed gap across five seeds")
fig.suptitle("Dialogue-level underconfidence (blue) and overconfidence (red)"); save(fig,"dialogue_signed_calibration_gap"); figs.append("dialogue_signed_calibration_gap")

fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
for col,ds in enumerate(("IEMOCAP","MELD")):
    names=[]
    for r in classes:
        if r["dataset"]==ds and r["grouping"]=="true_class" and r["method"]=="vanilla" and r["class_name"] not in names:names.append(r["class_name"])
    for row,method in enumerate(("vanilla","full_global_ts_in_sample")):
        arr=np.array([[np.mean([float(r["signed_gap"]) for r in classes if r["dataset"]==ds and r["grouping"]=="true_class" and r["method"]==method and r["class_name"]==name and int(r["seed"])==s]) for name in names] for s in seeds])
        ax=axes[row,col]; im=ax.imshow(arr,aspect="auto",cmap="coolwarm",vmin=-.4,vmax=.4); ax.set_xticks(range(len(names)),names,rotation=35,ha="right"); ax.set_yticks(range(5),list(map(str,seeds))); ax.set_title(f"{ds} — {'Vanilla' if row==0 else 'Full Global TS'}")
fig.colorbar(im,ax=axes.ravel().tolist(),label="Signed gap (confidence − accuracy)",shrink=.8); fig.suptitle("True-class conditional calibration direction"); save(fig,"class_conditional_signed_gap"); figs.append("class_conditional_signed_gap")

fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,ds in zip(axes,("IEMOCAP","MELD")):
    positions=[]; vals=[]; labels=[]; pos=1
    for s in seeds:
      for proportion in (.5,.75,1.0):
        x=[float(r["temperature"]) for r in sizes if r["dataset"]==ds and int(r["seed"])==s and float(r["dialogue_proportion"])==proportion]
        vals.append(x);positions.append(pos);labels.append(f"{s}\n{int(proportion*100)}%");pos+=1
      pos+=.5
    ax.boxplot(vals,positions=positions,widths=.7,showfliers=False);ax.set_xticks(positions,labels,rotation=65,fontsize=6);ax.axhline(1,color="0.4",ls="--",lw=1);ax.set(title=ds,ylabel="Fitted temperature",xlabel="Seed / dialogue fraction");ax.grid(axis="y",alpha=.2)
fig.suptitle("Temperature sensitivity to Validation dialogue count");save(fig,"temperature_sample_size_sensitivity");figs.append("temperature_sample_size_sensitivity")

fig,axes=plt.subplots(2,3,figsize=(12,7))
for row,ds in enumerate(("IEMOCAP","MELD")):
 for col,metric in enumerate(("nll","brier","ece15")):
    ax=axes[row,col]; x=np.arange(5); w=.25
    for j,(method,label,color) in enumerate((("vanilla","Vanilla","#4C78A8"),("lodo_global_ts","LODO TS","#F58518"),("regularized_lodo_ts","Reg. LODO TS","#54A24B"))):
        vals=[float(next(r[metric] for r in reg if r["dataset"]==ds and int(r["seed"])==s and r["method"]==method)) for s in seeds]
        ax.bar(x+(j-1)*w,vals,w,label=label,color=color)
    ax.set_xticks(x,list(map(str,seeds)),rotation=25);ax.set_title(f"{ds} — {metric.upper()}");ax.grid(axis="y",alpha=.2)
    if row==0 and col==2:ax.legend(frameon=False,fontsize=8)
fig.suptitle("Validation outer-OOF calibration comparison");save(fig,"regularized_oof_comparison");figs.append("regularized_oof_comparison")

fig,ax=plt.subplots(figsize=(9,5))
rr=[r for r in lodo if r["dataset"]=="IEMOCAP"]
by={}
for r in rr:by.setdefault(r["held_out_dialogue"],[]).append(abs(float(r["temperature_influence"])))
rank=sorted(((np.mean(v),k) for k,v in by.items()))
ax.barh(np.arange(len(rank)),[x[0] for x in rank],color="#E45756");ax.set_yticks(np.arange(len(rank)),[x[1] for x in rank]);ax.set(xlabel="Mean |T leave-dialogue-out − T full|",title="IEMOCAP dialogue influence on fitted temperature");ax.grid(axis="x",alpha=.2)
save(fig,"iemocap_dialogue_temperature_influence");figs.append("iemocap_dialogue_temperature_influence")

(F/"figure_manifest.json").write_text(json.dumps({"status":"complete","data":"Validation only","figure_stems":figs,"formats":["png","pdf","svg"]},indent=2)+"\n")
print(json.dumps({"figures":figs,"files":len(figs)*3},indent=2))
