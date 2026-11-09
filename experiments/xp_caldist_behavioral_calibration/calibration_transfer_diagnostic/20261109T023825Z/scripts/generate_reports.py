#!/usr/bin/env python
"""Generate required reports from frozen Validation-only diagnostic results."""
from __future__ import annotations
import csv,json,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;R=ROOT/"results";REP=ROOT/"reports"
def read(n):
 with (R/n).open(newline="") as f:return list(csv.DictReader(f))
def write(n,s):(REP/n).write_text(s.strip()+"\n")
def table(h,rows):return "|"+"|".join(h)+"|\n|"+"|".join(["---"]*len(h))+"|\n"+"\n".join("|"+"|".join(map(str,r))+"|" for r in rows)
seeds=[1475,67137,2024,3407,42];lodo=read("lodo_temperatures.csv");dlg=read("per_dialogue_metrics.csv");cls=read("classwise_calibration.csv");size=read("temperature_sample_size.csv");reg=read("regularization_oof_comparison.csv");folds=read("regularization_outer_folds.csv");boot=read("paired_bootstrap_oof.csv")

stability=[]
for ds in ("IEMOCAP","MELD"):
 for seed in seeds:
  rr=[r for r in lodo if r["dataset"]==ds and int(r["seed"])==seed];x=[float(r["lodo_temperature"]) for r in rr]
  top=max(rr,key=lambda r:abs(float(r["temperature_influence"])))
  stability.append([ds,seed,f'{float(rr[0]["full_validation_temperature"]):.4f}',f'{min(x):.4f}–{max(x):.4f}',f'{statistics.stdev(x):.4f}',top["held_out_dialogue"],f'{float(top["temperature_influence"]):+.4f}'])
stabtab=table(["Dataset","Seed","Full T","LODO range","LODO SD","Largest-influence dialogue","ΔT when left out"],stability)

sample=[]
for ds in ("IEMOCAP","MELD"):
 for prop in (.5,.75,1.0):
  within=[];ranges=[]
  for seed in seeds:
   x=[float(r["temperature"]) for r in size if r["dataset"]==ds and int(r["seed"])==seed and float(r["dialogue_proportion"])==prop]
   within.append(statistics.stdev(x) if len(x)>1 else 0);ranges.extend(x)
  sample.append([ds,f'{prop:.0%}',f'{statistics.mean(within):.4f}',f'{min(ranges):.4f}–{max(ranges):.4f}',len(ranges)])
sampletab=table(["Dataset","Dialogue fraction","Mean within-seed SD(T)","All runs range","Fits"],sample)

oof=[]
for ds in ("IEMOCAP","MELD"):
 for method in ("vanilla","lodo_global_ts","regularized_lodo_ts"):
  rr=[r for r in reg if r["dataset"]==ds and r["method"]==method]
  oof.append([ds,method,f'{statistics.mean(float(r["nll"]) for r in rr):.6f}',f'{statistics.mean(float(r["brier"]) for r in rr):.6f}',f'{statistics.mean(float(r["ece15"]) for r in rr):.6f}'])
ooftab=table(["Dataset","Outer-OOF method","NLL","Brier","ECE15"],oof)

classrows=[]
for ds in ("IEMOCAP","MELD"):
 names=[]
 for r in cls:
  if r["dataset"]==ds and r["grouping"]=="true_class" and r["class_name"] not in names:names.append(r["class_name"])
 for name in names:
  v=[r for r in cls if r["dataset"]==ds and r["grouping"]=="true_class" and r["class_name"]==name and r["method"]=="vanilla"]
  t=[r for r in cls if r["dataset"]==ds and r["grouping"]=="true_class" and r["class_name"]==name and r["method"]=="full_global_ts_in_sample"]
  classrows.append([ds,name,v[0]["support"],f'{statistics.mean(float(r["signed_gap"]) for r in v):+.4f}',f'{statistics.mean(float(r["signed_gap"]) for r in t):+.4f}',
                    f'{statistics.mean(float(b["mean_nll"])-float(a["mean_nll"]) for a,b in zip(v,t)):+.4f}',"yes" if any(r["small_support"]=="True" for r in v) else "no"])
classtab=table(["Dataset","True class","Support","Vanilla gap","Full-TS gap","Δ class NLL","Small support"],classrows)

# Dialogue extremes across model seeds.
dialogue_text=[]
for ds in ("IEMOCAP","MELD"):
 by=defaultdict(list)
 for r in dlg:
  if r["dataset"]==ds and r["method"]=="vanilla":by[r["dialogue_id"]].append(float(r["signed_gap"]))
 ranked=sorted((statistics.mean(v),k) for k,v in by.items())
 dialogue_text.append(f"- {ds}: strongest underconfidence `{ranked[0][1]}` ({ranked[0][0]:+.4f}); strongest overconfidence `{ranked[-1][1]}` ({ranked[-1][0]:+.4f}).")

# Reg-vs-unregularized differences and lambda counts.
regdiff=[]
for ds in ("IEMOCAP","MELD"):
 for metric in ("nll","brier","ece15"):
  dif=[]
  for seed in seeds:
   a=next(float(r[metric]) for r in reg if r["dataset"]==ds and int(r["seed"])==seed and r["method"]=="lodo_global_ts")
   b=next(float(r[metric]) for r in reg if r["dataset"]==ds and int(r["seed"])==seed and r["method"]=="regularized_lodo_ts");dif.append(b-a)
  regdiff.append([ds,metric,f'{statistics.mean(dif):+.6f}',f'{sum(x<0 for x in dif)}/5'])
regtab=table(["Dataset","Metric","Regularized−ordinary LODO","Improved seeds"],regdiff)
lcounts={ds:Counter(float(r["selected_lambda"]) for r in folds if r["dataset"]==ds) for ds in ("IEMOCAP","MELD")}

write("VALIDATION_TEMPERATURE_STABILITY.md",f"""
# Validation temperature stability

{stabtab}

IEMOCAP has visibly larger refitting variability. Across seeds, its LODO SD ranges from 0.0516 to 0.1362; MELD ranges only from 0.0016 to 0.0020. Leaving out `Ses02F_impro06` produces the largest absolute temperature change in four of five IEMOCAP seeds, reaching −0.3591. The remaining seed is most affected by `Ses02M_impro07`.

The concatenated outer-OOF results improve IEMOCAP NLL point estimates in 5/5 seeds, but all five paired dialogue-bootstrap NLL intervals include zero. MELD improves NLL, Brier, and ECE15 in 5/5 seeds and every corresponding interval excludes zero. This separates a weak, high-variance IEMOCAP calibration signal from a strong, stable MELD signal.
""")

write("DIALOGUE_CALIBRATION_ANALYSIS.md",f"""
# Dialogue calibration analysis

Each dialogue was summarized using fixed original predictions: size, true/predicted class composition, accuracy, mean confidence, signed gap, NLL, Brier, and error-confidence distribution. Positive signed gap means overconfidence.

{chr(10).join(dialogue_text)}

IEMOCAP contains sharply opposing dialogue-level directions. `Ses02F_impro06` has mean signed gap +0.5016 and also dominates leave-one-dialogue-out temperature influence, while `Ses01M_script03_1` is underconfident at −0.1685. MELD also contains dialogue heterogeneity, but 114 dialogues dilute any one dialogue's influence; its largest absolute LODO temperature shift is below 0.0068.

These results support composition sensitivity rather than a single anomalous metric calculation. Single-dialogue ECE remains noisy and is retained only descriptively; dialogue NLL and concatenated OOF metrics carry the primary interpretation.
""")

write("CLASS_CONDITIONAL_CALIBRATION.md",f"""
# Class-conditional calibration diagnostic

{classtab}

IEMOCAP classes do not share one calibration direction. Neutral (+0.2343) and frustrated (+0.1295) are overconfident on average, while sad and angry are approximately calibrated or mildly underconfident. A global T>1 improves aggregate Validation NLL mainly through neutral, frustrated, and happy, but increases mean class NLL for angry, excited, and sad.

MELD shows the opposite aggregate direction but still has class conflicts: neutral, surprise, anger, and joy are underconfident, whereas sadness, fear, and disgust are overconfident. Global T<1 strongly helps the high-support underconfident classes while worsening sadness, fear, and disgust. Disgust has only 22 samples and fear 40, so these estimates require caution.

The conflicts motivate future class-conditional research, but this phase supplies no fitted Classwise Temperature Scaling result. IEMOCAP's small dialogue count and per-seed class variability make an unregularized classwise calibrator especially prone to overfitting.
""")

write("TEMPERATURE_SAMPLE_SIZE_ANALYSIS.md",f"""
# Temperature sample-size analysis

{sampletab}

At 50% of Validation dialogues, IEMOCAP's mean within-seed SD(T) is 0.3291, compared with 0.0181 for MELD. At 75%, the values are 0.1800 and 0.0108. IEMOCAP subsets can even select T<1 although every full Validation fit selects T>1, demonstrating direction instability at small calibration-set sizes. MELD subsets remain consistently below one and concentrate near the full-data estimate.

The repeated-subset distributions directly include temperature refitting variation. They are distinct from the fixed-OOF dialogue bootstrap, which conditions on already fitted fold temperatures.
""")

write("REGULARIZED_TEMPERATURE_DIAGNOSTIC.md",f"""
# Regularized temperature diagnostic

The outer loop leaves one dialogue out. Within each outer training set, five deterministic grouped folds choose lambda without seeing the outer dialogue. The regularization objective is `NLL + lambda * log(T)^2`.

{ooftab}

{regtab}

Selected-lambda counts: IEMOCAP `{dict(sorted(lcounts['IEMOCAP'].items()))}`; MELD `{dict(sorted(lcounts['MELD'].items()))}`.

Regularization does not provide a consistent NLL/Brier improvement over ordinary LODO TS. For IEMOCAP it lowers mean ECE15 but worsens mean NLL by 0.003615 and Brier by 0.000874; NLL improves in only 1/5 seeds. MELD differences are negligible and mixed. Shrinking T toward one therefore does not resolve the transfer problem under this Validation-only nested protocol.
""")

write("FINAL_RESEARCH_DECISION.md",f"""
# Final research decision

{ooftab}

1. **Why IEMOCAP may fail to transfer:** only 12 Validation dialogues produce high temperature refitting variance, opposing dialogue/class calibration directions, and a weak OOF signal whose bootstrap intervals cross zero. Historical Test degradation is consistent with this unstable estimate plus calibration-distribution shift.
2. **Dialogue dominance:** yes. `Ses02F_impro06` is the largest temperature-influence dialogue in 4/5 seeds and is strongly overconfident; a few dialogues materially shape T.
3. **Sample-size sensitivity:** strong for IEMOCAP and modest for MELD. IEMOCAP's 50% subset within-seed SD(T) is 0.3291 versus MELD's 0.0181.
4. **Class directions:** they conflict in both datasets. IEMOCAP neutral/frustrated are overconfident while sad/angry are not; MELD's large neutral/surprise groups are underconfident while sadness/fear/disgust are overconfident.
5. **Regularization:** it does not consistently improve outer-OOF NLL or Brier and cannot be claimed as a solution.
6. **Why MELD is stable:** 114 Validation dialogues, much lower leave-dialogue-out influence, concentrated subset-temperature distributions, and a common underconfidence direction among its dominant classes.
7. **Classwise TS motivation:** class conflicts provide motivation, but IEMOCAP does not yet provide enough calibration data for an unconstrained classwise method. Any class-conditional model should be hierarchical/regularized and independently evaluated.
8. **Calibration data:** a larger, composition-balanced and independent calibration set is needed, especially for IEMOCAP.
9. **Behavior signals:** the route remains paused; no result here overturns the prior negative incremental-correctness finding.
10. **Best next single experiment:** run a preregistered session-grouped study on a larger independent calibration/evaluation partition, comparing Global TS with a strongly regularized hierarchical class-conditional temperature model. All method selection must occur inside the calibration portion, with one untouched evaluation partition. This phase does not execute it.

This is exploratory because known historical Test behavior motivated the question. No new Test prediction was read or evaluated, and these findings cannot be presented as an independently confirmed Test improvement.
""")

(ROOT/"README.md").write_text(f"""# CMERC Calibration Transfer Diagnostic

Validation-only diagnosis of why Global Temperature Scaling transfers differently for IEMOCAP and MELD. No model was trained, no checkpoint was changed, and no Test probability or label was read.

{ooftab}

IEMOCAP's 12-dialogue calibration set yields substantially higher leave-dialogue-out and subset-refit temperature variance, with opposing dialogue and class calibration directions. MELD's 114 dialogues yield a concentrated T<1 estimate and consistent OOF improvement. Nested shrinkage toward T=1 does not consistently improve NLL or Brier.

See `reports/PREREGISTRATION.md` and the six required reports. All result CSVs derive from the ten true Validation prediction files; `verification.json` records an independent full recomputation.
""")
print(json.dumps({"reports":7,"readme":True},indent=2))
