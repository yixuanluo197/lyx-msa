#!/usr/bin/env python
"""Generate the required research reports from frozen CSV/JSON artifacts."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent; RES=ROOT/"results"; REP=ROOT/"reports"
def read(name):
    with (RES/name).open(newline="") as f: return list(csv.DictReader(f))
def write(name,text): (REP/name).write_text(text.strip()+"\n")
def table(headers,rows): return "|"+"|".join(headers)+"|\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("|"+"|".join(map(str,r))+"|" for r in rows)
params=read("temperature_parameters.csv"); tests=read("per_seed_test_metrics.csv"); aggs=read("aggregate_test_metrics.csv"); boots=read("paired_bootstrap_results.csv")

param_table=table(["Dataset","Seed","T","Val NLL before","Val NLL after","Boundary"],[[r["dataset"],r["seed"],f'{float(r["temperature"]):.6f}',f'{float(r["validation_nll_before"]):.6f}',f'{float(r["validation_nll_after"]):.6f}',r["boundary_optimum"]] for r in params])
test_table=table(["Dataset","Seed","ECE15 before→after","Brier before→after","NLL before→after"],[[r["dataset"],r["seed"],f'{float(r["ece15_before"]):.6f}→{float(r["ece15_after"]):.6f}',f'{float(r["brier_before"]):.6f}→{float(r["brier_after"]):.6f}',f'{float(r["nll_before"]):.6f}→{float(r["nll_after"]):.6f}'] for r in tests])
agg_primary=[r for r in aggs if r["metric"] in ("ece15","brier","nll")]
agg_table=table(["Dataset","Metric","Vanilla mean±SD","Global TS mean±SD","Δ TS−Vanilla","Improved Seeds"],[[r["dataset"],r["metric"],f'{float(r["vanilla_mean"]):.6f}±{float(r["vanilla_sample_std"]):.6f}',f'{float(r["global_ts_mean"]):.6f}±{float(r["global_ts_sample_std"]):.6f}',f'{float(r["paired_difference_mean"]):+.6f}',r["direction_consistency"]] for r in agg_primary])

# Historical Experiment 6 anchor comparison.
historic={"IEMOCAP":{"seed":1475,"accuracy":.680837954405422,"mean_confidence":.7179599418076426,"ece10":.037121987402220454,"brier":.43016534432619563,"nll":.7789953495267861},"MELD":{"seed":67137,"accuracy":.6670498084291188,"mean_confidence":.4763837325374515,"ece10":.1906660758916673,"brier":.520741379938242,"nll":1.1187046025105363}}
comparison=[]
for dataset,h in historic.items():
    r=next(x for x in tests if x["dataset"]==dataset and int(x["seed"])==h["seed"])
    for metric in ("accuracy","mean_confidence","ece10","brier","nll"):
        current=float(r[f"{metric}_before"])
        comparison.append({"dataset":dataset,"seed":h["seed"],"metric":metric,"experiment6":h[metric],"current_vanilla":current,"absolute_difference":abs(current-h[metric]),"match_within_5e-7":abs(current-h[metric])<5e-7})
with (RES/"experiment6_comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0])); w.writeheader(); w.writerows(comparison)

write("GLOBAL_TEMPERATURE_SCALING.md",f"""
# Global Temperature Scaling

For each XP04 validation-selected checkpoint, this experiment fits one positive scalar temperature by minimizing multiclass NLL on the true Validation split. The calibrated distribution is `softmax(log(P_original) / T)`. The fit uses a preregistered 2,001-point log-spaced grid over [0.05, 20], containing T=1, followed by deterministic golden-section refinement. Test labels and Test metrics were unavailable to the fitting script.

{param_table}

All ten solutions were interior. Positive scalar scaling preserved every Test argmax and therefore preserved Accuracy, Weighted-F1, and Macro-F1 exactly.
""")

write("VALIDATION_FITTING_REPORT.md",f"""
# Validation fitting report

The read-only audit passed before fitting: all ten checkpoint hashes and twenty prediction artifacts matched historical provenance, the true split sizes were IEMOCAP 527/12 dialogues and MELD 1109/114 dialogues, and Validation/Test dialogue IDs were disjoint. Existing complete probabilities avoided GPU inference.

{param_table}

Validation NLL decreased for every checkpoint by construction. IEMOCAP selected T>1 for all seeds, while MELD selected T<1 for all seeds. The frozen parameter payload SHA-256 is `b3d07159fc930fc68d5e176147a87456114d6fd9336bbc7c421e81b3eb7d9c09`. No boundary expansion or Test-driven revision was performed.
""")

write("TEST_CALIBRATION_REPORT.md",f"""
# Test calibration report

The ten frozen temperatures were applied once to the corresponding Test probability vectors. Lower values are better for ECE15, multiclass Brier, and NLL.

{test_table}

MELD improved all three primary metrics for all five seeds, and every primary dialogue-bootstrap interval excluded zero in the improving direction. IEMOCAP did not show stable transfer: ECE15 improved for 1/5 seeds, Brier for 2/5, and NLL for 1/5; all IEMOCAP primary bootstrap intervals crossed zero.

The metric trade-offs are real. IEMOCAP seed42 improved ECE15 and Brier while NLL worsened; seed3407 improved Brier and NLL while ECE15 slightly worsened. Thus ECE changes alone do not establish comprehensive calibration improvement.

No Test prediction changed. Classification metrics match Vanilla CMERC to numerical equality for every seed.
""")

write("FIVE_SEED_COMPARISON.md",f"""
# Five-seed comparison

The summary unit is the model seed. The same Test utterances appear under all five model seeds, so predictions were never concatenated to claim a fivefold sample size. Standard deviations use `ddof=1`.

{agg_table}

IEMOCAP temperatures vary from 1.243830 to 2.132159 but consistently request confidence reduction on Validation. On Test, that direction mostly fails to improve the three primary metrics. MELD temperatures range from 0.515566 to 0.637564 and consistently sharpen its underconfident distributions; this direction transfers to Test across every seed and primary metric.

Dialogue-level paired bootstrap used 2,000 resamples per dataset/seed and kept all utterances from a sampled dialogue together. The interval conditions on the fixed fitted temperature and fixed model predictions; it does not cover uncertainty from refitting T.
""")

write("FINAL_RESEARCH_DECISION.md",f"""
# Final research decision

The outcome matches the preregistered dataset-specific case (Situation C), with an additional metric-trade-off warning for IEMOCAP.

{agg_table}

Global Temperature Scaling is a strong required baseline for MELD: it reduces mean ECE15 by 0.149244, Brier by 0.046616, and NLL by 0.118099, with 5/5 improving seeds and dialogue-bootstrap intervals below zero for every seed. MELD is systematically underconfident under the saved CMERC probabilities, so T<1 raises confidence.

Global TS is not a reliable Test calibration method for IEMOCAP under this Validation split. Mean ECE15 increases by 0.011228, Brier by 0.002337, and NLL by 0.012978. Although Validation selects T>1 for every seed, that confidence reduction does not transfer consistently. This is evidence of a Validation-to-Test calibration-distribution mismatch; it does not justify Test-tuned temperatures.

The next calibration study should keep Global TS as the reference and investigate the IEMOCAP split-transfer problem first, using validation-only resampling or nested fitting to quantify temperature stability. If a more flexible mechanism is later tested, class-conditional or regularized classwise temperature scaling is a better motivated next candidate than behavior-aware scaling, because prior behavioral signals did not provide stable incremental correctness information. This experiment does not execute that follow-up.
""")

readme=f"""# CMERC five-seed Global Temperature Scaling

This directory is a self-contained post-hoc calibration baseline for ten XP04 validation-selected CMERC checkpoints. It uses true Validation predictions to fit one scalar temperature per dataset/seed and evaluates the frozen parameter once on Test.

## Result

{agg_table}

MELD improves comprehensively in all five seeds. IEMOCAP does not improve consistently and degrades on average. All Test argmax predictions and classification metrics remain unchanged.

## Reproduction from saved probabilities

Run the scripts in order with `/data2/yb/conda_envs/cmerc-xp-repro/bin/python`: `audit_inputs.py`, `fit_temperatures.py`, `evaluate_test.py`, `verify_results.py`; figures use `python scripts/plot_results.py`. The fitting script reads Validation only. Complete original and calibrated distributions are stored in each run directory and in `results/per_utterance_calibrated_probabilities.csv`.

See `reports/PREREGISTRATION.md`, `reports/READ_ONLY_AUDIT.md`, `reports/VALIDATION_FITTING_REPORT.md`, `reports/TEST_CALIBRATION_REPORT.md`, `reports/FIVE_SEED_COMPARISON.md`, and `reports/FINAL_RESEARCH_DECISION.md`.
"""
(ROOT/"README.md").write_text(readme)
print(json.dumps({"reports":6,"experiment6_comparison_rows":len(comparison)},indent=2))
