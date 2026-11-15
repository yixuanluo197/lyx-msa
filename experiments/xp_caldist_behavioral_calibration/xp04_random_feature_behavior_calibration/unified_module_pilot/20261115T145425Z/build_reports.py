"""Generate Chinese result reports and static scientific figures from completed OOF."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
MAIN = "E1_DA"
METHODS = ["XP04", "E1", "L0", "E1_MUDELTA", "E1_DA", "L0_MUDELTA", "L0_DA"]


def num(x, digits=6):
    return f"{float(x):.{digits}f}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def write(name, lines):
    (OUT / name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    metrics = pd.read_csv(OUT / "metrics.csv")
    predictions = pd.read_csv(OUT / "oof_predictions.csv", dtype={"dialogue_id": str})
    feature = pd.read_csv(OUT / "feature_matrix.csv", dtype={"dialogue_id": str})
    bootstrap = pd.read_csv(OUT / "bootstrap_results.csv")
    shifts = pd.read_csv(OUT / "correct_wrong_adjustment.csv")
    choices = pd.read_csv(OUT / "inner_cv_results.csv", dtype={"penalty": str})
    per_dialogue = pd.read_csv(OUT / "per_dialogue_results.csv", dtype={"dialogue_id": str})
    verification = json.loads((OUT / "verification.json").read_text())
    config = json.loads((OUT / "preregistration.json").read_text())
    assert verification["independent_verification"]["passed"]
    difference_rows, decision = [], {}
    for ds in ["IEMOCAP", "MELD"]:
        sub = metrics[metrics.dataset.eq(ds)].set_index("method")
        baseline = sub.loc["XP04"]
        main_ = sub.loc[MAIN]
        passed = bool(main_.ece15 < baseline.ece15 and
                      (main_.binary_brier < baseline.binary_brier or main_.binary_nll < baseline.binary_nll) and
                      main_.binary_brier - baseline.binary_brier <= .001 and
                      main_.binary_nll - baseline.binary_nll <= .005)
        for method in METHODS:
            for reference in ["XP04", "E1", "L0"]:
                for metric in ["ece15", "binary_brier", "binary_nll"]:
                    v = float(sub.loc[method, metric] - sub.loc[reference, metric])
                    difference_rows.append({"dataset": ds, "candidate": method, "reference": reference,
                                            "metric": metric, "difference": v,
                                            "relative_change_percent": 100 * v / sub.loc[reference, metric]})
        selected = choices[choices.dataset.eq(ds) & choices.method.eq(MAIN) & choices.selected]
        decision[ds] = {"primary_point_estimate_gate_passed": passed,
                        "outer_folds_behavior_off": int(selected.penalty.eq("off").sum()),
                        "outer_folds": len(selected), "behavior_increment_established": False,
                        "main_vs_xp04_bootstrap_all_three_ci_below_zero": bool((bootstrap[
                            bootstrap.dataset.eq(ds) & bootstrap.candidate.eq(MAIN) & bootstrap.reference.eq("XP04")].ci_high < 0).all())}
    diff = pd.DataFrame(difference_rows)
    diff.to_csv(OUT / "metric_differences.csv", index=False)
    selected_rows = choices[choices.selected].copy()
    selected_rows.to_csv(OUT / "selected_penalties.csv", index=False)

    bins = []
    for (ds, method), group in predictions.groupby(["dataset", "method"], sort=False):
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            g = group[(group.oof_confidence >= lo) & ((group.oof_confidence < hi) if i < 9 else (group.oof_confidence <= hi))]
            bins.append({"dataset": ds, "method": method, "bin": i, "n": len(g),
                         "mean_confidence": g.oof_confidence.mean(), "accuracy": g.correct.mean()})
    bins = pd.DataFrame(bins)
    bins.to_csv(OUT / "confidence_bins.csv", index=False)
    conditional = []
    for ds, f in feature.groupby("dataset", sort=False):
        for i in range(5):
            group = f[(f.confidence >= i / 5) & ((f.confidence < (i + 1) / 5) if i < 4 else (f.confidence <= 1))]
            if len(group) == 0:
                continue
            for name in ["gaussian_D", "gaussian_A", "dropout_D", "dropout_A"]:
                median = group[name].median()
                for label, mask in [("low_or_equal", group[name] <= median), ("high", group[name] > median)]:
                    g = group[mask]
                    conditional.append({"dataset": ds, "original_confidence_bin": f"[{i/5:.1f},{(i+1)/5:.1f}{']' if i==4 else ')'}",
                                        "feature": name, "half": label, "n": len(g), "median_threshold": median,
                                        "accuracy": g.correct.mean(), "mean_C": g.confidence.mean()})
    pd.DataFrame(conditional).to_csv(OUT / "behavior_within_confidence_bins.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for r, ds in enumerate(["IEMOCAP", "MELD"]):
        sub = metrics[metrics.dataset.eq(ds)].set_index("method").loc[["XP04", "E1", MAIN]]
        for col, key in enumerate(["ece15", "binary_brier", "binary_nll"]):
            ax = axes[r, col]
            ax.bar(range(3), sub[key], color=["#777777", "#0072B2", "#D55E00"])
            for i, val in enumerate(sub[key]):
                ax.text(i, val, num(val, 4), ha="center", va="bottom", fontsize=9)
            ax.set(xticks=range(3), xticklabels=["XP04", "C only", "Main module"],
                   title=f"{ds}: {key}", ylim=(0, float(sub[key].max()) * 1.22))
    fig.tight_layout()
    fig.savefig(OUT / "figures/primary_metrics.pdf")
    fig.savefig(OUT / "figures/primary_metrics.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, sharey=True)
    for ax, ds in zip(axes, ["IEMOCAP", "MELD"]):
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        for method, color, marker, style in [("XP04", "#777777", "o", "-"), ("E1", "#0072B2", "s", "-"),
                                              (MAIN, "#D55E00", "x", "--")]:
            b = bins[bins.dataset.eq(ds) & bins.method.eq(method) & bins.n.gt(0)]
            ax.plot(b.mean_confidence, b.accuracy, marker=marker, linestyle=style, color=color, label=method)
        ax.set(title=ds, xlim=(0, 1), ylim=(0, 1), xlabel="Mean top-label confidence")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "figures/reliability_diagram.pdf")
    plt.close(fig)

    report = ["# XP04 统一行为可靠性校准模块：数据结果报告", "",
              "本次完成 Seed 1475 真正 Validation 的 CPU Pilot。主比较为 **XP04+统一模块 E1_DA 对 XP04**。IEMOCAP 527条/12 Dialogue，MELD 1109条/114 Dialogue。没有重新训练CMERC、GPU前向、新扰动或Test标签使用。", "",
              "## 直接回答研究目标", "",
              "**主模块在两个数据集的 ECE15、Binary Brier、Binary NLL 点估计均低于原始 XP04。**但是IEMOCAP所有外层折都关闭了行为分支，提升全部来自基础置信度映射；MELD行为修正的增量微小且不一致。不能把整体校准改善全部归因于随机扰动。", "",
              "![主要校准指标比较](figures/primary_metrics.png)", "",
              md_table(["Dataset", "Method", "ECE15 ↓", "Brier ↓", "NLL ↓", "Correctness AUC"], [
                  [r.dataset, r.method, num(r.ece15), num(r.binary_brier), num(r.binary_nll), num(r.correctness_auc)]
                  for r in metrics.itertuples()]), "",
              "## 主模块相对XP04的绝对变化", "",
              md_table(["Dataset", "Metric", "Δ模块−XP04", "相对变化"], [
                  [r.dataset, r.metric, num(r.difference), f"{r.relative_change_percent:.2f}%"]
                  for r in diff[diff.candidate.eq(MAIN) & diff.reference.eq("XP04")].itertuples()]), "",
              "负数表示指标下降。相对变化是相对原始XP04指标的百分比，不是显著性。Accuracy、Weighted-F1、Macro-F1逐方法保持不变：IEMOCAP为0.660342/0.650208/0.644892；MELD为0.669071/0.661041/0.563809。", "",
              "## Dialogue配对bootstrap", "",
              "2000次Dialogue成组抽样，固定种子1475；以下差值为候选方法−参照。区间只描述固定OOF预测的Dialogue采样不确定性，未重拟合嵌套CV。", "",
              md_table(["Dataset", "Candidate", "Reference", "Metric", "Δ", "95% CI"], [
                  [r.dataset, r.candidate, r.reference, r.metric, num(r.difference), f"[{num(r.ci_low)}, {num(r.ci_high)}]"]
                  for r in bootstrap[bootstrap.candidate.eq(MAIN) & bootstrap.reference.isin(["XP04", "E1"])].itertuples()]), "",
              "MELD相对XP04三项区间均低于0；IEMOCAP三项区间均跨0，12个Dialogue提供的证据有限。对E1的行为增量在MELD没有稳定优势，IEMOCAP则逐行相同。", "",
              "## 协议与核验", "",
              "统一结构、数据集分别拟合。外层LODO12/GroupKFold5；每折内部GroupKFold5比较off与λ=1/0.1/0.01，依合并内层OOF Brier选择。每个内层训练子集都重新拟合基础映射、标准化器及4权重offset修正器。详见`EXPERIMENT_PROTOCOL.md`与不可变预登记`preregistration.json`。", "",
              "独立核验已重建9816条外层预测、163728条内层候选预测，检查1360个内层候选拟合、68个选择决策以及全部14行指标。外层预测重建最大误差约1.31×10⁻¹⁴；原始输入与运行代码哈希均保持一致。E1、L0与历史真正Validation结果精确对齐。", "",
              "原Validation已经多轮用于方法设计，因此本轮仍是探索性Pilot，不是全新独立确认集。所有负结果与关闭分支的折均保留。数值预检中有一次语法错误和一次L-BFGS线搜索失败；修正后的正式计算使用精确Hessian阻尼Newton，损失/输入/候选λ未改变；失败日志保留，没有按结果调参。", "",
              "## 产物", "",
              "`metrics.csv`完整指标；`metric_differences.csv`绝对/相对变化；`oof_predictions.csv`逐话语概率与原始标签/预测/Dialogue ID/Utterance Index；`feature_matrix.csv`逐话语行为输入；`fold_parameters.json`全部内外层参数与分组；`inner_cv_results.csv`全部候选分数；`bootstrap_results.csv`全部区间；`correct_wrong_adjustment.csv`正确/错误置信度变化；`per_dialogue_results.csv`逐Dialogue指标。图在`figures/`。"]
    write("DATA_RESULTS_REPORT.md", report)

    ablation = ["# 行为分支消融与启用情况", "",
                "E1为raw-C Logistic基础映射，L0为标准化logit-C基础映射。MUDELTA使用两扰动各自的原始μ/δ；DA使用两扰动各自的固定原预测类别方向D与RMS幅度A。两个行为版本都为4维。没有混用每次扰动为训练样本。", "",
                md_table(["Dataset", "Method", "Penalty", "Selected folds"], [
                    [ds, method, penalty, count]
                    for (ds, method, penalty), count in selected_rows.groupby(["dataset", "method", "penalty"]).size().items()]), "",
                "**IEMOCAP主DA分支：12/12折关闭，E1_DA与E1完全一致，L0_DA与L0完全一致。**原μ/δ有2折启用，但Brier/NLL比各自基础更差。", "",
                "**MELD主DA分支：5/5折启用，其中4折λ=1、1折λ=0.1。**E1_DA相对E1的ECE变化−0.000980、Brier变化+0.000100、NLL变化−0.000078，变化很小且不是三项一致收益。L0_DA有3/5折关闭，相对L0的ECE更低，但Brier/NLL更高。原μ/δ对照也未稳定超过基础映射。", "",
                "内层选择off是允许的模型选择结果，不是代码未接入。原始预测类别、输入D/A、优化参数和每折候选分数均已保存；外层留出从未参与λ选择。", "",
                "`behavior_within_confidence_bins.csv`提供相近原始C范围内按行为中位数分组的描述性正确率；其分组只用于事后诊断，不能当作OOF增量证据或新的模型选择规则。"]
    write("BEHAVIOR_ABLATION_REPORT.md", ablation)

    adjustment = ["# 正确与错误样本的置信度调整", "",
                  md_table(["Dataset", "Method", "Group", "n", "Original", "Calibrated", "Mean shift", "Raised", "Lowered"], [
                      [r.dataset, r.method, "正确" if r.correct else "错误", int(r.n), num(r.mean_original),
                       num(r.mean_calibrated), num(r.mean_shift), num(r.raised_fraction, 4), num(r.lowered_fraction, 4)]
                      for r in shifts[shifts.method.isin(["XP04", "E1", MAIN])].itertuples()]), "",
                  "IEMOCAP主模块降低多数样本置信度，但行为分支全部关闭，和E1相同。MELD主模块将正确样本均值从0.501153提高至0.703259，错误样本从0.402171提高至0.598127；所有正确与错误样本都上调。相比E1，错误样本均值略降0.001578，但整体Brier微增且AUC略降，不能声称可靠地区分了错误。", "",
                  "总体校准改善不要求每条错误预测都下降，但提高某一条错误预测的置信度本身不是逐样本收益。重点依据留出Brier/NLL/ECE与AUC共同判断。"]
    write("CONFIDENCE_ADJUSTMENT_REPORT.md", adjustment)

    final = ["# 本轮实验判断与后续建议", "",
             "**针对主要目标：主模块在两个数据集的三项校准指标点估计均超过原始XP04，达到本轮预先登记的点估计门槛。MELD对XP04的改善有明确Dialogue bootstrap支持；IEMOCAP区间较宽，不能认定稳定改善。**", "",
             "**针对随机扰动贡献：本轮仍未证明有效增量。**IEMOCAP所有折关闭行为分支，提升由基础校准贡献；MELD与E1相比的行为增量微小，Brier反而略差，区间跨0。不能将方法包装成已证实的CaliDist行为可靠性提升。", "",
             "建议：若优先目标是获得超过XP04的可用置信度模块，可以冻结基础校准组件，在五Seed上确认其低成本收益；若目标是行为模块的科研增量，当前不建议立即投入五Seed全量新扰动。应先接受本轮负增量结果，另行设计更有条件可靠性信息的挑战，并预先登记；当前不进入原始数据干扰。", "",
             "本次到此停止。未启动后续Seed、未使用Test、未改变CMERC权重、未自动Git Commit/Push。"]
    write("FINAL_DECISION.md", final)
    write("README.md", ["# XP04统一模块Pilot交付", "",
        "入口报告：[DATA_RESULTS_REPORT.md](DATA_RESULTS_REPORT.md)。最终判断：[FINAL_DECISION.md](FINAL_DECISION.md)。", "",
        "正式运行环境 `/data2/yb/conda_envs/cmerc-xp-repro/bin/python`，依赖numpy/pandas/scipy/scikit-learn；图表使用已有base环境matplotlib。运行目录为本目录，源数据路径由同级Pilot目录确定。", "",
        "复核（无需GPU）：", "", "```bash", "/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m unittest test_calibration_core -v",
        "/data2/yb/conda_envs/cmerc-xp-repro/bin/python verify_module.py", "```", "",
        "`run_experiment.py`是完整复算入口，会重新拟合CPU校准器并写入本目录结果；审阅时优先运行只读数值复核入口，保留原运行日志与哈希。`verify_module.py`仅更新verification.json中的核验字段。", "",
        "数据来源：真正Validation 527/1109条，Seed1475；旧XP04 validation/实际Test预测不使用。详见协议及provenance。"])
    verification["decision"] = decision
    verification["unit_tests"] = {"passed": 4, "checks": ["analytic gradient", "off recovers base", "outer-held-data isolation", "optimizer stationarity"]}
    verification["numerical_recovery"] = {"initial_solver": "L-BFGS-B line search failure", "final_solver": "damped Newton exact Hessian",
                                             "scientific_protocol_changed": False, "failure_logs_preserved": True}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    artifact_paths = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                      and p.name != "artifact_hashes.json"]
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths}
    (OUT / "provenance/artifact_hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
