#!/usr/bin/env python3
"""Check score replay and write the single-Seed scoring result and evidence."""
import argparse
import csv
import json
import math
from pathlib import Path
import time

from verify_score_inputs import digest, load, require


RUN = Path(__file__).resolve().parents[1]
PROV = RUN / "provenance"
TABLE = RUN / "results/score_inputs.csv"
SCORE = RUN / "results/scoring"
METRIC_NAMES = {
    "binary_nll": "Binary NLL",
    "brier": "Binary Brier",
    "auroc_error": "AUROC（错误为正类）",
    "auprc_error": "AUPRC（错误为正类）",
}


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def predict(model, row):
    x = [(float(row[key]) - center) / scale for key, center, scale in
         zip(model["features"], model["scaler_mean"], model["scaler_scale"])]
    z = model["intercept"][0] + sum(w * value for w, value in zip(model["coefficient"][0], x))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def replay(parameters, scored):
    clean = rows(TABLE)
    cohort = {row["sample_id"]: row for row in clean}
    fit = {row["sample_id"] for row in clean if row["role"] == "score_fit"}
    covered_fit = {sid for sid in fit if cohort[sid]["D_pred"]}
    expected_eval = set(cohort) - fit
    require(len(scored) == 1407 and expected_eval == {row["sample_id"] for row in scored},
            "Expected the complete held-out Dialogue cohort")
    for name, model in parameters.items():
        expected = fit if name.endswith("_all_fit") else covered_fit
        actual = model["fit_sample_ids"]
        require(len(actual) == len(expected) and set(actual) == expected, f"Scorer fit set differs: {name}")
        require(set(model["fit_dialogue_ids"]) == {cohort[sid]["dialogue_id"] for sid in expected},
                f"Scorer fit Dialogue set differs: {name}")
    maximum = 0.0
    fallback = 0
    for row in scored:
        sid = row["sample_id"]
        require(row["original_correct"] == cohort[sid]["original_correct"], "Correctness changed in scoring")
        for strength in ("weak", "strong"):
            baseline = f"{strength}_baseline"
            full = parameters[f"{baseline}_all_fit"]
            expected = {f"{baseline}_R": predict(full, row)}
            augmented = parameters.get(f"{strength}_augmented_covered_fit")
            if augmented is not None and row["D_pred"]:
                expected[f"{strength}_augmented_policy_R"] = predict(augmented, row)
                expected[f"{strength}_matched_fit_baseline_R"] = predict(parameters[f"{baseline}_covered_fit"], row)
                require(row[f"{strength}_fallback_to_baseline"] == "False", "Covered score got fallback")
            else:
                require(row[f"{strength}_fallback_to_baseline"] == "True", "Missing score failed to fall back")
                require(row[f"{strength}_augmented_policy_R"] == row[f"{baseline}_R"], "Fallback differs from clean")
                fallback += strength == "strong"
            maximum = max(maximum, *(abs(float(row[key]) - value) for key, value in expected.items()))
    require(maximum <= 1e-12, "Stored logistic parameters do not reproduce predictions")
    return dict(max_score_replay_absolute_error=maximum, complete_eval_count=len(scored), fallback_count=fallback)


def decimal(value):
    return f"{value:.6f}" if value is not None else "NA"


def metric_rows(result):
    if "baseline" not in result:
        return ["| 全部 | NA | NA | NA | NA |"]
    output = []
    for key, label in METRIC_NAMES.items():
        ci = result["paired_dialogue_improvement"][key]
        output.append("| " + " | ".join((
            label, decimal(result["baseline"][key]), decimal(result["augmented"][key]),
            decimal(ci["improvement"]),
            f"[{decimal(ci['improvement_ci_low'])}, {decimal(ci['improvement_ci_high'])}]"
        )) + " |")
    return output


def write_report(score, details, reviewed, replay_audit, terminal):
    comparison = score["comparisons"]
    primary = comparison["strong_matched_fit_covered_eval"]
    full = comparison["strong_all_eval"]
    fit = details["score_fit"]
    evaluation = details["score_eval"]
    delta = primary.get("paired_dialogue_improvement", {}).get("binary_nll", {})
    ci = [delta.get("improvement_ci_low"), delta.get("improvement_ci_high")]
    if all(value is not None for value in ci):
        if ci[0] > 0:
            decision = "当前受覆盖子集上，Binary NLL 的区间支持文本 D_pred 有增量价值。证据限于这次单 Seed 和受覆盖条件。"
        elif ci[1] < 0:
            decision = "当前受覆盖子集上，加入文本 D_pred 后 Binary NLL 变差，区间支持负向结果。"
        else:
            decision = "当前 Binary NLL 区间跨零。这次数据无法确定文本 D_pred 的增量价值。"
    else:
        decision = "覆盖与结果支持不足，无法估计主对照的 Binary NLL 增量区间。"
    lines = [
        "# MELD Seed 67137 文本行为置信度评分",
        "",
        decision,
        "",
        "## 分组与覆盖",
        "",
        "从 MELD 原 Train 的 1,038 个 Dialogue 固定四组。新 RoBERTa、M³Net 和 CMERC 只在基础训练的 "
        "6,313 条话语上更新参数，另用 700 条内部话语选择模型。两个评分组的 Dialogue 和整条监督链隔离。",
        "官方 Validation 与 Test 均未用于训练、选择或评分。",
        "",
        "| 评分组 | Dialogue | 干净话语 | 干净预测正确 | 尝试改写 | 有效 K=2 | 覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, count, support in (("拟合", score["fit_dialogues"], fit), ("留出评估", score["eval_dialogues"], evaluation)):
        covered = support.get("covered", 0)
        lines.append(f"| {label} | {count} | {support['clean']} | {support['clean_correct']} | "
                     f"{support['attempted']} | {covered} | {covered/support['clean']:.2%} |")
    lines.extend([
        "",
        "在每个评分组预抽 96 个 Dialogue，各选一条目标话语，生成两版语义改写。选择时已锁定名单，"
        "生成者与盲审者均为 AI，审核隐藏模型输出和真实标签。FAIL 和 UNCERTAIN 原样留档。"
        "没有同义改写覆盖的话语继续进入完整留出评估，逐条回退到干净评分。",
        "",
        "## 正确性概率",
        "",
        "逻辑回归输出原预测正确的概率 R。干净基线用最大 Softmax 的 logit、干净 top-2 间隔与熵。"
        "新增项是干净原预测类别概率减去两版改写中该类别的平均概率。主对照在相同的受覆盖话语上拟合和评估两组模型。",
        "区间来自同一组 2,000 次留出 Dialogue 配对 Bootstrap，条件于本次固定评分模型和干扰版本。改善为正。",
        "",
        f"### 主对照（留出 {primary['n']} 条受覆盖话语）",
        "",
        "| 指标 | 干净基线 | 加 D_pred | 改善 | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
        *metric_rows(primary),
        "",
        f"### 完整留出队列（{full['n']} 条，缺失时回退）",
        "",
        "| 指标 | 干净基线 | 加 D_pred 的回退策略 | 改善 | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
        *metric_rows(full),
        "",
        "完整队列对照的干净基线使用全部评分拟合话语，增广模型仅使用有两版有效改写的话语。"
        "其中的变化同时受拟合样本覆盖与行为特征影响。",
        "",
        f"已用保存的模型参数重算评分，最大绝对误差 {replay_audit['max_score_replay_absolute_error']:.3g}。"
        f"完整留出有 {replay_audit['fallback_count']} 条使用干净回退。",
        "原有音频和视觉参与模型的干净输入，干预只替换目标话语的文本特征。"
        "本轮未做多分类概率重分配，未测多分类 ECE、Brier 或 NLL 的校准收益。",
        "",
    ])
    with (RUN / "REAL_SCORING_REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    verification = dict(status="COMPLETE_VERIFIED", scoring=terminal["status"], dataset="MELD", seed=67137,
                        ai_review_only=True, k=2, official_validation_used=False, test_used=False,
                        evaluation_scope=details, review_counts=reviewed["role_variant_verdicts"],
                        primary_binary_nll_improvement=delta, replay=replay_audit,
                        artifacts={str(p.relative_to(RUN)): digest(p) for p in (
                            RUN / "REAL_SCORING_REPORT.md", TABLE, SCORE / "evaluation_predictions.csv",
                            SCORE / "scoring_report.json", SCORE / "fitted_parameters.json",
                            PROV / "ai_review_lock.json", PROV / "scoring_execution.json", Path(__file__))})
    with (RUN / "verification.json").open("x") as handle:
        json.dump(verification, handle, indent=2)
    print(json.dumps(dict(status=verification["status"], primary=delta, replay=replay_audit)))


def main(wait):
    status_path = PROV / "scoring_execution.json"
    terminal = load(status_path)
    while wait and terminal["status"] in {"WAITING", "RUNNING"}:
        time.sleep(20)
        terminal = load(status_path)
    require(terminal["status"] == "COMPLETE", "The independent scoring chain is not complete")
    score = load(SCORE / "scoring_report.json")
    require(score["status"] == "COMPLETE_CPU_SCORING"
            and score["scalar_validation"]["max_softmax_definition_numerically_verified_for_all_rows"],
            "Incomplete frozen scorer validation")
    for name, expected in score["input_sha256"].items():
        require(digest(Path(name)) == expected, f"Scorer input changed: {name}")
    evidence = load(PROV / "scoring_input_declaration.json")["independent_probability_audit"]["roles"]
    reviewed = load(PROV / "ai_review_lock.json")
    audit = replay(load(SCORE / "fitted_parameters.json"), rows(SCORE / "evaluation_predictions.csv"))
    write_report(score, evidence, reviewed, audit, terminal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-scoring", action="store_true")
    main(parser.parse_args().wait_for_scoring)
