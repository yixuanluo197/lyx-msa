"""Create result narratives from frozen E3 CSVs; no fitting or data mutation."""

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
MAIN = ["E0", "E1", "L0", "E3-A", "E3-B", "E3-C", "E3-D"]
E3 = ["E3-A", "E3-B", "E3-C", "E3-D"]


def number(x, digits=5):
    return f"{x:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def write(name, lines):
    (OUT / name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    m = pd.read_csv(OUT / "e3_metrics.csv")
    ab = pd.read_csv(OUT / "e3_feature_ablation.csv")
    boot = pd.read_csv(OUT / "bootstrap_results.csv")
    shift = pd.read_csv(OUT / "correct_wrong_adjustment.csv")
    dialogue = pd.read_csv(OUT / "per_dialogue_results.csv")
    assert set(m.dataset) == {"IEMOCAP", "MELD"}
    assert all(set(m.query("dataset == @dataset").method) == set(MAIN + list(ab.query("dataset == @dataset").method))
               for dataset in ["IEMOCAP", "MELD"])

    results = ["# E3 Dialogue-level OOF 结果", "",
               "Seed 1475 真正 Validation。IEMOCAP 527 条/12 Dialogue，MELD 1109 条/114 Dialogue。每条话语一条训练记录；行为特征来自原 Pilot 已完成的扰动汇总。所有数字均为 Top-label 二元正确性评价，较低 ECE15/Brier/NLL 更好。E3 是 CaliDist-inspired 双向扩展，并非原版 CaliDist。", "",
               "## 主模型与对照", "",
               table(["Dataset", "Method", "ECE15", "Brier", "NLL", "Correctness AUC", "Mean C"], [
                   [r.dataset, r.method, number(r.ece15), number(r.binary_brier), number(r.binary_nll),
                    number(r.correctness_roc_auc), number(r.mean_confidence)]
                   for r in m[m.method.isin(MAIN)].itertuples()]), "",
               "IEMOCAP 的 Accuracy/Weighted-F1/Macro-F1 均为 0.66034/0.65021/0.64489；MELD 均为 0.66907/0.66104/0.56381。所有方法不改变预测类别，因此这些值逐方法相同。", "",
               "## E3 相对 E1 的 Dialogue 配对 bootstrap", "",
               "Δ=E3−E1，负数代表改善。2,000 次按 Dialogue 有放回抽样；95% percentile 区间。IEMOCAP 只有 12 个 Dialogue，区间应谨慎解释。", "",
               table(["Dataset", "Method", "Metric", "Δ", "95% CI"], [
                   [r.dataset, r.method, r.metric, number(r.estimate),
                    f"[{number(r.ci_low)}, {number(r.ci_high)}]"]
                   for r in boot[(boot.reference == "E1") & boot.method.isin(E3)].itertuples()]), "",
               "模型参数、训练折与留出 Dialogue ID、每折标准化器参数在 `fold_parameters.json`；逐话语结果在 `e3_oof_predictions.csv`。Bootstrap 的完整结果在 `bootstrap_results.csv`。"]
    write("E3_OOF_RESULTS.md", results)

    compare = ["# E1 与 E3 的严格对照", "",
               "E1 使用 Pilot 原始 C 的 Logistic Regression OOF；本次按完全相同的 Dialogue 折复算后，最大逐行差异 IEMOCAP 5.6×10⁻¹⁷、MELD 1.1×10⁻¹⁶。L0 使用 `logit(C)` 单特征与 E3 相同的标准化及 L2 Logistic Regression，是判断行为特征增量的必要链接函数对照。", "",
               table(["Dataset", "Method", "ΔECE vs E1", "ΔBrier vs E1", "ΔNLL vs E1", "ΔBrier vs L0", "ΔNLL vs L0"], [
                   [ds, method, *[number(float(row[k] - e1[k])) for k in ["ece15", "binary_brier", "binary_nll"]],
                    number(float(row.binary_brier - l0.binary_brier)), number(float(row.binary_nll - l0.binary_nll))]
                   for ds in ["IEMOCAP", "MELD"]
                   for e1, l0 in [(m.query("dataset == @ds and method == 'E1'").iloc[0],
                                  m.query("dataset == @ds and method == 'L0'").iloc[0])]
                   for method in ["L0", *E3]
                   for row in [m.query("dataset == @ds and method == @method").iloc[0]]]), "",
               "**IEMOCAP：**E3-A/B 的 ECE 低于 E1，但 Brier 和 NLL 均高于 E1；E3-C/D 扩充输入后 Brier/NLL 继续恶化。L0 也显示 ECE 与 Brier/NLL 的相反方向，说明不能只凭 ECE 判定收益。", "",
               "**MELD：**E3-A/B 相对 E1 的三项指标均有点估计改善，但 L0 单特征也同样改善 E1，且 L0 的 Brier/NLL 比 E3-A/B 更低。E3-A 相对 L0 仅 ECE 更低 0.00277，Brier 高 0.00078、NLL 高 0.00174；相对 L0 的 ECE 配对区间跨 0。不能将对 E1 的收益直接归因于 μ、δ 或 JS。", "",
               "E3-C/D 在 MELD 的 Brier/NLL 也比 E1 差。上述比较是单 Seed OOF，不是独立 Test 或五 Seed 结论。"]
    write("E1_VS_E3_COMPARISON.md", compare)

    ablation = ["# 行为特征消融", "",
                "全部特征组合与 L2 强度在 `E3_METHOD.md` 中先固定。下表的差值相对 L0=`logit(C)`；负数越好。`C_text/audio/visual` 单独加入该模态的 μ、δ、JS；`C_no_*` 从 E3-C 删除该模态特征，但仍保留 joint 特征，因此不构成完全排除该模态的因果实验。", "",
                table(["Dataset", "Method", "ECE15", "Brier", "NLL", "ΔBrier vs L0", "ΔNLL vs L0"], [
                    [r.dataset, r.method, number(r.ece15), number(r.binary_brier), number(r.binary_nll),
                     number(r.binary_brier - float(m.query("dataset == @r.dataset and method == 'L0'").iloc[0].binary_brier)),
                     number(r.binary_nll - float(m.query("dataset == @r.dataset and method == 'L0'").iloc[0].binary_nll))]
                    for r in ab.itertuples()]), "",
                "**μ 与 δ：**MELD 中 E3-A（μ+δ）的 Brier/NLL 都高于 L0；单独加 μ 或 δ 也没有稳定的三指标联合改善。IEMOCAP 同样没有可靠增量。", "",
                "**JS：**在 E3-A 上加入 JS（E3-B）并未一致改善三项指标。MELD 的探索性 `B_js`（只加 joint JS）点估计略优于 L0：ECE −0.01022、Brier −0.00019、NLL −0.00052，但其相对 L0 的 Dialogue bootstrap 区间三项均跨 0；这是多配置消融中的弱信号。", "",
                "**模态：**完整 E3-C 在两个数据集的 Brier/NLL 均较 E3-B 变差。MELD 的 `C_visual` 单模态点估计 Brier/NLL 略优于 L0，但 ECE 较差且 bootstrap 区间跨 0；MELD 有 95 条缺失视觉话语，其视觉行为零响应也限制解释。IEMOCAP 的视觉单模态也有些微 Brier/NLL 点估计改善，但同样不足以支持模态互补结论。", "",
                "**预测类别：**E3-D 在两数据集均明显恶化 Brier/NLL，当前样本量和模型设置下不建议加入预测类别 one-hot。", "",
                "相对 L0 的 bootstrap 是看过主结果后补充的诊断；覆盖全部固定特征组合，但未做多重比较校正，不能把最好的单项消融当作确认性结论。"]
    write("BEHAVIOR_FEATURE_ABLATION.md", ablation)

    adjustment = ["# 置信度调整与错误样本", "",
                  "对原始 CMERC C 计算上调/下调；正确与错误均使用原始预测是否正确来分组。真实标签只用于事后评价，不用于留出 Dialogue 的推理。", "",
                  table(["Dataset", "Method", "Group", "n", "E0 mean", "OOF mean", "Mean shift", "Raised", "Lowered"], [
                      [r.dataset, r.method, "correct" if r.correct else "wrong", int(r.n), number(r.mean_e0),
                       number(r.mean_calibrated), number(r.mean_shift), number(r.fraction_raised, 3),
                       number(r.fraction_lowered, 3)]
                      for r in shift[shift.method.isin(["E1", "L0", *E3])].itertuples()]), "",
                  "MELD E1 提高 100% 错误样本的置信度。E3-A 仍提高 98.9% 错误样本，E3-C 仍提高 94.6%；没有实现可靠的逐错误样本抑制。MELD E3-A 正确/错误均值约 0.716/0.577，与 L0 的 0.715/0.575 接近，行为特征未提供清晰的额外区分。", "",
                  "IEMOCAP E3-A/C 主要降低置信度，但 E3-C 的正确样本平均下降幅度约 0.104、错误样本仅约 0.017，Brier/NLL 因而变差。E3-D 的正确与错误样本均值几乎重合，正确性 AUC 约 0.516。", "",
                  "## Dialogue 层面 Brier/NLL 改善数量", "",
                  table(["Dataset", "Reference", "Method", "Brier wins", "NLL wins", "Dialogues"], [
                      [ds, reference, method,
                       int((x.binary_brier < y.binary_brier).sum()),
                       int((x.binary_nll < y.binary_nll).sum()), len(x)]
                      for ds in ["IEMOCAP", "MELD"]
                      for reference in ["E1", "L0"]
                      for method in E3
                      for x, y in [
                          (dialogue.query("dataset == @ds and method == @method").set_index("dialogue_id"),
                           dialogue.query("dataset == @ds and method == @reference").set_index("dialogue_id"))]]), "",
                  "逐话语增减在 `confidence_shift.csv`，逐 Dialogue 的完整指标在 `per_dialogue_results.csv`。"]
    write("CONFIDENCE_ADJUSTMENT_ANALYSIS.md", adjustment)

    final = ["# E3 Pilot 最终判断", "",
             "**结论：部分有效，但核心行为增量假设尚未得到支持。** MELD 的 E3-A/B 在这个 Seed 的 OOF 点估计上同时优于原始 E1；IEMOCAP 没有三指标一致收益。MELD 的优势不能直接归功于随机扰动行为，因为只用 `logit(C)` 的 L0 也优于 E1，而且 E3-A/B 的 Brier/NLL 反而不如 L0。", "",
             "1. IEMOCAP 原始 C 整体偏高；E3-A/B 将均值降至约 0.66，但 ECE 的下降伴随 Brier/NLL 恶化。E3-C/D 更差。", "2. MELD 原始 C 整体偏低；E3-A/B 将均值升至约 0.67，但大部分错误预测也被提高，且行为特征未稳定超越 L0。", "3. μ 与 δ 联合没有比 `logit(C)` 单特征带来一致增量；JS 在完整 E3-B 中无一致收益。单独 JS、视觉消融的微小 MELD 点估计信号均带宽区间且属于多配置探索。", "4. 完整模态输入 E3-C 没有展示互补收益；预测类别 one-hot 的 E3-D 明显恶化。", "5. MELD E3-A/B 相对 E1 的三项差值的 Dialogue bootstrap 95% 区间均跨 0，IEMOCAP 只有 12 个 Dialogue，不能把单 Seed OOF 点估计提升当成稳定结论。", "",
             "## 后续建议", "",
             "若研究者希望按“E3-A/B 在 MELD OOF 点估计优于 E1”的预定规则继续，建议**只针对 MELD**进行五 Seed 确认，并将 E1、L0、E3-A、E3-B 同场固定比较；五 Seed 不能把共享 Dialogue 当作独立数据集。IEMOCAP 的 E3-C/D 与类别 one-hot 暂不推进。只有行为组跨 Seed 在 ECE、Brier、NLL 上稳定超过 **L0 与 E1**，才可称为随机扰动行为的增量价值。当前不建议进入原始文本、音频、视频干扰阶段。", "",
             "本轮到此停止，等待研究者审核。未重新训练 CMERC，未运行 GPU 推理，未生成新扰动，未使用 Test，未训练五 Seed，未提交或推送 Git。"]
    write("FINAL_E3_DECISION.md", final)


if __name__ == "__main__":
    main()
