#!/usr/bin/env python3
"""Generate research reports from the frozen Validation OOF result tables."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent; R=ROOT/"results"
def read(name):
    with (R/name).open(newline="") as f: return list(csv.DictReader(f))
def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)
def write(name,text): (ROOT/name).write_text(text.strip()+"\n")

metrics=read("oof_metrics.csv"); temps=read("classwise_temperatures.csv"); outer=read("outer_fold_temperatures.csv")
support=read("predicted_class_support.csv"); boot=read("paired_bootstrap.csv"); classm=read("classwise_oof_metrics.csv")
seeds=[1475,67137,2024,3407,42]
def mrow(ds,seed,method): return next(r for r in metrics if r["dataset"]==ds and int(r["seed"])==seed and r["method"]==method)
def mean_metric(ds,method,key): return np.mean([float(r[key]) for r in metrics if r["dataset"]==ds and r["method"]==method])
def seed_table(ds):
    rows=[]
    for seed in seeds:
      e0,e1,e2,e3=[mrow(ds,seed,x) for x in ("E0","E1","E2","E3")]
      rows.append([seed,*[f"{float(x[k]):.6f}" for x in (e0,e1,e2,e3) for k in ("nll","brier","ece15")]])
    return table(["Seed"]+[f"{m} {k}" for m in ("E0","E1","E2","E3") for k in ("NLL","Brier","ECE15")],rows)
def delta_summary(ds,a,b):
    rows=[]
    for key in ("nll","brier","ece15"):
      values=np.array([float(mrow(ds,s,a)[key])-float(mrow(ds,s,b)[key]) for s in seeds])
      rows.append([key.upper(),f"{values.mean():+.6f}",f"{values.std(ddof=1):.6f}",f"{(values<0).sum()}/5"])
    return table([f"{a}-{b}","Mean delta","Seed SD","Improved seeds"],rows)
def class_delta(ds):
    groups=defaultdict(list)
    for name in sorted({r["class_name"] for r in classm if r["dataset"]==ds}):
      for seed in seeds:
        r1=next(r for r in classm if r["dataset"]==ds and int(r["seed"])==seed and r["grouping"]=="predicted_class" and r["class_name"]==name and r["method"]=="E1")
        r3=next(r for r in classm if r["dataset"]==ds and int(r["seed"])==seed and r["grouping"]=="predicted_class" and r["class_name"]==name and r["method"]=="E3")
        groups[name].append(float(r3["nll"])-float(r1["nll"]))
    return table(["Predicted class","Mean E3-E1 NLL","Improved seeds","Range"],[[name,f"{np.mean(v):+.6f}",f"{sum(x<0 for x in v)}/5",f"[{min(v):+.6f}, {max(v):+.6f}]"] for name,v in groups.items()])
def temp_summary(ds):
    rows=[]
    names=[]
    for r in temps:
      if r["dataset"]==ds and r["class_name"] not in names: names.append(r["class_name"])
    for name in names:
      rr=[r for r in temps if r["dataset"]==ds and r["method"]=="E3" and r["class_name"]==name]
      values=np.array([float(r["class_temperature"]) for r in rr]); offsets=np.array([float(r["absolute_log_offset"]) for r in rr]); sup=np.array([int(r["outer_train_support"]) for r in rr])
      seed_means=[np.mean([float(r["class_temperature"]) for r in rr if int(r["seed"])==s]) for s in seeds]
      rows.append([name,f"{sup.mean():.1f}",f"{values.mean():.3f}",f"{np.std(seed_means,ddof=1):.3f}",f"{offsets.mean():.4f}"])
    return table(["Predicted class","Mean train support","Mean E3 T","SD of seed means","Mean |log offset|"],rows)

mean_rows=[]
for ds in ("IEMOCAP","MELD"):
 for method in ("E0","E1","E2","E3"):
  mean_rows.append([ds,method,*[f"{mean_metric(ds,method,k):.6f}" for k in ("nll","brier","ece15")]])
mean_table=table(["Dataset","Method","NLL","Brier","ECE15"],mean_rows)

write("README.md",f"""
# CMERC Hierarchical Predicted-Class Temperature Scaling Pilot

本目录记录严格限制在真正 Validation Split 上的校准方法探索。十个 XP04 checkpoint 均复用已保存完整概率；未进行 GPU 推理、CMERC 重训练、checkpoint 修改或 Test 访问。

四种方法均以外层 Leave-One-Dialogue-Out OOF 概率评价：E0 原始 CMERC，E1 Global TS，E2 不正则化预测类别温度，E3 层次化预测类别温度。E3 的 λ 仅由外层训练对话内的 5 折 Grouped CV 选择；推理时仅由**原始预测类别**选择一个正标量温度。

## 五 Seed OOF 均值

{mean_table}

主要结论：IEMOCAP 上 E3 相对 E1 的 NLL 与 Brier 在 5/5 Seed 均退化；MELD 的差异接近零且方向不一致。E3 明显压缩了 E2 的类别温度偏移并改善其过拟合，但没有建立超越 Global TS 的净收益。因此当前结果不支持进入历史 Test 或把类别条件化校准作为正式方法。

## 复现

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_hierarchical_pilot.py --output .
python scripts/plot_results.py
python scripts/verify_results.py --output .
```

完整结果见 [FINAL_RESEARCH_DECISION.md](FINAL_RESEARCH_DECISION.md)，预注册见 [PREREGISTRATION.md](PREREGISTRATION.md)。
""")

write("PREDICTED_CLASS_DISTRIBUTION.md",f"""
# Predicted-Class Distribution Audit

校准器按原始预测类别选择温度，因此下表以预测类别为主要分组；真实类别分组仅保存在 `results/predicted_class_support.csv` 作辅助诊断。

## IEMOCAP

{temp_summary('IEMOCAP')}

五 Seed 的完整验证集预测类支持范围为：angry 116–144、excited 56–83、frustrated 58–94、happy 76–98、neutral 66–81、sad 84–87。IEMOCAP 没有验证集预测类别缺失，但每个外层训练折仅有 11 个对话，类别组成容易随留出对话改变。

## MELD

{temp_summary('MELD')}

完整验证集支持范围为：neutral 491–556；surprise 151–191；joy 146–175；anger 121–172；sadness 46–79；fear 18–33；disgust 15–22。fear 与 disgust 是明确的小样本类别，不能依赖不受约束类别温度。
""")

write("NESTED_CV_RESULTS.md",f"""
# Nested Dialogue-Level Cross-Validation Results

每个 Dataset×Seed 独立执行外层 LODO；每条话语恰好产生一次 OOF 预测。IEMOCAP 共 60 个外层折，MELD 共 570 个。每个外层折以内层 5 折 Grouped Dialogue CV、Inner OOF NLL 从固定候选中选择 λ。

## IEMOCAP per Seed

{seed_table('IEMOCAP')}

### E3 相对 E1

{delta_summary('IEMOCAP','E3','E1')}

## MELD per Seed

{seed_table('MELD')}

### E3 相对 E1

{delta_summary('MELD','E3','E1')}

所有 Accuracy、Weighted-F1、Macro-F1 在 E0–E3 间完全一致，因为每条话语只使用一个正温度。Bootstrap 以对话为单位重采样固定 OOF 预测；它没有覆盖每次重采样重新拟合校准器的额外不确定性。
""")

write("HIERARCHICAL_CALIBRATION_REPORT.md",f"""
# Hierarchical Calibration Report

## 正则化是否抑制过拟合

IEMOCAP 中 E2 的平均 `|log(T_k)-log(T_global)|` 为 0.3548，E3 降至 0.0806；MELD 从 0.1328 降至 0.0272。E3 相对 E2 的平均 NLL：IEMOCAP 改善 0.021516（3/5 Seed），MELD 改善 0.001955（5/5 Seed）。因此层次正则化确实限制了自由类别温度，但“比 E2 更稳”没有转化为“比 E1 更好”。

IEMOCAP 的 E3 相对 E1 平均 NLL +0.017228、Brier +0.004770，二者 0/5 Seed 改善。MELD 对应差值分别为 -0.000006 和 -0.000025，仅 3/5 Seed 改善，量级接近数值与抽样波动；ECE15 平均反而 +0.000521。

## λ 行为

IEMOCAP 的 60 个外层折中 λ=100 被选择 31 次；MELD 的 570 折中 λ=100 被选择 189 次、λ=1 被选择 257 次。大量外层折选择强收缩，说明内层 OOF 常偏好接近 Global T 的解。没有类别温度触及 [0.05,20] 边界。

## 解释

结果显示类别方向冲突存在，但当前验证数据不足以稳定估计可迁移的类别修正。IEMOCAP 的 12 个对话使嵌套拟合方差更高；MELD 的 Global TS 已捕获主要欠置信方向，类别温度的剩余收益极小。
""")

write("CLASSWISE_TEMPERATURE_ANALYSIS.md",f"""
# Classwise Temperature Analysis

温度由原始预测类别决定；下列 NLL 差值均为 E3−E1，负值代表层次类别温度优于 Global TS。

## IEMOCAP

{class_delta('IEMOCAP')}

angry 在 5/5 Seed 有小幅 NLL 改善，neutral 与 frustrated 分别为 4/5；但 sad 在 0/5 改善且平均明显退化，happy、excited 也不稳定。整体样本加权结果仍使 E3 在 5/5 Seed 输给 E1，不能据此部署 angry 专用温度。

## MELD

{class_delta('MELD')}

没有 MELD 类别达到跨五 Seed 一致改善。fear 与 disgust 支持量最少，E3 已强制向 Global T 收缩；fear、sadness 的类别 NLL 在 0/5 Seed 改善。MELD 的类别温度五 Seed均值波动小于 IEMOCAP，但这主要反映稳定的全局欠置信方向，并未形成额外类别收益。

## 稳定性

{temp_summary('IEMOCAP')}

{temp_summary('MELD')}

IEMOCAP 类别温度的 Seed 间波动仍大；MELD 较稳定但各类温度接近同一个全局温度。支持量与偏移图见 `figures/support_vs_temperature_offset.*`。
""")

write("INDEPENDENT_SPLIT_FEASIBILITY.md","""
# Independent Calibration/Evaluation Split Feasibility

## 审计结果

- IEMOCAP 的 Dialogue ID 包含 Session 信息，可按 Session 分组；MELD 的冻结产物含 Dialogue ID，但当前特征/配置审计未建立可靠 Episode 映射。
- XP04 checkpoint 已见过原始 Training Split。把其中样本划作 calibration 或 evaluation 不能称为模型未见数据。
- 当前 Validation 已用于 checkpoint 选择，并已用于多轮校准方法探索，因此适合作探索性 OOF 分析，不足以承担最终确认。
- 历史 Test 未参与训练，但已被多轮实验分析，不再是对当前新假设完全新鲜的确认集。
- 现有数据划分中没有识别出第四个、模型未见且未用于方法开发的样本池。

## 可行路径

真正独立的训练、校准、评价三分组需要：（1）按 Session/Dialogue 重新设计分组并重新训练 CMERC；或（2）引入外部模型未见数据。IEMOCAP 可优先按 Session 设计；MELD 至少按 Dialogue，若能从原始元数据可靠恢复 Episode，应按 Episode 避免跨对话内容泄漏。本阶段未重新划分、未重训练。
""")

write("FINAL_RESEARCH_DECISION.md",f"""
# Final Research Decision

## 决策

**暂停 Hierarchical Predicted-Class TS，不进入历史 Test 或正式五 Seed独立评价。保留 Global TS 作为标准基线。**

{mean_table}

1. **类别条件化是否优于 Global TS？** 否。IEMOCAP E3 的 NLL/Brier 在 5/5 Seed 均比 E1 差；MELD 的平均变化近零且方向不一致。
2. **层次正则化是否抑制过拟合？** 是。它将类别 log-temperature 平均偏移缩小约 77%（IEMOCAP）和 80%（MELD），并普遍优于 E2；但仍未超越 E1。
3. **IEMOCAP 是否有值得继续验证的类别收益？** angry 有 5/5 Seed 的小幅类内 NLL改善，neutral/frustrated 有弱信号；sad 明显退化。鉴于整体退化和仅 12 个对话，这只是后续独立数据的假设，不足以立项正式 Test。
4. **MELD 的 Global TS 是否足够？** 对当前 Validation OOF 是。E1 已大幅改善 E0，E3 的增量约为零。
5. **哪些类别应独立、哪些应共享？** 当前没有类别具备足够证据采用独立温度。fear/disgust 等低支持类必须共享或强收缩；其余类别也应暂用 Global T。
6. **温度是否稳定？** MELD 类别温度较稳定但高度接近 Global T；IEMOCAP 跨 Seed 和对话折波动明显。
7. **是否存在超越 ECE 分箱波动的真实收益？** 未建立。IEMOCAP 的 NLL/Brier 明确反对 E3；MELD 的 NLL/Brier变化极小，ECE也不一致。
8. **当前 Validation 是否足够支持方法选择？** 足够否定当前 E3 作为优于 E1 的候选；不足以精确估计 IEMOCAP 类别温度或作确认性推广。
9. **能否构建真正独立划分？** 现有 checkpoint 下不能。需要分组重划并重训练，或引入外部未见数据。
10. **下一步是否进行正式五 Seed独立评价？** 当前不值得。下一项最有价值的单个实验是预先固定 Global TS，并在重新训练后形成的独立 grouped calibration/evaluation split 上确认；若资源有限，先停止新增校准复杂度。

Bootstrap 区间来自固定 OOF 概率的对话级重采样，不包含校准器重新拟合的不确定性。IEMOCAP 只有 12 个对话，区间尤其应保守解释。
""")

write("PREREGISTRATION_RESULTS_NOTE.md",f"""
# Preregistration Execution Note

预注册条件未改变：固定 λ 候选、5 折内层 Grouped Dialogue CV、外层 LODO、原始预测类别选择器及 [0.05,20] 温度边界均按计划执行。共生成 8,180 条 OOF 话语记录；未读取 Test 预测或标签。
""")
print("reports generated")
