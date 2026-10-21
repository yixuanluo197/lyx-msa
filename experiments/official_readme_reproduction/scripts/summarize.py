"""Validate all registered runs and export traceable tables/reports."""
import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def csv_file(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, digits=4):
    return "待完成" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete",action="store_true")
    args = parser.parse_args()
    state = read(ROOT/"state/status.json")
    xp = read(ROOT/"provenance/xp04_verified_metrics.json")["runs"]
    inputs = read(ROOT/"provenance/inputs.json")
    official = read(ROOT/"provenance/official_source.json")
    for name, expected in official["source_sha256"].items():
        assert hashlib.sha256((ROOT/"official_code"/name).read_bytes()).hexdigest()==expected,name
    for name, expected in inputs["source_hashes"].items():
        if name.startswith(("official_code/","runtime_code/","configs/")) or name in ("scripts/observe_run.py","scripts/supervise.py"):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    rows, comparison, curves, audits, loss_curves, xp_curves = [], [], [], {}, [], []
    for file in sorted((ROOT/"configs").glob("*.json")):
        cfg = read(file)
        ident = cfg["run_id"]
        run = ROOT/"checkpoints"/ident
        record = state["runs"][ident]
        metric_path = ROOT/"results"/ident/"metrics.json"
        metric = read(metric_path) if metric_path.exists() else None
        old = next(x for x in xp if x["dataset"]==cfg["dataset"] and x["model"]==cfg["variant"])
        paper = {("IEMOCAP","baseline"):69.61,("IEMOCAP","cmerc"):71.98,
                 ("MELD","baseline"):65.37,("MELD","cmerc"):66.85}[(cfg["dataset"],cfg["variant"])]
        epochs = jsonl(run/"epochs.jsonl")
        batches = jsonl(run/"batches.jsonl")
        tests = [r for r in epochs if r["split"]=="test"]
        row = {"run_id":ident,"dataset":cfg["dataset"],"model":cfg["variant"],"seed":cfg["seed"],
               "status":record["status"],"selection_protocol":"official_test_selection",
               "host":state["host"],"gpu_uuid":state["gpu_uuid"],"pid":record.get("pid"),
               "run_class":"compatibility_patched_with_external_observer",
               "epochs_planned":cfg["epochs"],"epochs_completed":len(tests),
               "train_dialogues_full":inputs["datasets"][cfg["dataset"]]["train_dialogues"],
               "weighted_f1_percent":metric["weighted_f1_percent"] if metric else None,
               "official_wf1_rounded_percent":metric["official_wf1_percent"] if metric else None,
               "accuracy_percent":metric["accuracy_percent"] if metric else None,
               "macro_f1_percent":metric["macro_f1_percent"] if metric else None,
               "ece_15_percent":metric["calibration"]["ece"] if metric else None,
               "cel_sum":metric["cel"]["total"] if metric else None,
               "best_epoch":metric["selected_epoch_1based"] if metric else None,
               "final_epoch_wf1_percent":tests[-1]["wf1_percent"] if record["status"]=="completed" else None,
               "metrics_source":str(metric_path.relative_to(ROOT)) if metric else None}
        audit = {"status":record["status"],"completed_epochs":len(tests),"checks":{}}
        if record["status"]=="completed":
            assert len(tests)==cfg["epochs"]
            assert [r["epoch"] for r in tests]==list(range(1,cfg["epochs"]+1))
            first_best = max(tests,key=lambda r:r["official_wf1_percent"])
            saved_best = read(run/"best_test.json")
            assert first_best==saved_best
            assert all((run/f).exists() for f in ("best.pt","latest.pt","final.pt"))
            actual_args = read(run/"effective_args.json")
            assert actual_args["epochs"]==cfg["epochs"] and actual_args["seed"]==cfg["seed"]
            assert actual_args["calibrate"]==(cfg["variant"]=="cmerc")
            assert actual_args["contrastlearning"]==(cfg["variant"]=="cmerc")
            assert actual_args["courselearning"]==(cfg["variant"]=="cmerc")
            assert all(b["losses_finite"] for b in batches)
            assert all(b["gradients_finite"] for b in batches if b["train"])
            data = inputs["datasets"][cfg["dataset"]]
            import pickle
            difficulty = pickle.load((ROOT/"official_code/caldiff"/(cfg["dataset"]+"_diff.pkl")).open("rb"))
            ranked = sorted(difficulty,key=difficulty.get)
            coverage = []
            for epoch in range(1,cfg["epochs"]+1):
                pool = set(data["train_ids"])
                ratio = min(1,round((.15 if cfg["dataset"]=="IEMOCAP" else .4)*epoch,2)) if cfg["variant"]=="cmerc" else 1
                if cfg["variant"]=="cmerc":
                    pool = set(ranked[:int(ratio*len(ranked))])
                train_batches = [b for b in batches if b["train"] and b["epoch"]==epoch]
                ids = [key for b in train_batches for key in b["ids"]]
                assert len(ids)==len(set(ids))==len(pool) and set(ids)==pool,(ident,epoch)
                test_ids = [key for b in batches if b["split"]=="test" and b["epoch"]==epoch for key in b["ids"]]
                assert test_ids==data["test_ids"],(ident,epoch,"test order")
                coverage.append({"epoch":epoch,"fraction":ratio,"dialogues":len(ids)})
            audit["checks"]={"complete_epochs":True,"first_best_rounded_test_selection":True,
                "best_and_final_present":True,"finite_losses_and_gradients":True,
                "exact_train_curriculum_coverage_every_epoch":True,"fixed_complete_test_order_every_epoch":True}
            audit["curriculum_coverage"] = coverage
            selected_train = [b for b in batches if b["train"] and b["epoch"]==saved_best["epoch"]]
            row.update({"selected_train_"+key:statistics.mean(b[key] for b in selected_train)
                        for key in ("classification_loss","rank_loss","Mscl_tav","Cscl_c","total_loss")})
            runtime_status = read(run/"status.json")
            row.update(peak_allocated_mib=runtime_status["peak_allocated_mib"],
                peak_reserved_mib=runtime_status["peak_reserved_mib"],
                elapsed_seconds=runtime_status["finished_at_unix"]-runtime_status["started_at_unix"])
            if metric:
                assert metric["replay_matches_training_test_wf1"]
                assert metric["selected_epoch_1based"]==first_best["epoch"]
                assert metric["num_test_utterances"]==data["test_utterances"]
                audit["checks"]["independent_checkpoint_replay_matches"] = True
            (ROOT/"configs"/(ident+".effective.json.snapshot")).write_text(json.dumps(actual_args,indent=2)+"\n")
        if not args.allow_incomplete:
            assert record["status"]=="completed" and metric is not None,(ident,record["status"])
        rows.append(row)
        comparison.append({"dataset":cfg["dataset"],"model":cfg["variant"],"seed":cfg["seed"],
            "readme_selection":"official_test_selection","xp04_selection":"validation",
            "readme_wf1_percent":row["weighted_f1_percent"],"xp04_wf1_percent":old["weighted_f1_percent"],
            "paper_wf1_percent":paper,
            "readme_minus_xp04_pp":row["weighted_f1_percent"]-old["weighted_f1_percent"] if metric else None,
            "readme_minus_paper_pp":row["weighted_f1_percent"]-paper if metric else None,
            "readme_accuracy_percent":row["accuracy_percent"],"xp04_accuracy_percent":old["accuracy_percent"],
            "readme_macro_f1_percent":row["macro_f1_percent"],"xp04_macro_f1_percent":old["macro_f1_percent"],
            "readme_cel_sum":row["cel_sum"],"xp04_cel_sum":old["cel"]["total"],
            "readme_ece_15_percent":row["ece_15_percent"],"xp04_ece_15_percent":old["calibration"]["ece"],
            "readme_best_epoch":row["best_epoch"],"xp04_best_epoch":old["selected_epoch_1based"]})
        for e in epochs:
            curves.append(dict(run_id=ident,**e))
            selected = [b for b in batches if b["epoch"]==e["epoch"] and b["split"]==e["split"]]
            loss_row = {"run_id":ident,"epoch":e["epoch"],"split":e["split"],
                        "batch_count":len(selected),"dialogues":sum(len(b["ids"]) for b in selected),
                        "utterances":sum(sum(b["lengths"]) for b in selected)}
            for field in ("classification_loss","rank_loss","Mscl_tav","Cscl_c","total_loss"):
                vals = [b[field] for b in selected if field in b]
                loss_row[field+"_batch_mean"] = statistics.mean(vals) if vals else None
            loss_curves.append(loss_row)
        old_curve = ROOT.parents[1]/"experiment_records/04_xp_mainline_validation/runs"/cfg["variant"]/f"{cfg['dataset']}_validation_seed{cfg['seed']}"/"metrics.jsonl"
        for e in jsonl(old_curve):
            xp_curves.append(dict(run_id=ident,source=str(old_curve),**e))
        audits[ident] = audit
    csv_file(ROOT/"results/per_run.csv",rows)
    csv_file(ROOT/"results/comparison.csv",comparison)
    csv_file(ROOT/"results/epoch_curves.csv",curves)
    csv_file(ROOT/"results/epoch_loss_curves.csv",loss_curves)
    csv_file(ROOT/"results/xp04_epoch_curves.csv",xp_curves)
    resources = jsonl(ROOT/"logs/resources.jsonl")
    summary = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":state["phase"],
        "official_commit":official["commit"],"source_manifest":"provenance/official_source.json",
        "input_manifest":"provenance/inputs.json","environment_manifest":"provenance/environment.json",
        "compatibility_diff":"provenance/compatibility.patch","xp04_reference":"provenance/xp04_verified_metrics.json",
        "formal_runs_started":state["formal_runs_started"],"formal_runs_completed":state["formal_runs_completed"],
        "single_seed_per_dataset":True,"statistical_significance_test":"not applicable: one preregistered seed per dataset",
        "run_class":"compatibility_patched_with_external_observer","selection_protocol":"official_test_selection",
        "independent_of_test_selection":False,"runs":rows,"comparison":comparison,"validation":audits,
        "resource_snapshots":len(resources),"snapshots_with_external_gpu_process":sum(bool(r["unknown"]) for r in resources),
        "limitations":["different train pools and model selection from XP04; descriptive differences only",
            "author difficulty generation history unknown","torch-sparse differs from README",
            "GPU concurrency affects runtime; GPU resources sampled at 10s intervals",
            "cuDNN internal dropout state not covered by general RNG snapshots"]}
    (ROOT/"results/metrics_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    report = ["# 四组正式训练报告", "",f"更新：{summary['created_at_utc']}。状态：`{state['phase']}`；启动{state['formal_runs_started']}/4，完整训练完成{state['formal_runs_completed']}/4。", "",
        "4个固定 run 均按 README 参数从随机初始化训练；只使用 IEMOCAP seed1475、MELD seed67137。预检与正式训练分开。源码两处兼容修复和外部观测见报告03，不属于完全未修改源码运行。", "",
        "| 数据集 | 模型 | Seed | 状态 | 完成/计划 Epoch | 最佳 Epoch | W-F1 % |", "|---|---|---:|---|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['dataset']} | {r['model']} | {r['seed']} | {r['status']} | {r['epochs_completed']}/{r['epochs_planned']} | {r['best_epoch'] or '待完成'} | {number(r['weighted_f1_percent'])} |")
    report += ["", "## 可追溯证据", "", "完整启动 argv、覆盖环境变量、PID、GPU UUID、启动前显存与退出码见 `state/status.json`；监督器启动见 `provenance/launch.json`。全部 epoch 标准输出见 `logs/<run_id>.log`。实际解析参数在 `checkpoints/<run_id>/effective_args.json`，训练完成后另存 `configs/*.effective.json.snapshot`。", "",
        "各 run 的 `checkpoints/<run_id>/` 包含 `best.pt`、`latest.pt`、`final.pt`、`best_test.json`、逐轮 `epochs.jsonl` 和逐 Batch `batches.jsonl`。checkpoint保存model/Adam/进度及Python、NumPy、Torch/CUDA RNG；cuDNN内部dropout缓存不保证可序列化，未发生续训。", "",
        "`official_test_selection`：按官方四舍五入到两位的测试W-F1、严格大于时记录首次最佳轮。观察器将该轮模型固定保存，避免原脚本best_label/pred被最低test_loss覆盖所造成的打印混淆；原始日志仍原样保留。", "",
        "`results/<run_id>/metrics.json` 是固定最佳checkpoint的独立进程重放，显式不再选epoch。它不独立于此前的测试选模，不能称为无测试泄漏评估。CEL为五种移除的正真类log概率差总和，ECE为15等宽bin百分比；Baseline也只在训练结束后启用移除前向做诊断。", "",
        "## 验收与资源", "",f"机器验收见 `results/metrics_summary.json`：完整轮数、首次最高rounded test-F1、best/final产物、所有Batch有限loss/gradient、每轮完整训练课程ID覆盖、完整固定测试顺序和checkpoint重放一致性。当前资源快照{len(resources)}条，其中发现外部GPU进程的快照{summary['snapshots_with_external_gpu_process']}条；离散观测不能证明所有时刻独占。", "",
        "显存峰值和实际耗时逐run在 `results/per_run.csv`。同卡并发影响速度，耗时不是严谨性能benchmark。`results/epoch_curves.csv` 包含训练/空验证/测试完整曲线，`epoch_loss_curves.csv` 包含各Loss逐轮batch均值及实际对话/句数，`xp04_epoch_curves.csv` 提供对应XP04训练/验证曲线与来源；分项原始值在batches.jsonl，未把分类loss误称总loss。", "",
        "单面板诊断曲线在 `results/figures/`，每run各有W-F1和Loss的PNG/SVG。W-F1图明确区分README test与XP04 validation，两条曲线不是同一评估协议；Loss图中的正则项乘以原固定README系数，便于与总Loss对照。绘图仅使用既有CPU环境，版本及CSV来源哈希见 `results/figures/manifest.json`，不修改训练环境。", "",
        "未因分数改变seed、超参数、训练轮数或反复重训。`logs/preflight_*` 保留2次原始兼容失败及4项成功单Batch诊断，均不计正式训练。最终以状态文件实际记录为准。"]
    (ROOT/"reports/04_FOUR_RUN_TRAINING_REPORT.md").write_text("\n".join(report)+"\n")
    report = ["# XP04 与官方 README 实训对照", "",f"更新：{summary['created_at_utc']}。分类指标和 ECE 使用百分比，差值为百分点；CEL 为总和，Epoch 为轮数。新实训未完成的单元格不填推测值。", "",
        "| 数据集 | 模型 | 官方 README 实训 W-F1 | XP04 同 Seed W-F1 | 论文报告 W-F1 | 新实训−XP04 |", "|---|---|---:|---:|---:|---:|"]
    for r in comparison:
        report.append(f"| {r['dataset']} | {r['model']} | {number(r['readme_wf1_percent'])} | {r['xp04_wf1_percent']:.4f} | {r['paper_wf1_percent']:.2f} | {number(r['readme_minus_xp04_pp'])} |")
    report += ["", "| 数据集 | README CMERC−Baseline | XP04同Seed CMERC−Baseline | 论文报告差值 |", "|---|---:|---:|---:|"]
    for dataset in ("IEMOCAP","MELD"):
        base = next(r for r in comparison if r["dataset"]==dataset and r["model"]=="baseline")
        full = next(r for r in comparison if r["dataset"]==dataset and r["model"]=="cmerc")
        gain = full["readme_wf1_percent"]-base["readme_wf1_percent"] if full["readme_wf1_percent"] is not None and base["readme_wf1_percent"] is not None else None
        report.append(f"| {dataset} | {number(gain)} | {full['xp04_wf1_percent']-base['xp04_wf1_percent']:.4f} | {full['paper_wf1_percent']-base['paper_wf1_percent']:.2f} |")
    report += ["", "官方README实训为 `official_test_selection`，XP04为 `validation`。两者训练池、batch与RNG等也不同，表中的差值只能作描述性对照，不能当作同协议下的泛化提升或特定设置的因果效应。", "",
        "| 数据集/模型 | Acc 新/XP04 | Macro F1 新/XP04 | ECE % 新/XP04 | CEL总和 新/XP04 | 最佳Epoch 新/XP04 |", "|---|---:|---:|---:|---:|---:|"]
    for r in comparison:
        report.append(f"| {r['dataset']}/{r['model']} | {number(r['readme_accuracy_percent'])}/{r['xp04_accuracy_percent']:.4f} | {number(r['readme_macro_f1_percent'])}/{r['xp04_macro_f1_percent']:.4f} | {number(r['readme_ece_15_percent'])}/{r['xp04_ece_15_percent']:.4f} | {number(r['readme_cel_sum'])}/{r['xp04_cel_sum']:.4f} | {r['readme_best_epoch'] or '待完成'}/{r['xp04_best_epoch']} |")
    report += ["", "XP04四项均从真实归档独立评估文件、训练final_test与manifest重新核验；完整精度、loss、epoch和来源SHA见 `provenance/xp04_verified_metrics.json`。本批精度来自 `results/<run_id>/metrics.json`；官方选轮使用的两位精度另列 `results/per_run.csv`。", "",
        "官方IEMOCAP训练120对话，XP04为108训练+12验证；官方MELD1152训练，XP04为1038训练+114验证。测试对话相同，但不能消除训练与选模差异。官方IEMOCAP CL前7轮对话数18/36/54/72/90/108/120，XP04为16/32/48/64/81/97/108；官方MELD前3轮460/921/1152，XP04为415/830/1038。实际逐轮覆盖在metrics_summary中核验。", "",
        "官方训练分类loss为IEMOCAP加权NLL或MELD原作者FocalLoss；CEL是另一个校准指标，不是分类cross entropy，也不能与平均loss混比。分项loss、最佳轮和曲线见报告04及CSV。", "",
        "每个数据集只有一个预先指定Seed。单Seed不能代表XP04五Seed均值，不估计跨Seed标准差、置信区间或显著性；论文报告值只是参考，不保证与当前单Seed具有同等统计口径。"]
    report += ["", "## 训练曲线", "", "下图文件为CPU生成的单面板诊断图，原始CSV与来源哈希同时保留。W-F1图的README测试曲线参与选轮，XP04验证曲线参与选轮；二者用途及训练数据不同。Loss图中的CC/MSCL/CSCL为乘固定系数后的batch均值。", "", "| 数据集/模型 | W-F1曲线 | Loss曲线 |", "|---|---|---|"]
    for r in rows:
        ident = r["run_id"]
        report.append(f"| {r['dataset']}/{r['model']} | [PNG](../results/figures/{ident}_wf1.png) / [SVG](../results/figures/{ident}_wf1.svg) | [PNG](../results/figures/{ident}_loss.png) / [SVG](../results/figures/{ident}_loss.svg) |")
    (ROOT/"reports/05_XP04_VS_README_COMPARISON.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"status":summary["status"],"completed":summary["formal_runs_completed"],"replayed":sum(r["weighted_f1_percent"] is not None for r in rows),"rows":len(rows)}))


if __name__ == "__main__":
    main()
