"""Independently cross-check frozen inputs, raw stdout, checkpoints and exports."""
import hashlib
import json
import math
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(2**20),b""):
            h.update(data)
    return h.hexdigest()


def main():
    summary = json.loads((ROOT/"results/metrics_summary.json").read_text())
    status = json.loads((ROOT/"state/status.json").read_text())
    assert summary["formal_runs_completed"]==4 and summary["formal_runs_started"]==4
    assert not status["active"] and status["phase"]=="completed"
    inputs = json.loads((ROOT/"provenance/inputs.json").read_text())
    for filename,meta in inputs["data"].items():
        assert digest(Path(meta["path"]))==meta["sha256"],filename
    checks = {}
    for row in summary["runs"]:
        ident = row["run_id"]
        cfg = json.loads((ROOT/"configs"/(ident+".json")).read_text())
        run = ROOT/"checkpoints"/ident
        epochs = [json.loads(line) for line in (run/"epochs.jsonl").read_text().splitlines()]
        tests = [r for r in epochs if r["split"]=="test"]
        text = (ROOT/"logs"/(ident+".log")).read_text()
        printed = re.findall(r"epoch: (\d+), train_loss: ([\d.]+), train_acc: ([\d.]+), train_fscore: ([\d.]+), test_loss: ([\d.]+), test_acc: ([\d.]+), test_fscore: ([\d.]+), time: ([\d.]+) sec",text)
        assert len(printed)==cfg["epochs"],(ident,len(printed))
        for line,record in zip(printed,tests):
            assert int(line[0])==record["epoch"]
            assert float(line[4])==record["official_loss"]
            assert float(line[5])==record["official_accuracy_percent"]
            assert float(line[6])==record["official_wf1_percent"]
        batches = [json.loads(line) for line in (run/"batches.jsonl").read_text().splitlines()]
        train = [b for b in batches if b["train"]]
        for b in train:
            total = b["classification_loss"]
            if cfg["variant"]=="cmerc":
                args = json.loads((run/"effective_args.json").read_text())
                total += args["rank_coff"]*b["rank_loss"]+args["mscl_coff"]*b["Mscl_tav"]+args["cscl_coff"]*b["Cscl_c"]
            assert math.isclose(total,b["total_loss"],rel_tol=1e-5,abs_tol=1e-6),(ident,b["epoch"],b["batch"])
        best = torch.load(run/"best.pt",map_location="cpu")
        assert best["epoch"]==row["best_epoch"]
        latest = torch.load(run/"latest.pt",map_location="cpu")
        final = torch.load(run/"final.pt",map_location="cpu")
        assert latest["epoch"]==final["epoch"]==cfg["epochs"]
        assert latest["model"].keys()==final["model"].keys()
        assert all(torch.equal(latest["model"][k],final["model"][k]) for k in latest["model"])
        metrics = json.loads((ROOT/"results"/ident/"metrics.json").read_text())
        assert metrics["checkpoint_sha256"]==digest(run/"best.pt")
        assert metrics["replay_labels_identical"] and metrics["replay_predictions_identical"] and metrics["replay_probabilities_close"]
        checks[ident] = {"raw_stdout_matches_epoch_metrics":True,"weighted_loss_identity_every_training_batch":True,
            "best_epoch_matches":True,"latest_equals_final_model":True,"exact_replay_labels_predictions":True,
            "checkpoint_hashes":{name:digest(run/name) for name in ("best.pt","latest.pt","final.pt")},
            "training_batch_count":len(train),
            "mscl_zero_batches":sum(b["Mscl_tav"]==0 for b in train),"cscl_zero_batches":sum(b["Cscl_c"]==0 for b in train)}
        del best,latest,final
    for filename in ("01_OFFICIAL_SOURCE_AUDIT.md","02_README_PROTOCOL_COMPARISON.md","03_ENVIRONMENT_PREFLIGHT.md",
                     "04_FOUR_RUN_TRAINING_REPORT.md","05_XP04_VS_README_COMPARISON.md","06_F1_GAP_DIAGNOSIS_AND_NEXT_STEPS.md"):
        assert (ROOT/"reports"/filename).is_file(),filename
    result = {"status":"passed","all_four_formal_runs":True,"data_hashes_unchanged":True,"runs":checks,
        "limits":["does not make test-selected results unbiased","single seed per dataset","diagnostic gradients are a single batch"]}
    (ROOT/"provenance/final_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"passed","formal_runs":len(checks),"data_hashes_unchanged":True}))


if __name__ == "__main__":
    main()
