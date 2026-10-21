"""Evaluate a fixed official-test-selected checkpoint; no epoch/config selection."""
import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    opts = parser.parse_args()
    run = ROOT/"checkpoints"/opts.run_id
    config = json.loads((ROOT/"configs"/(opts.run_id+".json")).read_text())
    status = json.loads((run/"status.json").read_text())
    if status["status"] != "completed" or status["completed_test_epochs"] != config["epochs"]:
        raise RuntimeError("Refusing evaluation of incomplete training")
    out = ROOT/"results"/opts.run_id
    out.mkdir(exist_ok=False)
    code = ROOT/"runtime_code"
    os.chdir(code)
    sys.path.insert(0,str(code))
    argv = list(config["argv"])
    argv[argv.index("--epochs")+1] = "0"
    sys.argv = [str(code/"train_cm.py")]+argv
    g = runpy.run_path(str(code/"train_cm.py"),run_name="__main__")
    payload = torch.load(run/"best.pt",map_location="cpu")
    g["model"].load_state_dict(payload["model"],strict=True)
    # Enable removals only for post-training measurement, even for the baseline.
    # No optimizer step and no additional model/config choice is performed.
    g["args"].calibrate = True
    g["args"].contrastlearning = False
    captured = {name:[] for name in ("full","modal","intra_context","inter_context")}
    def hook(name):
        def record(module, inputs, output):
            captured[name].append(output[0].detach().cpu().numpy())
        return record
    handles = [g[model].register_forward_hook(hook(name)) for model,name in (
        ("model","full"),("model_mm","modal"),("model_intra","intra_context"),("model_inter","inter_context"))]
    loader_fn = g["get_IEMOCAP_loaders"] if config["dataset"]=="IEMOCAP" else g["get_MELD_loaders"]
    _,_,loader = loader_fn(valid=0.0,batch_size=16,num_workers=2,epoch_ratio=-1)
    started = time.time()
    with torch.no_grad():
        result = g["train_or_eval_graph_model"](g["model"],g["model_mm"],g["model_intra"],g["model_inter"],
            g["loss_function"],loader,payload["epoch"]-1,True,g["args"].modals,train=False,dataset=config["dataset"])
    for handle in handles:
        handle.remove()
    labels = np.asarray(result[2]); predictions = np.asarray(result[3])
    full = np.concatenate(captured["full"]).astype(np.float64)
    # model_mm is invoked in the frozen order text/audio/visual for every batch.
    removed = {name:np.concatenate(captured["modal"][offset::3]).astype(np.float64)
               for offset,name in enumerate(("text","audio","visual"))}
    removed.update({name:np.concatenate(captured[name]).astype(np.float64)
                    for name in ("intra_context","inter_context")})
    metric_file = ROOT.parents[1]/"experiment_records/04_xp_mainline_validation/code/evaluation_metrics.py"
    spec = importlib.util.spec_from_file_location("xp04_frozen_evaluation_metrics",metric_file)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    row = {"run_id":opts.run_id,"dataset":config["dataset"],"model":config["variant"],"seed":config["seed"],
           "selection_protocol":"official_test_selection","selected_epoch_1based":payload["epoch"],
           "independent_checkpoint_replay":True,"independent_of_test_selection":False,
           "accuracy_percent":float(accuracy_score(labels,predictions)*100),
           "weighted_f1_percent":float(f1_score(labels,predictions,average="weighted")*100),
           "macro_f1_percent":float(f1_score(labels,predictions,average="macro")*100),
           "official_wf1_percent":result[4], "official_classification_loss":result[0],
           "num_test_utterances":len(labels),"calibration":metrics.calibration_metrics(labels,np.exp(full)),
           "cel":metrics.confidence_enhancement_level(labels,full,removed),
           "checkpoint_sha256":hashlib.sha256((run/"best.pt").read_bytes()).hexdigest(),
           "evaluation_metric_source":str(metric_file),"evaluation_metric_source_sha256":hashlib.sha256(metric_file.read_bytes()).hexdigest(),
           "evaluation_protocol":"same frozen model and official test batches; no_grad; diagnostic removals enabled only after training",
           "elapsed_seconds":time.time()-started,
           "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
           "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
    observed = json.loads((run/"best_test.json").read_text())
    row["replay_matches_training_test_wf1"] = abs(row["weighted_f1_percent"]-observed["wf1_percent"])<1e-8
    if not row["replay_matches_training_test_wf1"]:
        raise RuntimeError(("Replay prediction mismatch",row["weighted_f1_percent"],observed["wf1_percent"]))
    observed_arrays = np.load(run/"best_predictions.npz")
    row["replay_labels_identical"] = bool(np.array_equal(labels,observed_arrays["labels"]))
    row["replay_predictions_identical"] = bool(np.array_equal(predictions,observed_arrays["probs"].argmax(1)))
    row["replay_probabilities_close"] = bool(np.allclose(np.exp(full),observed_arrays["probs"],rtol=1e-5,atol=1e-7))
    row["replay_max_probability_difference"] = float(np.abs(np.exp(full)-observed_arrays["probs"]).max())
    if not all(row[k] for k in ("replay_labels_identical","replay_predictions_identical","replay_probabilities_close")):
        raise RuntimeError("Checkpoint replay differs from saved training-time predictions")
    np.savez_compressed(out/"predictions.npz",labels=labels,full_log_probs=full,**removed)
    (out/"metrics.json").write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:row[k] for k in ("run_id","selected_epoch_1based","weighted_f1_percent","accuracy_percent","macro_f1_percent","replay_matches_training_test_wf1")}))


if __name__ == "__main__":
    main()
