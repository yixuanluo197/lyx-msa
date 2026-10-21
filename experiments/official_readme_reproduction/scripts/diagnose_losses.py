"""One training batch at a fixed CMERC checkpoint; gradients only, no update."""
import argparse
import hashlib
import json
import math
import os
import runpy
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticDone(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    opts = parser.parse_args()
    cfg = json.loads((ROOT/"configs"/(opts.run_id+".json")).read_text())
    assert cfg["variant"]=="cmerc"
    out = ROOT/"results"/opts.run_id/"gradient_diagnostic.json"
    if out.exists():
        raise RuntimeError("Refusing to overwrite diagnostic")
    code = ROOT/"runtime_code"
    os.chdir(code)
    sys.path.insert(0,str(code))
    argv = list(cfg["argv"])
    argv[argv.index("--epochs")+1] = "0"
    sys.argv = [str(code/"train_cm.py")]+argv
    g = runpy.run_path(str(code/"train_cm.py"),run_name="__main__")
    ckpt = ROOT/"checkpoints"/opts.run_id/"best.pt"
    saved = torch.load(ckpt,map_location="cpu")
    g["model"].load_state_dict(saved["model"],strict=True)
    params = [p for p in g["model"].parameters() if p.requires_grad]
    before = [p.detach().cpu().clone() for p in params]
    ratio = min(1,round((.15 if cfg["dataset"]=="IEMOCAP" else .4)*saved["epoch"],2))
    loader_fn = g["get_IEMOCAP_loaders"] if cfg["dataset"]=="IEMOCAP" else g["get_MELD_loaders"]
    loader,_,_ = loader_fn(valid=0.0,batch_size=16,num_workers=2,epoch_ratio=ratio)
    script = code/"train_cm.py"
    lines = script.read_text().splitlines()
    graph_start = next(i+1 for i,l in enumerate(lines) if l.startswith("def train_or_eval_graph_model("))
    backward_line = next(i+1 for i,l in enumerate(lines) if i+1>graph_start and l.strip()=="loss.backward()")
    result = {"run_id":opts.run_id,"checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "selected_epoch":saved["epoch"],"batch_size":16,"curriculum_fraction":ratio,
        "scope":"diagnostic at fixed checkpoint, first current training batch; no optimization or config/epoch selection",
        "auxiliary_snapshot":"copied from fixed main checkpoint at diagnostic function entry; not historical epoch-start auxiliaries",
        "training_mode":True,"test_data_used":False,"optimizer_steps":0}
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=str(script) or frame.f_code.co_name!="train_or_eval_graph_model":
            return None
        if event=="line" and frame.f_lineno==backward_line:
            loc = frame.f_locals
            losses = {"classification":loc["loss_function"](loc["log_prob"],loc["label"]),
                      "cc":loc["rank_loss"],"mscl":loc["Mscl_tav"],"cscl":loc["Cscl_c"]}
            coefs = {"classification":1,"cc":g["args"].rank_coff,"mscl":g["args"].mscl_coff,"cscl":g["args"].cscl_coff}
            result.update(ids=loc["data"][-1],lengths=loc["lengths"],components={},pseudo_labels={})
            classification_grads = None
            for name,loss in losses.items():
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    result["components"][name] = {"value":float(loss),"requires_grad":False,"gradient_l2":0.0}
                    continue
                grads = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                finite = all(bool(torch.isfinite(v).all()) for v in grads if v is not None)
                norm = math.sqrt(sum(float(v.detach().double().square().sum()) for v in grads if v is not None))
                row = {"value":float(loss.detach()),"requires_grad":True,"coefficient":coefs[name],
                    "gradient_l2":norm,"weighted_gradient_l2":norm*coefs[name],"all_gradients_finite":finite,
                    "parameter_tensors_with_gradient":sum(v is not None for v in grads)}
                if classification_grads is None:
                    classification_grads = [v.detach().cpu() if v is not None else None for v in grads]
                    class_norm = norm
                else:
                    dot = sum(float((v.detach().cpu().double()*ref.double()).sum()) for v,ref in zip(grads,classification_grads) if v is not None and ref is not None)
                    row["cosine_with_classification"] = dot/(norm*class_norm) if norm*class_norm else None
                result["components"][name] = row
                del grads
            for key in ("Mscl_pred_t","Mscl_pred_a","Mscl_pred_v","Cscl_pred_intra","Cscl_pred_inter"):
                values = loc[key].detach()
                counts = torch.bincount(values,minlength=2).cpu().tolist()
                result["pseudo_labels"][key] = {"counts_0_1":counts,"positive_fraction":counts[1]/sum(counts),
                    "same_label_pair_count":sum(n*(n-1) for n in counts),
                    "interpretation":"constructed target distribution, not independently measured label quality"}
            result["main_train_accuracy_percent"] = float((loc["pred_"]==loc["labels_"]).float().mean()*100)
            raise DiagnosticDone()
        return trace
    try:
        sys.settrace(trace)
        g["train_or_eval_graph_model"](g["model"],g["model_mm"],g["model_intra"],g["model_inter"],g["loss_function"],
            loader,saved["epoch"]-1,True,g["args"].modals,g["optimizer"],True,dataset=cfg["dataset"])
        raise RuntimeError("Diagnostic did not intercept before backward/update")
    except DiagnosticDone:
        pass
    finally:
        sys.settrace(None)
    result["model_parameters_unchanged"] = all(torch.equal(old,p.detach().cpu()) for old,p in zip(before,params))
    assert result["model_parameters_unchanged"]
    result["limitations"] = ["single training batch; not a longitudinal gradient study",
        "BatchNorm running buffers may update in train mode in this throwaway process; saved checkpoint is unchanged",
        "pseudo-label balance is not an oracle estimate of counterfactual calibration target quality"]
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"run_id":opts.run_id,"components":result["components"],"model_parameters_unchanged":result["model_parameters_unchanged"]}))


if __name__ == "__main__":
    main()
