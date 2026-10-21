"""Observe the frozen README entry point without changing its training operations.

Trace hooks read locals only. Checkpoint writes occur after an epoch evaluation.
Preflight stops after one successful optimizer step and is never a formal run.
"""
import argparse
import json
import os
import random
import runpy
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


class PreflightDone(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true")
    opts = parser.parse_args()
    config = json.loads(Path(opts.config).read_text())
    output = Path(opts.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = Path(opts.code).resolve()
    script = code / "train_cm.py"
    lines = script.read_text().splitlines()
    # Resolve observation sites from the frozen source, not hardcoded line numbers.
    def site(text):
        matches = [i + 1 for i, line in enumerate(lines) if line.strip() == text]
        if len(matches) != 1:
            raise RuntimeError((text, matches))
        return matches[0]

    config_site = site("today = datetime.datetime.now()")
    loss_site = site("preds.append(torch.argmax(log_prob, 1).cpu().numpy())")
    backward_site = site("loss.backward()") if lines.count("            loss.backward()") == 1 else None
    # Both graph and non-graph functions contain backward; choose graph range.
    graph_start = next(i+1 for i,l in enumerate(lines) if l.startswith("def train_or_eval_graph_model("))
    backward_site = next(i+1 for i,l in enumerate(lines) if i+1 > graph_start and l.strip()=="loss.backward()")
    step_site = next(i+1 for i,l in enumerate(lines) if i+1 > graph_start and l.strip()=="optimizer.step()")
    state = {"status": "running", "formal": not opts.preflight,
             "config": config, "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "pid": os.getpid(), "started_at_unix": time.time(), "code": str(code),
             "selection_protocol": "official_test_selection", "observer": str(Path(__file__).resolve())}
    batches, epochs = [], []
    current = {}
    last_main = {}
    best = None
    def write_json(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    def checkpoint(name, g, epoch):
        torch.save({"model": g["model"].state_dict(),
                    "optimizer": g["optimizer"].state_dict(), "epoch": epoch,
                    "args": vars(g["args"]), "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "selection_protocol": "official_test_selection"}, output/name)

    def trace(frame, event, arg):
        nonlocal best, current, last_main
        if frame.f_code.co_filename != str(script):
            return None
        loc = frame.f_locals
        if frame.f_code.co_name == "<module>":
            last_main = frame.f_globals
            if event == "line" and frame.f_lineno == config_site:
                write_json("effective_args.json", vars(loc["args"]))
            return trace
        if frame.f_code.co_name != "train_or_eval_graph_model":
            return None
        if event == "line" and opts.preflight and state.pop("waiting_for_step_return", False):
            write_json("single_batch.json", current["batches"][-1])
            state["optimizer_step_completed"] = True
            raise PreflightDone()
        if event == "call":
            current = {"epoch": loc["epoch"]+1, "split": "train" if loc["train"] else "eval", "batches": []}
        if event == "line" and frame.f_lineno == loss_site:
            row = {"epoch": loc["epoch"]+1, "train": loc["train"],
                   "batch": len(current["batches"]), "classification_loss": float(loc["loss"].detach()),
                   "ids": loc["data"][-1], "lengths": loc["lengths"]}
            for name in ("rank_loss", "Mscl_tav", "Cscl_c"):
                v = loc.get(name, 0)
                row[name] = float(v.detach()) if torch.is_tensor(v) else float(v)
            row["losses_finite"] = all(np.isfinite(row[k]) for k in ("classification_loss", "rank_loss", "Mscl_tav", "Cscl_c"))
            if not row["losses_finite"]:
                raise RuntimeError("Non-finite loss in frozen official computation")
            current["batches"].append(row)
            # Only detached CPU predictions retained; no extra model forward.
            current.setdefault("probs", []).append(loc["log_prob"].detach().exp().cpu().numpy())
        if event == "line" and frame.f_lineno == backward_site:
            current["batches"][-1]["total_loss"] = float(loc["loss"].detach())
        if event == "line" and frame.f_lineno == step_site:
            grads = [p.grad for p in loc["model"].parameters() if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            current["batches"][-1]["gradients_finite"] = finite
            if not finite:
                raise RuntimeError("Non-finite gradient")
            state["waiting_for_step_return"] = opts.preflight
        if event == "return" and isinstance(arg, tuple):
            split = "train" if loc["train"] else ("test" if len(loc["dataloader"].dataset)>0 and len(loc["dataloader"])>0 else "validation_empty")
            row = {"epoch": loc["epoch"]+1, "split": split, "official_loss": arg[0] if np.isfinite(arg[0]) else None,
                   "official_accuracy_percent": arg[1] if np.isfinite(arg[1]) else None,
                   "official_wf1_percent": arg[4] if np.isfinite(arg[4]) else None}
            if len(arg[2]):
                probs = np.concatenate(current["probs"])
                labels = np.asarray(arg[2]); preds = np.asarray(arg[3])
                conf = probs.max(1); correct = preds == labels
                ece = 0.0
                for lower in np.linspace(0, 1, 16)[:-1]:
                    mask = (conf > lower) & (conf <= lower+1/15)
                    if mask.any():
                        ece += mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
                row.update(wf1_percent=float(f1_score(labels,preds,average="weighted")*100),
                           accuracy_percent=float(accuracy_score(labels,preds)*100),
                           macro_f1_percent=float(f1_score(labels,preds,average="macro")*100),
                           cross_entropy_nll=float(-np.log(np.clip(probs[np.arange(len(labels)),labels],1e-30,1)).mean()),
                           cel=sum(b["rank_loss"] for b in current["batches"]) if frame.f_globals["args"].calibrate or frame.f_globals["args"].contrastlearning else None,
                           ece_15_percent=float(ece*100))
                if split == "test":
                    if best is None or row["official_wf1_percent"] > best["official_wf1_percent"]:
                        best = dict(row)
                        checkpoint("best.pt", frame.f_globals, row["epoch"])
                        write_json("best_test.json", row)
                        np.savez_compressed(output/"best_predictions.npz", labels=labels, probs=probs)
                    checkpoint("latest.pt", frame.f_globals, row["epoch"])
            epochs.append(row)
            with (output/"epochs.jsonl").open("a") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
            with (output/"batches.jsonl").open("a") as f:
                for batch in current["batches"]:
                    f.write(json.dumps(dict(batch,split=split),allow_nan=False)+"\n")
        return trace

    os.chdir(code)
    sys.path.insert(0, str(code))
    sys.argv = [str(script)]+config["argv"]
    write_json("status.json", state)
    try:
        sys.settrace(trace)
        runpy.run_path(str(script), run_name="__main__")
        state["status"] = "completed"
        checkpoint("final.pt", last_main, config["epochs"])
    except PreflightDone:
        state["status"] = "preflight_passed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        traceback.print_exc()
    finally:
        sys.settrace(None)
        state["finished_at_unix"] = time.time()
        state["completed_test_epochs"] = sum(r["split"] == "test" for r in epochs)
        if torch.cuda.is_initialized():
            state["peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
            state["peak_reserved_mib"] = torch.cuda.max_memory_reserved()/2**20
        write_json("status.json", state)
    return 0 if state["status"] in ("completed", "preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
