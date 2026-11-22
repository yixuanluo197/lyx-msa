"""Run the frozen Train-only inference and CPU scoring in the original environments."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from freeze_pilot import PRIVATE, PROV, RUN, digest, load, dump


HOST = "user23"
GPU = "GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a"
ENVS = Path("/home/yangbin/miniconda3/envs")
TEXT = ENVS / "TEXT_repro/bin/python"
MODEL = ENVS / "multimodalerc310/bin/python"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    assert socket.gethostname() == HOST
    rows = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits"],text=True).splitlines()
    selected = next(r for r in rows if GPU in r)
    index, _, used, temperature = selected.split(", ")
    if index != "1" or int(used) > 300 or int(temperature) > 80:
        raise RuntimeError("Original GPU unavailable at launch")
    protocol = RUN / "EXPERIMENT_PROTOCOL.json"
    gates = [PROV / name for name in ("SOURCE_LOCK.json","GENERATION_LOCK.json","REVIEW_LOCK.json","INDEPENDENCE_GATE.json")]
    source, generation, review, independent = map(load, gates)
    assert independent["status"] == "CONFIRMATION_PAUSED"
    assert source["status"] == "SOURCE_EXPORT_VERIFIED" and not source["old_score_eval_exported"]
    assert generation["n_targets"] == review["n_targets"] == source["n_targets"] == 135
    assert review["status"] == "AI_REVIEW_LOCKED" and not review["human_review"]
    assert digest(protocol) == source["protocol_sha256"] == review["protocol_sha256"]
    assert digest(PRIVATE/"reviewed.jsonl") == review["reviewed_sha256"]
    stages = {
        "encode":[TEXT,RUN/"scripts/infer_pilot.py","--stage","encode","--device","cuda:0"],
        "infer":[MODEL,RUN/"scripts/infer_pilot.py","--stage","infer","--device","cuda:0"],
        "score":[TEXT,RUN/"scripts/score_pilot.py"],
    }
    scripts = sorted({Path(command[1]).resolve() for command in stages.values()})
    assets = {str(path):digest(path) for path in (protocol,*gates,*scripts,Path(__file__))}
    execution = PROV / "PILOT_EXECUTION.json"
    state = dict(status="RUNNING",host=HOST,gpu=selected,pid=os.getpid(),started_at=stamp(),
                 scope="TRAIN_ONLY_EXPLORATORY",source_hashes=assets,stages=[])
    dump(execution,state)

    def save():
        tmp=execution.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,indent=2)+"\n")
        tmp.replace(execution)

    gpu_env = dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
    cpu_env = dict(gpu_env,CUDA_VISIBLE_DEVICES="")
    try:
        for stage, parts in stages.items():
            if any(digest(Path(path)) != sha for path, sha in assets.items()):
                raise RuntimeError("Frozen driver, protocol or candidate lock changed")
            command = [str(x) for x in parts]
            item = dict(stage=stage,command=command,started_at=stamp())
            state["stages"].append(item)
            with (RUN/"logs"/f"{stage}.log").open("x") as log:
                child = subprocess.Popen(command,env=cpu_env if stage=="score" else gpu_env,
                                         stdout=log,stderr=subprocess.STDOUT)
                item["pid"] = child.pid
                save()
                item.update(exit_code=child.wait(),ended_at=stamp())
                save()
            if item["exit_code"] != 0:
                raise RuntimeError(f"Stop on {stage} failure. See logs/{stage}.log")
        state["status"] = "COMPLETE"
    except Exception as error:
        state.update(status="FAILED",error=str(error))
        raise
    finally:
        state["ended_at"] = stamp()
        save()


if __name__ == "__main__":
    main()
