#!/usr/bin/env python3
"""Run frozen K5 re-extraction, inference and analysis in their original environments."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess

RUN=Path(__file__).resolve().parents[1]
ENVS=Path("/home/yangbin/miniconda3/envs")
STAGES=(("encode","TEXT_repro","run_k5.py",["--mode","encode","--run-dir",str(RUN),"--device","cuda:0"]),
        ("infer","multimodalerc310","run_k5.py",["--mode","infer","--run-dir",str(RUN),"--device","cuda:0"]),
        ("analyze","TEXT_repro","analyze_k5.py",[]))
GPU="GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def binding():
    for rel,h in json.loads((RUN/"provenance/cohort_lock.json").read_text()).items():
        if digest(RUN/rel)!=h: raise ValueError(f"Cohort drift: {rel}")
    lock=json.loads((RUN/"provenance/input_lock.json").read_text())
    if digest(RUN/"local_only/matched_text_reviewed.jsonl")!=lock["final_private_inputs_sha256"]:
        raise ValueError("Reviewed inputs changed")
    if any(digest(RUN/path)!=h for path,h in lock["source_part_sha256"].items()):
        raise ValueError("Reviewed source material changed")
    return {p.name:digest(p) for p in (RUN/"scripts").glob("*.py")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--session",required=True)
    args=parser.parse_args()
    if socket.gethostname()!="user23":
        raise ValueError("K5 uses the same local server as the frozen baseline")
    inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader,nounits"],text=True).splitlines()
    devices=[r.split(", ") for r in inventory]
    row=next((r for r in devices if r[1]==GPU),None)
    if not row or row[0]!="0" or int(row[2])>300:
        raise ValueError("Original healthy GPU 0 not idle")
    frozen=binding()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="2",PYTHONDONTWRITEBYTECODE="1")
    path=RUN/"provenance/execution.json"
    state=dict(session=args.session, host=socket.gethostname(), pid=os.getpid(), gpu=row,
               started_at=now(), status="RUNNING", source_script_hashes=frozen, stages=[])
    with path.open("x") as stream:
        json.dump(state,stream,indent=2)
    def save():
        temp=path.with_suffix(".tmp")
        temp.write_text(json.dumps(state,indent=2)+"\n")
        temp.replace(path)
    try:
        for stage,environment,script,opts in STAGES:
            if binding()!=frozen:
                raise RuntimeError("K5 scripts changed during execution")
            cmd=[str(ENVS/environment/"bin/python"),str(RUN/"scripts"/script),*opts]
            status=dict(name=stage,command=cmd,started_at=now())
            state["stages"].append(status)
            save()
            with (RUN/"logs"/(stage+".log")).open("x") as stream:
                p=subprocess.Popen(cmd, stdout=stream,stderr=subprocess.STDOUT,env=env)
                status["pid"]=p.pid
                save()
                status.update(exit_code=p.wait(),ended_at=now())
            save()
            if status["exit_code"]:
                raise RuntimeError(f"{stage} exited {status['exit_code']}; see local log")
        state["status"]="COMPLETE"
    except Exception as error:
        state.update(status="FAILED",error=str(error))
        raise
    finally:
        state["completed_at"]=now()
        save()


if __name__=="__main__":
    main()
