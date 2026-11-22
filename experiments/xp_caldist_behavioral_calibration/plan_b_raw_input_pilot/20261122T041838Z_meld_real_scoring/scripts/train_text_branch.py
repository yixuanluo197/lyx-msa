#!/usr/bin/env python3
"""Persist tracked text smoke/training/extraction for the isolated base cohort."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import time

RUN=Path(__file__).resolve().parents[1]
PARENT=RUN.parent/"20261120T143131Z"
TEXT=Path("/home/yangbin/miniconda3/envs/TEXT_repro/bin/python")
PRETRAINED=Path("/home/yangbin/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59")
GPU="GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a"


def dg(path):
    h=sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()

def verified_smoke(out):
    proof=json.loads((out/"smoke_completion.json").read_text())
    evidence=proof["gradient_verification"]
    assert proof["status"]=="COMPLETE" and proof["optimizer_steps"]==2
    assert proof["official_validation_accessed"] is False
    assert evidence["all_backbone_parameters_trainable"] is True
    for role in ("backbone","classifier"):
        entry=evidence[role]
        assert entry["gradient_present"] and entry["gradient_finite"]
        assert entry["gradient_l2"]>0 and entry["max_parameter_update"]>0
    return dg(out/"smoke_completion.json")

def main():
    assert socket.gethostname()=="user23"
    gpu=next(r for r in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True).splitlines() if GPU in r)
    assert gpu.split(", ")[0]=="1" and int(gpu.split(", ")[2])<300
    lock=json.loads((RUN/"provenance/data_lock.json").read_text())
    data=RUN/"local_only/base_data/dataset.pt"
    enc=PARENT/"scripts/text_encoder.py"
    base_cmd=[str(TEXT),str(enc),"--dataset","MELD","--data",str(data),"--pretrained",str(PRETRAINED)]
    assert dg(data)==lock["output_hashes"]["local_only/base_data/dataset.pt"] and lock["base_bundle_score_dialogues"]==0
    source_hashes={str(f):dg(f) for f in (data,enc,*[PRETRAINED/name for name in ("model.safetensors","vocab.json","merges.txt","config.json")],
        RUN/"configs/REAL_SCORING_PROTOCOL.json",RUN/"provenance/data_lock.json",Path(__file__))}
    path=RUN/"provenance/text_execution.json"
    state=dict(status="RUNNING",session="meld_score_text_20261122",host="user23",gpu=gpu,
        pid=os.getpid(),started_at=now(),no_official_validation=True,no_Test=True,
        source_hashes=source_hashes,stages=[])
    modes=("smoke","train","extract")
    first_attempt=RUN/"provenance/text_execution_smoke_wrapper_failed.json"
    if first_attempt.is_file():
        prior=json.loads(first_attempt.read_text())
        assert prior["status"]=="FAILED" and len(prior["stages"])==1
        assert prior["stages"][0]["mode"]=="smoke" and prior["stages"][0]["exit_code"]==0
        assert all(prior["source_hashes"][str(f)]==source_hashes[str(f)] for f in (data,enc,PRETRAINED/"model.safetensors"))
        state["recovered_smoke_proof_sha256"]=verified_smoke(RUN/"local_only/text_smoke")
        state["prior_execution_sha256"]=dg(first_attempt)
        modes=("train","extract")
    with path.open("x") as f:json.dump(state,f,indent=2)
    def save():
        temp=path.with_suffix(".tmp")
        temp.write_text(json.dumps(state,indent=2)+"\n");temp.replace(path)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    try:
        for mode in modes:
            if not all(dg(Path(f))==h for f,h in source_hashes.items()):
                raise RuntimeError("A frozen text source changed")
            out=RUN/"local_only"/("text_smoke" if mode=="smoke" else "text_model")
            cmd=base_cmd+["--mode",mode,"--output",str(out)]
            if mode=="extract":
                cmd+=["--features",str(RUN/"local_only/base_data/features/text_features.pt")]
            stage=dict(mode=mode,started_at=now(),cmd=cmd)
            state["stages"].append(stage);save()
            with (RUN/"logs"/f"text_{mode}.log").open("x") as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                stage["pid"]=proc.pid;save()
                stage.update(exit_code=proc.wait(),ended_at=now());save()
            if stage["exit_code"]!=0:
                raise RuntimeError(f"Text {mode} failed, check local log")
            if mode=="smoke":
                state["smoke_proof_sha256"]=verified_smoke(out)
        state["status"]="COMPLETE"
    except Exception as error:
        state.update(status="FAILED",error=str(error));raise
    finally:
        state["ended_at"]=now();save()

if __name__=="__main__":
    main()
