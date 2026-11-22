"""Execute the two frozen-model inference stages and the CPU exploratory score."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from audit_inputs import BASE, RUN, SCORE, digest, read_json


HOST = "user23"
GPU = "GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a"
ENVS = Path("/home/yangbin/miniconda3/envs")
ENC = ENVS / "TEXT_repro/bin/python"
MM = ENVS / "multimodalerc310/bin/python"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    if socket.gethostname() != HOST:
        raise RuntimeError("Use the existing Plan B model host")
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    selected = next(row for row in gpu_info if GPU in row)
    index, _, memory, temperature = selected.split(", ")
    if index != "1" or int(memory) >= 300 or int(temperature) > 80:
        raise RuntimeError("The selected GPU is busy or unhealthy")
    protocol = RUN / "EXPERIMENT_PROTOCOL.json"
    infer = RUN / "scripts/infer_corruption.py"
    score = RUN / "scripts/compare_scores.py"
    inputs = RUN / "scripts/bind_scoring.py"
    audit = RUN / "provenance/INPUT_AUDIT.json"
    review = RUN / "provenance/CORRUPTION_REVIEW_LOCK.json"
    candidates = RUN / "local_only/corruption_reviewed.jsonl"
    if read_json(audit)["status"] != "INPUTS_VERIFIED" or read_json(review)["status"] != "AI_REVIEW_LOCKED":
        raise RuntimeError("Complete the input and contextual review audits first")
    if digest(candidates) != read_json(review)["reviewed_sha256"]:
        raise ValueError("Reviewed variant hash mismatch")
    for rel, sha in read_json(audit)["sha256_relative_to_pilot_parent"].items():
        if digest(BASE / rel) != sha:
            raise ValueError(f"Source audit changed: {rel}")
    sources = {str(p): digest(p) for p in (protocol, infer, score, inputs, audit, review, candidates, Path(__file__))}
    output = RUN / "provenance/PILOT_EXECUTION.json"
    info = dict(status="RUNNING", host=HOST, gpu=selected, pid=os.getpid(),
                started_at=stamp(), source_hashes=sources, steps=[])
    with output.open("x") as handle:
        json.dump(info, handle, indent=2)

    def save():
        pending = output.with_suffix(".tmp")
        pending.write_text(json.dumps(info, indent=2) + "\n")
        pending.replace(output)

    device_env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    cpu_env = dict(device_env, CUDA_VISIBLE_DEVICES="")
    steps = [
        ("encode", [ENC, infer, "--stage", "encode", "--device", "cuda:0"], device_env),
        ("infer", [MM, infer, "--stage", "infer", "--device", "cuda:0"], device_env),
        ("bind_scoring", [ENC, inputs], cpu_env),
        ("scoring", [ENC, score, "--manifest", SCORE / "local_only/SPLIT_DIALOGUES.csv",
                     "--utterance-manifest", SCORE / "local_only/SPLIT_UTTERANCES.csv",
                     "--split-lock", SCORE / "SPLIT_LOCK.json"], cpu_env),
    ]
    try:
        for stage, raw_command, env in steps:
            if any(digest(Path(p)) != sha for p, sha in sources.items()):
                raise ValueError("A frozen execution input changed")
            command = list(map(str, raw_command))
            entry = dict(stage=stage, started_at=stamp(), command=command)
            info["steps"].append(entry)
            with (RUN / "logs" / f"pilot_{stage}.log").open("x") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                entry.update(pid=proc.pid)
                save()
                entry.update(exit_code=proc.wait(), ended_at=stamp())
                save()
            if entry["exit_code"]:
                raise RuntimeError(f"Stop at failed stage {stage}. See pilot_{stage}.log.")
        info["status"] = "COMPLETE"
    except Exception as error:
        info.update(status="FAILED", error=str(error))
        raise
    finally:
        info["ended_at"] = stamp()
        save()


if __name__ == "__main__":
    main()
