#!/usr/bin/env python3
"""Wait for independent training and locked review, then run held-out scoring."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import time


RUN = Path(__file__).resolve().parents[1]
SCORER = RUN.parent / "20261121T175233Z_reliability_readiness/scripts/score_heldout.py"
ENVS = Path("/home/yangbin/miniconda3/envs")
TEXT = ENVS / "TEXT_repro/bin/python"
MM = ENVS / "multimodalerc310/bin/python"
GPU = "GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    if socket.gethostname() != "user23":
        raise RuntimeError("Use the existing Plan B server")
    protocol = RUN / "configs/REAL_SCORING_PROTOCOL.json"
    path = RUN / "provenance/scoring_execution.json"
    review = RUN / "provenance/ai_review_lock.json"
    downstream = RUN / "provenance/downstream_execution.json"
    script = RUN / "scripts/run_score_predictions.py"
    audit = RUN / "scripts/verify_score_inputs.py"
    state = dict(status="WAITING", host=socket.gethostname(), gpu=GPU,
                 pid=os.getpid(), started_at=stamp(), stages=[])
    with path.open("x") as handle:
        json.dump(state, handle, indent=2)

    def save():
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(state, indent=2) + "\n")
        pending.replace(path)

    try:
        while True:
            status = json.loads(downstream.read_text())["status"] if downstream.is_file() else None
            if status == "COMPLETE" and review.is_file():
                break
            if status == "FAILED":
                raise RuntimeError("Downstream supervisor did not complete")
            time.sleep(20)
        if digest(SCORER) != json.loads(protocol.read_text())["scorer_sha256"]:
            raise ValueError("Frozen CPU scorer changed")
        gpu = next(row for row in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True).splitlines() if GPU in row)
        if int(gpu.split(", ")[2]) >= 300:
            raise RuntimeError("The selected GPU already has an active workload")
        sources = {str(p): digest(p) for p in (protocol, review, SCORER, script, audit, downstream, Path(__file__))}
        state.update(status="RUNNING", source_hashes=sources, gpu_at_launch=gpu)
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1")
        cohort = RUN / "local_only"
        results = RUN / "results"
        commands = {
            "encode": [TEXT, script, "--stage", "encode", "--device", "cuda:0"],
            "infer": [MM, script, "--stage", "infer", "--device", "cuda:0"],
            "audit": [TEXT, audit],
            "fit_and_evaluate": [TEXT, SCORER, "--input", results / "score_inputs.csv",
                                 "--provenance", RUN / "provenance/scoring_input_declaration.json",
                                 "--manifest", cohort / "SPLIT_DIALOGUES.csv",
                                 "--utterance-manifest", cohort / "SPLIT_UTTERANCES.csv",
                                 "--split-lock", RUN / "SPLIT_LOCK.json",
                                 "--output", results / "scoring", "--allow-declared-real"],
        }
        for stage, parts in commands.items():
            if any(digest(Path(file)) != sha for file, sha in sources.items()):
                raise ValueError("A frozen scoring source changed")
            command = list(map(str, parts))
            record = dict(stage=stage, command=command, started_at=stamp())
            state["stages"].append(record)
            with (RUN / "logs" / f"scoring_{stage}.log").open("x") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                save()
                record.update(exit_code=process.wait(), ended_at=stamp())
                save()
            if record["exit_code"]:
                raise RuntimeError(f"Scoring stage failed: {stage}")
        state["status"] = "COMPLETE"
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        raise
    finally:
        state["ended_at"] = stamp()
        save()


if __name__ == "__main__":
    main()
