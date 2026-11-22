#!/usr/bin/env python3
"""Supervise the new MELD downstream branch on the existing local GPU host."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import time


RUN = Path(__file__).resolve().parents[1]
PYTHON = "/home/yangbin/miniconda3/envs/multimodalerc310/bin/python"
GPU = "GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a"


def timestamp():
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
    path = RUN / "provenance/downstream_execution.json"
    text_path = RUN / "provenance/text_execution.json"
    protocol = RUN / "configs/REAL_SCORING_PROTOCOL.json"
    adapter = RUN / "scripts/downstream_branch.py"
    metadata = dict(status="WAITING_FOR_TEXT", host=socket.gethostname(), gpu=GPU,
                    pid=os.getpid(), started_at=timestamp(), stages=[], text_run=str(text_path))
    with path.open("x") as handle:
        json.dump(metadata, handle, indent=2)

    def save():
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(metadata, indent=2) + "\n")
        pending.replace(path)

    try:
        while True:
            status = json.loads(text_path.read_text())["status"] if text_path.is_file() else None
            if status == "COMPLETE":
                break
            if status == "FAILED":
                raise RuntimeError("Text supervisor did not complete")
            time.sleep(20)
        gpu = next(row for row in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True).splitlines() if GPU in row)
        if int(gpu.split(", ")[2]) >= 300:
            raise RuntimeError("The selected GPU already has an active workload")
        sources = {str(p): digest(p) for p in (protocol, adapter, text_path, Path(__file__))}
        metadata.update(status="RUNNING", source_hashes=sources, gpu_at_launch=gpu)
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1")
        for stage in ("audit", "smoke", "train_m3net", "build_difficulty", "train_cmerc", "evaluate"):
            if any(digest(Path(file)) != sha for file, sha in sources.items()):
                raise RuntimeError("A frozen downstream source changed")
            command = [PYTHON, str(adapter), stage, "--protocol", str(protocol), "--device", "cuda:0"]
            record = dict(stage=stage, started_at=timestamp(), command=command)
            metadata["stages"].append(record)
            with (RUN / "logs" / f"downstream_{stage}.log").open("x") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                save()
                record.update(exit_code=process.wait(), ended_at=timestamp())
                save()
            if record["exit_code"]:
                raise RuntimeError(f"Downstream stage failed: {stage}")
        metadata["status"] = "COMPLETE"
    except Exception as error:
        metadata.update(status="FAILED", error=str(error))
        raise
    finally:
        metadata["ended_at"] = timestamp()
        save()


if __name__ == "__main__":
    main()
