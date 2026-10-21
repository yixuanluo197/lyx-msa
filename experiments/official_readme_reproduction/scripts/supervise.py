"""Persist the four preregistered runs; never retry based on observed scores."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data2/yb/conda_envs/cmerc-xp-repro/bin/python"
GPU = "GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456"
ORDER = ["IEMOCAP_baseline_seed1475", "MELD_baseline_seed67137",
         "MELD_cmerc_seed67137", "IEMOCAP_cmerc_seed1475"]


def write(name, value):
    path = ROOT/name
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def main():
    lock = (ROOT/"state/supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any((ROOT/"checkpoints"/run).exists() for run in ORDER):
        raise RuntimeError("Existing formal output: refusing automatic duplicate/restart")
    manifest = json.loads((ROOT/"provenance/inputs.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        if name.startswith(("official_code/", "runtime_code/", "configs/")):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    for pattern in ("preflight_compat_IEMOCAP_baseline", "preflight_compat_IEMOCAP_cmerc_seed1475",
                    "preflight_compat_MELD_baseline_seed67137", "preflight_compat_MELD_cmerc_seed67137"):
        assert json.loads((ROOT/"logs"/pattern/"status.json").read_text())["status"] == "preflight_passed"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
               LD_LIBRARY_PATH="/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # No PYTHONHASHSEED, allocator, RNG, model or optimizer overrides.
    env.pop("PYTHONHASHSEED", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    state = {"active": True, "phase": "formal_training", "started_at_unix": time.time(),
             "host": os.uname().nodename, "gpu_uuid": GPU, "supervisor_pid": os.getpid(),
             "runs": {run: {"status": "planned"} for run in ORDER},
             "automatic_retraining": False, "selection_protocol": "official_test_selection"}
    processes, logs = {}, {}
    last_launch = 0
    while True:
        for run, proc in list(processes.items()):
            result = proc.poll()
            if result is None:
                continue
            logs.pop(run).close()
            record = state["runs"][run]
            record.update(returncode=result, finished_at_unix=time.time())
            output = ROOT/"checkpoints"/run
            recorded = json.loads((output/"status.json").read_text()) if (output/"status.json").exists() else {}
            expected = json.loads((ROOT/"configs"/(run+".json")).read_text())["epochs"]
            success = result == 0 and recorded.get("status") == "completed" and recorded.get("completed_test_epochs") == expected and all((output/f).exists() for f in ("best.pt", "final.pt", "latest.pt", "best_test.json"))
            record["status"] = "completed" if success else "failed"
            record["completed_test_epochs"] = recorded.get("completed_test_epochs", 0)
            record["error"] = recorded.get("error")
            del processes[run]
            print(run, record["status"], record.get("error"), flush=True)
        gpu = subprocess.run(["nvidia-smi", "--id="+GPU,
            "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True, capture_output=True)
        known_pids = {p.pid for p in processes.values()}
        unknown = []
        for line in apps.stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields)>=3 and fields[0] == GPU and int(fields[1]) not in known_pids:
                unknown.append(fields)
        state["last_gpu"] = gpu.stdout.strip()
        state["unknown_gpu_processes"] = unknown
        with (ROOT/"logs/resources.jsonl").open("a") as f:
            f.write(json.dumps({"time":time.time(),"gpu":gpu.stdout.strip(),"apps":apps.stdout.strip(),"known_pids":sorted(known_pids),"unknown":unknown})+"\n")
        if not unknown and gpu.returncode == 0 and len(processes)<2 and time.time()-last_launch>=15:
            fields = [x.strip() for x in gpu.stdout.split(",")]
            free, utilization = int(fields[2]), int(fields[3])
            for run in ORDER:
                if state["runs"][run]["status"] != "planned":
                    continue
                # The same-dataset baseline must finish first; author difficulty remains fixed.
                if "cmerc" in run:
                    dependency = run.replace("cmerc", "baseline")
                    if state["runs"][dependency]["status"] == "failed":
                        state["runs"][run].update(status="blocked", error="same-dataset baseline failed; no protocol change/retry")
                        continue
                    if state["runs"][dependency]["status"] != "completed":
                        continue
                budget = 9500 if run.startswith("IEMOCAP_cmerc") else 8000 if run.startswith("IEMOCAP") else 2500
                if free < budget or (not processes and utilization>10):
                    continue
                argv = [PYTHON, "-u", str(ROOT/"scripts/observe_run.py"), "--config", str(ROOT/"configs"/(run+".json")),
                        "--code", str(ROOT/"runtime_code"), "--output", str(ROOT/"checkpoints"/run)]
                log = (ROOT/"logs"/(run+".log")).open("x")
                proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
                processes[run], logs[run] = proc, log
                state["runs"][run].update(status="running",pid=proc.pid,argv=argv,
                    environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
                    started_at_unix=time.time(),gpu_before_start=gpu.stdout.strip(), reserved_admission_mib=budget)
                last_launch = time.time()
                print(run,"started",proc.pid,flush=True)
                break
        state["formal_runs_started"] = sum("pid" in v for v in state["runs"].values())
        state["formal_runs_completed"] = sum(v["status"] == "completed" for v in state["runs"].values())
        if all(v["status"] in ("completed","failed","blocked") for v in state["runs"].values()):
            state.update(active=False,phase="completed" if state["formal_runs_completed"]==4 else "blocked",completed_at_unix=time.time())
            write("state/status.json",state)
            return 0 if state["formal_runs_completed"]==4 else 1
        write("state/status.json",state)
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
