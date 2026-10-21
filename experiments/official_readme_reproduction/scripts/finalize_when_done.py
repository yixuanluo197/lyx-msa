"""After all four trainings finish, replay fixed checkpoints and validate exports."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU = "GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456"
PYTHON = "/data2/yb/conda_envs/cmerc-xp-repro/bin/python"


def save(state):
    path = ROOT/"state/finalization.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,indent=2)+"\n")
    tmp.replace(path)


def main():
    if (ROOT/"state/finalization.json").exists():
        raise RuntimeError("Existing finalization state; refuse duplicate launch")
    state = {"active":True,"phase":"waiting_for_four_trainings","pid":os.getpid(),"commands":[],"started_at_unix":time.time()}
    save(state)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,
        LD_LIBRARY_PATH="/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    def run(argv,log_name):
        record = {"argv":argv,"started_at_unix":time.time(),"log":log_name,
            "environment":{k:env[k] for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
            "script_sha256":hashlib.sha256(Path(argv[2]).read_bytes()).hexdigest() if len(argv)>2 and Path(argv[2]).is_file() else None}
        state["commands"].append(record)
        save(state)
        with (ROOT/"logs"/log_name).open("x") as log:
            proc = subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
            record["pid"] = proc.pid
            save(state)
            record["returncode"] = proc.wait()
        record["finished_at_unix"] = time.time()
        save(state)
        if record["returncode"]:
            raise RuntimeError((log_name,record["returncode"]))
    try:
        while True:
            training = json.loads((ROOT/"state/status.json").read_text())
            if not training["active"]:
                if training["formal_runs_completed"] != 4:
                    raise RuntimeError("Training incomplete/failed; no automatic retraining")
                break
            time.sleep(20)
        for ident in training["runs"]:
            state.update(phase="fixed_checkpoint_replay",run_id=ident)
            save(state)
            while True:
                query = subprocess.run(["nvidia-smi","--id="+GPU,"--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True)
                processes = subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],capture_output=True,text=True)
                if query.returncode==0 and processes.returncode==0:
                    free,util = [int(x.strip()) for x in query.stdout.split(",")]
                    if free>=11500 and util<=10 and GPU not in processes.stdout:
                        break
                time.sleep(10)
            run([PYTHON,"-u",str(ROOT/"scripts/evaluate_saved.py"),"--run-id",ident],ident+"_checkpoint_replay.log")
            if "cmerc" in ident:
                state["phase"] = "training_batch_gradient_diagnostic"
                save(state)
                run([PYTHON,"-u",str(ROOT/"scripts/diagnose_losses.py"),"--run-id",ident],ident+"_gradient_diagnostic.log")
        state["phase"] = "validating_and_exporting"
        save(state)
        run([PYTHON,"-u",str(ROOT/"scripts/summarize.py")],"final_summary_validation.log")
        state.update(active=False,phase="completed",completed_at_unix=time.time())
        save(state)
        print("Four trainings, fixed-checkpoint replays, gradient diagnostics, and numerical exports completed",flush=True)
    except BaseException as error:
        state.update(active=False,phase="blocked",error=repr(error),completed_at_unix=time.time())
        save(state)
        raise


if __name__ == "__main__":
    main()
