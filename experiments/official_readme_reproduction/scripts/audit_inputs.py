"""Record current environment, original data coverage, and immutable input hashes."""
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def command(argv):
    p = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main():
    env = {"time_unix": time.time(), "python": sys.version, "executable": sys.executable,
           "host": platform.node(), "platform": platform.platform(),
           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "packages": {}, "environment": {k: os.environ.get(k) for k in (
               "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED", "PYTORCH_CUDA_ALLOC_CONF")}}
    for n in ("torch", "torch-geometric", "torch-scatter", "torch-sparse", "numpy",
              "scipy", "scikit-learn", "pandas", "ipdb", "tqdm"):
        env["packages"][n] = metadata.version(n)
    import torch_sparse
    import torch_scatter
    env["torch_sparse_import"] = torch_sparse.__version__
    env["torch_scatter_import"] = torch_scatter.__version__
    env["gpu"] = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version", "--format=csv"])
    env["gpu_processes"] = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"])
    env["process_owners"] = command(["ps", "-eo", "user,pid,ppid,etime,args"])
    env["disk"] = command(["df", "-h", str(ROOT)])
    cuda_file = Path(torch.__file__).parent/"cuda/__init__.py"
    env["inherited_torch_cuda_file"] = {"path": str(cuda_file), "sha256": digest(cuda_file),
        "known_patch": "XP04 inherited UUID-NVML enumeration compatibility; environment unmodified by this batch"}
    (ROOT/"provenance/environment.json").write_text(json.dumps(env, indent=2)+"\n")

    result = {"data": {}, "datasets": {}, "source_hashes": {}}
    for f in sorted((ROOT/"official_code/data").iterdir()):
        result["data"][f.name] = {"path": str(f.resolve()), "size_bytes": f.stat().st_size,
                                  "sha256": digest(f)}
    for dataset, name in (("IEMOCAP", "IEMOCAP_features.pkl"), ("MELD", "MELD_features_raw1.pkl")):
        data = pickle.load((ROOT/"official_code/data"/name).open("rb"), encoding="latin1")
        labels, train, test = data[2], list(data[7]), list(data[8])
        diff_file = ROOT/"official_code/caldiff"/(dataset+"_diff.pkl")
        difficulty = pickle.load(diff_file.open("rb"))
        entry = {"train_ids": train, "test_ids": test, "train_dialogues": len(train),
                 "test_dialogues": len(test), "train_utterances": sum(len(labels[k]) for k in train),
                 "test_utterances": sum(len(labels[k]) for k in test),
                 "train_test_overlap": list(set(train)&set(test)), "raw_train_type": type(data[7]).__name__,
                 "difficulty": {"sha256": digest(diff_file), "count": len(difficulty),
                     "missing_train_ids": list(set(train)-set(difficulty)),
                     "outside_train_ids": list(set(difficulty)-set(train)),
                     "test_overlap_ids": list(set(test)&set(difficulty)),
                     "all_finite": all(math.isfinite(v) for v in difficulty.values()),
                     "source": "official git object at pinned commit", "generation_history": "unknown"},
                 "fixed_order_batch_capacity": {}}
        for split, keys in (("train", train), ("test", test)):
            batches = []
            for start in range(0, len(keys), 16):
                ids = keys[start:start+16]
                length = sum(len(labels[k]) for k in ids)
                batches.append({"batch": start//16, "ids": ids, "utterances": length,
                                "hyperedges": length+3*len(ids), "nodes": 3*length,
                                "incidences": 6*length})
            entry["fixed_order_batch_capacity"][split] = batches
        fractions = [.15, .30, .45, .60, .75, .90, 1] if dataset=="IEMOCAP" else [.4,.8,1]
        entry["curriculum_expected"] = [{"epoch": i+1,"fraction": f,"dialogues": int(f*len(train))} for i,f in enumerate(fractions)]
        # Every feature stream must have a row for every utterance in every used dialogue.
        roberta = pickle.load((ROOT/"official_code/data"/("iemocap_features_roberta.pkl" if dataset=="IEMOCAP" else "meld_features_roberta.pkl")).open("rb"), encoding="latin1")
        features = roberta[2:6] if dataset=="IEMOCAP" else roberta[3:7]
        issues = []
        for key in train+test:
            expected = len(labels[key])
            for idx, stream in enumerate([*features, data[4], data[5], data[1]]):
                if key not in stream or len(stream[key]) != expected:
                    issues.append({"id": key, "stream": idx,"expected": expected})
        entry["feature_alignment_issues"] = issues
        if dataset=="MELD":
            entry["roberta_splits"] = {"train":list(roberta[8]), "test":list(roberta[9]), "valid":list(roberta[10])}
            entry["raw_train_equals_roberta_train_union_valid"] = set(train) == set(roberta[8])|set(roberta[10])
        result["datasets"][dataset] = entry
    for folder in ("official_code", "runtime_code", "scripts", "configs"):
        for f in sorted((ROOT/folder).rglob("*")):
            if f.is_file() and not f.is_symlink() and "__pycache__" not in f.parts:
                result["source_hashes"][str(f.relative_to(ROOT))] = digest(f)
    (ROOT/"provenance/inputs.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"feature_alignment_issues": {k:len(v["feature_alignment_issues"]) for k,v in result["datasets"].items()}, "host": env["host"], "packages":env["packages"]}))


if __name__ == "__main__":
    main()
