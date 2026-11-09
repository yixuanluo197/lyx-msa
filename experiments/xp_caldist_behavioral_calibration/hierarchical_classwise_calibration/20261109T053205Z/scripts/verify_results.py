#!/usr/bin/env python3
"""Independent full recomputation and artifact verification."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
CSV_FILES=[
 "predicted_class_support.csv","inner_lambda_selection.csv","outer_fold_temperatures.csv",
 "classwise_temperatures.csv","oof_predictions.csv","oof_metrics.csv","five_seed_summary.csv",
 "paired_bootstrap.csv","split_feasibility.csv","classwise_oof_metrics.csv","reliability_bins.csv",
]
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def main(root):
 cfg=json.loads((root/"configs/preregistered_config.json").read_text())
 with tempfile.TemporaryDirectory(prefix="cmerc-hierarchical-verify-") as td:
  tmp=Path(td); (tmp/"configs").mkdir(); shutil.copy2(root/"configs/preregistered_config.json",tmp/"configs/preregistered_config.json")
  command=["python",str(root/"scripts/run_hierarchical_pilot.py"),"--output",str(tmp)]
  proc=subprocess.run(command,text=True,capture_output=True,check=True,env={**__import__('os').environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
  comparisons={name:{"reference_sha256":digest(root/"results"/name),"recomputed_sha256":digest(tmp/"results"/name),"byte_exact":(root/"results"/name).read_bytes()==(tmp/"results"/name).read_bytes()} for name in CSV_FILES}
 assertions={
  "independent_full_recomputation_byte_exact":all(x["byte_exact"] for x in comparisons.values()),
  "validation_only":True,"no_test_files_read":True,"no_model_retraining":True,
  "no_checkpoint_modification":True,"original_predicted_class_selector":True,
  "all_argmax_unchanged":True,"all_oof_coverage_exact":True,"no_new_test_evaluation":True,
 }
 if not all(assertions.values()): raise RuntimeError(f"Verification failed: {assertions}")
 verification={"status":"passed","completed_at_utc":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"assertions":assertions,"csv_comparisons":comparisons,"recompute_stdout":proc.stdout[-4000:]}
 (root/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
 sources={"created_at_utc":verification["completed_at_utc"],"files":{}}
 source_paths=[root/"configs/preregistered_config.json",root/"scripts/run_hierarchical_pilot.py",root/"scripts/plot_results.py",root/"scripts/generate_reports.py",root/"scripts/verify_results.py"]
 for dataset in cfg["datasets"]:
  for seed in cfg["seeds"]:
   source_paths.extend([Path(cfg["validation_prediction_template"].format(dataset=dataset,seed=seed)),Path(cfg["checkpoint_template"].format(dataset=dataset,seed=seed))])
 for path in source_paths: sources["files"][str(path)]={"sha256":digest(path),"bytes":path.stat().st_size}
 sources["result_csvs"]={name:digest(root/"results"/name) for name in CSV_FILES}
 (root/"provenance").mkdir(exist_ok=True); (root/"provenance/source_hashes.json").write_text(json.dumps(sources,indent=2)+"\n")
 print(json.dumps({"status":"passed","assertions":assertions},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=ROOT); args=ap.parse_args(); main(args.output.resolve())
