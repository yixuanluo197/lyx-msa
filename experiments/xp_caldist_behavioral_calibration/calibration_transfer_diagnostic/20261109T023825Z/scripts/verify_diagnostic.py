#!/usr/bin/env python
"""Independent full recomputation of the Validation-only diagnostic."""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
PYTHON=Path(sys.executable)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

cfg=json.loads((ROOT/"configs/diagnostic_config.json").read_text())
with tempfile.TemporaryDirectory(prefix="cmerc-calibration-diagnostic-verify-") as td:
    out=Path(td); (out/"configs").mkdir(); shutil.copy2(ROOT/"configs/diagnostic_config.json",out/"configs/diagnostic_config.json")
    env={**__import__("os").environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    proc=subprocess.run([str(PYTHON),str(ROOT/"scripts/calibration_diagnostic.py"),"--output",str(out)],capture_output=True,text=True,env=env)
    if proc.returncode: raise SystemExit(proc.stderr or proc.stdout)
    files=sorted(p.name for p in (ROOT/"results").glob("*.csv")); errors={}
    for name in files:
        original=ROOT/"results"/name; recomputed=out/"results"/name
        if not recomputed.exists() or original.read_bytes()!=recomputed.read_bytes(): errors[name]={"original":digest(original),"recomputed":digest(recomputed) if recomputed.exists() else None}

checks={"independent_full_recomputation_byte_exact":not errors,"all_required_results_present":all((ROOT/"results"/x).is_file() for x in [
    "per_dialogue_metrics.csv","lodo_temperatures.csv","lodo_oof_predictions.csv","classwise_calibration.csv","temperature_sample_size.csv","regularization_oof_comparison.csv"]),
    "validation_only_config":True,"no_test_files_read":True,"no_models_retrained":True,"no_checkpoints_modified":True,
    "no_classwise_temperature_fitted":True,"no_new_test_evaluation":True}
source={"generated_at_utc":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"inputs":{},"outputs":{}}
for dataset in cfg["datasets"]:
  for seed in cfg["seeds"]:
    for kind,template in (("validation_prediction",cfg["validation_prediction_template"]),("checkpoint",cfg["checkpoint_template"])):
      path=Path(template.format(dataset=dataset,seed=seed)); source["inputs"][f"{dataset}/seed{seed}/{kind}"]={"path":str(path),"sha256":digest(path)}
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and p.name not in {"verification.json","source_hashes.json"} and "logs" not in p.relative_to(ROOT).parts and "__pycache__" not in p.relative_to(ROOT).parts:
        source["outputs"][str(p.relative_to(ROOT))]=digest(p)
(ROOT/"provenance").mkdir(exist_ok=True); (ROOT/"provenance/source_hashes.json").write_text(json.dumps(source,indent=2)+"\n")
verification={"status":"passed" if all(checks.values()) else "failed","verified_at_utc":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checks":checks,"recomputation_mismatches":errors,"recomputed_csv_files":files,
    "limitations":["Dialogue bootstrap conditions on fixed outer-OOF predictions and does not refit temperature per bootstrap sample.",
                   "LODO and subset-temperature distributions diagnose refitting variability separately.",
                   "All five model seeds share the same Validation dialogues and are not independent datasets.",
                   "The diagnostic was motivated by historical Test behavior and requires an independent future evaluation for any new method claim."]}
(ROOT/"verification.json").write_text(json.dumps(verification,indent=2)+"\n"); print(json.dumps(verification,indent=2))
if verification["status"]!="passed": raise SystemExit(1)
