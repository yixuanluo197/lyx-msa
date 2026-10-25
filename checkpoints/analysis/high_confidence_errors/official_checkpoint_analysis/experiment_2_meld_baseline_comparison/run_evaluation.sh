#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
meld_root=/data2/yb/paper/cmerc_reproduction_20261013
meld_output="$meld_root/checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_meld_baseline_comparison"
meld_reference="$meld_root/checkpoints/analysis/official_checkpoint_verification"
cd "$meld_root"
"$meld_root/scripts/with_cmerc_env.sh" python -u "$meld_output/source_code/verify_meld_baseline.py" \
 --runtime-code "$meld_reference/code_runtime" --input-root "$meld_root" \
 --reference-dir "$meld_reference" --output-dir "$meld_output/baseline_inference"
