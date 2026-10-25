#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
m3net_root=/data2/yb/paper/cmerc_reproduction_20261013
m3net_output="$m3net_root/checkpoints/analysis/m3net_official_checkpoint_verification"
m3net_reference="$m3net_root/checkpoints/analysis/official_checkpoint_verification"
cd "$m3net_root"
"$m3net_root/scripts/with_cmerc_env.sh" python -u "$m3net_output/source_code/verify_m3net_official_checkpoint.py" \
 --runtime-code "$m3net_reference/code_runtime" --input-root "$m3net_root" \
 --reference-dir "$m3net_reference" --output-dir "$m3net_output/inference"
