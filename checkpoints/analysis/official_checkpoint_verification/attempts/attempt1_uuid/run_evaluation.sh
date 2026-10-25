#!/usr/bin/env bash
set -euo pipefail
verification_output=/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/analysis/official_checkpoint_verification
verification_input=/data2/yb/paper/cmerc_reproduction_20261013
export CUDA_VISIBLE_DEVICES=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$verification_input"
for verification_dataset in IEMOCAP MELD; do
  "$verification_input/scripts/with_cmerc_env.sh" python -u "$verification_output/source_code/verify_author_checkpoints.py" \
    --dataset "$verification_dataset" \
    --runtime-code "$verification_output/code_runtime" \
    --input-root "$verification_input" \
    --output-dir "$verification_output"
done
