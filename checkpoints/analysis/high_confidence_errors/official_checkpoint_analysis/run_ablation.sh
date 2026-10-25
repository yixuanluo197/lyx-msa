#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ablation_root=/data2/yb/paper/cmerc_reproduction_20261013
ablation_output="$ablation_root/checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis"
ablation_reference="$ablation_root/checkpoints/analysis/official_checkpoint_verification"
cd "$ablation_root"
for ablation_dataset in IEMOCAP MELD; do
  for ablation_modality in text audio visual; do
    "$ablation_root/scripts/with_cmerc_env.sh" python -u "$ablation_output/source_code/run_official_modality_ablation.py" \
      --dataset "$ablation_dataset" --condition "w/o_$ablation_modality" \
      --runtime-code "$ablation_reference/code_runtime" --input-root "$ablation_root" \
      --reference-dir "$ablation_reference" --output-dir "$ablation_output/$ablation_dataset/inference/without_$ablation_modality"
  done
done
