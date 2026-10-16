#!/usr/bin/env bash
# Bounded checks only. Every replay gets a new directory; original evidence stays.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CHECK_PYTHON="${CHECK_PYTHON:-/home/yangbin/miniconda3/envs/multimodalerc310/bin/python}"
CHECK_PYTEST_SITE="${CHECK_PYTEST_SITE:-/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages}"
export CHECK_PYTEST_SITE
CHECK_REPLAY="reports/causal_review_followup/replays/$(date -u +%Y%m%dT%H%M%S%NZ)"
mkdir -p reports/causal_review_followup/replays
mkdir "$CHECK_REPLAY"
CUDA_VISIBLE_DEVICES='' "$CHECK_PYTHON" -c \
  'import os,sys;sys.path.append(os.environ["CHECK_PYTEST_SITE"]);import pytest;raise SystemExit(pytest.main(["-q","tests","-ra"]))' \
  > "$CHECK_REPLAY/cpu_tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$CHECK_PYTHON" scripts/verify_causal_regression.py \
  --output-dir "$CHECK_REPLAY/cpu" > "$CHECK_REPLAY/cpu_causality.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$CHECK_PYTHON" scripts/verify_training_integration.py \
  --device cpu --output-dir "$CHECK_REPLAY" > "$CHECK_REPLAY/cpu_training.log" 2>&1
if [[ -f data/IEMOCAP_features.pkl && -f data/meld_features_roberta.pkl ]]; then
  CUDA_VISIBLE_DEVICES='' "$CHECK_PYTHON" scripts/verify_real_feature_followup.py \
    --output "$CHECK_REPLAY/real_cpu.json" > "$CHECK_REPLAY/real_cpu.log" 2>&1
else
  echo 'SKIP real features: required local files are unavailable.' > "$CHECK_REPLAY/real_cpu.log"
fi
if [[ "${RUN_CUDA_CHECKS:-0}" == 1 ]]; then
  # This default is local user23 GPU 3. Verify health/resources before opting in.
  CHECK_GPU="${CHECK_GPU:-3}"
  nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv > "$CHECK_REPLAY/gpu_inventory.csv"
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$CHECK_PYTHON" -c \
    'import os,sys;sys.path.append(os.environ["CHECK_PYTEST_SITE"]);import pytest;raise SystemExit(pytest.main(["-q","tests","-ra"]))' \
    > "$CHECK_REPLAY/gpu_tests.log" 2>&1
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$CHECK_PYTHON" scripts/verify_causal_regression.py \
    --device cuda:0 --output-dir "$CHECK_REPLAY/gpu" > "$CHECK_REPLAY/gpu_causality.log" 2>&1
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CHECK_PYTHON" \
    scripts/verify_training_integration.py --device cuda:0 --output-dir "$CHECK_REPLAY" \
    > "$CHECK_REPLAY/gpu_training.log" 2>&1
  if [[ -f data/IEMOCAP_features.pkl && -f data/meld_features_roberta.pkl ]]; then
    CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$CHECK_PYTHON" scripts/verify_real_feature_followup.py \
      --device cuda:0 --output "$CHECK_REPLAY/real_gpu.json" > "$CHECK_REPLAY/real_gpu.log" 2>&1
  fi
fi
echo "Evidence: $CHECK_REPLAY"
