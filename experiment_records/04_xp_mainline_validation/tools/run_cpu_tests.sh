#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/code
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro
exec bash scripts/with_cmerc_env.sh /data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest -q tests --ignore=tests/test_multiseed_scheduler.py --ignore=tests/test_xp_scheduler.py --ignore=tests/test_caldiff_batching.py -k 'not test_causal_checkpoint_roundtrip_and_architecture_rejection'
