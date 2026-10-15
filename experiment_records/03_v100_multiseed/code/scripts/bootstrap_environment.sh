#!/usr/bin/env bash
set -euo pipefail

CMERC_ENV_PREFIX="${CMERC_ENV_PREFIX:-/data2/yb/conda_envs/cmerc-repro}"
CONDA_SH="${CONDA_SH:-/home/qinjinghui/anaconda3/etc/profile.d/conda.sh}"

# shellcheck disable=SC1090
source "$CONDA_SH"

conda create \
  --prefix "$CMERC_ENV_PREFIX" \
  --override-channels \
  --channel conda-forge \
  'python=3.10.13=hd12c33a_1_cpython' \
  'pip=24.2' \
  --yes

PYTHON="$CMERC_ENV_PREFIX/bin/python"

"$PYTHON" -m pip install --no-cache-dir \
  'torch==1.13.1+cu117' \
  'torchvision==0.14.1+cu117' \
  --extra-index-url https://download.pytorch.org/whl/cu117

"$PYTHON" -m pip install --no-cache-dir \
  'nvidia-cusparse-cu11==11.7.5.86' \
  'numpy==1.26.4' \
  'scipy==1.11.4' \
  'pandas==1.5.3' \
  'scikit-learn==1.3.2' \
  'ipdb==0.13.13' \
  'tqdm==4.67.1' \
  'pytest==8.3.5' \
  'psutil==7.1.0' \
  'tensorboardX==2.6.4'

"$PYTHON" -m pip install --no-cache-dir \
  'torch-scatter==2.1.0+pt113cu117' \
  'torch-sparse==0.6.15+pt113cu117' \
  --find-links https://data.pyg.org/whl/torch-1.13.0+cu117.html

"$PYTHON" -m pip install --no-cache-dir 'torch-geometric==2.4.0'

