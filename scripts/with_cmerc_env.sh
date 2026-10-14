#!/usr/bin/env bash
set -euo pipefail

CMERC_ENV_PREFIX="${CMERC_ENV_PREFIX:-/data2/yb/conda_envs/cmerc-repro}"
CUSPARSE_LIB="$CMERC_ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib"

if [[ ! -x "$CMERC_ENV_PREFIX/bin/python" ]]; then
  echo "CMERC environment not found: $CMERC_ENV_PREFIX" >&2
  exit 2
fi

if [[ ! -d "$CUSPARSE_LIB" ]]; then
  echo "CUDA 11 cuSPARSE runtime not found: $CUSPARSE_LIB" >&2
  exit 3
fi

export PATH="$CMERC_ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUSPARSE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$@"

