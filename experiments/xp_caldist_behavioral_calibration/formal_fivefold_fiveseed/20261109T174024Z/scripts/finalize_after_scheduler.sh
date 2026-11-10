#!/usr/bin/env bash
set -euo pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${formal_root}/../../../.." && pwd)"
scheduler_pid="${1:?scheduler PID is required}"
log_dir="${formal_root}/logs"
mkdir -p "${log_dir}"

while kill -0 "${scheduler_pid}" 2>/dev/null; do
  sleep 60
done

export LD_LIBRARY_PATH="/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

cd "${repo_root}"
/data2/yb/conda_envs/cmerc-xp-repro/bin/python \
  "${formal_root}/scripts/finalize_formal.py" \
  --formal-root "${formal_root}" \
  --teacher-pilot "${repo_root}/experiments/xp_caldist_behavioral_calibration/curriculum_teacher_pilot/20261109T154754Z" \
  --cmerc-pilot "${repo_root}/experiments/xp_caldist_behavioral_calibration/independent_cmerc_pilot/20261109T164030Z" \
  >"${log_dir}/finalize_formal.stdout.log" \
  2>"${log_dir}/finalize_formal.stderr.log"
