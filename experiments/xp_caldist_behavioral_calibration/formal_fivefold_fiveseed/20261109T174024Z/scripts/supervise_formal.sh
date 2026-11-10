#!/usr/bin/env bash
set -euo pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${formal_root}/../../../.." && pwd)"
initial_scheduler_pid="${1:?initial scheduler PID is required}"
python_bin=/data2/yb/conda_envs/cmerc-xp-repro/bin/python
teacher_pilot="${repo_root}/experiments/xp_caldist_behavioral_calibration/curriculum_teacher_pilot/20261109T154754Z"
cmerc_pilot="${repo_root}/experiments/xp_caldist_behavioral_calibration/independent_cmerc_pilot/20261109T164030Z"
log_dir="${formal_root}/logs"
mkdir -p "${log_dir}"

while kill -0 "${initial_scheduler_pid}" 2>/dev/null; do sleep 60; done
while pgrep -f "${formal_root}/scripts/(teacher_runner|generate_fold_local_difficulty|cmerc_runner|calibration_only)\.py" >/dev/null 2>&1; do
  sleep 30
done

export LD_LIBRARY_PATH="/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd "${repo_root}"

if ! "${python_bin}" - "${formal_root}/state/scheduler/task_ledger.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))["tasks"].values()
raise SystemExit(0 if all(row["status"] in {"completed", "reused"} for row in rows) else 1)
PY
then
  "${python_bin}" "${formal_root}/scripts/formal_scheduler.py" \
    --formal-root "${formal_root}" --teacher-pilot "${teacher_pilot}" --cmerc-pilot "${cmerc_pilot}" \
    --gpu-uuid GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248 \
    --gpu-uuid GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a \
    --gpu-uuid GPU-d17e4326-9d3b-2025-fbf8-c9b0d16623a0 \
    --gpu-uuid GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456 \
    --max-attempts 3 --recover-ledger --execute \
    >"${log_dir}/formal_scheduler_recovery.stdout.log" \
    2>"${log_dir}/formal_scheduler_recovery.stderr.log"
fi

"${python_bin}" "${formal_root}/scripts/finalize_formal.py" \
  --formal-root "${formal_root}" --teacher-pilot "${teacher_pilot}" --cmerc-pilot "${cmerc_pilot}" \
  >"${log_dir}/finalize_formal.stdout.log" 2>"${log_dir}/finalize_formal.stderr.log"
