#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${CMERC_OUTPUT_DIR:-$repo_dir/runs/formal}
log_root=${CMERC_LOG_DIR:-$repo_dir/logs/training}
official_run="$run_root/cmerc/MELD_official_test_seed67137"
validation_run="$run_root/cmerc/MELD_validation_seed67137"
mkdir -p "$log_root"

epoch_count() {
  local path=$1
  [[ -f "$path" ]] || { echo 0; return; }
  wc -l < "$path"
}

find_live_run() {
  local protocol=$1
  pgrep -f -- "train_cm.py.*--Dataset MELD.*--selection-protocol $protocol.*$run_root/cmerc" | head -n 1 || true
}

official_pid=$(find_live_run official_test)
if [[ -n "$official_pid" ]]; then
  official_is_child=0
  echo "Attached to existing MELD CMERC official_test PID $official_pid"
elif (( $(epoch_count "$official_run/metrics.jsonl") == 0 )); then
  bash "$repo_dir/scripts/run_training.sh" MELD cmerc official_test --num-workers 0 \
    >"$log_root/MELD_cmerc_official_test.log" 2>&1 &
  official_pid=$!
  official_is_child=1
  echo "Started MELD CMERC official_test as PID $official_pid"
elif (( $(epoch_count "$official_run/metrics.jsonl") < 40 )); then
  echo "Incomplete MELD official_test run has no live process" >&2
  exit 3
else
  official_pid=
  official_is_child=0
  echo "MELD CMERC official_test was already complete"
fi

iemocap_finished() {
  local metrics
  for metrics in \
    "$run_root/cmerc/IEMOCAP_official_test_seed1475/metrics.jsonl" \
    "$run_root/cmerc/IEMOCAP_validation_seed1475/metrics.jsonl"; do
    if (( $(epoch_count "$metrics") == 80 )); then
      return 0
    fi
  done
  return 1
}

baseline_slot_released() {
  (( $(epoch_count "$run_root/baseline/MELD_official_test_seed67137/metrics.jsonl") == 40 )) &&
    (( $(epoch_count "$run_root/baseline/MELD_validation_seed67137/metrics.jsonl") == 40 ))
}

while ! iemocap_finished && ! baseline_slot_released; do
  if [[ -n "$official_pid" ]] && ! kill -0 "$official_pid" 2>/dev/null; then
    if (( official_is_child )); then
      wait "$official_pid"
    fi
    official_pid=
    if (( $(epoch_count "$official_run/metrics.jsonl") != 40 )); then
      echo "MELD official_test terminated before 40 epochs" >&2
      exit 4
    fi
  fi
  sleep 30
done

validation_pid=$(find_live_run validation)
if [[ -n "$validation_pid" ]]; then
  validation_is_child=0
  echo "Attached to existing MELD CMERC validation PID $validation_pid"
elif (( $(epoch_count "$validation_run/metrics.jsonl") == 0 )); then
  bash "$repo_dir/scripts/run_training.sh" MELD cmerc validation --num-workers 0 \
    >"$log_root/MELD_cmerc_validation.log" 2>&1 &
  validation_pid=$!
  validation_is_child=1
  echo "Started MELD CMERC validation as PID $validation_pid"
elif (( $(epoch_count "$validation_run/metrics.jsonl") < 40 )); then
  echo "Incomplete MELD validation run has no live process" >&2
  exit 3
else
  validation_pid=
  validation_is_child=0
  echo "MELD CMERC validation was already complete"
fi

wait_for_run() {
  local pid=$1
  local is_child=$2
  local run_dir=$3
  local label=$4
  [[ -n "$pid" ]] || return 0
  if (( is_child )); then
    wait "$pid"
  else
    while (( $(epoch_count "$run_dir/metrics.jsonl") < 40 )); do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "$label terminated before 40 epochs" >&2
        exit 4
      fi
      sleep 30
    done
  fi
}

wait_for_run "$official_pid" "$official_is_child" "$official_run" "MELD official_test"
wait_for_run "$validation_pid" "$validation_is_child" "$validation_run" "MELD validation"
echo "Completed both MELD CMERC protocols"
