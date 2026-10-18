#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${CMERC_RUN_DIR:-$repo_dir/runs/formal}
result_root=${CMERC_RESULT_DIR:-$repo_dir/results/formal}

run_is_complete() {
  local dataset=$1
  local variant=$2
  local protocol=$3
  local seed expected run_dir metrics
  if [[ "$dataset" == IEMOCAP ]]; then
    seed=1475
    expected=80
  else
    seed=67137
    expected=40
  fi
  run_dir="$run_root/$variant/${dataset}_${protocol}_seed${seed}"
  metrics="$run_dir/metrics.jsonl"
  [[ -f "$metrics" && -f "$run_dir/best.pt" ]] || return 1
  (( $(wc -l < "$metrics") == expected ))
}

all_runs_complete() {
  local dataset variant protocol
  for dataset in IEMOCAP MELD; do
    for variant in baseline cmerc; do
      for protocol in official_test validation; do
        run_is_complete "$dataset" "$variant" "$protocol" || return 1
      done
    done
  done
}

formal_training_running() {
  pgrep -f -- "train_cm.py.*--output-dir $run_root/" >/dev/null
}

while ! all_runs_complete || formal_training_running; do
  sleep 30
done

CMERC_RUN_DIR="$run_root" CMERC_RESULT_DIR="$result_root" \
  bash "$repo_dir/scripts/evaluate_training_matrix.sh" \
  IEMOCAP baseline official_test \
  IEMOCAP baseline validation \
  IEMOCAP cmerc official_test \
  IEMOCAP cmerc validation \
  MELD baseline official_test \
  MELD baseline validation \
  MELD cmerc official_test \
  MELD cmerc validation

"${CMERC_PYTHON:-python}" "$repo_dir/scripts/summarize_results.py" \
  --run-root "$run_root" \
  --result-root "$result_root" \
  --output "$result_root/summary.json"

echo "All formal runs evaluated and summarized"
