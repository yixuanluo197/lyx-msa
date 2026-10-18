#!/usr/bin/env bash
set -euo pipefail

explicit_seeds=""
while [[ ${1:-} == --* ]]; do
  case "$1" in
    --seed|--seeds) explicit_seeds=${2:?Missing seed list}; shift 2 ;;
    *) echo "Unsupported option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$explicit_seeds" && ! "$explicit_seeds" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Seeds must be comma-separated nonnegative integers" >&2; exit 2
fi
if (( $# == 0 || $# % 3 != 0 )); then
  echo "Usage: $0 [--seeds 1475,67137,2024] DATASET VARIANT PROTOCOL [...]" >&2
  exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${CMERC_RUN_DIR:-$repo_dir/runs/formal}
result_root=${CMERC_RESULT_DIR:-$repo_dir/results/formal}
log_root=${CMERC_EVAL_LOG_DIR:-$repo_dir/logs/formal_evaluation}
python_bin=${CMERC_PYTHON:-python}
mkdir -p "$result_root" "$log_root"

while (( $# )); do
  dataset=$1
  variant=$2
  protocol=$3
  shift 3

  case "$dataset" in
    IEMOCAP) seed=1475 ;;
    MELD) seed=67137 ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
  esac
  case "$variant" in baseline|cmerc) ;; *) echo "Unsupported variant: $variant" >&2; exit 2 ;; esac
  case "$protocol" in official_test|validation) ;; *) echo "Unsupported protocol: $protocol" >&2; exit 2 ;; esac

  IFS=',' read -r -a requested_seeds <<< "${explicit_seeds:-$seed}"
  for seed in "${requested_seeds[@]}"; do
  checkpoint="$run_root/$variant/${dataset}_${protocol}_seed${seed}/best.pt"
  output_dir="$result_root/$variant/$dataset/$protocol"
  log_path="$log_root/${dataset}_${variant}_${protocol}.log"
  if [[ -n "$explicit_seeds" ]]; then
    output_dir="$output_dir/seed${seed}"
    log_path="$log_root/${dataset}_${variant}_${protocol}_seed${seed}.log"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing completed-run checkpoint: $checkpoint" >&2
    exit 3
  fi
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "Refusing to overwrite existing evaluation: $output_dir" >&2
    exit 3
  fi

  # noclobber reserves the log atomically before inference starts.
  (set -o noclobber
  "$python_bin" "$repo_dir/scripts/evaluate_checkpoint.py" \
    --dataset "$dataset" \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --require-noncausal \
    --device cuda:0 --batch-size 16 --num-workers 0 \
    >"$log_path" 2>&1)
  echo "Evaluated $dataset $variant $protocol -> $output_dir"
  done
done
