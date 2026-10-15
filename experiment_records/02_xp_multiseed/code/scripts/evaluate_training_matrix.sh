#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 || $# % 3 != 0 )); then
  echo "Usage: $0 DATASET VARIANT PROTOCOL [DATASET VARIANT PROTOCOL ...]" >&2
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

  checkpoint="$run_root/$variant/${dataset}_${protocol}_seed${seed}/best.pt"
  output_dir="$result_root/$variant/$dataset/$protocol"
  log_path="$log_root/${dataset}_${variant}_${protocol}.log"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing completed-run checkpoint: $checkpoint" >&2
    exit 3
  fi
  if [[ -e "$output_dir/metrics.json" ]]; then
    echo "Refusing to overwrite existing evaluation: $output_dir" >&2
    exit 3
  fi

  "$python_bin" "$repo_dir/scripts/evaluate_checkpoint.py" \
    --dataset "$dataset" \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --device cuda:0 --batch-size 16 --num-workers 0 \
    >"$log_path" 2>&1
  echo "Evaluated $dataset $variant $protocol -> $output_dir"
done
