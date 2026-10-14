#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 || $# % 3 != 0 )); then
  echo "Usage: $0 DATASET VARIANT PROTOCOL [DATASET VARIANT PROTOCOL ...]" >&2
  exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${CMERC_OUTPUT_DIR:-$repo_dir/runs/formal}
log_root=${CMERC_LOG_DIR:-$repo_dir/logs/training}
mkdir -p "$output_root" "$log_root"

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

  run_dir="$output_root/$variant/${dataset}_${protocol}_seed${seed}"
  log_path="$log_root/${dataset}_${variant}_${protocol}.log"
  if [[ -e "$run_dir/latest.pt" || -e "$run_dir/metrics.jsonl" ]]; then
    echo "Refusing to overwrite an existing run: $run_dir" >&2
    exit 3
  fi

  echo "Starting $dataset $variant $protocol; log=$log_path"
  CMERC_OUTPUT_DIR="$output_root" \
    bash "$repo_dir/scripts/run_training.sh" "$dataset" "$variant" "$protocol" \
    --num-workers 0 >"$log_path" 2>&1
  echo "Completed $dataset $variant $protocol"
done
