#!/usr/bin/env bash
set -euo pipefail

seeds=""
split_seed=""
rng_policy=legacy
while (( $# )) && [[ $1 == --* ]]; do
  case "$1" in
    --seeds|--split-seed|--rng-policy)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --seeds) seeds=$2 ;;
        --split-seed) split_seed=$2 ;;
        --rng-policy) rng_policy=$2 ;;
      esac
      shift 2
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if (( $# == 0 || $# % 3 != 0 )); then
  echo "Usage: $0 [--seeds N,N,N] [--split-seed N] [--rng-policy legacy|continuous] DATASET VARIANT PROTOCOL [...]" >&2
  exit 2
fi
case "$rng_policy" in legacy|continuous) ;; *) echo "Invalid rng-policy" >&2; exit 2 ;; esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${CMERC_OUTPUT_DIR:-$repo_dir/runs/formal}
log_root=${CMERC_LOG_DIR:-$repo_dir/logs/training}
mkdir -p "$output_root" "$log_root"

while (( $# )); do
  dataset=$1
  variant=$2
  protocol=$3
  shift 3

  case "$dataset" in IEMOCAP) default_seed=1475 ;; MELD) default_seed=67137 ;; *) echo "Invalid dataset" >&2; exit 2 ;; esac
  IFS=',' read -r -a run_seeds <<< "${seeds:-$default_seed}"
  for seed in "${run_seeds[@]}"; do
    if [[ ! $seed =~ ^[0-9]+$ ]]; then echo "Invalid seed: $seed" >&2; exit 2; fi
    run_split_seed=${split_seed:-$seed}
    if [[ ! $run_split_seed =~ ^[0-9]+$ ]]; then echo "Invalid split-seed" >&2; exit 2; fi
    run_dir="$output_root/$variant/${dataset}_${protocol}_seed${seed}"
    log_path="$log_root/${dataset}_${variant}_${protocol}_seed${seed}.log"
    if [[ -e "$run_dir" || -e "$log_path" ]]; then
      echo "Refusing to overwrite existing run/log: $run_dir ; $log_path" >&2
      exit 3
    fi
    set -o noclobber
    exec {run_log}>"$log_path"
    set +o noclobber
    echo "Starting $dataset $variant $protocol seed=$seed split_seed=$run_split_seed; log=$log_path"
    bash "$repo_dir/scripts/run_training.sh" "$dataset" "$variant" "$protocol" \
      --seed "$seed" --split-seed "$run_split_seed" --rng-policy "$rng_policy" \
      --output-dir "$output_root/$variant" --num-workers 0 >&"$run_log" 2>&1
    exec {run_log}>&-
    echo "Completed $dataset $variant $protocol seed=$seed"
  done
done
