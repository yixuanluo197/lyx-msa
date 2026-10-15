#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 IEMOCAP|MELD baseline|cmerc official_test|validation [--seed N] [--split-seed N] [--rng-policy legacy|continuous] [train_cm.py args...]" >&2
  exit 2
fi

dataset=$1
variant=$2
protocol=$3
shift 3

case "$dataset" in
  IEMOCAP)
    default_seed=1475
    dataset_args=(--dropout 0.5 --epochs 80 --num_K 4)
    cmerc_args=(--rank_coff 0.005 --mscl_coff 0.05 --cscl_coff 0.05 --epoch_ratio 0.15)
    ;;
  MELD)
    default_seed=67137
    dataset_args=(--dropout 0.4 --epochs 40 --num_K 3)
    cmerc_args=(--rank_coff 0.002 --mscl_coff 0.15 --cscl_coff 0.15 --epoch_ratio 0.4)
    ;;
  *)
    echo "Unsupported dataset: $dataset" >&2
    exit 2
    ;;
esac

case "$variant" in
  baseline)
    variant_args=()
    ;;
  cmerc)
    variant_args=(--calibrate --contrastlearning --courselearning --scheduler_steps 1 "${cmerc_args[@]}")
    ;;
  *)
    echo "Unsupported variant: $variant" >&2
    exit 2
    ;;
esac

case "$protocol" in
  official_test|validation) ;;
  *)
    echo "Unsupported selection protocol: $protocol" >&2
    exit 2
    ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${CMERC_PYTHON:-python}
output_root=${CMERC_OUTPUT_DIR:-$repo_dir/runs/train_cm}

seed=$default_seed
split_seed=$default_seed
rng_policy=legacy
explicit_split_seed=false
extra_args=()
while (( $# )); do
  case "$1" in
    --seed|--split-seed|--rng-policy)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --seed) seed=$2 ;;
        --split-seed) split_seed=$2; explicit_split_seed=true ;;
        --rng-policy) rng_policy=$2 ;;
      esac
      shift 2
      ;;
    --seed=*|--split-seed=*|--rng-policy=*)
      case "$1" in
        --seed=*) seed=${1#*=} ;;
        --split-seed=*) split_seed=${1#*=}; explicit_split_seed=true ;;
        --rng-policy=*) rng_policy=${1#*=} ;;
      esac
      shift
      ;;
    --causal|--causal=*)
      echo "This isolated experiment entry is for original non-causal CMERC only" >&2
      exit 2
      ;;
    *) extra_args+=("$1"); shift ;;
  esac
done
if ! $explicit_split_seed; then split_seed=$seed; fi
if [[ ! $seed =~ ^[0-9]+$ || ! $split_seed =~ ^[0-9]+$ ]]; then
  echo "seed and split-seed must be nonnegative integers" >&2; exit 2
fi
case "$rng_policy" in legacy|continuous) ;; *) echo "Invalid rng-policy: $rng_policy" >&2; exit 2 ;; esac
run_dir="$output_root/$variant"

cd "$repo_dir"
exec "$python_bin" -u train_cm.py \
  --base-model GRU \
  --lr 0.0001 \
  --batch-size 16 \
  --graph_type hyper \
  --graph_construct direct \
  --multi_modal \
  --mm_fusion_mthd concat_DHT \
  --modals avl \
  --Dataset "$dataset" \
  --norm BN \
  --num_L 3 \
  --selection-protocol "$protocol" \
  --output-dir "$run_dir" \
  --seed "$seed" \
  --split-seed "$split_seed" \
  --rng-policy "$rng_policy" \
  "${dataset_args[@]}" \
  "${variant_args[@]}" \
  "${extra_args[@]}"
