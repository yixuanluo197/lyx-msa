#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 IEMOCAP|MELD baseline|cmerc official_test|validation [train_cm.py args...]" >&2
  exit 2
fi

dataset=$1
variant=$2
protocol=$3
shift 3

case "$dataset" in
  IEMOCAP)
    dataset_args=(--dropout 0.5 --epochs 80 --num_K 4 --seed 1475)
    cmerc_args=(--rank_coff 0.005 --mscl_coff 0.05 --cscl_coff 0.05 --epoch_ratio 0.15)
    ;;
  MELD)
    dataset_args=(--dropout 0.4 --epochs 40 --num_K 3 --seed 67137)
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
  --output-dir "$output_root/$variant" \
  "${dataset_args[@]}" \
  "${variant_args[@]}" \
  "$@"
