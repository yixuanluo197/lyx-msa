#!/usr/bin/env bash
set -euo pipefail
# Run through scripts/run_single.py for lock, resource admission, probe and one formal launch.
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -u /data2/yb/paper/lyx-msa/CMERC/experiments/hscl_ablation_validation/20261025T080937Z/scripts/watch_training.py --run-id MELD_wo_hscl_seed67137
