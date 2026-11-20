# Pilot IO amendment and remaining runtime steps

This amendment changes JSONL physical record boundaries and completion provenance only. Original text bytes, encoders, scientific protocol, perturbations, thresholds and statistical analysis remain unchanged. `SMOKE_COMPLETION_PROVENANCE_ERRATA.json` corrects historical Train-smoke provenance without modifying its reports, cached features or numerical results. New production caches bind the current source hashes and actual CMERC `completion.json`.

Synthetic regression evidence: `jsonl_lf_io_fix_verification.json` and `jsonl_lf_io_regression.py`. Both intended environments pass Unicode JSONL preservation and completion-path/input-guard checks.

The actual frozen selections are `results/pilot/{DS}/sample_manifest.csv`; media/reviews/transformation manifests are under `local_only/pilot/{DS}`. Both frozen CMERC checkpoints are `local_only/data/{DS}/runs/cmerc/best.pt`; baseline locks are `configs/BASELINE_LOCK_{DS}.json`. Text models are `local_only/text_models/{DS}`. No selection rerun is needed.

Use these commands from the project root **after** the attributed context reviews are complete. `REVIEW` must point to the completed local JSONL containing all 480 fixed text variants, with accurate reviewer/method/status. Missing or failed variants remain in the manifest; do not remove them. Choose a healthy GPU UUID and retain the shell launch command/log. The commands below are a handoff, and were not executed by this IO amendment.

```bash
D=/data2/yb/paper/lyx-msa/experiments/xp_caldist_behavioral_calibration/plan_b_raw_input_pilot/20261120T143131Z
DS=IEMOCAP
REVIEW="$D/local_only/pilot/$DS/text_reviews.jsonl"
GPU_UUID='<assigned healthy GPU UUID>'

/home/yangbin/miniconda3/envs/TEXT_repro/bin/python "$D/scripts/raw_pilot.py" validate-text \
  --dataset "$DS" --dataset-dir "$D/local_only/data/$DS" \
  --text-features "$D/local_only/features/$DS/text_features.pt" \
  --audio-features "$D/local_only/features/$DS/audio_features.pt" \
  --checkpoint "$D/local_only/data/$DS/runs/cmerc/best.pt" \
  --baseline-lock "$D/configs/BASELINE_LOCK_$DS.json" \
  --output "$D/results/pilot/$DS" --local-dir "$D/local_only/pilot/$DS" \
  --text-model "$D/local_only/text_models/$DS" --text-review "$REVIEW"

CUDA_VISIBLE_DEVICES="$GPU_UUID" /home/yangbin/miniconda3/envs/TEXT_repro/bin/python "$D/scripts/extract_pilot_features.py" \
  --mode pilot --dataset "$DS" --dataset-dir "$D/local_only/data/$DS" \
  --text-features "$D/local_only/features/$DS/text_features.pt" \
  --audio-features "$D/local_only/features/$DS/audio_features.pt" \
  --checkpoint "$D/local_only/data/$DS/runs/cmerc/best.pt" \
  --text-model "$D/local_only/text_models/$DS" \
  --output "$D/local_only/pilot/$DS/encoder_features.pt" \
  --local-dir "$D/local_only/pilot/$DS" --pilot-output "$D/results/pilot/$DS" \
  --baseline-lock "$D/configs/BASELINE_LOCK_$DS.json" --text-review "$REVIEW" \
  --audio-review "$D/local_only/pilot/$DS/audio_label_review.csv" --device cuda:0

CUDA_VISIBLE_DEVICES="$GPU_UUID" /home/yangbin/miniconda3/envs/multimodalerc310/bin/python "$D/scripts/raw_pilot.py" infer \
  --dataset "$DS" --dataset-dir "$D/local_only/data/$DS" \
  --text-features "$D/local_only/features/$DS/text_features.pt" \
  --audio-features "$D/local_only/features/$DS/audio_features.pt" \
  --checkpoint "$D/local_only/data/$DS/runs/cmerc/best.pt" \
  --baseline-lock "$D/configs/BASELINE_LOCK_$DS.json" \
  --output "$D/results/pilot/$DS" --local-dir "$D/local_only/pilot/$DS" \
  --precomputed-features "$D/local_only/pilot/$DS/encoder_features.pt" \
  --modality both --device cuda:0
```

Repeat the three steps with `DS=MELD` and its completed review path. Audio DSP manifests already exist and are consumed as fixed assets; this handoff does not regenerate them. Retain `PENDING`/`UNCERTAIN` audio emotion reviews unless an actual attributed listening review is available. Signal QC alone permits only the separate exploratory analysis.

After both datasets have inference CSVs, run the frozen analysis once. Use either each dataset's `both` file or its two separate modality files, never duplicate the same variants across inputs.

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python "$D/scripts/pilot_analysis.py" \
  --predictions "$D/results/pilot/IEMOCAP/perturbed_predictions_both.csv" \
                "$D/results/pilot/MELD/perturbed_predictions_both.csv" \
  --output "$D/results/pilot_analysis" --bootstrap 2000 --seed 20261120
```

The analysis writes complete-family diagnostics, strict attributed-label-review main tables, separate signal-QC-only audio exploration, failed/incomplete records, correct/error comparisons, and exact Dialogue × confidence-bin comparisons with support diagnostics. An empty strict audio main table reflects missing label confirmation; it is not evidence of zero audio response. New feature extraction must finish before CMERC inference, and inference must finish before analysis/reporting. Do not change source files after production caches bind their hashes.
