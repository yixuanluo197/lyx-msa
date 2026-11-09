# Fold-local Difficulty Report

Both final Teachers were reloaded strictly into `Model`, modality-removal `Model`, `Model_Intra`, and `Model_Inter`; all missing/unexpected key lists are empty. Difficulty generation used FP32, `eval()`, `no_grad()`, and Batch=1. Only approved Fold 1 Training dialogues were loaded.

| Dataset | Dialogues | Utterances | Masked | Difficulty min | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 84 | 4,216 | 23 | 0.096888 | 0.349933 | 0.841138 |
| MELD | 852 | 8,176 | 0 | 0.127362 | 0.346920 | 0.686961 |

For every dialogue, the JSON records valid-label count, total length, speaker term, five view gaps, total gap, primary valid-count difficulty, and total-length sensitivity control. The primary formula is `(total_gap + speaker_term) / (valid_label_count + speaker_term)`. No dialogue has zero valid labels, NaN, or Inf.

The six-view NPZ archives store complete normalized class distributions for full, text, audio, visual, intra, and inter views, plus dialogue offsets, labels, and supervision masks. Independent verification confirms every probability row sums to one within `atol=1e-6, rtol=1e-5`.

Difficulty IDs exactly equal Training IDs and are disjoint from Development, Calibration, and Evaluation. The actual curriculum selection audit passes for ratios 0.15, 0.25, 0.50, 0.75, and 1.00, selecting by difficulty and restoring approved source order without introducing any non-Training dialogue.
