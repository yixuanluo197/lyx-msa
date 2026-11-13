# XP04 random-feature Pilot

Seed 1475, true Validation only. IEMOCAP: 527 utterances/12 dialogues; MELD: 1109/114.
Fresh XP04 `best.pt` inference matched the independently saved true Validation probabilities. Historical XP04 `validation/` prediction files were excluded because their rows are Test.

Gaussian uses `x + 0.05 × Std_train × ε`; dropout replaces 5% of selected coordinates with the corresponding Training mean. Exactly 5% of nonconstant dimensions (rounded to the nearest integer) are selected for each feature array. T1–T4 are perturbed together for Text. All conditions use 10 independent deterministic repeats. The original model and source features were read-only.

The model returns log-probabilities. Probabilities were computed with `exp`, with no second softmax.
Each target variant is evaluated in its original mixed Dialogue batch and retains the dialogue, labels, `qmask`, `umask`, and other utterances. Visual all-zero inputs were skipped for that modality and retained in the dataset.

## Verification

| Dataset | Max original probability error | Perturbed rows | Behavior rows | Skipped visual utterances |
|---|---:|---:|---:|---:|
| IEMOCAP | 2.98e-07 | 42160 | 4216 | 1 |
| MELD | 1.79e-07 | 88720 | 8872 | 95 |

See `verification.json`, per-dataset `validation_parity.json`, and `provenance/source_hashes.json`. This is a single-seed technical and calibration Pilot, not five-seed evidence.

## Execution provenance

Earlier local attempts were explicitly excluded: the first used a Bernoulli mask instead of exactly 5% coordinates; the second exceeded graph-index capacity when duplicating long dialogues; the third showed that duplicate-dialogue batches change XP04 predictions. This run perturbed one target at a time inside the original mixed-dialogue batch. MELD was restarted on idle local GPU 0 after two simultaneous jobs saturated local GPU 3; IEMOCAP continued on GPU 3. The only source edit during IEMOCAP execution widened the permitted local device assertion from GPU 3 to GPU 0/3 so MELD could run; no model, feature, perturbation, or analysis logic changed.
