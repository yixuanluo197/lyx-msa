# Independent numerical verification

The verifier imports no calibration fitting, prediction, or candidate-selection functions. It independently rebuilds all design matrices, train-only scaling, serialized predictions, inner pooled Brier selection, offset objectives/gradients, metrics, and shared-dialogue bootstrap.

## Verified outputs

- 8,180 original utterance records; 73,620 OOF records; 90 per-seed metric rows.
- 85 outer folds; 425 inner folds; 510 new outer fits and 170 historical E1/L0 reconstructions.
- 21,420 serialized fit reconstructions; 176 controlled static fallbacks.
- 60 paired-bootstrap rows and 270 literal dialogue-resampling metric checks.
- Maximum outer probability reconstruction error: 1.1102230246251565e-16.
- 36 constant-column cases and 1,692 held values differing from constant training columns checked; category vocabulary uses original predicted classes only.

## L-BFGS precision audit

The first verifier attempted an extra mean-gradient threshold of 2e-5. It failed because sklearn L-BFGS can terminate on relative objective change, without satisfying that extra gradient threshold. The captured first traceback is preserved in `../logs/calibration_verification_initial.log`; that file explicitly records reconstruction from the captured original log output after the active log was reused.

The registered model and all saved OOF predictions remain unchanged. Independent exact-Hessian Newton optimization was used only as a numerical sensitivity audit for the same convex logistic objective. Across 8,076 unique serialized LR models, 40 exceeded the diagnostic gradient threshold (95 reconstruction calls). Maximum mean-gradient was 4.2072349485437754e-05. Maximum mean-objective improvement under refinement was 3.6559856031459503e-08.

Across all checked LR models, maximum training probability change was 0.0003075714660988327 and maximum held probability change was 0.0009730094365710462. These are conservative maxima across candidates, including models never selected. Offset refinements are assessed for the static base only; registered correction weights are not replaced or retrained.

For the primary S_DA-minus-S comparison, all six dataset/metric mean directions and all improved-seed counts were unchanged by independently refining the selected joint LR fits. The largest per-seed primary metric-difference movement was 1.487753912621237e-06 (MELD binary NLL). Thus the numerical refinement does not alter the reported failure to meet the primary decision criteria. Detailed values are saved in `independent_verification.json`.

Refinement predictions are neither substituted into reported performance nor treated as an additional candidate model. Original OOF, selection, metric, and bootstrap reconstruction checks were retained in full.
