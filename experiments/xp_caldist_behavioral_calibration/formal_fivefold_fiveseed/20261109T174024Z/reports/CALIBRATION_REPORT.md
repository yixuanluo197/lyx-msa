# Calibration Report

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

All 50 Calibration jobs fitted E0–E3 using only the corresponding Calibration role. Full probability replay, Grouped OOF coverage, unknown-mask exclusion, positive temperatures, and argmax invariance passed.

## IEMOCAP Calibration OOF mean

| method | nll | brier | ece15 | accuracy |
|---|---:|---:|---:|---:|
| E0 | 0.910854 | 0.482383 | 0.097971 | 0.654907 |
| E1 | 0.859424 | 0.469072 | 0.046267 | 0.654907 |
| E2 | 0.850022 | 0.464920 | 0.042451 | 0.654907 |
| E3 | 0.859715 | 0.466174 | 0.046493 | 0.654907 |

E2 minus E1: {'nll': -0.00940204524696242, 'brier': -0.004152495565240444, 'ece15': -0.003815575597681531}.

## MELD Calibration OOF mean

| method | nll | brier | ece15 | accuracy |
|---|---:|---:|---:|---:|
| E0 | 1.085430 | 0.501360 | 0.255187 | 0.708574 |
| E1 | 0.883899 | 0.412449 | 0.033507 | 0.708574 |
| E2 | 0.877696 | 0.410143 | 0.029266 | 0.708574 |
| E3 | 0.880181 | 0.410533 | 0.028651 | 0.708574 |

E2 minus E1: {'nll': -0.006202906228211069, 'brier': -0.002305816619012524, 'ece15': -0.0042411428073492775}.

Hierarchical TS shows lower internal OOF NLL/Brier/ECE15 than Global TS on average for both datasets. This does not establish an Evaluation improvement.
