# Confidence distribution diagnostic

Seed 1475, true Validation only; no model or perturbation rerun.
Top-label confidence and binary correctness are analyzed. Sample standard deviations use ddof=1.
Dialogue-cluster percentile intervals use 2,000 resamples with fixed RNG seed 1475; they describe uncertainty across these Validation dialogues, not five-seed stability.

## IEMOCAP

n=527, dialogues=12, accuracy=0.660, mean E0 confidence=0.728, gap (accuracy−confidence)=-0.068; dialogue bootstrap 95% CI [-0.170, 0.022].
Correct: n=348, mean=0.760, median=0.752, SD=0.179, Q25/Q75/Q90=0.618/0.930/0.987, C<0.5=0.086.
Wrong: n=179, mean=0.666, median=0.633, SD=0.190, Q25/Q75/Q90=0.508/0.828/0.969, C<0.5=0.218.
Confidence separates correctness to some extent: mean difference=0.093; ROC AUC=0.645.
E0 occupied bins: 7; underconfident bins: 2 ([0.3, 0.4), [0.7, 0.8)); overconfident bins: 5 ([0.4, 0.5), [0.5, 0.6), [0.6, 0.7), [0.8, 0.9), [0.9, 1.0]). Sparse bins should not be overinterpreted.

| E0 interval | n | Mean C | Accuracy | Gap |
|---|---:|---:|---:|---:|
| [0.3, 0.4) | 14 | 0.373 | 0.429 | 0.055 |
| [0.4, 0.5) | 55 | 0.455 | 0.436 | -0.018 |
| [0.5, 0.6) | 87 | 0.549 | 0.529 | -0.020 |
| [0.6, 0.7) | 96 | 0.653 | 0.646 | -0.007 |
| [0.7, 0.8) | 71 | 0.748 | 0.789 | 0.041 |
| [0.8, 0.9) | 65 | 0.854 | 0.692 | -0.161 |
| [0.9, 1.0] | 139 | 0.966 | 0.784 | -0.182 |

See `figures/correct_wrong_distributions.pdf` and `figures/reliability_diagram.pdf`.

## MELD

n=1109, dialogues=114, accuracy=0.669, mean E0 confidence=0.468, gap (accuracy−confidence)=0.201; dialogue bootstrap 95% CI [0.172, 0.229].
Correct: n=742, mean=0.501, median=0.511, SD=0.118, Q25/Q75/Q90=0.410/0.590/0.650, C<0.5=0.469.
Wrong: n=367, mean=0.402, median=0.387, SD=0.111, Q25/Q75/Q90=0.323/0.481/0.568, C<0.5=0.807.
Confidence separates correctness to some extent: mean difference=0.099; ROC AUC=0.730.
E0 occupied bins: 8; underconfident bins: 7 ([0.2, 0.3), [0.3, 0.4), [0.4, 0.5), [0.5, 0.6), [0.6, 0.7), [0.7, 0.8), [0.8, 0.9)); overconfident bins: 1 ([0.1, 0.2)). Sparse bins should not be overinterpreted.

| E0 interval | n | Mean C | Accuracy | Gap |
|---|---:|---:|---:|---:|
| [0.1, 0.2) | 2 | 0.191 | 0.000 | -0.191 |
| [0.2, 0.3) | 102 | 0.264 | 0.402 | 0.138 |
| [0.3, 0.4) | 268 | 0.354 | 0.466 | 0.113 |
| [0.4, 0.5) | 272 | 0.450 | 0.669 | 0.219 |
| [0.5, 0.6) | 277 | 0.550 | 0.819 | 0.270 |
| [0.6, 0.7) | 160 | 0.640 | 0.881 | 0.241 |
| [0.7, 0.8) | 27 | 0.721 | 0.926 | 0.205 |
| [0.8, 0.9) | 1 | 0.804 | 1.000 | 0.196 |

See `figures/correct_wrong_distributions.pdf` and `figures/reliability_diagram.pdf`.
