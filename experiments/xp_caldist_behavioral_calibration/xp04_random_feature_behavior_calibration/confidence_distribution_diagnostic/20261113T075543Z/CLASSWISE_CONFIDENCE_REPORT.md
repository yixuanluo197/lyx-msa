# Classwise confidence diagnostic

Class IDs follow the frozen XP04 evaluation code. `true` condition accuracy is recall; `predicted` condition accuracy is precision. A true-class gap is descriptive and cannot be used at inference. Classes with n<20 are flagged; classwise gaps are exploratory, without multiplicity-adjusted inference.

## IEMOCAP: predicted-class calibration gaps

Positive gap means observed precision exceeds mean top-label confidence within this predicted class. Separation is mean C(correct)−mean C(wrong).

| Class | n | Correct | Mean C | Precision | Gap | Correct C | Wrong C | Separation | Small n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| neutral | 73 | 52 | 0.592 | 0.712 | 0.120 | 0.625 | 0.512 | 0.112 | no |
| frustrated | 58 | 34 | 0.595 | 0.586 | -0.009 | 0.623 | 0.556 | 0.067 | no |
| angry | 142 | 109 | 0.811 | 0.768 | -0.044 | 0.848 | 0.691 | 0.157 | no |
| happy | 98 | 54 | 0.635 | 0.551 | -0.084 | 0.658 | 0.607 | 0.051 | no |
| excited | 69 | 39 | 0.749 | 0.565 | -0.184 | 0.795 | 0.690 | 0.105 | no |
| sad | 87 | 60 | 0.882 | 0.690 | -0.192 | 0.863 | 0.925 | -0.062 | no |

### IEMOCAP: true-class descriptive breakdown

Here conditional accuracy is recall; the gap is not a deployable top-label calibration target.

| Class | n | Correct | Mean C | Recall | Gap | Correct C | Wrong C | Small n |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| happy | 81 | 54 | 0.675 | 0.667 | -0.008 | 0.658 | 0.709 | no |
| sad | 66 | 60 | 0.849 | 0.909 | 0.060 | 0.863 | 0.711 | no |
| neutral | 123 | 52 | 0.659 | 0.423 | -0.236 | 0.625 | 0.684 | no |
| angry | 130 | 109 | 0.794 | 0.838 | 0.044 | 0.848 | 0.516 | no |
| excited | 58 | 39 | 0.747 | 0.672 | -0.074 | 0.795 | 0.648 | no |
| frustrated | 69 | 34 | 0.657 | 0.493 | -0.164 | 0.623 | 0.690 | no |

See `classwise_confidence.csv` and `figures/classwise_mean_confidence.pdf`.

## MELD: predicted-class calibration gaps

Positive gap means observed precision exceeds mean top-label confidence within this predicted class. Separation is mean C(correct)−mean C(wrong).

| Class | n | Correct | Mean C | Precision | Gap | Correct C | Wrong C | Separation | Small n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| neutral | 491 | 375 | 0.497 | 0.764 | 0.267 | 0.513 | 0.445 | 0.068 | no |
| sadness | 57 | 34 | 0.408 | 0.596 | 0.188 | 0.442 | 0.358 | 0.085 | no |
| anger | 168 | 100 | 0.419 | 0.595 | 0.177 | 0.468 | 0.345 | 0.123 | no |
| surprise | 167 | 104 | 0.475 | 0.623 | 0.147 | 0.506 | 0.424 | 0.082 | no |
| fear | 33 | 15 | 0.327 | 0.455 | 0.127 | 0.361 | 0.299 | 0.061 | no |
| joy | 175 | 105 | 0.481 | 0.600 | 0.119 | 0.532 | 0.405 | 0.127 | no |
| disgust | 18 | 9 | 0.422 | 0.500 | 0.078 | 0.419 | 0.425 | -0.006 | yes |

### MELD: true-class descriptive breakdown

Here conditional accuracy is recall; the gap is not a deployable top-label calibration target.

| Class | n | Correct | Mean C | Recall | Gap | Correct C | Wrong C | Small n |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| neutral | 470 | 375 | 0.485 | 0.798 | 0.313 | 0.513 | 0.377 | no |
| surprise | 150 | 104 | 0.472 | 0.693 | 0.222 | 0.506 | 0.393 | no |
| fear | 40 | 15 | 0.370 | 0.375 | 0.005 | 0.361 | 0.375 | no |
| sadness | 111 | 34 | 0.418 | 0.306 | -0.112 | 0.442 | 0.407 | no |
| joy | 163 | 105 | 0.495 | 0.644 | 0.149 | 0.532 | 0.429 | no |
| disgust | 22 | 9 | 0.411 | 0.409 | -0.002 | 0.419 | 0.405 | no |
| anger | 153 | 100 | 0.456 | 0.654 | 0.198 | 0.468 | 0.432 | no |

See `classwise_confidence.csv` and `figures/classwise_mean_confidence.pdf`.
