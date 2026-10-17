# IEMOCAP validation: Causal Baseline vs CMERC

Samples: 527; dialogues: 12. Delta = CMERC − Baseline.

Classification and calibration use percentages; their deltas are percentage points. CEL uses summed log-probability differences.
`prefix_mean_v1` is a prefix smoothing view, not complete modality removal.

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 62.998102 | 63.187856 | +0.189753 |
| weighted_f1 | 62.905474 | 63.393519 | +0.488045 |
| macro_f1 | 61.469316 | 62.706679 | +1.237363 |
| f1.happy | 58.571429 | 64.197531 | +5.626102 |
| f1.sad | 74.666667 | 74.509804 | -0.156863 |
| f1.neutral | 56.768559 | 53.266332 | -3.502227 |
| f1.angry | 78.260870 | 76.315789 | -1.945080 |
| f1.excited | 55.944056 | 57.333333 | +1.389277 |
| f1.frustrated | 44.604317 | 50.617284 | +6.012967 |
| ece | 21.367332 | 15.216119 | -6.151213 |
| mce | 28.359297 | 73.788282 | +45.428985 |
| rmsce | 22.212391 | 17.222625 | -4.989766 |
| cel.total | 489.961756 | 318.726103 | -171.235653 |
| cel.text | 96.638190 | 75.306723 | -21.331467 |
| cel.audio | 117.823623 | 122.475720 | +4.652098 |
| cel.visual | 6.192530 | 2.117760 | -4.074770 |
| cel.intra_context | 122.151204 | 39.130805 | -83.020399 |
| cel.inter_context | 147.156211 | 79.695095 | -67.461115 |

## Preregistered relative-position thirds

Position = (zero-based utterance index + 0.5) / complete dialogue length. Early [0, 1/3), middle [1/3, 2/3), late [2/3, 1]. Length-one dialogues belong to middle.

### early (177 utterances)

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 51.977401 | 56.497175 | +4.519774 |
| weighted_f1 | 52.124575 | 56.634339 | +4.509764 |
| macro_f1 | 51.916186 | 55.153001 | +3.236815 |
| f1.happy | 61.818182 | 75.000000 | +13.181818 |
| f1.sad | 70.000000 | 60.000000 | -10.000000 |
| f1.neutral | 50.909091 | 60.000000 | +9.090909 |
| f1.angry | 60.714286 | 50.000000 | -10.714286 |
| f1.excited | 45.833333 | 54.545455 | +8.712121 |
| f1.frustrated | 22.222222 | 31.372549 | +9.150327 |
| ece | 28.806972 | 18.918251 | -9.888721 |
| mce | 40.423717 | 73.788282 | +33.364566 |
| rmsce | 30.300719 | 21.386577 | -8.914142 |

### middle (173 utterances)

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 64.161850 | 63.583815 | -0.578035 |
| weighted_f1 | 63.589599 | 63.233200 | -0.356400 |
| macro_f1 | 63.005102 | 62.218605 | -0.786497 |
| f1.happy | 58.823529 | 51.282051 | -7.541478 |
| f1.sad | 79.310345 | 83.333333 | +4.022989 |
| f1.neutral | 61.538462 | 51.851852 | -9.686610 |
| f1.angry | 68.888889 | 71.428571 | +2.539683 |
| f1.excited | 73.469388 | 70.588235 | -2.881152 |
| f1.frustrated | 36.000000 | 44.827586 | +8.827586 |
| ece | 18.833274 | 15.717175 | -3.116099 |
| mce | 25.491246 | 30.952606 | +5.461360 |
| rmsce | 20.515495 | 18.637629 | -1.877867 |

### late (177 utterances)

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 72.881356 | 69.491525 | -3.389831 |
| weighted_f1 | 72.863976 | 68.681185 | -4.182790 |
| macro_f1 | 68.561294 | 64.902482 | -3.658812 |
| f1.happy | 54.901961 | 61.016949 | +6.114988 |
| f1.sad | 73.076923 | 75.471698 | +2.394775 |
| f1.neutral | 62.962963 | 40.000000 | -22.962963 |
| f1.angry | 95.327103 | 92.000000 | -3.327103 |
| f1.excited | 47.826087 | 45.454545 | -2.371542 |
| f1.frustrated | 77.272727 | 75.471698 | -1.801029 |
| ece | 18.390918 | 14.810954 | -3.579964 |
| mce | 60.792953 | 23.268767 | -37.524186 |
| rmsce | 19.998344 | 16.444992 | -3.553352 |

## Training resource summary

| Run | Epochs | Best epoch | Validation W-F1 | Seconds | Peak allocated GPU MB |
|---|---:|---:|---:|---:|---:|
| baseline | 80 | 57 | 62.910000 | 215.83 | 3424.01 |
| cmerc | 80 | 48 | 63.390000 | 1502.78 | 4065.14 |

Resource differences (CMERC − Baseline):

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| elapsed_seconds | 215.830000 | 1502.780000 | +1286.950000 |
| max_gpu_memory_mb | 3424.010000 | 4065.140000 | +641.130000 |

Full per-epoch loss and convergence curves are retained in the companion JSON. Recorded loss components are raw terms; coefficient-weighted optimization contributions must be read from the training diagnostics.
