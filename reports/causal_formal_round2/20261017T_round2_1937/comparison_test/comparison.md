# IEMOCAP test: Causal Baseline vs CMERC

Samples: 1623; dialogues: 31. Delta = CMERC − Baseline.

Classification and calibration use percentages; their deltas are percentage points. CEL uses summed log-probability differences.
`prefix_mean_v1` is a prefix smoothing view, not complete modality removal.

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 64.510166 | 65.249538 | +0.739372 |
| weighted_f1 | 63.958180 | 65.059289 | +1.101108 |
| macro_f1 | 62.150484 | 63.065101 | +0.914617 |
| f1.happy | 40.366972 | 45.323741 | +4.956769 |
| f1.sad | 68.837209 | 75.319149 | +6.481940 |
| f1.neutral | 64.677804 | 63.198960 | -1.478845 |
| f1.angry | 63.841808 | 55.762082 | -8.079726 |
| f1.excited | 73.981191 | 73.344652 | -0.636539 |
| f1.frustrated | 61.197917 | 65.442021 | +4.244104 |
| ece | 18.520027 | 14.550467 | -3.969560 |
| mce | 33.974854 | 24.416502 | -9.558352 |
| rmsce | 19.257726 | 15.614315 | -3.643410 |
| cel.total | 1489.332581 | 966.648720 | -522.683861 |
| cel.text | 269.851847 | 261.855853 | -7.995994 |
| cel.audio | 453.735504 | 284.526046 | -169.209458 |
| cel.visual | 16.826484 | 7.724921 | -9.101563 |
| cel.intra_context | 335.022504 | 180.315654 | -154.706850 |
| cel.inter_context | 413.896242 | 232.226246 | -181.669997 |

## Preregistered relative-position thirds

Position = (zero-based utterance index + 0.5) / complete dialogue length. Early [0, 1/3), middle [1/3, 2/3), late [2/3, 1]. Length-one dialogues belong to middle.

### early (540 utterances)

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 62.222222 | 61.111111 | -1.111111 |
| weighted_f1 | 61.341273 | 60.412819 | -0.928454 |
| macro_f1 | 53.803730 | 51.064509 | -2.739221 |
| f1.happy | 16.666667 | 27.027027 | +10.360360 |
| f1.sad | 57.142857 | 63.235294 | +6.092437 |
| f1.neutral | 61.309524 | 58.193980 | -3.115544 |
| f1.angry | 46.666667 | 17.142857 | -29.523810 |
| f1.excited | 77.366255 | 74.336283 | -3.029972 |
| f1.frustrated | 63.670412 | 66.451613 | +2.781201 |
| ece | 20.460856 | 17.498728 | -2.962127 |
| mce | 66.902366 | 31.365658 | -35.536708 |
| rmsce | 21.814851 | 18.626813 | -3.188038 |

### middle (543 utterances)

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 66.298343 | 67.403315 | +1.104972 |
| weighted_f1 | 66.200751 | 67.549553 | +1.348801 |
| macro_f1 | 64.226021 | 64.512003 | +0.285982 |
| f1.happy | 46.428571 | 45.945946 | -0.482625 |
| f1.sad | 72.992701 | 80.263158 | +7.270457 |
| f1.neutral | 70.250896 | 67.938931 | -2.311965 |
| f1.angry | 61.904762 | 52.083333 | -9.821429 |
| f1.excited | 75.121951 | 74.594595 | -0.527357 |
| f1.frustrated | 58.657244 | 66.246057 | +7.588813 |
| ece | 16.690405 | 13.442743 | -3.247662 |
| mce | 67.169711 | 68.458299 | +1.288588 |
| rmsce | 18.014213 | 15.964131 | -2.050083 |

### late (540 utterances)

| Metric | Baseline | CMERC | Delta |
|---|---:|---:|---:|
| accuracy | 65.000000 | 67.222222 | +2.222222 |
| weighted_f1 | 64.431818 | 67.067793 | +2.635975 |
| macro_f1 | 64.286284 | 67.007448 | +2.721164 |
| f1.happy | 47.368421 | 55.384615 | +8.016194 |
| f1.sad | 74.251497 | 80.219780 | +5.968283 |
| f1.neutral | 62.780269 | 64.423077 | +1.642808 |
| f1.angry | 71.428571 | 68.115942 | -3.312629 |
| f1.excited | 68.421053 | 70.786517 | +2.365464 |
| f1.frustrated | 61.467890 | 63.114754 | +1.646864 |
| ece | 19.239080 | 12.948422 | -6.290659 |
| mce | 32.147515 | 20.296182 | -11.851332 |
| rmsce | 19.858145 | 14.625342 | -5.232802 |

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
