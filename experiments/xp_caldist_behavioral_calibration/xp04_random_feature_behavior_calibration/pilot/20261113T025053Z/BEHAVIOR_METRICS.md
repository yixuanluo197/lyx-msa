# Behavioral metrics

Original CaliDist equations were checked against [the ICML paper](https://proceedings.mlr.press/v306/jawad26a.html) (Section 3, Fig. 2) and [official code](https://github.com/anas-jawad/CaliDist/blob/aa490893831c5c27a891e8d28c9da38b7e7f6057/src/calidist/evaluation/calidist.py):
`μ = mean(1[pred_r ≠ pred_0] × conf_r)`, `δ = |conf_0 − mean(conf_r)|`, and `λ_raw = (1−μ)/(δ+1e−10)`.
The equations are original; random perturbation of CMERC features is this Pilot’s adaptation of the paper’s semantic distractors. JS is an added diagnostic and does not enter the original Reliability Score.

| Dataset | Corruption | Modality | Mean μ | Mean δ | Mean JS | Mean flip rate | Mean confidence change |
|---|---|---|---:|---:|---:|---:|---:|
| IEMOCAP | dropout | audio | 0.00478256 | 0.00617047 | 0.000301234 | 0.0110057 | -0.00232658 |
| IEMOCAP | dropout | joint | 0.00591236 | 0.00629748 | 0.000311343 | 0.0134725 | -0.00278273 |
| IEMOCAP | dropout | text | 0.00147091 | 0.00143901 | 8.80464e-06 | 0.00322581 | -0.000444095 |
| IEMOCAP | dropout | visual | 0.000403342 | 0.000251656 | 4.29092e-07 | 0.000948767 | -5.4387e-05 |
| IEMOCAP | gaussian | audio | 0.00110583 | 0.00067871 | 1.36565e-05 | 0.00265655 | -2.47194e-05 |
| IEMOCAP | gaussian | joint | 0.00110057 | 0.000696731 | 1.39614e-05 | 0.00265655 | -1.29603e-05 |
| IEMOCAP | gaussian | text | 0.00048376 | 0.000104557 | 2.75016e-07 | 0.00113852 | 1.12279e-05 |
| IEMOCAP | gaussian | visual | 0.000322466 | 3.24154e-05 | 1.78506e-08 | 0.000759013 | 2.31425e-06 |
| MELD | dropout | audio | 0.00119349 | 0.0013899 | 1.4491e-05 | 0.00396754 | -0.000365964 |
| MELD | dropout | joint | 0.00327596 | 0.00479706 | 8.84421e-05 | 0.0110911 | -0.00383748 |
| MELD | dropout | text | 0.00267343 | 0.00438384 | 7.07409e-05 | 0.0091073 | -0.00348844 |
| MELD | dropout | visual | 0.000590268 | 0.000382366 | 1.93457e-06 | 0.00198377 | 2.03731e-05 |
| MELD | gaussian | audio | 0.000299445 | 0.000145847 | 4.28296e-07 | 0.00108206 | -4.6936e-06 |
| MELD | gaussian | joint | 0.000724231 | 0.000375784 | 2.71881e-06 | 0.0025248 | -5.64674e-05 |
| MELD | gaussian | text | 0.000650313 | 0.000334134 | 2.21859e-06 | 0.00225428 | -5.07258e-05 |
| MELD | gaussian | visual | 0.000149545 | 5.06748e-05 | 8.11351e-08 | 0.000450857 | -1.23036e-06 |

A low flip rate does not imply correctness; perturbation sensitivity is diagnostic only. The full 10-repeat probabilities and per-utterance metrics are in the CSV files.
