# Top-label OOF calibration comparison

All ECE15, Brier, and NLL values use binary prediction correctness; they are not multiclass scores.
E1 fits an L2 logistic mapping from original confidence in each training fold. E2 follows the [official CaliDist equations](https://github.com/anas-jawad/CaliDist/blob/aa490893831c5c27a891e8d28c9da38b7e7f6057/src/calidist/evaluation/calidist.py): training-fold λ min/max, 100×100 α/β grid minimizing training-fold Brier, and `C_cal = C × sigmoid(β(λ−α))`. The held-out dialogue does not contribute labels or λ bounds. No temperature scaling was used.

| Dataset | Arm | Corruption | Modality | ECE15 | Binary Brier | Binary NLL | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP | E0 | none | none | 0.10497 | 0.22230 | 0.71951 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E1 | none | none | 0.08977 | 0.22121 | 0.63706 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | dropout | audio | 0.10578 | 0.22191 | 0.67334 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | dropout | joint | 0.08905 | 0.22048 | 0.65225 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | dropout | text | 0.10492 | 0.22106 | 0.65638 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | dropout | visual | 0.08798 | 0.22068 | 0.64822 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | gaussian | audio | 0.10710 | 0.22065 | 0.65029 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | gaussian | joint | 0.09293 | 0.22105 | 0.65427 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | gaussian | text | 0.10054 | 0.22070 | 0.64775 | 0.66034 | 0.65021 | 0.64489 |
| IEMOCAP | E2 | gaussian | visual | 0.08834 | 0.22077 | 0.64842 | 0.66034 | 0.65021 | 0.64489 |
| MELD | E0 | none | none | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E1 | none | none | 0.04323 | 0.19276 | 0.56858 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | dropout | audio | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | dropout | joint | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | dropout | text | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | dropout | visual | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | gaussian | audio | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | gaussian | joint | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | gaussian | text | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |
| MELD | E2 | gaussian | visual | 0.20136 | 0.23342 | 0.65945 | 0.66907 | 0.66104 | 0.56381 |

E2 can only lower the original confidence. Fold membership and fitted parameters are saved in each dataset’s `fold_parameters.json`; every held-out dialogue is disjoint from its training dialogues.
