# Auxiliary-view core diagnostics

prefix_mean_v1 formula max_abs_diff=2.38418579e-07; inverse max_abs_diff=4.76837158e-07.
`m_t=(x_1+...+x_t)/t`, `x_t=t*m_t-(t-1)*m_(t-1)` (one-based t). The prefix sequence is invertible up to floating point; prefix_mean_v1 is a prefix smoothing view, not complete modality absence. Only valid utterances contribute; padding is excluded and output padding is zero.

60 same-weight eval probability-vector checks pass: 12 backbone/av/residue configurations × three prefix-mean views and two GNN-edge views. All use strict state_dict loading. The first utterance has no prior temporal edge and its prefix mean equals itself. Agreement compares all four class probabilities, not argmax.

24 Model_Intra/Inter historical-gradient checks pass. Graph source/target incidence is identical across auxiliary context views; only same-modality temporal GNN edges are removed according to speaker. Both encoder history and hypergraph history remain. These are GNN-edge ablations, not complete context removal.

No strict modality-absence policy is added. Train-function synchronization, dropout and weight lag are reported separately in AUXILIARY_VIEW_DIAGNOSTIC.md / TRAINING_INTEGRATION.md.

Full numeric first-utterance vectors-comparison errors and historical gradient maxima are in CAUSALITY_REGRESSION.json.
