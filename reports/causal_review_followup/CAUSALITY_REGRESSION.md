# Causal CMERC regression evidence

Status: **passed**. Real model configurations: 144/144.
Device `cpu`, float32, seed 20261016, atol 2e-05, rtol 1e-05.
Relative differences use denominator max(|reference|, 1e-8); acceptance uses PyTorch combined atol + rtol rule.

Each row executes Model / Model_Intra / Model_Inter and their real HyperGCN. All GRU/LSTM/Transformer × audio/visual encoder × residue × train/eval × IEMOCAP/MELD settings use num_L=2, num_K=3. Lengths are 4, 2, 1 with repeated and differing speakers. No trained weights or datasets are loaded.

Tests compare full log-probability vectors and graph hidden representations for individually perturbed future text/audio/visual/speaker and their joint perturbation. Every nonempty prefix is inferred separately (suffix addition/deletion), then dialogue reorder, other-dialogue changes, single-dialogue inference, finite padding values, and longer right padding are compared. Scalar gradient checks sample times 0 and 1, categories 0 and 3 and one hidden coordinate; every tested scalar has exactly zero future feature gradients and nonzero visible-feature influence. This requirement does not apply to a loss containing future training examples.

Executed normalization hooks confirm feature-only LayerNorm; hooks inspect actual causal and padding masks at every executed Transformer. Unused legacy modules are not inferred to execute. Graph source membership and target delivery are checked separately; incoming-degree prefix invariance is exercised numerically.

| Configuration | mode | max absolute difference | max relative difference | future gradient max |
| --- | --- | ---: | ---: | ---: |
| Model/GRU/False/False/IEMOCAP | eval | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model/GRU/False/False/MELD | eval | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model/GRU/False/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model/GRU/False/False/MELD | train(dropout=0) | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model/GRU/False/True/IEMOCAP | eval | 2.38418579e-07 | 0.000220653135 | 0 |
| Model/GRU/False/True/MELD | eval | 2.38418579e-07 | 0.000220653135 | 0 |
| Model/GRU/False/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 0.000220653135 | 0 |
| Model/GRU/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 0.000220653135 | 0 |
| Model/GRU/True/False/IEMOCAP | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/False/MELD | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/False/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/False/MELD | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/True/IEMOCAP | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/True/MELD | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/True/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/GRU/True/True/MELD | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model/LSTM/False/False/IEMOCAP | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/False/MELD | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/True/IEMOCAP | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/True/MELD | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/False/True/MELD | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model/LSTM/True/False/IEMOCAP | eval | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/False/MELD | eval | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/False/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/False/MELD | train(dropout=0) | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/True/IEMOCAP | eval | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/True/MELD | eval | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/True/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/LSTM/True/True/MELD | train(dropout=0) | 1.1920929e-07 | 1.23318807e-06 | 0 |
| Model/Transformer/False/False/IEMOCAP | eval | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/False/MELD | eval | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/True/IEMOCAP | eval | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/True/MELD | eval | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/True/IEMOCAP | train(dropout=0) | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/False/True/MELD | train(dropout=0) | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model/Transformer/True/False/IEMOCAP | eval | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model/Transformer/True/False/MELD | eval | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model/Transformer/True/False/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model/Transformer/True/False/MELD | train(dropout=0) | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model/Transformer/True/True/IEMOCAP | eval | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model/Transformer/True/True/MELD | eval | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model/Transformer/True/True/IEMOCAP | train(dropout=0) | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model/Transformer/True/True/MELD | train(dropout=0) | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Intra/GRU/False/False/IEMOCAP | eval | 1.1920929e-07 | 9.0412334e-07 | 0 |
| Model_Intra/GRU/False/False/MELD | eval | 1.1920929e-07 | 9.0412334e-07 | 0 |
| Model_Intra/GRU/False/False/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 9.0412334e-07 | 0 |
| Model_Intra/GRU/False/False/MELD | train(dropout=0) | 1.1920929e-07 | 9.0412334e-07 | 0 |
| Model_Intra/GRU/False/True/IEMOCAP | eval | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Intra/GRU/False/True/MELD | eval | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Intra/GRU/False/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Intra/GRU/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Intra/GRU/True/False/IEMOCAP | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/False/MELD | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/False/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/False/MELD | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/True/IEMOCAP | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/True/MELD | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/True/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/GRU/True/True/MELD | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Intra/LSTM/False/False/IEMOCAP | eval | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/False/MELD | eval | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/True/IEMOCAP | eval | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/True/MELD | eval | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/False/True/MELD | train(dropout=0) | 3.57627869e-07 | 1.74368215e-05 | 0 |
| Model_Intra/LSTM/True/False/IEMOCAP | eval | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/False/MELD | eval | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/False/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/False/MELD | train(dropout=0) | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/True/IEMOCAP | eval | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/True/MELD | eval | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/True/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/LSTM/True/True/MELD | train(dropout=0) | 1.1920929e-07 | 3.95949655e-06 | 0 |
| Model_Intra/Transformer/False/False/IEMOCAP | eval | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/False/MELD | eval | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/True/IEMOCAP | eval | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/True/MELD | eval | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/True/IEMOCAP | train(dropout=0) | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/False/True/MELD | train(dropout=0) | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Intra/Transformer/True/False/IEMOCAP | eval | 5.36441803e-07 | 4.6120399e-06 | 0 |
| Model_Intra/Transformer/True/False/MELD | eval | 5.36441803e-07 | 4.6120399e-06 | 0 |
| Model_Intra/Transformer/True/False/IEMOCAP | train(dropout=0) | 5.36441803e-07 | 4.6120399e-06 | 0 |
| Model_Intra/Transformer/True/False/MELD | train(dropout=0) | 5.36441803e-07 | 4.6120399e-06 | 0 |
| Model_Intra/Transformer/True/True/IEMOCAP | eval | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Intra/Transformer/True/True/MELD | eval | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Intra/Transformer/True/True/IEMOCAP | train(dropout=0) | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Intra/Transformer/True/True/MELD | train(dropout=0) | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Inter/GRU/False/False/IEMOCAP | eval | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model_Inter/GRU/False/False/MELD | eval | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model_Inter/GRU/False/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model_Inter/GRU/False/False/MELD | train(dropout=0) | 2.38418579e-07 | 1.78104222e-06 | 0 |
| Model_Inter/GRU/False/True/IEMOCAP | eval | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Inter/GRU/False/True/MELD | eval | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Inter/GRU/False/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Inter/GRU/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 0.000220653135 | 0 |
| Model_Inter/GRU/True/False/IEMOCAP | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/False/MELD | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/False/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/False/MELD | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/True/IEMOCAP | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/True/MELD | eval | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/True/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/GRU/True/True/MELD | train(dropout=0) | 7.15255737e-07 | 5.6074241e-05 | 0 |
| Model_Inter/LSTM/False/False/IEMOCAP | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/False/MELD | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/True/IEMOCAP | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/True/MELD | eval | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/False/True/MELD | train(dropout=0) | 3.57627869e-07 | 0.000901306921 | 0 |
| Model_Inter/LSTM/True/False/IEMOCAP | eval | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/False/MELD | eval | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/False/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/False/MELD | train(dropout=0) | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/True/IEMOCAP | eval | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/True/MELD | eval | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/True/IEMOCAP | train(dropout=0) | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/LSTM/True/True/MELD | train(dropout=0) | 1.1920929e-07 | 1.17376601e-06 | 0 |
| Model_Inter/Transformer/False/False/IEMOCAP | eval | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/False/MELD | eval | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/True/IEMOCAP | eval | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/True/MELD | eval | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/True/IEMOCAP | train(dropout=0) | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/False/True/MELD | train(dropout=0) | 9.53674316e-07 | 1.99442111e-05 | 0 |
| Model_Inter/Transformer/True/False/IEMOCAP | eval | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model_Inter/Transformer/True/False/MELD | eval | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model_Inter/Transformer/True/False/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model_Inter/Transformer/True/False/MELD | train(dropout=0) | 4.76837158e-07 | 3.92170705e-06 | 0 |
| Model_Inter/Transformer/True/True/IEMOCAP | eval | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Inter/Transformer/True/True/MELD | eval | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Inter/Transformer/True/True/IEMOCAP | train(dropout=0) | 9.53674316e-07 | 1.75112946e-05 | 0 |
| Model_Inter/Transformer/True/True/MELD | train(dropout=0) | 9.53674316e-07 | 1.75112946e-05 | 0 |

Six independent faulty fixtures (not production mutations) must trigger the same close-comparison detector:
- bidirectional_encoder: detected=True, max_abs_diff=1.06028104, future perturbation.
- cross_time_normalization: detected=True, max_abs_diff=1.76943922, future perturbation.
- missing_causal_attention_mask: detected=True, max_abs_diff=2.08076048, future perturbation.
- hyperedge_delivery_to_past: detected=True, max_abs_diff=36.6666679, future perturbation.
- future_outdegree_normalization: detected=True, max_abs_diff=5.33333302, suffix deletion.
- whole_dialogue_mean_replacement: detected=True, max_abs_diff=17, future perturbation.

GPU nondeterminism: this invocation records only its selected device. A separate GPU invocation must preserve its own output directory. Normal-dropout training integration and restore validation are separate evidence. On CUDA, ordinary forward comparisons keep the default cuDNN-enabled backend; only the diagnostic scalar-gradient forward/backward locally disables cuDNN to use ATen RNN. Model weights and eval/train mode remain unchanged. Gradient-forward predictions are compared to ordinary forward predictions with the original unchanged tolerances. Separate GRU/LSTM fixtures profile actual aten::_cudnn_rnn execution, reproduce the eval-backward restriction, and verify finite nonzero ATen gradients without switching model mode.

Re-run from `CMERC/`:

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python scripts/verify_causal_regression.py --device cpu
```

Detailed per-check numbers and scalar-gradient records: `CAUSALITY_REGRESSION.json`.
Raw output: `logs/causality_cpu.log`.

## Cross-device final evidence

| Device | configurations | max_abs_diff | max_rel_diff | future gradient maximum |
| --- | --- | ---: | ---: | ---: |
| cpu | 144/144 | 9.53674316e-07 | 0.000901306921 | 0 |
| gpu | 144/144 | 7.15255737e-07 | 0.000450755004 | 0 |

GPU details: `gpu_fixed/CAUSALITY_REGRESSION.json`. Initial failure evidence is retained in `gpu/` and `logs/causality_gpu.log`. Eval RNN gradients use a locally declared ATen backend; normal GPU forward stays on its default backend and is compared at the same tolerances. Six small faulty fixtures and auxiliary algebra diagnostics in that GPU invocation remain CPU and are labelled as such. No threshold was relaxed.
