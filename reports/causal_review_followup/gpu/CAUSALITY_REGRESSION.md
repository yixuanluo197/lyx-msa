# Causal CMERC regression evidence

Status: **failed**. Real model configurations: 96/144.
Device `cuda:0`, float32, seed 20261016, atol 2e-05, rtol 1e-05.
Relative differences use denominator max(|reference|, 1e-8); acceptance uses PyTorch combined atol + rtol rule.

Each row executes Model / Model_Intra / Model_Inter and their real HyperGCN. All GRU/LSTM/Transformer × audio/visual encoder × residue × train/eval × IEMOCAP/MELD settings use num_L=2, num_K=3. Lengths are 4, 2, 1 with repeated and differing speakers. No trained weights or datasets are loaded.

Tests compare full log-probability vectors and graph hidden representations for individually perturbed future text/audio/visual/speaker and their joint perturbation. Every nonempty prefix is inferred separately (suffix addition/deletion), then dialogue reorder, other-dialogue changes, single-dialogue inference, finite padding values, and longer right padding are compared. Scalar gradient checks sample times 0 and 1, categories 0 and 3 and one hidden coordinate; every tested scalar has exactly zero future feature gradients and nonzero visible-feature influence. This requirement does not apply to a loss containing future training examples.

Executed normalization hooks confirm feature-only LayerNorm; hooks inspect actual causal and padding masks at every executed Transformer. Unused legacy modules are not inferred to execute. Graph source membership and target delivery are checked separately; incoming-degree prefix invariance is exercised numerically.

| Configuration | mode | max absolute difference | max relative difference | future gradient max |
| --- | --- | ---: | ---: | ---: |
| Model/GRU/False/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model/GRU/False/False/MELD | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model/GRU/False/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model/GRU/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model/GRU/True/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model/GRU/True/False/MELD | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model/GRU/True/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model/GRU/True/True/MELD | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model/LSTM/False/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 0.000450755004 | 0 |
| Model/LSTM/False/False/MELD | train(dropout=0) | 2.38418579e-07 | 0.000450755004 | 0 |
| Model/LSTM/False/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 0.000450755004 | 0 |
| Model/LSTM/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 0.000450755004 | 0 |
| Model/LSTM/True/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 7.53208133e-06 | 0 |
| Model/LSTM/True/False/MELD | train(dropout=0) | 3.57627869e-07 | 9.59126464e-06 | 0 |
| Model/LSTM/True/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 9.50858157e-06 | 0 |
| Model/LSTM/True/True/MELD | train(dropout=0) | 3.57627869e-07 | 9.50858157e-06 | 0 |
| Model/Transformer/False/False/IEMOCAP | eval | 7.15255737e-07 | 5.19553532e-06 | 0 |
| Model/Transformer/False/False/MELD | eval | 7.15255737e-07 | 5.19553532e-06 | 0 |
| Model/Transformer/False/False/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 5.19553532e-06 | 0 |
| Model/Transformer/False/False/MELD | train(dropout=0) | 7.15255737e-07 | 5.19553532e-06 | 0 |
| Model/Transformer/False/True/IEMOCAP | eval | 7.15255737e-07 | 7.3306619e-06 | 0 |
| Model/Transformer/False/True/MELD | eval | 7.15255737e-07 | 7.3306619e-06 | 0 |
| Model/Transformer/False/True/IEMOCAP | train(dropout=0) | 7.15255737e-07 | 7.3306619e-06 | 0 |
| Model/Transformer/False/True/MELD | train(dropout=0) | 5.96046448e-07 | 7.3306619e-06 | 0 |
| Model/Transformer/True/False/IEMOCAP | eval | 4.76837158e-07 | 5.14768226e-06 | 0 |
| Model/Transformer/True/False/MELD | eval | 4.76837158e-07 | 4.81631469e-06 | 0 |
| Model/Transformer/True/False/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 5.14768135e-06 | 0 |
| Model/Transformer/True/False/MELD | train(dropout=0) | 4.76837158e-07 | 5.14768226e-06 | 0 |
| Model/Transformer/True/True/IEMOCAP | eval | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model/Transformer/True/True/MELD | eval | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model/Transformer/True/True/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model/Transformer/True/True/MELD | train(dropout=0) | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Intra/GRU/False/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model_Intra/GRU/False/False/MELD | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model_Intra/GRU/False/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model_Intra/GRU/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 3.64535526e-06 | 0 |
| Model_Intra/GRU/True/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Intra/GRU/True/False/MELD | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Intra/GRU/True/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Intra/GRU/True/True/MELD | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Intra/LSTM/False/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 8.40441771e-06 | 0 |
| Model_Intra/LSTM/False/False/MELD | train(dropout=0) | 3.57627869e-07 | 8.40441771e-06 | 0 |
| Model_Intra/LSTM/False/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 1.36384715e-05 | 0 |
| Model_Intra/LSTM/False/True/MELD | train(dropout=0) | 3.57627869e-07 | 1.36384715e-05 | 0 |
| Model_Intra/LSTM/True/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 9.59126464e-06 | 0 |
| Model_Intra/LSTM/True/False/MELD | train(dropout=0) | 3.57627869e-07 | 7.53208133e-06 | 0 |
| Model_Intra/LSTM/True/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 9.50858157e-06 | 0 |
| Model_Intra/LSTM/True/True/MELD | train(dropout=0) | 3.57627869e-07 | 7.53208133e-06 | 0 |
| Model_Intra/Transformer/False/False/IEMOCAP | eval | 5.96046448e-07 | 5.87857858e-06 | 0 |
| Model_Intra/Transformer/False/False/MELD | eval | 5.96046448e-07 | 6.71837597e-06 | 0 |
| Model_Intra/Transformer/False/False/IEMOCAP | train(dropout=0) | 5.96046448e-07 | 5.30335183e-06 | 0 |
| Model_Intra/Transformer/False/False/MELD | train(dropout=0) | 5.96046448e-07 | 5.30335183e-06 | 0 |
| Model_Intra/Transformer/False/True/IEMOCAP | eval | 5.96046448e-07 | 7.3306619e-06 | 0 |
| Model_Intra/Transformer/False/True/MELD | eval | 5.96046448e-07 | 7.3306619e-06 | 0 |
| Model_Intra/Transformer/False/True/IEMOCAP | train(dropout=0) | 5.96046448e-07 | 7.3306619e-06 | 0 |
| Model_Intra/Transformer/False/True/MELD | train(dropout=0) | 5.96046448e-07 | 7.3306619e-06 | 0 |
| Model_Intra/Transformer/True/False/IEMOCAP | eval | 4.76837158e-07 | 6.91302375e-06 | 0 |
| Model_Intra/Transformer/True/False/MELD | eval | 4.76837158e-07 | 6.91302375e-06 | 0 |
| Model_Intra/Transformer/True/False/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 6.91302375e-06 | 0 |
| Model_Intra/Transformer/True/False/MELD | train(dropout=0) | 4.76837158e-07 | 6.91302375e-06 | 0 |
| Model_Intra/Transformer/True/True/IEMOCAP | eval | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Intra/Transformer/True/True/MELD | eval | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Intra/Transformer/True/True/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Intra/Transformer/True/True/MELD | train(dropout=0) | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Inter/GRU/False/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 2.70915934e-05 | 0 |
| Model_Inter/GRU/False/False/MELD | train(dropout=0) | 2.38418579e-07 | 2.70915934e-05 | 0 |
| Model_Inter/GRU/False/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 2.70915934e-05 | 0 |
| Model_Inter/GRU/False/True/MELD | train(dropout=0) | 2.38418579e-07 | 2.70915934e-05 | 0 |
| Model_Inter/GRU/True/False/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Inter/GRU/True/False/MELD | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Inter/GRU/True/True/IEMOCAP | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Inter/GRU/True/True/MELD | train(dropout=0) | 2.38418579e-07 | 1.4019216e-05 | 0 |
| Model_Inter/LSTM/False/False/IEMOCAP | train(dropout=0) | 3.87430191e-07 | 0.000450755004 | 0 |
| Model_Inter/LSTM/False/False/MELD | train(dropout=0) | 3.87430191e-07 | 0.000450755004 | 0 |
| Model_Inter/LSTM/False/True/IEMOCAP | train(dropout=0) | 3.87430191e-07 | 0.000450755004 | 0 |
| Model_Inter/LSTM/False/True/MELD | train(dropout=0) | 3.87430191e-07 | 0.000450755004 | 0 |
| Model_Inter/LSTM/True/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 9.59126464e-06 | 0 |
| Model_Inter/LSTM/True/False/MELD | train(dropout=0) | 3.57627869e-07 | 9.59126464e-06 | 0 |
| Model_Inter/LSTM/True/True/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 7.53208133e-06 | 0 |
| Model_Inter/LSTM/True/True/MELD | train(dropout=0) | 3.57627869e-07 | 7.53208133e-06 | 0 |
| Model_Inter/Transformer/False/False/IEMOCAP | eval | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/False/MELD | eval | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/False/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/False/MELD | train(dropout=0) | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/True/IEMOCAP | eval | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/True/MELD | eval | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/True/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/False/True/MELD | train(dropout=0) | 4.76837158e-07 | 9.61400565e-06 | 0 |
| Model_Inter/Transformer/True/False/IEMOCAP | eval | 3.57627869e-07 | 4.81631469e-06 | 0 |
| Model_Inter/Transformer/True/False/MELD | eval | 3.57627869e-07 | 4.81631469e-06 | 0 |
| Model_Inter/Transformer/True/False/IEMOCAP | train(dropout=0) | 3.57627869e-07 | 4.81631469e-06 | 0 |
| Model_Inter/Transformer/True/False/MELD | train(dropout=0) | 3.57627869e-07 | 4.81631469e-06 | 0 |
| Model_Inter/Transformer/True/True/IEMOCAP | eval | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Inter/Transformer/True/True/MELD | eval | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Inter/Transformer/True/True/IEMOCAP | train(dropout=0) | 4.76837158e-07 | 1.7510989e-05 | 0 |
| Model_Inter/Transformer/True/True/MELD | train(dropout=0) | 4.76837158e-07 | 1.7510989e-05 | 0 |

Six independent faulty fixtures (not production mutations) must trigger the same close-comparison detector:
- bidirectional_encoder: detected=True, max_abs_diff=1.06028104, future perturbation.
- cross_time_normalization: detected=True, max_abs_diff=1.76943922, future perturbation.
- missing_causal_attention_mask: detected=True, max_abs_diff=2.08076048, future perturbation.
- hyperedge_delivery_to_past: detected=True, max_abs_diff=36.6666679, future perturbation.
- future_outdegree_normalization: detected=True, max_abs_diff=5.33333302, suffix deletion.
- whole_dialogue_mean_replacement: detected=True, max_abs_diff=17, future perturbation.

GPU nondeterminism: this invocation records only its selected device. A separate GPU invocation must preserve its own output directory. Normal-dropout training integration and restore validation are separate evidence.

Re-run from `CMERC/`:

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python scripts/verify_causal_regression.py --device cpu
```

Detailed per-check numbers and scalar-gradient records: `CAUSALITY_REGRESSION.json`.
Raw output: `logs/causality_cpu.log`.

Failures:
```json
[
  {
    "name": "model_0",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_1",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_4",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_5",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_8",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_9",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_12",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_13",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_16",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_17",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_20",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_21",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_24",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_25",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_28",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_29",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_48",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_49",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_52",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_53",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_56",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_57",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_60",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_61",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_64",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_65",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_68",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_69",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_72",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_73",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_76",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_77",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_96",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_97",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_100",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_101",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_104",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_105",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_108",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_109",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_112",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_113",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_116",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_117",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_120",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_121",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_124",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  },
  {
    "name": "model_125",
    "status": "failed",
    "error": "cudnn RNN backward can only be called in training mode",
    "traceback": "Traceback (most recent call last):\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 52, in run\n    value = function()\n  File \"/data2/yb/paper/lyx-msa/CMERC/scripts/verify_causal_regression.py\", line 61, in <lambda>\n    value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))\n  File \"/data2/yb/paper/lyx-msa/CMERC/tests/test_followup_causality.py\", line 178, in check_config\n    gradients = torch.autograd.grad(output[time, category], tensors, retain_graph=True)\n  File \"/home/yangbin/miniconda3/envs/multimodalerc310/lib/python3.10/site-packages/torch/autograd/__init__.py\", line 411, in grad\n    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\nRuntimeError: cudnn RNN backward can only be called in training mode\n"
  }
]
```
