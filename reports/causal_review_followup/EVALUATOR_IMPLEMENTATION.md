# Evaluator fixes and evidence

Scope: `scripts/evaluate_checkpoint.py`, `tests/test_followup_evaluator.py`. Read `CAUSAL_README.md` and `docs/GPT_REVIEW_GUIDE.md` before implementation. Other agents own training/provenance/model regression work. No commit/push, official test evaluation, or formal training was performed by this slice.

## Confirmed issues

1. **Unindexed default CUDA device fails before inference.** Original main called `torch.cuda.set_device(torch.device('cuda'))`. On the existing PyTorch 2.2.2+cu121 environment with `CUDA_VISIBLE_DEVICES=3`, the isolated original expression raises `ValueError: Expected a torch.device with a specified index or an integer, but got:cuda`. Raw evidence: `logs/evaluator_original_device_probe.log`. New `resolve_device` at lines 226–242 uses public `is_available/current_device/device_count` APIs to turn unindexed CUDA into a concrete logical index, rejects indices outside visible devices, permits causal CPU, and preserves the official CUDA requirement. Main calls this resolver before `set_device` at lines 629–632. Physical host GPU 3 is logical cuda:0 in this run; host GPU UUID is `GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456`.
2. **Causal evaluator previously built fixed GRU defaults.** Checkpoints with LSTM/Transformer, different num_L/num_K, residue, or A/V encoders could not round-trip. New `causal_model_arguments` at lines 245–299 reads only an explicit whitelist from trusted checkpoint args, validates types/ranges and protocol, and constructs all three real networks. Missing/incompatible metadata fails clearly. `main` reads checkpoint metadata before constructing models at lines 639–667 and retains `load_state_dict(..., strict=True)`. Official noncausal construction stays fixed to its existing defaults.

## Supported architecture boundary

Causal `hyper/concat_DHT/avl`, effective LN, IEMOCAP or MELD; GRU, LSTM, Transformer; positive num_L/num_K/Deep_GCN_nlayers up to 128; boolean residue, av_using_lstm, speaker/modal flags; finite dropout in [0,1). Transformer D_g must be divisible by 4. Production dimensions follow `train_cm.py`: text 1024; D_g 512 (IEMOCAP) or 1024 (MELD); graph_hidden_size 512; audio/visual/speaker/class counts follow dataset defaults.

For explicit synthetic diagnostic architectures only, `checkpoint.args.architecture` can override positive integer D_g/graph_hidden_size/D_m_a/D_m_v/n_speakers (maximum 4096). Unknown dimension fields, bool-as-int, invalid sizes and unsupported configs are rejected. Model text normalization still requires 1024 text features, and class count remains dataset-defined. The evaluator accepts only `prefix_mean_v1`; missing policy in historical causal metadata is interpreted as that already-existing implementation policy. This whitelist constrains model construction, **not pickle deserialization**: checkpoint inputs must be trusted.

## Data and output boundary

Added `--split train|validation|test`, `--split-seed`, `--valid-fraction`, `--max-dialogues` at lines 607–612. Existing default test behavior stays compatible. IEMOCAP train/validation uses the existing dialogue split helper; MELD uses official split IDs. No curriculum/difficulty is loaded during inference. Output metrics now record effective model constructor arguments, split/seed/fraction and replacement policy at lines 577–587. Auxiliary passes remain same-weight eval-mode passes. `prefix_mean_v1` is a causal smoothing view, not full modality removal.

## Regression execution

- Red before implementation: `logs/evaluator_red.log`: **25 failed, 1 skipped**. Failures identify missing resolver/checkpoint-aware architecture APIs; CUDA skipped under explicitly hidden devices.
- CPU after implementation, including original evaluator utility tests: `logs/evaluator_cpu.log`: **30 passed, 1 skipped**, 5.14 s. GPU-specific entry test intentionally skipped. CPU same-weight main/reference log-probability max_abs_diff=0, max_rel_diff=0.
- GPU execution: `logs/evaluator_gpu_before.log` records GPU availability immediately before launch; host GPU 3 was idle. `logs/evaluator_gpu.log`: **31 passed**, 5.93 s. The actual main function exercised its default CUDA argument, checkpoint loading, strict architecture construction, all six inference passes, probability checks, metrics, CSV and NPZ writing. Only the data loader was replaced with a synthetic validation loader; Model/Model_Intra/Model_Inter and graph calculation were real. Float32, synthetic input seed 310, same saved LSTM/residue/A/V weights: GPU versus CPU reference max_abs_diff=2.384185791015625e-07, max_rel_diff=1.1714276837437865e-07, within unchanged atol=2e-5/rtol=1e-5.
- Twelve real-network save/load round-trips cover 3 encoders × residue on/off × A/V encoder on/off with nondefault num_L=1, num_K=2. Strictly deleting a state key is rejected. Comparisons assert exact full log-probability vectors for main and all five auxiliary views. The added auxiliary equality assertions were re-run on CPU in `logs/evaluator_cpu_final.log`: 30 passed, 1 skipped in 5.39 s; main/reference CPU differences remain zero.
- This slice does not claim production-dimension trained checkpoint accuracy or formal CL readiness; parent-owned real-feature checks are reported separately.

Re-run from the CMERC directory, using the existing environments without installing/upgrading anything:

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c 'import sys; sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "-s", "tests/test_followup_evaluator.py", "tests/test_evaluate_checkpoint.py"]))'
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=3 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c 'import sys; sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "-s", "tests/test_followup_evaluator.py", "tests/test_evaluate_checkpoint.py"]))'
```


## Follow-up review: checkpoint-bound evaluation split

**Confirmed MEDIUM after the first patch:** the added train/validation CLI initially defaulted split_seed to 0 and could reconstruct IEMOCAP validation IDs different from a checkpoint trained with split seed 1475. A subset limit did not correct that provenance mismatch. This was a defect in the newly added diagnostic path, not a property of model forward causality.

Minimal counterexample: causal checkpoint args.split_seed=1475 + `--split validation` without a split seed. Before correction parse_args returned zero. `logs/evaluator_split_red.log` preserves **5 failed, 26 deselected** before this correction.

Correction: `resolve_evaluation_data` (scripts/evaluate_checkpoint.py:314) resolves `--split-seed` default None from checkpoint args.split_seed or source data_provenance.split_seed, cross-checks both sources, and rejects any conflicting user value. Historical validation checkpoints with only args.seed are explicitly identified as `legacy_checkpoint.args.seed`; this is the old code's documented seed==split-seed convention, not a generic assumption for arbitrary checkpoints. Missing seed evidence requires an explicit diagnostic seed. Missing source provenance additionally requires `--diagnostic-data-reason`, and outputs retain `diagnostic_unverified` instead of implying data provenance was verified. A known official_test selection protocol is rejected for the held-out train/validation diagnostic route.

`verify_evaluation_data` (scripts/evaluate_checkpoint.py:375) rebuilds the full current source provenance using the existing helper and compares feature-file SHA256, ordered train/validation ID hashes, counts, dataset and selection protocol before the loader takes any `--max-dialogues` subset (main:659–670). Existing causal source provenance is checked rather than bypassed by a diagnostic reason. For the current training code, the absent valid_fraction metadata defaults to its production 0.1; any different data partition is rejected by ID/count hashes. Checkpoints that explicitly record valid_fraction also reject conflicting overrides immediately. Official test membership and noncausal legacy behavior retain compatibility.

Output `evaluation_data` records effective split seed, seed source, source provenance, verification status and diagnostic reason. The main-entry regression now inherits seed 310 from the saved synthetic checkpoint without a CLI split override. Additional tests reject explicit seed conflicts, internally inconsistent checkpoint metadata, teacher official_test provenance, changed feature hashes and changed split IDs. One test writes real temporary feature bytes and pickled synthetic split metadata and invokes the real provenance helper: modifying the feature bytes or validation fraction is rejected. No formal dataset split is evaluated in these new tests.

Final follow-up GPU run: `logs/evaluator_split_gpu.log`: **37 passed**, 5.62 s, using host GPU 3 after the saved `evaluator_split_gpu_before.log` resource check. The actual default-CUDA main entry and inherited split seed pass; GPU/CPU float32 probability differences remain max_abs=2.384185791015625e-07, max_rel=1.1714276837437865e-07. CPU final execution: `logs/evaluator_split_cpu_final.log`: **36 passed, 1 CUDA skip**, 5.19 s; main/reference CPU differences remain zero. py_compile and evaluator diff whitespace checks passed. All previous logs remain intact.
