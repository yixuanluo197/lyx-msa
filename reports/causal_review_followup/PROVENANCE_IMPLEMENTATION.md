# Causal curriculum provenance follow-up

This subtask concerns difficulty provenance and immutable output only. It does not train a teacher or produce official experimental difficulty values. Files read first: `CAUSAL_README.md`, `docs/GPT_REVIEW_GUIDE.md`; then `dataloader.py`, `caldiff/cal_diff.py`, existing causal protocol tests, and loader/checkpoint call sites in `train_cm.py` (read only).

## Confirmed defects and minimal changes

- **Confirmed: a causal wrapper alone established no teacher/split provenance.** Baseline `91a1bffa763d57015896dee1014f0d11c5d09029`, `dataloader.py:88–99`, accepted `{'causal': True, 'dataset': 'IEMOCAP', 'difficulties': {'train-a': .2}}`. This has no evidence that the teacher excluded validation training examples or that selection avoided the official test. `logs/provenance_baseline_defect.log` executes the old source and records that acceptance followed by an intentional assertion failure. The first regression test collection before the new API existed is retained as `logs/provenance_before.log`; this import failure is distinguished from the actual baseline defect demonstration.
- **Fixed: all causal difficulty readers require explicit artifact paths and strict manifests.** `dataloader.py:88`, `:108`, `:167` now route through `causal_provenance.resolve_causal_difficulty` with expected data/architecture and optional pinned digest. Causal Dataset curriculum construction fails before feature deserialization if no path is supplied. Old provenance-free causal wrappers are intentionally no longer compatible; noncausal legacy readers retain their original format. No unrecorded research override exists.
- **Fixed: causal generation no longer overwrites a fixed filename.** `caldiff/cal_diff.py:455` writes a new `Dataset_causal_diff_<full SHA256>.pkl` (or an explicitly supplied new `--difficulty-output`), paired with `.manifest.json`. Both use exclusive creation and reject existing paths. The original noncausal difficulty file is unaffected.
- **Fixed: source checkpoint and source training split are checked independently of emitted IDs.** `causal_provenance.py:132` requires causal LN architecture, validation selection, explicit training/split seeds, completed epoch and best epoch metadata, matching training/validation ID and feature hashes, and the named replacement policy. The source checkpoint's actual file SHA256 is recorded. `:173` and `:250` recheck that source file exists and still matches its digest when an artifact is read; missing/replaced teachers are rejected. A white list snapshot records the source training protocol and architecture.
- **Fixed: train-only values cannot disguise validation-contaminated teacher metadata.** The teacher's saved `training_data_provenance` must exactly match the requested source training split, validation split, feature bytes and split seed. The emitted difficulty dictionary must independently match all ordered training IDs. `official_test`, different splits, missing provenance, stale file digests, incompatible architecture, replacement policies and requested teacher seeds are rejected. Generation admission uses the actual `torch.load` result before any model inference (`caldiff/cal_diff.py:563`).
- **Fixed: causal generation evaluates only the exact validation-protocol training IDs.** The causal branch (`caldiff/cal_diff.py:842`) uses deterministic IEMOCAP split IDs or official MELD train IDs with validation IDs removed. It does not use test dialogues in difficulty inference.

## Artifact/API contract

`resolve_causal_difficulty(path, dataset, expected_data_provenance, expected_architecture=None, replacement_policy='prefix_mean_v1', expected_sha256=None, expected_teacher_seed=None)` returns `(difficulty_dictionary, {'path': absolute_path, 'sha256': artifact_file_hash, 'manifest': embedded_manifest})`. Training can pin the returned hash into run/checkpoint metadata; resume supplies `expected_sha256`. The training owner binds `expected_teacher_seed` to the requested training seed so paired-seed experiments cannot silently choose a different teacher seed.

`build_data_provenance` hashes **ordered** training and validation IDs and complete feature files, records counts and split seed, and rejects overlapping IDs. `data_provenance_from_files` reads stored split metadata. For the explicitly named legacy `official_test` protocol, its `validation_ids_sha256` field denotes the actual selection/test IDs, with `selection_protocol='official_test'`; this permits honest legacy checkpoint/run recording while strict causal difficulty admission rejects it. It does not turn test-selected teachers into validation-selected teachers.

The replacement policy stays `prefix_mean_v1`. No zero-mask policy or scientific interpretation change was introduced. JSON manifests and pickle/checkpoint files are trusted local research artifacts, not a security boundary against an adversary who can rewrite code and all audit files.

## Executed verification

- Regression tests: `tests/test_followup_provenance.py` contains 21 tests. These include all teacher provenance rejection cases, source teacher replacement/missing cases, split/content identity, overlap, seed binding, immutable output, and default causal admission.
- Together with the existing real difficulty batching mathematics tests: **25 passed**, CPU, `logs/provenance_after.log`. No cases skipped in this subset.
- Real file metadata hashing succeeded: IEMOCAP validation split seed 1475 gives 108 training and 12 validation dialogues; MELD gives 1038 training and 114 validation dialogues. Full feature file SHA256 and ordered ID SHA256 values are in `logs/provenance_real_metadata.log`. The optional legacy protocol entries in that log are metadata-only auditing; no test-set feature inference, loss, tuning or selection was run by this subtask.
- All artifact unit fixtures explicitly contain synthetic checkpoint bytes and synthetic feature bytes. They verify protocol consistency only and are never offered as official teachers or official CL artifacts. No causal teacher with verified real trained weights has been created here; real causal CL remains blocked until a compliant teacher is available.

Re-run the subset from `CMERC`:

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c 'import sys; sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_followup_provenance.py", "tests/test_caldiff_batching.py"]))'
```

Full-suite, training checkpoint integration, auxiliary-view and causal numerical verification are reported by their respective task owners. No dependencies were installed, no experiments launched, and no commit or push was made by this subtask.
