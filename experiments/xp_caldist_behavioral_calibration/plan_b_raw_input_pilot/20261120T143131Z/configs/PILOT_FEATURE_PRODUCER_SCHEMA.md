# Pinned encoder producer / CMERC consumer boundary

`extract_pilot_features.py` runs only under TEXT_repro (Torch 2.2.0+cu118,
Transformers 4.42.3). It does not import downstream/PyG or execute CMERC inference.
`raw_path_smoke.py` and `raw_pilot.py` consume its verified CPU FP32 tensors in the
separate pinned CMERC runtime. Model/feature identity is checked across stages.

## CLI

```text
--mode smoke|pilot --dataset IEMOCAP|MELD
--dataset-dir DATA_DIR --text-features TEXT_PT --audio-features AUDIO_PT
--checkpoint SELECTED_CMERC_PT --text-model TRAINED_TEXT_OUTPUT
--output NEW_CACHE_PT --local-dir LOCAL_ASSET_DIR --device cuda:0
[--baseline-lock LOCK_JSON --pilot-output FIXED_SELECTION_DIR
 --text-review REVIEWED_TEXT_JSONL --audio-review AUDIO_REVIEW_CSV]
[--ffmpeg EXISTING_FFMPEG]
```

Smoke requires an already completed selected baseline and uses exactly the first
declared Train Fit Dialogue's first available audio target. It never indexes
Validation or internal dev. Its local directory should be
`local_only/raw_smoke/DATASET/encoder_stage`; the CMERC consumer uses sibling
`cmerc_stage`. It generates only the fixed suffix ` Page two.` and pitch +0.5
semitones; no label-preservation PASS is assigned.

Pilot additionally requires the frozen baseline lock, unchanged 120-target sample
manifest/selection lock, all 480 fixed text rows, all 720 fixed audio rows and their
review records. Audio assets must already exist in
`LOCAL_ASSET_DIR/audio_transformations.jsonl`; this producer does not redesign
transformations or select another cohort. Pilot execution is launched only by the
root after its prerequisites are satisfied. Code/synthetic tests do not authorize
an independent Validation run.

## Output contract

```text
{
  metadata: {
    format: PLAN_B_PRECOMPUTED_FEATURES_V2,
    mode: smoke|pilot, dataset: DATASET, status: COMPLETE,
    runtime: require_runtime('encoder'),
    scientific_protocol_sha256: SHA(PLAN_B_PROTOCOL.json),
    input_sha256: {absolute_resolved_path: SHA256, ...},
    encoder_provenance: {text: {...}, audio: {...}},
    encoder_states_unchanged: true, input_files_unchanged: true,
    encoder_states_before: {...}, encoder_states_after: {...},
    baseline_lock_sha256: SHA or null,
    sample_manifest_sha256: SHA or null, ...
  },
  clean_controls: {sample_id: {text: record, audio: record}},
  variants: [record, ...],
  selection: {...}  # smoke only
}
```

Success records use `status=EXTRACTED` and `feature=CPU FloatTensor[4,1024]` for text
or `[768]` for audio. Both `target_feature_sha256` and
`original_feature_sha256` use `raw_pilot.array_hash` (dtype, shape and contiguous
bytes). Failure records retain all available identifiers, QC and provenance,
`status=FAILED`, an error string, and main eligibility false. Every planned variant
is retained; `metadata.status=COMPLETE` means production finished, not that every
variant passed. Counts explicitly disclose extracted/failed conditions.

Text records store actual post-truncation input IDs locally, including BOS/EOS;
`actual_token_ids_sha256 = array_hash(np.asarray(actual_ids, dtype=np.int64))`.
The original and changed UTF-8 strings' hashes are distinguished from the
extractor's ASCII-normalized string hash. Full raw strings are in reviewed local
assets only. Audio records bind source, decoded clean and transformed file hashes,
decoded waveform hashes, transformation parameters, and encoder decoding metadata.
QC alone never establishes a preserved emotion label. Main eligibility additionally
requires the shared attributed review rule; smoke is always ineligible.

The sibling `CACHE.manifest.json` binds the complete serialized cache SHA and
duplicates provenance metadata for inspection. All source files are rehashed and
full loaded encoder state dictionaries/BN buffers are fingerprinted before/after
extraction. Changed inputs, weights or buffers prevent a COMPLETE cache. Existing
cache/sidecar paths are never overwritten.

## Synthetic checks

Run `TEXT_repro/bin/python scripts/test_extract_pilot_features.py` with script
directory on Python's import path. Tests load no dataset, real model, raw audio or
Validation sample; they verify tensor shape/dtype/finiteness, actual-token hashes,
failure retention, fixed-variant completeness, immutable file binding and detection
of model/BN mutations. Real smoke and pilot are separate root-launched actions.
