# Audio feature extraction audit

## Established facts

- IEMOCAP audio is labeled `IS10`, 1,582 dimensions (`train_cm.py:999-1002`), one vector per utterance.
- MELD audio is labeled `MELD_audio`, 300 dimensions, one vector per utterance.
- Actual pickle arrays are finite; IEMOCAP has 0 all-zero rows, MELD has 1 all-zero row. See `modality_feature_statistics.csv`.
- The loader passes the vectors directly to a learned linear projection and optional recurrent path (`model_cm.py:642-646,802-807`). Dialogue padding is applied only at batching (`dataloader.py:149-151,210-212`).

## Missing provenance

Neither the frozen CMERC tree nor its traced M3NET feature archive contains an audio extraction or pickle-writing script. The following are therefore UNVERIFIED: extractor/tool, model/config version, feature names/order, sample rate/channel conversion, segment boundaries, frame parameters, pooling, normalization, missing-audio policy, runtime command, and producer checkpoint hash. The symbol `IS10` is not sufficient evidence to assert a particular openSMILE configuration.

No utterance-level timestamps or frame sequence survive in the pickle. Structural length equality proves alignment inside the pickle but not correct slicing of the original waveform.

## Scientific interpretation

The current vectors do not retain enough attributable information to implement or validate true pitch-shift versus timbre transformation. Arbitrary coordinate edits are possible but semantically opaque. Physical acoustic perturbation requires raw audio, a pinned re-extractor, a segment manifest, and comparison against a newly generated clean baseline.
