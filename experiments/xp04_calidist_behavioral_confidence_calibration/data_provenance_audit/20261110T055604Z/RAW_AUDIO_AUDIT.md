# Raw audio audit

## XP04 dependency status

XP04 reads only utterance-level audio vectors through `dataloader.py:106-111,157-163`. It does not read WAV, FLAC, MP4 audio tracks, timestamps, or a raw-audio manifest. Therefore original audio is **MISSING from the proven XP04 dependency chain**.

## Local candidate raw media

### IEMOCAP

The local release candidate under `/data2/yb/paper/IEMOCAP_extract/IEMOCAP_full_release` contains 10,039 sentence WAVs and 151 dialogue WAVs (Apple `._*` resource forks excluded); all 7,433 XP04 utterance IDs have a sentence WAV. Five-session sampling found 16 kHz, mono, 16-bit PCM for sentence WAV and 16 kHz, stereo, 16-bit PCM for dialogue WAV. Annotation segment durations for the 7,433 rows range 0.585-34.1388 s, mean 4.5880 s. This is not an all-file header/decode audit. Because XP04 never references this directory and no raw-to-pickle manifest exists, correspondence to the published 1,582-D vectors is CANDIDATE/UNVERIFIED even where filenames match.

### MELD

The candidate MELD bundle has no standalone WAV/FLAC. Audio is embedded in standard-named `dia{Dialogue_ID}_utt{Utterance_ID}.mp4` files. A bounded metadata audit of eight files found AAC (`mp4a`), 48 kHz, stereo audio; this is a sample, not proof for all files. Two expected standard MP4s are missing, yet their XP04 audio and visual vectors are nonzero, further proving the bundle was not consumed at runtime.

## Feasibility

- Pitch shift, timbre transformation, and other controlled signal perturbations require attributable raw media plus deterministic segmentation and feature re-extraction. **Not currently reproducible against exact XP04 features.**
- Numerical noise, coordinate dropout, whole-vector zeroing, or mean replacement can be performed on the existing vectors, but must be described as **feature-space perturbations**, never pitch/timbre manipulation.
- Large media were not loaded into memory; only file inventory/metadata and selected bounded checks were used.
