# Raw video audit

## Proven XP04 inputs

No MP4/AVI, frame directory, face crop, landmark, action-unit table, bounding box, tracking record, or frame-to-utterance timestamp manifest is referenced by XP04. The loader consumes one 342-D visual vector per utterance (`dataloader.py:138,194`).

## Local candidates

- IEMOCAP: a local release candidate contains 151 session/dialogue AVI files whose stems match all 151 XP04 dialogue IDs. One per session was inspected: 720×480, about 29.97 fps, DivX 5 video with embedded 22,050 Hz stereo PCM. This is a five-file sample, not a full decode audit. No extracted image frames were found. MOCAP head/rotated marker files exist for many but not all XP04 utterances; they are neither OpenFace AUs nor proved inputs to the 342-D vectors.
- MELD: `/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw` contains 13,706/13,708 expected standard-named utterance MP4 files. Naming is `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`. The two missing files are train `dia125_utt3` and dev `dia110_utt7`; both have nonzero XP04 visual vectors. The directory also contains resource-fork and extra files, which are excluded from the expected count.
- No standalone frames, face crops, landmarks, or AU outputs were found in the candidate raw bundle.

All 3,760 MELD utterances with zero 342-D visual vectors have a matching MP4 in the candidate bundle. Therefore zero visual vectors cannot be explained by missing MP4s. Face absence, detector failure, target-speaker selection, tracking failure, or deliberate placeholder behavior remain UNVERIFIED because no upstream status log exists.

## Feasibility

Current 342-D vectors do not preserve frame order or a coordinate-to-face-region map. They cannot support facial-dynamics smoothing, local eye/mouth occlusion, or expression editing. Such interventions require raw video/frame provenance, face detection/alignment/tracking, frame sampling, region metadata, and deterministic re-extraction. Existing vectors support only opaque feature-space perturbations.
