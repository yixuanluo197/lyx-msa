# Raw text audit

## IEMOCAP

The XP04 feature pickle itself preserves the full utterance string, utterance ID, speaker code, numeric label, and dialogue key. Frozen loader evidence is `CMERC/experiment_records/04_xp_mainline_validation/code/dataloader.py:106-114`.

A local IEMOCAP release candidate exists under `/data2/yb/paper/IEMOCAP_extract/IEMOCAP_full_release`, with transcript files in `Session*/dialog/transcriptions/` and annotations in `Session*/dialog/EmoEvaluation/`. It is **not referenced by XP04**, so it is candidate raw provenance rather than a proven producer input.

Full cross-check against that candidate found:

- all 7,433 XP04 utterance IDs in the transcripts;
- 7,433/7,433 exact utterance-text matches;
- speaker suffixes consistent with the XP04 `M`/`F` speaker field;
- all six known class labels consistent;
- 53 raw annotation label `xxx` utterances are nevertheless retained by XP04 and encoded as numeric `0` (`happy`): 51 train, 1 validation, 1 test.

Thus ID/text alignment is strong, but unknown-label handling is not an identity mapping and must be treated as a frozen preprocessing decision.

## MELD

Candidate raw CSVs are at `/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw/{train,dev,test}_sent_emo.csv`. Fields are `Sr No.`, `Utterance`, `Speaker`, `Emotion`, `Sentiment`, `Dialogue_ID`, `Utterance_ID`, `Season`, `Episode`, `StartTime`, and `EndTime`.

Mapping to XP04 internal dialogue keys is:

- train: `key = Dialogue_ID`;
- dev: `key = 1039 + Dialogue_ID`;
- test: `key = 1153 + Dialogue_ID`.

All 13,708 utterance IDs and emotion labels align. Text aligns in 13,707/13,708 cases. The exception is train dialogue 556, utterance 6 (`train_sent_emo.csv:5246`): XP04 stores `What do I do?`, while the candidate CSV stores a different utterance and speaker. Both frozen raw and RoBERTa pickles share the XP04 version. Consequently the candidate CSV is not proven to be the exact producer version.

MELD speaker names are not retained in XP04. They are converted to a per-dialogue, first-appearance 9-way one-hot slot. The conversion script is MISSING; the mapping was reconstructed by cross-check.

## Can semantic perturbations be applied?

Yes at the raw-text level for both datasets because the full strings are retained, but modified strings **must be re-encoded**. Existing T1–T4 cannot represent an added question, Assertion, Probe, or Sample Corruption. Before running such perturbations, pin the exact/final replacement encoder, tokenizer assets, checkpoint hash, ASCII policy for MELD, ID manifest, and output schema. The exact historical encoder binary is currently UNVERIFIED.
