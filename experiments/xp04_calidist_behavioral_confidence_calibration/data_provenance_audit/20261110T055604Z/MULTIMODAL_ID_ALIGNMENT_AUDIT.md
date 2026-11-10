# Multimodal ID alignment audit

## Method

All dialogue dictionaries were traversed utterance by utterance. The audit compared IDs/order, sentences, speakers, labels, modality lengths and dimensions, finite values, split membership, duplicate composite IDs, and zero rows. It did not infer alignment from equal file counts. The complete frozen-pickle row table is `utterance_alignment_audit.csv`.

## IEMOCAP

- 151 dialogues, 7,433 utterances; XP04 split 108 train / 12 validation / 31 test dialogues.
- Within every dialogue, ID, sentence, speaker, label, legacy text, T1-T4, audio, and visual lengths agree.
- Raw-feature and RoBERTa pickles agree on all 7,433 sentences, speakers, and numeric labels.
- No duplicate composite utterance ID, split overlap, unexplained missing feature row, or non-finite value was found.
- Candidate original transcripts contain every XP04 ID and exactly match every sentence.
- Important exception: 53 candidate raw annotations are `xxx`, while XP04 labels them `0` (`happy`); this is a preprocessing-label anomaly, not a row shift.
- Zero rows: audio 0; visual 79. Their upstream reasons are UNVERIFIED.

## MELD

- 1,432 dialogues / 13,708 utterances: train 1,038/9,989; validation 114/1,109; test 280/2,610.
- Frozen raw-feature and RoBERTa pickles agree on all internal sentences, speaker slots, and emotion labels.
- All modality lengths/dimensions align; no duplicate composite ID, split overlap, non-finite value, or internal missing row was found.
- Candidate CSV aligns on all dialogue/utterance IDs and labels, but has one text and speaker-version mismatch at train dialogue 556, utterance 6. This is a real content mismatch, not accepted as aligned.
- Candidate MP4 filenames align to 13,706/13,708 rows. The two absent standard files have nonzero frozen vectors, demonstrating that candidate raw-media availability is separate from frozen feature availability.
- Zero rows: audio 1; visual 3,760 (train 3,407, validation 95, test 258). All 3,760 zero-visual rows have a candidate standard MP4. Root cause remains UNVERIFIED.
- Thirty-one candidate raw dialogues contain an adjacent `StartTime` decrease, so timestamp monotonicity is not universal.

## Speaker, labels, padding and masks

- IEMOCAP speaker is `M`/`F`; XP04 converts it to `[1,0]`/`[0,1]` at `dataloader.py:140-141`.
- MELD stores a valid 9-way one-hot speaker slot. Original names are not retained.
- IEMOCAP numeric mapping is documented at `dataloader.py:113`; MELD observed mapping is 0 neutral, 1 surprise, 2 fear, 3 sadness, 4 joy, 5 disgust, 6 anger.
- `__getitem__` creates one mask value per real utterance (`dataloader.py:142,197`); `collate_fn` pads dialogue sequences (`:149-151,210-212`). This is dialogue-level padding, distinct from tokenizer padding used during RoBERTa extraction.

## Verdict

The **frozen modalities are internally row-aligned and safe for existing-feature interventions**. Alignment from external raw media to the exact frozen A/V vectors is not provenance-closed. Text can be reconstructed with two explicit exceptions: IEMOCAP unknown-label remapping and the one MELD candidate CSV version mismatch.
