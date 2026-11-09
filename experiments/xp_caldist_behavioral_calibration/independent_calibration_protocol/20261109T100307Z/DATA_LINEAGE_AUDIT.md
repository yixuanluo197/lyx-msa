# Data Lineage Audit

## Authoritative inputs

IEMOCAP uses the XP04 frozen `IEMOCAP_features.pkl` and four-array RoBERTa pickle. The feature pickle contains dialogue IDs, utterance IDs, speaker sex markers, six-class labels, text/audio/visual features, sentences, and the original train/test dialogue sets. The raw IEMOCAP release is present under `/data2/yb/paper/IEMOCAP_extract/IEMOCAP_full_release/Session1..5`; however, the repository does not contain a complete reproducible feature-extraction pipeline for all frozen inputs.

MELD uses the XP04 frozen feature pickles plus the metadata snapshot in `/data2/yb/paper/feature_sources/mmalign_20261017`. Its `acquisition.json` pins `declare-lab/MELD` commit `e8cedf27b5d2877e198332c957127e16eb214afe` and hashes the CSV files. All 1,432 CMERC dialogues and 13,708 utterances align with those CSVs by split, dialogue, utterance order, length, and label. The official training CSV contains raw dialogue 60, which is absent from the CMERC feature namespace; the feature-backed audit therefore contains 1,038 rather than 1,039 train dialogues.

The MELD feature-to-metadata mapping is:

- train: CMERC global dialogue ID = raw Dialogue_ID;
- validation: global ID = raw Dialogue_ID + 1039;
- test: global ID = raw Dialogue_ID + 1153.

`sample_lineage.csv` uses collision-free dialogue names such as `MELD_validation_0004`. Speaker names come from the official MELD CSV. The nine-dimensional pickle speaker vector is a dialogue-local role index and is not a global person identity.

## Counts

| Dataset | Utterances | Dialogues | Highest grouping | Classes |
|---|---:|---:|---:|---:|
| IEMOCAP | 7,433 | 151 | 5 Sessions | 6 |
| MELD | 13,708 | 1,432 | 190 Season–Episode groups | 7 |

The unified index includes `dataset`, `dialogue_id`, `session_id`, `episode_id`, `speaker_id`, `utterance_index`, `emotion_label`, `original_split`, and five historical-exposure flags.

## Frozen IEMOCAP label discrepancy

Raw EmoEvaluation files contain 7,380 utterances with one of the six target primary labels. The frozen XP04 pickle contains 7,433 six-class examples. Fifty-three utterances whose raw primary label is `xxx` are represented as class 0 (`happy`) in the upstream preprocessing product: 8 in Session 1, 8 in Session 2, 36 in Session 3, and 1 in Session 5. The same pickle hash is present in another local reproduction, so this is a frozen upstream preprocessing discrepancy rather than local corruption.

The candidate protocol retains the XP04 pickle labels to keep the future experiment comparable with CMERC. Correcting those 53 labels would define a new dataset protocol and must be reviewed separately.

## Reproducibility boundary

All primary inputs and code snapshots are SHA-256 recorded in `provenance/source_hashes.json`. The current audit can reconstruct IDs, labels, groups, and split candidates without model inference. It cannot claim end-to-end raw-media feature regeneration because the complete original extractors and versions are unavailable.
