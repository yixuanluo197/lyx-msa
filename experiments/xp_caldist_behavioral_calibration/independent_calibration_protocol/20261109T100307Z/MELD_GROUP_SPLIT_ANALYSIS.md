# MELD Group Split Analysis

## Episode recovery

Episode ID is reliably recoverable. The pinned official CSVs contain `Season`, `Episode`, `Dialogue_ID`, `Utterance_ID`, `Speaker`, and `Emotion`. Group IDs use the composite `SxxEyy`; Episode number alone is not unique across Seasons.

The complete feature-backed pool has 13,708 utterances, 1,432 dialogues, and 190 Season–Episode groups. Class support is:

| Neutral | Surprise | Fear | Sadness | Joy | Disgust | Anger |
|---:|---:|---:|---:|---:|---:|---:|
| 6,436 | 1,636 | 358 | 1,002 | 2,308 | 361 | 1,607 |

Among 190 Episodes, 135 contain fear, 132 contain disgust, and 92 contain all seven classes. Episode-level grouping is therefore viable despite the two rare classes.

## Existing split leakage at Episode level

The existing train/validation/test dialogue IDs are disjoint, but their Episode groups are not:

- train ∩ validation: 81 Episodes;
- train ∩ test: 142 Episodes;
- validation ∩ test: 67 Episodes;
- 67 Episodes occur in all three splits.

The original MELD split cannot support Episode-independent confirmation. Any Episode-level protocol must regroup the full feature pool and retrain CMERC.

## Candidate design

The audit deterministically balances 190 Episode groups into ten atomic blocks using utterance and seven-class loads. Protocol A assigns six blocks to Training, one to Development, one to Calibration, and two to Evaluation. It retains 8,197 training utterances (59.8%), 1,435 Calibration utterances, and 2,713 Evaluation utterances.

Protocol B rotates five folds over the same ten blocks:

- Evaluation: two blocks, 2,665–2,798 utterances;
- Calibration: one block, 1,309–1,406 utterances;
- Development: one block, 1,284–1,454 utterances;
- Training: six blocks, 8,176–8,274 utterances (59.6%–60.4%).

Every role in every fold contains all seven classes. Calibration has at least 26 fear/disgust examples depending on the fold; Evaluation has at least 61. These supports permit Global TS and strongly regularized predicted-class diagnostics, but rare-class temperature estimates remain uncertain.

Dialogue-level grouping is technically feasible and offers 1,432 groups, but it allows different scenes from the same Friends Episode to cross roles. Episode-level grouping is the recommended stronger isolation unit.
