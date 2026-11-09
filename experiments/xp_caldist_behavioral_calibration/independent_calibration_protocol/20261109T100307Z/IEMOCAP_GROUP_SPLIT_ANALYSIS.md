# IEMOCAP Group Split Analysis

## Session inventory

| Session | Dialogues | Utterances | Happy | Sad | Neutral | Angry | Excited | Frustrated | XP04 exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ses01 | 28 | 1,373 | 143 | 194 | 384 | 229 | 143 | 280 | train/validation |
| Ses02 | 30 | 1,356 | 125 | 197 | 362 | 137 | 210 | 325 | train/validation |
| Ses03 | 32 | 1,569 | 171 | 305 | 320 | 240 | 151 | 382 | train/validation |
| Ses04 | 30 | 1,512 | 65 | 143 | 258 | 327 | 238 | 481 | train/validation |
| Ses05 | 31 | 1,623 | 144 | 245 | 384 | 170 | 299 | 381 | test |

Every Session contains all six classes. The smallest Calibration/Evaluation class support in a one-Session holdout is 65 (`happy`, Ses04), so strict Session-level Calibration and Evaluation are feasible.

## Protocol A: fixed strict four-way split

The candidate fixed split assigns Ses04+Ses05 to Training, Ses03 to Development, Ses02 to Calibration, and Ses01 to Evaluation. All four roles are Session-disjoint. Training has 3,135 utterances, only 42.2% of the complete pool. This provides the cleanest four-way Session isolation but discards too much training information and evaluates only one Session condition. It is retained as a comparison, not recommended.

## Protocol B: rotating Session cross-fitting

Five folds set Evaluation to Session *i* and Calibration to Session *(i+1) mod 5*. Within the other three Sessions, exactly three whole dialogues per Session form Development; all remaining dialogues form Training. Development dialogue selection is deterministic (`seed=20261109`) and optimizes class distribution and utterance count without model outputs.

Across folds:

- Training: 80–84 dialogues and 3,819–4,216 utterances (51.4%–56.7% of the full corpus);
- Development: 9 dialogues and 479–526 utterances;
- Calibration: 1,356–1,623 utterances, all six classes;
- Evaluation: 1,356–1,623 utterances, all six classes;
- minimum Development class support: 28;
- minimum Calibration/Evaluation class support: 65.

Calibration and Evaluation are strictly Session-isolated from all other roles. Training and Development share the three remaining Sessions but never share a dialogue. With only five Sessions, also isolating Development by Session would leave only two Sessions for Training, which is the main weakness of Protocol A.

Protocol B is the preferred IEMOCAP design. It measures Session transfer directly and ensures each Session is evaluated once, while preserving more training data than a four-Session-role fixed split.
