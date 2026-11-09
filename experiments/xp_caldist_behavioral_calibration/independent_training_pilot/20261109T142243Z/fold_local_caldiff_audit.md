# Fold-local Curriculum Difficulty Fidelity Audit

## Decision

**Failed the preregistered fidelity gate. Training was not started.**

## Evidence

1. XP04 did not freeze a difficulty generator. `experiment_records/04_xp_mainline_validation/preflight/test_scope.json` explicitly excludes `test_caldiff_batching.py` because XP04 used the fixed-author artifacts and the frozen export did not contain the generator.
2. The upstream workflow first trains an M³Net model and then passes `YOUR_M3NET_MODEL` to `caldiff/cal_diff.py`. The original generator loads that checkpoint before scoring. A random model, old XP04 checkpoint or feature-only heuristic is therefore not faithful.
3. The identity, training split, checkpoint-selection rule and hash of the teacher used for the published `IEMOCAP_diff.pkl` and `MELD_diff.pkl` are not published.
4. The public IEMOCAP command specifies batch size 16, but the original generator assigns the aggregate batch loss only to `data[-1][0]`. It can write at most 8 keys for 120 IEMOCAP dialogues, while the supplied artifact contains all 120. The MELD README calls `train_cm.py` rather than the difficulty tool. These commands cannot reproduce the supplied complete artifacts.
5. A fold-local teacher would add two baseline teacher trainings before the two requested CMERC Pilots. The task does not define a Training-only teacher endpoint. Selecting the teacher with Development would violate the explicit rule that fold-local difficulty may not use Development labels or predictions; using the final epoch would be a new, unregistered choice.
6. Unknown-label masking also requires an explicit masked true-class-gap adaptation. The published generator has no supervision-mask input. The exact score formula is recoverable, but the required teacher-generation protocol is not.

The recoverable score is the sum, over text/audio/visual and intra/inter-context removals, of `max(log p_removed(y) - log p_full(y), 0)`, combined with the original speaker term. Recovering the formula alone is insufficient to claim faithful difficulty reproduction.

## Rejected substitutions

- Old author difficulty: violates Training-only scope and lacks new Training dialogue coverage.
- Old XP04 checkpoint: has seen data assigned to new Calibration/Evaluation roles.
- Randomly initialized model: contradicts the upstream checkpoint-loading workflow.
- Disable curriculum: changes the approved XP04 training protocol.
- Feature norm or loss proxy: changes the difficulty algorithm.

No difficulty artifact was generated, and no GPU training process was launched.
