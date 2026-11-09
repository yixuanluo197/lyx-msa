# Final Protocol Recommendation

## Unique recommendation

Adopt **Protocol B: five-fold group-level rotating cross-fitting** for both datasets, subject to researcher approval of the candidate manifests.

For IEMOCAP, Fold *i* uses Session *i* for Evaluation and the next Session cyclically for Calibration. The other three Sessions supply Training and Development; nine whole dialogues, three from each remaining Session, form Development. For MELD, use the ten fixed balanced Season–Episode blocks: two blocks for Evaluation, the next one for Calibration, the next one for Development, and six for Training. Each dataset/seed/fold gets a fresh CMERC checkpoint.

This design is preferable because every high-level group is evaluated, Calibration and Evaluation are model-unseen and mutually isolated, all roles cover every class, and it exposes group-dependent calibration transfer rather than hiding it in one favorable split.

## Direct answers

1. **IEMOCAP strict Session-level Calibration/Evaluation:** yes. Every Session covers all six classes. A fully Session-isolated four-role split is possible but leaves only two Sessions for Training; rotating Session Calibration/Evaluation with dialogue-isolated Development is recommended.
2. **MELD Episode recovery:** yes. All 1,432 feature dialogues align with the pinned official metadata; use composite Season–Episode IDs.
3. **Dialogue fallback:** feasible but unnecessary and weaker, because it allows an Episode to cross roles.
4. **Training size:** adequate for a controlled retraining study, though smaller than XP04. Protocol B retains 51.4–56.7% of IEMOCAP and 59.6–60.4% of MELD utterances for Training.
5. **Calibration support:** adequate overall. IEMOCAP has at least 65 examples in every class; MELD Calibration has at least 26 examples in the rarest class across candidate folds.
6. **Evaluation support:** adequate. MELD Evaluation has at least 61 examples in every class; every IEMOCAP Session covers all six classes.
7. **Retraining:** mandatory for every fold and seed. Old checkpoints saw portions of the proposed Calibration/Evaluation pool.
8. **Best design:** Protocol B for both datasets, Session-level for IEMOCAP and Season–Episode-level for MELD.
9. **Model count:** 25 IEMOCAP plus 25 MELD models: 5 folds × 5 seeds × 2 datasets = 50.
10. **Compute/storage:** approximately 28.8 GPU-hours, 36.0 with 25% contingency, and about 15.1 GB for full run artifacts. Recheck resources at execution time.
11. **Reusable historical results:** all documented historical results remain exploratory evidence and engineering baselines.
12. **Non-confirmatory results:** prior Validation calibration and repeatedly inspected historical Test results cannot be called independent confirmation of a newly selected method.
13. **External data:** not required to execute the recommended model-unseen protocol, but required for a stronger researcher-unseen confirmation. No immediately usable local CMERC-compatible external MERC cohort was verified.
14. **Final protocol:** freeze the candidate manifests after review; retrain CMERC from scratch per fold/seed using Training only, select checkpoints with Development only, fit E1/E2/E3 on Calibration only, and open Evaluation once for E0–E3 comparison.

## Decision boundary

No candidate assignment has been promoted to a formal split. No training or calibration should start until this report and `candidate_split_manifests.json` are reviewed. If approved, the next step is a split-materialization and training preregistration task, not immediate model execution.
