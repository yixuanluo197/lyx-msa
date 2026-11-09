# Final Teacher Pilot Decision

**Decision: the two Fold 1 Teacher pilots and real Fold-local Difficulty generation pass. The project is technically ready to instantiate the two CMERC Pilot runners, subject to preserving the audited capacity-safe batching and mask adapter. No CMERC training was started in this stage.**

1. **Training-only Runner:** passed its role tripwire and exact-coverage audit.
2. **Training access:** IEMOCAP used exactly 84 Fold 1 Training dialogues; MELD used exactly 852. No Development, Calibration, Evaluation, or Test loader was constructed.
3. **IEMOCAP Teacher:** completed the fixed 80th epoch from random initialization.
4. **MELD Teacher:** completed the fixed 40th epoch from random initialization.
5. **Six-view adapter:** passed real-model Batch-1 reference parity at `atol=1e-6, rtol=1e-5` in the XP04 environment.
6. **Unknown labels:** all 53 remain frozen in the manifest; the 23 present in Fold 1 Training were retained as context and excluded before every label-dependent operation. A real dialogue containing eight unknowns passed parity and sentinel invariance.
7. **Difficulty coverage:** 84/84 IEMOCAP and 852/852 MELD Training dialogues have finite decompositions and complete six-view probabilities.
8. **Role exclusion:** Difficulty IDs are strictly disjoint from Development, Calibration, and Evaluation.
9. **Curriculum sampler:** actual Fold 1 difficulties pass exact-ID and source-order checks across five curriculum ratios.
10. **CMERC losses:** classification, five rank, three MSCL, and two CSCL paths pass mask-aware backward tests; empty and singleton cases are handled.
11. **Readiness:** the Teacher/Difficulty blockers are closed. The next authorized experiment can be the two CMERC Fold 1 Pilot models only; the remaining 48 CMERC models and eight Teachers still require a later decision.

The initial capacity failure is a useful protocol finding. Formal CMERC Pilot training must retain XP04's length-only `capacity_safe` batching (`edge_capacity=1000`, `node_capacity=5200`) rather than plain shuffle. This changes only the legal batch permutation, keeps batch size 16 and every Training dialogue, and uses no labels or held-out data.
