# Final Pilot Decision

## Current decision

**Hold before manifest freeze and training.** This is the explicit preregistered gate, not a runtime failure.

- Protocol B researcher approval: received.
- Five-fold split precheck: passed for both datasets.
- Fold 1 DataLoader dry-run: passed for both datasets.
- GPU preflight: four eligible GPUs at capture time; launch-time recheck required.
- IEMOCAP label policy: awaiting researcher decision.
- Approved manifest: intentionally absent until that decision.
- Training/calibration: not started.
- Evaluation label access/forward: none.

The recommended main policy is **Unknown-Label Mask Policy**. Once the researcher explicitly confirms Legacy or Unknown Mask, the next execution must freeze the unchanged candidate assignments, implement the selected supervision semantics, generate Training-only fold curriculum difficulty, rerun guards, and then launch the two independent Pilot jobs.

The project does not yet satisfy the acceptance criteria for the formal 50-model run because no Pilot training or calibration has occurred.
