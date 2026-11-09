# Final Pilot Decision

## Outcome

The formal Protocol B manifest is approved and frozen, and the Unknown-Label Mask core passed unit and frozen-data alignment checks. The Pilot is **blocked before training** by the user-defined curriculum fidelity stop condition.

No accepted evidence identifies how the upstream fold-local teacher should be trained and selected without using Development, and the published generation command cannot reproduce the complete official difficulty artifact. Proceeding would require choosing a new teacher protocol or relaxing an explicit constraint.

## Acceptance status

- Manifest frozen: passed.
- Dialogue/Session/Episode isolation: passed.
- 53-row mask alignment: passed.
- Masked-loss core tests: passed, 5/5.
- Faithful fold-local difficulty: failed due missing teacher-generation provenance.
- IEMOCAP/MELD training: not started.
- Calibration: not started.
- Evaluation use: labels were read only for the explicitly required manifest class-support audit; no model forward, metric, training, calibration or method selection used them.
- GPU usage: 0 GPU-hours.
- Ready for formal 50-model run: no.

## Required research decision

The scientifically clean continuation would preregister two additional Training-only M³Net teacher jobs, use a fixed final epoch rather than Development selection, generate mask-aware difficulty with fixed-order batch size 1, and then train the two CMERC Pilots. This is a protocol amendment and adds teacher models, so it was not inferred or executed automatically.
