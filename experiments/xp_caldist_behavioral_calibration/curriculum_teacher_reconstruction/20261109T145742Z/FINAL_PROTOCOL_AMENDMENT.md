# Final Protocol Amendment

## Decision

**Retain Curriculum and adopt one Training-only fixed-final-epoch M³Net-family teacher per Dataset × Fold. Do not start the Teacher Pilot yet.** The scientific protocol is now defined, but two implementation gates remain: a role-scoped teacher runner must pass leakage tripwires, and the frozen real-model batch-one removal adapter must pass end-to-end parity tests.

## Answers

1. A (`CMERC/train_cm.py`) is the CMERC-documented Teacher entry; it trains the M³Net-family architecture. External B is upstream evidence and weight-compatible, while C is unsupported. The unpublished historical checkpoint save/selection chain still cannot be recovered.
2. The five-view gap and speaker formula are reproducible. The published batch16 implementation is defective, so its execution behavior is not faithfully reproducible as a valid per-dialogue artifact.
3. Fixed final epoch is a reasonable leakage-free Protocol B amendment because it removes Development/Evaluation selection. It is not claimed as the unpublished author selection rule.
4. Use CMERC's teacher commands: 80 epochs for IEMOCAP and 40 for MELD. The external MELD value of 15 is a documented conflict.
5. Keep unknown utterances in context, exclude them before true-class gather, and normalize the primary score by valid-label count plus the original full-qmask speaker term. Keep total length only as a frozen sensitivity control.
6. Real-model batch1 and batch16 equivalence has not been demonstrated, and static padding/indexing risks make equivalence unsafe to assume; formal generation is fixed to batch1.
7. Pure mask helpers cover the named objects, but the complete runner is not integrated. A rank tie-gradient mismatch was found and corrected only in the new experiment-local core.
8. The protocol is ready for implementation review, not for launching two teachers. The two gates above must pass first.
9. The formal experiment comprises 10 teachers and 50 CMERC models; the Pilot comprises 2+2.
10. Curriculum should be retained. NoCurriculum remains an ablation, not the primary method.

## Author method versus amendment

Author-recoverable components: M³Net-family architecture, feature inputs, five removal views, positive true-class log-probability gap, speaker term, 80/40 CMERC commands, and curriculum ratios.

Protocol B amendments: fold-local Training-only teachers, fixed final epoch, one teacher shared across five CMERC seeds, unknown-label mask, valid-count denominator, total-length sensitivity control, batch-one scoring, strict hashes, and prohibition of any Development/Calibration/Evaluation teacher access.

No teacher/CMERC training or formal difficulty generation was performed. The approved manifest was not modified.
