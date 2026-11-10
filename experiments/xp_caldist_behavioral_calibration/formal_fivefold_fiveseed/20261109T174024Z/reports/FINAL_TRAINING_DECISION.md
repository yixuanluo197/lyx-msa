# Final Training Decision

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

The formal training and Calibration stage passed all gates: 10/10 Teachers, 10/10 Difficulty artifacts, 50/50 CMERC checkpoints, and 50/50 Calibration packages are complete; all Teacher and CMERC checkpoints strict reload; Curriculum and unknown-label masking audits pass; and Evaluation remains sealed.

Calibration-internal OOF diagnostics favor Hierarchical TS over Global TS on average: IEMOCAP E2−E1 NLL -0.009402, Brier -0.004152, ECE15 -0.003816; MELD NLL -0.006203, Brier -0.002306, ECE15 -0.004241. These values motivate retaining both methods for the sealed Evaluation; they are not evidence of independent generalization.

**Decision:** the artifacts satisfy the prerequisites for a separately authorized final independent Evaluation. Do not run Evaluation until the researcher approves this readiness report.
