# Figure contract — Experiment 6

Backend: Python/matplotlib only. Archetype: quantitative grid.

Claim: compare observed confidence–accuracy gaps of these fixed Baseline and CMERC checkpoints; the figure alone makes no universal improvement or significance claim.

Evidence: upper panel plots nonempty-bin mean maximum-class probability against empirical accuracy with equal 0–1 axes and an ideal diagonal; lower panel reports all ten bin counts. Empty bins have no reliability point. Connected points are guides, not fitted estimates.

Support: IEMOCAP/baseline: n=1623, nonempty bins=8; IEMOCAP/cmerc: n=1623, nonempty bins=8; MELD/baseline: n=2610, nonempty bins=6; MELD/cmerc: n=2610, nonempty bins=8.

Exports: 4.6 × 5.8 inch, PNG 300 dpi for Word; editable-text SVG and TrueType PDF. Baseline uses blue circles, CMERC orange squares; the legend is outside the axes.

Review risks: sparse bins are unstable; no confidence intervals or paired significance are estimated. This is descriptive analysis of the current local validation weights, one checkpoint per model. No retraining, checkpoint selection, probability clipping or calibration fitting is performed.
