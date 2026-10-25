# Official checkpoint reliability figure contract

Backend: Python / matplotlib exclusively. Archetype: quantitative grid.

Claim: describe the confidence–accuracy gaps of the two fixed author checkpoints on their complete test sets; no Baseline improvement claim.

Evidence: nonempty-bin mean confidence versus accuracy on equal [0,1] axes, ideal calibration diagonal, plus all ten bin supports below. Empty bins have no reliability point. Lines only guide the eye.

Exports: 4.6 × 5.6 inches; PNG 300 dpi for Word, editable-text SVG and TrueType PDF. Source data are each dataset/calibration_bins.csv.

Risks: sparse bins are unstable; no fitted calibration, confidence intervals, cross-seed claims or causal modality-conflict claims.

- IEMOCAP: n=1623; nonempty bins=7
- MELD: n=2610; nonempty bins=8
