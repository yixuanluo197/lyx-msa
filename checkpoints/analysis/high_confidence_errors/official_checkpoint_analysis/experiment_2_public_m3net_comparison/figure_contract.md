# Public M³Net–CMERC reliability comparison figure contract

Backend: the existing Python / matplotlib workflow exclusively.
Archetype: quantitative grid; fixed public-checkpoint descriptive comparison.
Claim: show each model's confidence–accuracy deviation and bin support under
one identical IEMOCAP test protocol, without attributing it to training modules.

Upper panel: empirical accuracy versus mean maximum probability for each
nonempty bin; identical [0,1] axes and ideal diagonal. Connected points only
guide the eye and are not fitted calibration curves. Empty bins have no point.
Lower panel: all ten equal-width-bin supports, grouped by fixed model.

Source: calibration_bins.csv; n=1623 valid utterances per model. No new model
inference, smoothing, relabeling, probability clipping or recalibration.
Exports: PNG at 300 dpi for Word; editable-text SVG; TrueType PDF;
6.2 × 6.8 inches, white background, two distinguishable colors and markers.
Risks: sparse-bin uncertainty is not estimated; no confidence intervals,
seed-level inference, significance or strict training-pair / causal claims.
