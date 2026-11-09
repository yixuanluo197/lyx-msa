# CMERC Calibration Transfer Diagnostic

Validation-only diagnosis of why Global Temperature Scaling transfers differently for IEMOCAP and MELD. No model was trained, no checkpoint was changed, and no Test probability or label was read.

|Dataset|Outer-OOF method|NLL|Brier|ECE15|
|---|---|---|---|---|
|IEMOCAP|vanilla|1.033637|0.513962|0.136499|
|IEMOCAP|lodo_global_ts|0.948872|0.509061|0.134696|
|IEMOCAP|regularized_lodo_ts|0.952487|0.509935|0.129934|
|MELD|vanilla|1.132528|0.527992|0.209270|
|MELD|lodo_global_ts|0.995087|0.472355|0.029213|
|MELD|regularized_lodo_ts|0.995189|0.472346|0.029857|

IEMOCAP's 12-dialogue calibration set yields substantially higher leave-dialogue-out and subset-refit temperature variance, with opposing dialogue and class calibration directions. MELD's 114 dialogues yield a concentrated T<1 estimate and consistent OOF improvement. Nested shrinkage toward T=1 does not consistently improve NLL or Brier.

See `reports/PREREGISTRATION.md` and the six required reports. All result CSVs derive from the ten true Validation prediction files; `verification.json` records an independent full recomputation.
