# Final research decision

The outcome matches the preregistered dataset-specific case (Situation C), with an additional metric-trade-off warning for IEMOCAP.

|Dataset|Metric|Vanilla mean±SD|Global TS mean±SD|Δ TS−Vanilla|Improved Seeds|
|---|---|---|---|---|---|
|IEMOCAP|ece15|0.067169±0.024055|0.078398±0.031217|+0.011228|1/5|
|IEMOCAP|brier|0.443966±0.015820|0.446304±0.013811|+0.002337|2/5|
|IEMOCAP|nll|0.818251±0.034006|0.831229±0.020622|+0.012978|1/5|
|MELD|ece15|0.186665±0.029827|0.037421±0.000930|-0.149244|5/5|
|MELD|brier|0.525287±0.007938|0.478671±0.007576|-0.046616|5/5|
|MELD|nll|1.130949±0.016868|1.012850±0.021256|-0.118099|5/5|

Global Temperature Scaling is a strong required baseline for MELD: it reduces mean ECE15 by 0.149244, Brier by 0.046616, and NLL by 0.118099, with 5/5 improving seeds and dialogue-bootstrap intervals below zero for every seed. MELD is systematically underconfident under the saved CMERC probabilities, so T<1 raises confidence.

Global TS is not a reliable Test calibration method for IEMOCAP under this Validation split. Mean ECE15 increases by 0.011228, Brier by 0.002337, and NLL by 0.012978. Although Validation selects T>1 for every seed, that confidence reduction does not transfer consistently. This is evidence of a Validation-to-Test calibration-distribution mismatch; it does not justify Test-tuned temperatures.

The next calibration study should keep Global TS as the reference and investigate the IEMOCAP split-transfer problem first, using validation-only resampling or nested fitting to quantify temperature stability. If a more flexible mechanism is later tested, class-conditional or regularized classwise temperature scaling is a better motivated next candidate than behavior-aware scaling, because prior behavioral signals did not provide stable incremental correctness information. This experiment does not execute that follow-up.
