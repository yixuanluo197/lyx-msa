# Five-seed comparison

The summary unit is the model seed. The same Test utterances appear under all five model seeds, so predictions were never concatenated to claim a fivefold sample size. Standard deviations use `ddof=1`.

|Dataset|Metric|Vanilla mean±SD|Global TS mean±SD|Δ TS−Vanilla|Improved Seeds|
|---|---|---|---|---|---|
|IEMOCAP|ece15|0.067169±0.024055|0.078398±0.031217|+0.011228|1/5|
|IEMOCAP|brier|0.443966±0.015820|0.446304±0.013811|+0.002337|2/5|
|IEMOCAP|nll|0.818251±0.034006|0.831229±0.020622|+0.012978|1/5|
|MELD|ece15|0.186665±0.029827|0.037421±0.000930|-0.149244|5/5|
|MELD|brier|0.525287±0.007938|0.478671±0.007576|-0.046616|5/5|
|MELD|nll|1.130949±0.016868|1.012850±0.021256|-0.118099|5/5|

IEMOCAP temperatures vary from 1.243830 to 2.132159 but consistently request confidence reduction on Validation. On Test, that direction mostly fails to improve the three primary metrics. MELD temperatures range from 0.515566 to 0.637564 and consistently sharpen its underconfident distributions; this direction transfers to Test across every seed and primary metric.

Dialogue-level paired bootstrap used 2,000 resamples per dataset/seed and kept all utterances from a sampled dialogue together. The interval conditions on the fixed fitted temperature and fixed model predictions; it does not cover uncertainty from refitting T.
