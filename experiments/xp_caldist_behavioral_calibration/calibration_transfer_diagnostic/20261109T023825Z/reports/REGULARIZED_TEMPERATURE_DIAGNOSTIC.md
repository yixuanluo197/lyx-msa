# Regularized temperature diagnostic

The outer loop leaves one dialogue out. Within each outer training set, five deterministic grouped folds choose lambda without seeing the outer dialogue. The regularization objective is `NLL + lambda * log(T)^2`.

|Dataset|Outer-OOF method|NLL|Brier|ECE15|
|---|---|---|---|---|
|IEMOCAP|vanilla|1.033637|0.513962|0.136499|
|IEMOCAP|lodo_global_ts|0.948872|0.509061|0.134696|
|IEMOCAP|regularized_lodo_ts|0.952487|0.509935|0.129934|
|MELD|vanilla|1.132528|0.527992|0.209270|
|MELD|lodo_global_ts|0.995087|0.472355|0.029213|
|MELD|regularized_lodo_ts|0.995189|0.472346|0.029857|

|Dataset|Metric|Regularized−ordinary LODO|Improved seeds|
|---|---|---|---|
|IEMOCAP|nll|+0.003615|1/5|
|IEMOCAP|brier|+0.000874|2/5|
|IEMOCAP|ece15|-0.004762|3/5|
|MELD|nll|+0.000102|1/5|
|MELD|brier|-0.000008|3/5|
|MELD|ece15|+0.000644|1/5|

Selected-lambda counts: IEMOCAP `{0.0: 4, 0.001: 6, 0.01: 17, 0.1: 26, 1.0: 7}`; MELD `{0.0: 39, 0.001: 378, 0.01: 153}`.

Regularization does not provide a consistent NLL/Brier improvement over ordinary LODO TS. For IEMOCAP it lowers mean ECE15 but worsens mean NLL by 0.003615 and Brier by 0.000874; NLL improves in only 1/5 seeds. MELD differences are negligible and mixed. Shrinking T toward one therefore does not resolve the transfer problem under this Validation-only nested protocol.
