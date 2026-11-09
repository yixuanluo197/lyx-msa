# Test calibration report

The ten frozen temperatures were applied once to the corresponding Test probability vectors. Lower values are better for ECE15, multiclass Brier, and NLL.

|Dataset|Seed|ECE15 before→after|Brier before→after|NLL before→after|
|---|---|---|---|---|
|IEMOCAP|1475|0.039775→0.057357|0.430165→0.433382|0.778995→0.801204|
|IEMOCAP|67137|0.059767→0.075311|0.460153→0.463423|0.834569→0.854251|
|IEMOCAP|2024|0.083467→0.117928|0.426408→0.434371|0.787884→0.821047|
|IEMOCAP|3407|0.099577→0.100399|0.443640→0.442092|0.860326→0.842847|
|IEMOCAP|42|0.053260→0.040994|0.459466→0.458252|0.829479→0.836793|
|MELD|1475|0.172453→0.036737|0.524121→0.484871|1.127431→1.031107|
|MELD|67137|0.190666→0.036189|0.520741→0.475387|1.118705→1.002331|
|MELD|2024|0.192643→0.038243|0.529392→0.478320|1.137250→1.008591|
|MELD|3407|0.148186→0.037651|0.515798→0.486795|1.114567→1.036987|
|MELD|42|0.229377→0.038284|0.536383→0.467982|1.156790→0.985234|

MELD improved all three primary metrics for all five seeds, and every primary dialogue-bootstrap interval excluded zero in the improving direction. IEMOCAP did not show stable transfer: ECE15 improved for 1/5 seeds, Brier for 2/5, and NLL for 1/5; all IEMOCAP primary bootstrap intervals crossed zero.

The metric trade-offs are real. IEMOCAP seed42 improved ECE15 and Brier while NLL worsened; seed3407 improved Brier and NLL while ECE15 slightly worsened. Thus ECE changes alone do not establish comprehensive calibration improvement.

No Test prediction changed. Classification metrics match Vanilla CMERC to numerical equality for every seed.
