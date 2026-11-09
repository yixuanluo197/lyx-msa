# Class-conditional calibration diagnostic

|Dataset|True class|Support|Vanilla gap|Full-TS gap|Δ class NLL|Small support|
|---|---|---|---|---|---|---|
|IEMOCAP|happy|81|+0.0433|-0.0801|-0.2247|no|
|IEMOCAP|sad|66|-0.0187|-0.1168|+0.0045|no|
|IEMOCAP|neutral|123|+0.2343|+0.1211|-0.2993|no|
|IEMOCAP|angry|130|-0.0031|-0.1046|+0.0403|no|
|IEMOCAP|excited|58|+0.0555|-0.0406|+0.0167|no|
|IEMOCAP|frustrated|69|+0.1295|+0.0087|-0.1494|no|
|MELD|neutral|470|-0.3392|-0.1038|-0.2941|no|
|MELD|surprise|150|-0.2345|-0.0236|-0.1516|no|
|MELD|fear|40|+0.0472|+0.2047|+0.1984|no|
|MELD|sadness|111|+0.0822|+0.2765|+0.1974|no|
|MELD|joy|163|-0.1425|+0.0710|-0.0767|no|
|MELD|disgust|22|+0.0644|+0.2425|+0.2453|no|
|MELD|anger|153|-0.1704|+0.0292|-0.1017|no|

IEMOCAP classes do not share one calibration direction. Neutral (+0.2343) and frustrated (+0.1295) are overconfident on average, while sad and angry are approximately calibrated or mildly underconfident. A global T>1 improves aggregate Validation NLL mainly through neutral, frustrated, and happy, but increases mean class NLL for angry, excited, and sad.

MELD shows the opposite aggregate direction but still has class conflicts: neutral, surprise, anger, and joy are underconfident, whereas sadness, fear, and disgust are overconfident. Global T<1 strongly helps the high-support underconfident classes while worsening sadness, fear, and disgust. Disgust has only 22 samples and fear 40, so these estimates require caution.

The conflicts motivate future class-conditional research, but this phase supplies no fitted Classwise Temperature Scaling result. IEMOCAP's small dialogue count and per-seed class variability make an unregularized classwise calibrator especially prone to overfitting.
