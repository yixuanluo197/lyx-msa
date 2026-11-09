# Temperature sample-size analysis

|Dataset|Dialogue fraction|Mean within-seed SD(T)|All runs range|Fits|
|---|---|---|---|---|
|IEMOCAP|50%|0.3291|0.8669–2.9554|500|
|IEMOCAP|75%|0.1800|1.0211–2.5098|500|
|IEMOCAP|100%|0.0000|1.2438–2.1322|5|
|MELD|50%|0.0181|0.4699–0.6860|500|
|MELD|75%|0.0108|0.4921–0.6687|500|
|MELD|100%|0.0000|0.5156–0.6376|5|

At 50% of Validation dialogues, IEMOCAP's mean within-seed SD(T) is 0.3291, compared with 0.0181 for MELD. At 75%, the values are 0.1800 and 0.0108. IEMOCAP subsets can even select T<1 although every full Validation fit selects T>1, demonstrating direction instability at small calibration-set sizes. MELD subsets remain consistently below one and concentrate near the full-data estimate.

The repeated-subset distributions directly include temperature refitting variation. They are distinct from the fixed-OOF dialogue bootstrap, which conditions on already fitted fold temperatures.
