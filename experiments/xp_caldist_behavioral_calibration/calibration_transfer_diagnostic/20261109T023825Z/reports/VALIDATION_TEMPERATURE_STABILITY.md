# Validation temperature stability

|Dataset|Seed|Full T|LODO range|LODO SD|Largest-influence dialogue|ΔT when left out|
|---|---|---|---|---|---|---|
|IEMOCAP|1475|1.3830|1.1511–1.4496|0.0808|Ses02F_impro06|-0.2319|
|IEMOCAP|67137|1.4914|1.2484–1.5744|0.0843|Ses02F_impro06|-0.2430|
|IEMOCAP|2024|2.0398|1.7052–2.1516|0.1335|Ses02M_impro07|-0.3346|
|IEMOCAP|3407|2.1322|1.7731–2.2439|0.1362|Ses02F_impro06|-0.3591|
|IEMOCAP|42|1.2438|1.1402–1.3115|0.0516|Ses02F_impro06|-0.1036|
|MELD|1475|0.5858|0.5803–0.5899|0.0018|1115|-0.0054|
|MELD|67137|0.5665|0.5610–0.5706|0.0019|1115|-0.0056|
|MELD|2024|0.5513|0.5454–0.5548|0.0017|1115|-0.0059|
|MELD|3407|0.6376|0.6308–0.6420|0.0020|1115|-0.0068|
|MELD|42|0.5156|0.5107–0.5196|0.0016|1147|-0.0049|

IEMOCAP has visibly larger refitting variability. Across seeds, its LODO SD ranges from 0.0516 to 0.1362; MELD ranges only from 0.0016 to 0.0020. Leaving out `Ses02F_impro06` produces the largest absolute temperature change in four of five IEMOCAP seeds, reaching −0.3591. The remaining seed is most affected by `Ses02M_impro07`.

The concatenated outer-OOF results improve IEMOCAP NLL point estimates in 5/5 seeds, but all five paired dialogue-bootstrap NLL intervals include zero. MELD improves NLL, Brier, and ECE15 in 5/5 seeds and every corresponding interval excludes zero. This separates a weak, high-variance IEMOCAP calibration signal from a strong, stable MELD signal.
