# Final research decision

|Dataset|Outer-OOF method|NLL|Brier|ECE15|
|---|---|---|---|---|
|IEMOCAP|vanilla|1.033637|0.513962|0.136499|
|IEMOCAP|lodo_global_ts|0.948872|0.509061|0.134696|
|IEMOCAP|regularized_lodo_ts|0.952487|0.509935|0.129934|
|MELD|vanilla|1.132528|0.527992|0.209270|
|MELD|lodo_global_ts|0.995087|0.472355|0.029213|
|MELD|regularized_lodo_ts|0.995189|0.472346|0.029857|

1. **Why IEMOCAP may fail to transfer:** only 12 Validation dialogues produce high temperature refitting variance, opposing dialogue/class calibration directions, and a weak OOF signal whose bootstrap intervals cross zero. Historical Test degradation is consistent with this unstable estimate plus calibration-distribution shift.
2. **Dialogue dominance:** yes. `Ses02F_impro06` is the largest temperature-influence dialogue in 4/5 seeds and is strongly overconfident; a few dialogues materially shape T.
3. **Sample-size sensitivity:** strong for IEMOCAP and modest for MELD. IEMOCAP's 50% subset within-seed SD(T) is 0.3291 versus MELD's 0.0181.
4. **Class directions:** they conflict in both datasets. IEMOCAP neutral/frustrated are overconfident while sad/angry are not; MELD's large neutral/surprise groups are underconfident while sadness/fear/disgust are overconfident.
5. **Regularization:** it does not consistently improve outer-OOF NLL or Brier and cannot be claimed as a solution.
6. **Why MELD is stable:** 114 Validation dialogues, much lower leave-dialogue-out influence, concentrated subset-temperature distributions, and a common underconfidence direction among its dominant classes.
7. **Classwise TS motivation:** class conflicts provide motivation, but IEMOCAP does not yet provide enough calibration data for an unconstrained classwise method. Any class-conditional model should be hierarchical/regularized and independently evaluated.
8. **Calibration data:** a larger, composition-balanced and independent calibration set is needed, especially for IEMOCAP.
9. **Behavior signals:** the route remains paused; no result here overturns the prior negative incremental-correctness finding.
10. **Best next single experiment:** run a preregistered session-grouped study on a larger independent calibration/evaluation partition, comparing Global TS with a strongly regularized hierarchical class-conditional temperature model. All method selection must occur inside the calibration portion, with one untouched evaluation partition. This phase does not execute it.

This is exploratory because known historical Test behavior motivated the question. No new Test prediction was read or evaluated, and these findings cannot be presented as an independently confirmed Test improvement.
