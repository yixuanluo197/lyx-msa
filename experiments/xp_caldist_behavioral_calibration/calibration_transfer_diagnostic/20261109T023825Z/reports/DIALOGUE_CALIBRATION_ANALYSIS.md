# Dialogue calibration analysis

Each dialogue was summarized using fixed original predictions: size, true/predicted class composition, accuracy, mean confidence, signed gap, NLL, Brier, and error-confidence distribution. Positive signed gap means overconfidence.

- IEMOCAP: strongest underconfidence `Ses01M_script03_1` (-0.1685); strongest overconfidence `Ses02F_impro06` (+0.5016).
- MELD: strongest underconfidence `1077` (-0.6377); strongest overconfidence `1120` (+0.4138).

IEMOCAP contains sharply opposing dialogue-level directions. `Ses02F_impro06` has mean signed gap +0.5016 and also dominates leave-one-dialogue-out temperature influence, while `Ses01M_script03_1` is underconfident at −0.1685. MELD also contains dialogue heterogeneity, but 114 dialogues dilute any one dialogue's influence; its largest absolute LODO temperature shift is below 0.0068.

These results support composition sensitivity rather than a single anomalous metric calculation. Single-dialogue ECE remains noisy and is retained only descriptively; dialogue NLL and concatenated OOF metrics carry the primary interpretation.
