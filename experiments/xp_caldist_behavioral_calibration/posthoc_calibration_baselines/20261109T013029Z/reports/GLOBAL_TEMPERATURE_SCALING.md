# Global Temperature Scaling

For each XP04 validation-selected checkpoint, this experiment fits one positive scalar temperature by minimizing multiclass NLL on the true Validation split. The calibrated distribution is `softmax(log(P_original) / T)`. The fit uses a preregistered 2,001-point log-spaced grid over [0.05, 20], containing T=1, followed by deterministic golden-section refinement. Test labels and Test metrics were unavailable to the fitting script.

|Dataset|Seed|T|Val NLL before|Val NLL after|Boundary|
|---|---|---|---|---|---|
|IEMOCAP|1475|1.383009|0.922797|0.884580|False|
|IEMOCAP|67137|1.491371|0.949092|0.888989|False|
|IEMOCAP|2024|2.039805|1.157486|0.950796|False|
|IEMOCAP|3407|2.132159|1.192453|0.956892|False|
|IEMOCAP|42|1.243830|0.946356|0.928812|False|
|MELD|1475|0.585767|1.124931|1.000168|False|
|MELD|67137|0.566549|1.125896|0.987872|False|
|MELD|2024|0.551300|1.139171|0.987888|False|
|MELD|3407|0.637564|1.105165|1.009397|False|
|MELD|42|0.515566|1.167475|0.983893|False|

All ten solutions were interior. Positive scalar scaling preserved every Test argmax and therefore preserved Accuracy, Weighted-F1, and Macro-F1 exactly.
