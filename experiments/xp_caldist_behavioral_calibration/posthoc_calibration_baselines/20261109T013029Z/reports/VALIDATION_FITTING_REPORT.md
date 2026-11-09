# Validation fitting report

The read-only audit passed before fitting: all ten checkpoint hashes and twenty prediction artifacts matched historical provenance, the true split sizes were IEMOCAP 527/12 dialogues and MELD 1109/114 dialogues, and Validation/Test dialogue IDs were disjoint. Existing complete probabilities avoided GPU inference.

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

Validation NLL decreased for every checkpoint by construction. IEMOCAP selected T>1 for all seeds, while MELD selected T<1 for all seeds. The frozen parameter payload SHA-256 is `b3d07159fc930fc68d5e176147a87456114d6fd9336bbc7c421e81b3eb7d9c09`. No boundary expansion or Test-driven revision was performed.
