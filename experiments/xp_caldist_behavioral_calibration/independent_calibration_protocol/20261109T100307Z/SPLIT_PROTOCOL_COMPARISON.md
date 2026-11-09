# Split Protocol Comparison

| Protocol | Independence | Training retained | Models at 5 seeds | Strength | Main cost |
|---|---|---:|---:|---|---|
| A: fixed four-way grouped | Model-, calibration-, and evaluation-unseen | IEM 42.2%; MELD 59.8% | 10 total | Simple, lowest compute | One split; IEM trains on only two Sessions |
| B: rotating grouped cross-fitting | Same, repeated across all held-out groups | IEM 51.4–56.7%; MELD 59.6–60.4% | 50 total | Tests group dependence and uses every group for Evaluation | More training and storage; folds share data |
| C: external independent data | Potential researcher- and method-unseen confirmation | Unknown | Unknown | Strongest confirmation if compatible | No locally verified CMERC-compatible external MERC cohort |

Protocol A is useful as a resource-constrained pilot. Its single split cannot separate method behavior from the choice of IEMOCAP Session, and strict Session separation leaves only two training Sessions.

Protocol B is the unique recommendation for a formal internal study. IEMOCAP uses Session holdouts; MELD uses Season–Episode holdouts. Results must be aggregated by fold and seed, with group-aware uncertainty, rather than pooling repeated predictions as independent utterances.

Protocol C was audited locally without downloading data. Other IEMOCAP copies are duplicate or alternative-label preprocessing, not external cohorts. No local EmoryNLP/DailyDialog-style dataset with verified CMERC-compatible text, audio, visual, dialogue, and label inputs was found. A new external MERC dataset would strengthen researcher-unseen confirmation but would introduce domain and label-space changes; it should be a separate follow-up rather than a substitute for Protocol B.
