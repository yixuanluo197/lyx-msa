# Mask-aware Difficulty Analysis

For valid primary-label positions `V`, five removal views `r`, and speaker term `S = max(active speaker index)+1` over the full qmask:

`R = sum_{u in V,r} max(log p_removed(y_u) - log p_full(y_u), 0)`

The primary amendment is `D_valid=(R+S)/(|V|+S)`. The preregistered sensitivity control is `D_total=(R+S)/(N_total+S)`.

Unknown utterances remain in all four text arrays, audio, visual, GRU sequence, graph, qmask, and intervention means. The implementation filters by `supervision_mask` before gathering the true class, so a sentinel label is never indexed. A dialogue with zero valid labels is a hard error.

`D_total` systematically lowers the score for dialogues containing more unknown labels while contributing no corresponding gap; it is retained only to quantify that effect. The synthetic arithmetic fixture in `difficulty_formula_comparison.csv` demonstrates the mechanism and is explicitly not real-model evidence. The main formula is frozen before any Calibration/Evaluation result.

The “intra” and “inter” names are narrower than full context deletion: they remove same-speaker or cross-speaker edges in the high-frequency GNN branch. The hypergraph and recurrent contexts remain.
