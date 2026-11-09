# IEMOCAP Fold 1 Teacher Pilot

The Training-only M³Net-family Teacher completed the fixed final epoch 80 with seed 1475. It used 84 Training dialogues, 4,216 contextual utterances, and 4,193 supervision-eligible utterances. The 23 Fold 1 Training `xxx` utterances stayed in the full context and never entered classification loss.

- Loss: unweighted NLL, matching the documented Teacher command.
- Optimizer: Adam, learning rate `1e-4`, weight decay `3e-5`.
- Architecture: GRU, dropout 0.5, BN, hypergraph, `num_L=3`, `num_K=4`.
- Training loss: 1.696597 at epoch 1 and 0.508493 at epoch 80.
- Runtime: 339.09 seconds on `GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248` (TITAN Xp).
- Peak CUDA allocation/reservation: 6,515,353,088 / 12,383,682,560 bytes.
- Capacity audit: maximum 1 permutation retry, 1,000 edges, and 2,856 nodes.
- Final checkpoint SHA-256: `48b93d53c17fb32331376d8984aaee46293f07b03addeda055cfb6e7b3f7ba7a`.

The final checkpoint is the fixed endpoint. No Development, Calibration, Evaluation, or Test metric was computed or used for selection.
