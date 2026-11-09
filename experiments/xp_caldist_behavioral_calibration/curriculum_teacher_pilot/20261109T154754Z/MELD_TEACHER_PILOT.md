# MELD Fold 1 Teacher Pilot

The Training-only M³Net-family Teacher completed the fixed final epoch 40 with seed 67137. It used exactly 852 Training dialogues and 8,176 utterances.

- Loss: frozen CMERC FocalLoss, matching the documented MELD Teacher path.
- Optimizer: Adam, learning rate `1e-4`, weight decay `3e-5`.
- Architecture: GRU, dropout 0.4, BN, hypergraph, `num_L=3`, `num_K=3`.
- Training loss: 4.447777 at epoch 1 and 2.578298 at epoch 40.
- Runtime: 165.00 seconds on `GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a` (TITAN Xp).
- Peak CUDA allocation/reservation: 567,781,888 / 775,946,240 bytes.
- Capacity audit: zero permutation retries, at most 289 edges and 723 nodes.
- Final checkpoint SHA-256: `5408e7bfb3e3341e4f070775762587c583022844804b8c3b5bbf8d3622b55707`.

The final checkpoint is the fixed endpoint. No Development, Calibration, Evaluation, or Test metric was computed or used for selection.
