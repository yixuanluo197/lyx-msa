# Unknown-Label Supervision Mask Verification

The mask implementation lives in `code/protocol_b_masking.py` and applies only after the complete dialogue forward pass. It preserves all context inputs and filters student logits, all five teacher views, hidden representations and labels before any ground-truth access.

Covered label-dependent paths:

- classification NLL/Focal input rows;
- five true-class rank-loss branches;
- three MSCL relation labels and pools;
- two CSCL relation labels and pools;
- fold-local class weights;
- optimizer-step guard for an empty supervised batch;
- safe zero contrastive term for a singleton supervised batch.

The implementation avoids two silent errors: slicing `hidden` inside `torch.no_grad()` would detach MSCL/CSCL gradients, and calling Adam on an empty supervised batch could still change weights through weight decay.

Test-first evidence:

- Red phase: 5 tests failed because the implementation module did not exist.
- Green phase: all 5 tests passed.
- Frozen-feature alignment: all 53 mask locations matched exact dialogue/utterance positions; source structure hashes before and after were identical.

This verifies the mask core and immutable data alignment. It was not integrated into a training runner because the earlier curriculum-fidelity gate failed and the task explicitly requires stopping before training in that case.
