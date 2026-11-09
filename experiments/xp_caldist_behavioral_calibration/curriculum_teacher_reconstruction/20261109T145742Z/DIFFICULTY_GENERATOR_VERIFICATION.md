# Difficulty Generator Verification

The experiment-local generator is `scripts/generate_fold_local_difficulty.py`. It provides manifest/hash guards, strict checkpoint metadata loading, a frozen CMERC model factory, a role-scoped Training-artifact contract, six complete-dialogue forwards, mask-before-gather reduction, per-view decomposition, exact Training coverage, JSON output, curriculum selection, and the XP04 tie-gradient convention.

Formal invocation requires and verifies separate SHA-256 values for the approved manifest, Teacher checkpoint, role-scoped Training artifact, Teacher config, and frozen source manifest. The checkpoint must bind the Training artifact/config hashes, seed, Training-only access role, fixed-final-epoch selection, architecture, and epoch. Output embeds these hashes, the complete effective dataset config, and architecture.

TDD evidence: the first run failed because the implementation file did not exist; after implementation, 20 CPU tests passed. Tests cover strict loading, architecture/final-epoch checks, Training-only IDs, forbidden held-out IDs, coverage, six full-context forwards through mock models, five views, unknown labels, zero-valid failure, finite values, speaker term, context preservation, reproducible hashes, curriculum restriction, and arithmetic partition invariance.

`scripts/verify_m3net_schema.py` preserves the real construction-only compatibility check. Its CPU log and JSON show strict loading of 85 IEMOCAP and 83 MELD tensors with zero missing, unexpected, or shape-mismatched keys. It performs no forward or optimization.

## Batch finding

Batch size 1 and 16 are **not established as equivalent for the real model**. The author's batch16 generator aggregates a whole batch and writes the value only to one dialogue. Even after fixing the reducer, padded temporal means, un-packed bidirectional GRU behavior, and batch-position-indexed hyperedge parameters can change neural outputs. Accordingly:

- the formal generator is fixed to batch size 1;
- `difficulty_batch_invariance.csv` separates the passing post-forward arithmetic test from the untested/not-guaranteed real-forward claim;
- no formal difficulty was generated in this round.

The real adapter is implemented but intentionally unexecuted because no fold-local Teacher checkpoint or approved role-scoped Training feature artifact exists yet. A real fixed-dialogue B1/B16 comparison and role-access tripwire remain gates before approving it for formal generation.
