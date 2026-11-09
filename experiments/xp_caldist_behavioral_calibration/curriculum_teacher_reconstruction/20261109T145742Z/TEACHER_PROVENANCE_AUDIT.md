# Teacher Provenance Audit

## Audited sources

- External M³Net: `https://github.com/feiyuchen7/M3NET`, clean commit `4d0fa88cd4f92c72b053527e877763f54e29ccb2`; remote `HEAD` was independently checked.
- Official CMERC source: commit `815c7c060914f5b55db816c0ba3662b11d08552f`.
- Local CMERC audit source: commit `8447d8d1bfca03ca2275d9196e38ce44d20892e3` plus separately versioned experiment-local files.

## Identity finding

For the requested A/B/C decision, **A (`CMERC/train_cm.py`) is the documented CMERC Teacher training entry**. The recoverable architecture is the M³Net model family. External entry B establishes the upstream family and compatible weights; there is no evidence for C.

CMERC's prose says to train M³Net and links the external repository, while the command immediately below runs `train_cm.py`. External `model.py` and official CMERC `model.py` are byte-identical (`SHA-256 ee84329b...`). A CPU schema check loaded external M³Net weights strictly into `model_cm.Model(causal=False)`: 85 tensors for IEMOCAP and 83 for MELD, with no missing, unexpected, or shape-mismatched keys. The forward return interfaces differ (five versus six values), so the external object cannot replace the caldiff object without an adapter even though weights are compatible.

The training provenance cannot be recovered from compatibility. External M³Net evaluates Test every epoch, updates its reported best values from Test, and contains no `torch.save` call. It therefore does not provide a leakage-free, closed checkpoint-production recipe. The external MELD README specifies 15 epochs, whereas CMERC specifies 40. The actual author teacher checkpoint hash and selection rule are unpublished.

The formal statement is therefore: **A is the documented entry and its architecture/checkpoint schema are recoverable; the exact checkpoint save/selection chain and the historical Teacher weights are not.** Weight compatibility alone cannot prove whether the unpublished historical file was produced by A or a compatible B-derived runner.

## Difficulty generator identity

`caldiff/cal_diff.py` constructs CMERC `model_cm.Model`, `Model_Intra`, and `Model_Inter`, loads a raw or wrapped state dict, runs full plus five removal views, and computes true-class log-probability gaps. The MELD CMERC README incorrectly invokes `train_cm.py` for step 2; its arguments and `--ckpt_path` show that `caldiff/cal_diff.py` was intended.
