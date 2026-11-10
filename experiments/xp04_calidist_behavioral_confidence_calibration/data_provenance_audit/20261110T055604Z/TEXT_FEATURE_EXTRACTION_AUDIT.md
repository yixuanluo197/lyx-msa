# Text feature extraction audit

## What is proved

XP04 loads four offline 1,024-D arrays from the RoBERTa pickle (`code/dataloader.py:110-111,134-137,161-163,190-193`). It never runs a tokenizer or encoder during training/evaluation.

Structurally matching producer scripts exist at:

`/data2/yb/paper/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction/`

The IEMOCAP script `roberta_feature_extract_iemocap.py:8-29` and MELD script `roberta_feature_extract_meld.py:8-30` load `checkpoint_best.pt`, encode every complete utterance, pad token IDs with `pad_idx=1`, request all hidden states, take the BOS position `[:,0,:]` from the final four states, detach to NumPy, and pickle the arrays. MELD first strips non-ASCII characters at line 22.

The matching fine-tuning scripts initialize from `roberta.large/model.pt`, set `--arch roberta_large`, `--max-positions 512`, `--shorten-method truncate`, train for 30 epochs, and select accuracy (`roberta_train_iemocap.sh:7-31`; MELD equivalent). No encoder-freezing option is present, so this is a fine-tuned, not frozen-pretrained, encoder. `eval()` in extraction disables training behavior; it does not imply the fine-tuning phase froze parameters.

## Answers to the required questions

1. Model: Fairseq `roberta_large`, 24 transformer layers and width 1024. Exact Fairseq package version/tokenizer asset hashes are UNVERIFIED.
2. The matching pipeline fine-tunes pretrained RoBERTa-large separately per dataset.
3. Encoder parameters are not frozen by the matching training command.
4. Each stored row represents the complete utterance string; MELD applies ASCII-ignore first.
5. Fine-tuning declares maximum position 512 and truncation. The extraction script has no explicit `max_length`; exact overlength behavior for the historical checkpoint is UNVERIFIED.
6. Four outputs are the BOS vectors of the final four hidden states.
7. They are created at IEMOCAP lines 24-27 and MELD lines 25-28, then saved at lines 29/30.
8. No upstream normalization is shown in the producer. XP04 independently applies four BN modules under its `norm=BN` configuration (`model_cm.py:591-599,751-772`).
9. The normalized arrays are averaged, projected/encoded by the language branch, then fused with audio/visual in CMERC (`model_cm.py:772,800-824`).

## Provenance limitation

The producer code, pickle schema, dimensions, and contents are structurally consistent. However, the published pickle lacks a manifest linking it to the exact producer commit, Fairseq version, tokenizer files, fine-tuned checkpoint hash, and command. Therefore exact historical reproduction is **UNVERIFIED**, not falsely claimed as hash-closed.
