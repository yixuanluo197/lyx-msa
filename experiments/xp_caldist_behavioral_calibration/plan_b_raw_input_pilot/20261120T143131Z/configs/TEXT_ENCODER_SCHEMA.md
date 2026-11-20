# Plan B supervised text encoder contract

This is a new supervised feature-production pipeline for a new downstream baseline.
It does not initialize from an XP04 student, consume teacher features, or claim to
restore the historical CMERC encoder.

## CLI

`text_encoder.py --mode smoke|train|extract --dataset IEMOCAP|MELD --data DATASET_PT --output TRAIN_DIR --pretrained PUBLIC_ROBERTA_DIRECTORY [--features FEATURES_PT] [--resume]`

- `smoke`: public initialization, longest 32 Fit utterances, exactly two optimizer
  steps, no internal-dev evaluation or large model save. Use a separate empty output.
- `train`: independent public initialization and new native classification head;
  CE training on Fit labels only, selected by Train-internal Dialogue-disjoint dev
  macro F1. Ties are not improvement; at most ten epochs, patience two. The selected
  encoder is **not** refitted on all effective Train.
- `extract`: load the locked best classifier, extract full effective Train and
  Validation text vectors once, and never compute official Validation selection
  metrics. Requires `--features` pointing to a new file.
- `--resume`: formal training only, restoring a full completed-epoch checkpoint.
  It restores model, AdamW, warmup/decay scheduler, Python/NumPy/Torch/CUDA RNG and
  sampler epoch/order/offset. Interrupted later CSV rows are retained in an archive
  before returning to the saved boundary. Loading public weights is not resume.

## Input dataset.pt

Required keys: `dataset`, `seed`, `class_names`, `splits`,
`fit_dialogue_ids`, `inner_dev_dialogue_ids`.

`splits` has exactly `train` and `validation`, each a list of dialogue dictionaries.
Each dialogue has `dialogue_id`, `utterance_ids`, `labels`, `texts`, and other
modalities consumed by other modules. Training indexes only Train Fit and internal
dev text/labels; extraction indexes text and utterance IDs only. No Test split is
accepted. Seed is 1475 for IEMOCAP and 67137 for MELD.

## Fixed training and text rules

Public initializer SHA-256 is checked against
`047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`.
Only the standard `classifier.dense` and `classifier.out_proj` weights/biases may be
newly initialized; missing backbone weights are an error. All backbone parameters
are trainable. AdamW lr 1e-5, weight decay .01, microbatch 4, accumulation 4, FP32,
gradient clip 1, dropout .1 and nonreentrant gradient checkpointing are fixed.
Warmup is ceil(10% of the fixed ten-epoch optimizer budget), followed by linear
decay. The first optimizer step has warmup lr zero; gradient/update probes continue
until a positive-lr update and record both backbone and classifier.

Preprocessing follows the stated COSMIC reference convention: IEMOCAP preserves the
original string; MELD removes non-ASCII characters with `errors='ignore'`. No
case/whitespace normalization or context concatenation is added. The public slow
RoBERTa byte BPE uses `add_prefix_space=False`, BOS=0, EOS=2, PAD=1; total length
including special tokens is at most 512, right truncation and right padding.
This is an explicitly specified new pipeline, not proof of historical equivalence.

## Saved training outputs

- `training_config.json`: frozen protocol/hash, input/model/tokenizer/code hashes,
  command, software/device details, Fit/dev token/truncation statistics.
- `initial_loading_info.json`: exact allowed missing classification-head tensors.
- `tokenizer/`: saved tokenizer configuration and byte-BPE assets.
- `optimizer_steps.csv`, `history.json`, `progress.json`,
  `gradient_update_verification.json`: complete training and selection records.
- `latest_training.pt`: full completed-epoch training state.
- `best_classifier/`: complete reloadable selected classifier, config and tokenizer.
- `best_model_lock.json`: selection epoch/score, selected model and latest-state
  hashes. No extraction from an unlocked training directory.

## Shared clean and perturbation API

`load_encoder(model_path, device='cuda:0') -> (RobertaModel, RobertaTokenizer)`

`model_path` can be the locked training directory or its `best_classifier` child.
The returned encoder remembers `planb_dataset` in its config.

`extract_texts(model, tokenizer, texts, dataset=None, batch_size=4) -> Tensor[N,4,1024]`

Returns CPU FP32 vectors in order `[-1,-2,-3,-4]`, BOS position zero. It always sets
eval mode, disables TF32, rejects non-FP32 parameters, and applies the same scalar
preprocessing/tokenization for clean and perturbed strings. No extra feature
normalization is performed. Calls are utterance-independent; a downstream pilot
can replace exactly one target utterance's four vectors and keep all other inputs
unchanged. Passing no dataset uses the saved encoder dataset identity.

## Feature output

`text_features.pt` is a dictionary:

```text
{
  'train':      {dialogue_id: FloatTensor[N,4,1024], ...},
  'validation': {dialogue_id: FloatTensor[N,4,1024], ...}
}
```

Utterance order follows the input dialogue exactly. The sibling `.manifest.json`
records feature/model/data/code/tokenizer hashes, ordered utterance IDs, per-dialogue
tensor shape/hash, and split-level truncation/ASCII-change counts. Sibling
`.truncation.csv` records per-utterance token counts and flags without raw text.
All model weights, complete features and source text stay local.
