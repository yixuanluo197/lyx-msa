#!/usr/bin/env python3
"""Prepare only effective XP04 Train targets; never index labels or held-out splits."""
import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaTokenizer

ROOT = Path('/LOCAL/lyx-msa')
XP = ROOT / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
HF = Path('/LOCAL_HOME/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59')
COSMIC = Path('/LOCAL/05_reproduction/external_repos/code_references/erc_task_modules/conv-emotion/COSMIC/feature-extraction')
SPECS = {
    'IEMOCAP': dict(seed=1475, count=5283, dialogues=108, slots=[2, 3, 4, 5], text_slot=6,
                    filename='iemocap_features_roberta.pkl',
                    sha256='4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06'),
    'MELD': dict(seed=67137, count=9989, dialogues=1038, slots=[3, 4, 5, 6], text_slot=7,
                 filename='meld_features_roberta.pkl',
                 sha256='58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare_tokenizer(out):
    directory = out / 'local_only/tokenizer'
    if directory.exists():
        raise RuntimeError('Refusing to overwrite tokenizer: ' + str(directory))
    encoder = json.loads((COSMIC / 'encoder.json').read_text())
    vocabulary = json.loads((HF / 'vocab.json').read_text())
    dictionary = {line.rsplit(' ', 1)[0]: index + 4 for index, line in
                  enumerate((COSMIC / 'dict.txt').read_text().splitlines())}
    mismatches = [token for token, identifier in encoder.items()
                  if dictionary.get(str(identifier)) != vocabulary.get(token)]
    assert len(encoder) == 50257 and not mismatches, 'COSMIC/HF BPE token IDs differ'
    assert sha256(COSMIC / 'vocab.bpe') == sha256(HF / 'merges.txt'), 'BPE merges differ'
    tokenizer = RobertaTokenizer.from_pretrained(str(HF), local_files_only=True,
                                                add_prefix_space=False)
    assert (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id,
            tokenizer.unk_token_id, tokenizer.mask_token_id) == (0, 1, 2, 3, 50264)
    tokenizer.save_pretrained(directory)
    protocol = dict(
        implementation='transformers.RobertaTokenizer (slow byte-level BPE)',
        pretrained_snapshot=str(HF),
        reference_implementation=str(COSMIC),
        all_cosmic_byte_bpe_token_ids_match_hf=True,
        compared_byte_bpe_token_count=len(encoder),
        cosmic_dictionary_special_token_offset=4,
        merges_identical_sha256=sha256(HF / 'merges.txt'),
        add_prefix_space=False, strip_whitespace=False,
        iemocap_text_preprocessing='Stored original text unchanged',
        meld_text_preprocessing="text.encode('ascii', errors='ignore').decode('utf-8')",
        add_special_tokens=True, bos_id=0, eos_id=2, pad_id=1,
        max_total_tokens=512, truncation='right; preserve BOS and EOS',
        padding='dynamic right padding per minibatch',
        layer_order=[-1, -2, -3, -4], pooling='BOS at token position 0',
        normalization=None, emotion_labels_read=False,
        official_validation_read=False, test_values_read=False,
        reference_note='Matching reference BPE assets and processing; no claim of original CMERC encoder recovery',
        source_files={str(path): sha256(path) for path in
                      [COSMIC / 'encoder.json', COSMIC / 'dict.txt', COSMIC / 'vocab.bpe',
                       COSMIC / 'roberta_feature_extract_iemocap.py',
                       COSMIC / 'roberta_feature_extract_meld.py', HF / 'vocab.json',
                       HF / 'merges.txt', HF / 'tokenizer_config.json']})
    write_json(out / 'configs/TOKENIZATION_PROTOCOL.json', protocol)
    return tokenizer


def prepare_dataset(out, dataset, tokenizer):
    spec = SPECS[dataset]
    path = out / 'local_only' / dataset / 'train_prepared.pt'
    if path.exists():
        raise RuntimeError('Refusing to overwrite prepared Train: ' + str(path))
    config_path = XP / ('configs/effective/%s_cmerc_validation_seed%d.json' % (dataset, spec['seed']))
    config = json.loads(config_path.read_text())
    # Only Train IDs are selected. No label, Validation or Test dictionaries are indexed.
    dialogues = config['provenance']['split']['train_ids']
    assert len(dialogues) == spec['dialogues'] and len(set(dialogues)) == len(dialogues)
    feature_path = XP / 'code/data' / spec['filename']
    assert sha256(feature_path) == spec['sha256'], 'Official pkl hash mismatch'
    with feature_path.open('rb') as stream:
        package = pickle.load(stream, encoding='latin1')
    selected = sorted(dialogues, key=str)
    random.Random(spec['seed']).shuffle(selected)
    dev_dialogues = set(selected[:math.ceil(len(selected) * .1)])
    rows, input_ids, target_chunks = [], [], []
    fit_indices, dev_indices = [], []
    raw_lengths, truncated_rows, nonascii_rows, empty_rows = [], [], [], []
    for dialogue in dialogues:
        texts = package[spec['text_slot']][dialogue]
        layers = [np.asarray(package[slot][dialogue], dtype=np.float32) for slot in spec['slots']]
        assert all(layer.shape == (len(texts), 1024) for layer in layers)
        target_chunks.append(np.stack(layers, axis=1))
        for utterance_index, original in enumerate(texts):
            assert isinstance(original, str)
            processed = original.encode('ascii', errors='ignore').decode('utf-8') if dataset == 'MELD' else original
            full_ids = tokenizer.encode(processed, add_special_tokens=True, truncation=False)
            tokens = tokenizer.encode(processed, add_special_tokens=True, truncation=True, max_length=512)
            index = len(rows)
            assert len(tokens) <= 512 and tokens[0] == 0 and tokens[-1] == 2
            subset = 'inner_dev' if dialogue in dev_dialogues else 'fit'
            (dev_indices if subset == 'inner_dev' else fit_indices).append(index)
            row = dict(row_index=index, dataset=dataset, split='train', inner_subset=subset,
                       dialogue_id=dialogue, utterance_index=utterance_index,
                       original_text_sha256=hashlib.sha256(original.encode('utf-8')).hexdigest(),
                       processed_text_sha256=hashlib.sha256(processed.encode('utf-8')).hexdigest(),
                       tokens_before_truncation=len(full_ids), tokens_after_truncation=len(tokens))
            rows.append(row)
            input_ids.append(tokens)
            raw_lengths.append(len(full_ids))
            if len(full_ids) > 512:
                truncated_rows.append(index)
            if original != processed:
                nonascii_rows.append(index)
            if not processed:
                empty_rows.append(index)
    targets = np.concatenate(target_chunks).astype(np.float32)
    assert targets.shape == (spec['count'], 4, 1024) and np.isfinite(targets).all()
    assert len(rows) == spec['count']
    assert set(fit_indices).isdisjoint(dev_indices)
    assert len(fit_indices) + len(dev_indices) == spec['count']
    prepared = dict(input_ids=input_ids, targets=torch.from_numpy(targets), rows=rows,
                    fit_indices=torch.tensor(fit_indices, dtype=torch.long),
                    dev_indices=torch.tensor(dev_indices, dtype=torch.long),
                    dataset=dataset, seed=spec['seed'], layer_order=[-1, -2, -3, -4])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared, path)
    with (path.parent / 'train_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Full strings remain in this local-only artifact, never in public reports.
    local_text_path = path.parent / 'train_texts.jsonl'
    with local_text_path.open('w') as stream:
        for row in rows:
            original = package[spec['text_slot']][row['dialogue_id']][row['utterance_index']]
            processed = original.encode('ascii', errors='ignore').decode('utf-8') if dataset == 'MELD' else original
            stream.write(json.dumps(dict(row_index=row['row_index'], original_text=original,
                                         processed_text=processed), ensure_ascii=False) + '\n')
    report = dict(dataset=dataset, seed=spec['seed'], total_train_rows=len(rows),
                  total_train_dialogues=len(dialogues), fit_rows=len(fit_indices),
                  inner_dev_rows=len(dev_indices), fit_dialogues=len(dialogues)-len(dev_dialogues),
                  inner_dev_dialogues=len(dev_dialogues), inner_split_seed=spec['seed'],
                  inner_split_algorithm='sort dialogue IDs, Python random.Random(seed).shuffle; first ceil(10%) held out',
                  split_has_no_shared_dialogues=True, target_shape=list(targets.shape),
                  target_dtype=str(targets.dtype), targets_all_finite=True,
                  layer_order=[-1, -2, -3, -4], pkl_feature_slots=spec['slots'],
                  max_untruncated_tokens=max(raw_lengths), token_length_quantiles={
                      str(q): float(np.quantile(raw_lengths, q)) for q in [0, .5, .9, .95, .99, 1]},
                  truncated_rows=truncated_rows, nonascii_changed_rows=nonascii_rows,
                  empty_processed_text_rows=empty_rows, labels_indexed=False,
                  official_validation_values_indexed=False, test_values_indexed=False,
                  shared_container_note='Unpickling shared official package materializes dictionaries; only effective Train text and four target feature dictionaries are indexed',
                  prepared_train_path=str(path), prepared_train_sha256=sha256(path),
                  config_path=str(config_path), config_sha256=sha256(config_path),
                  official_pkl_path=str(feature_path.resolve()), official_pkl_sha256=spec['sha256'],
                  target_layer_mean=[float(targets[:, i].mean()) for i in range(4)],
                  target_layer_std=[float(targets[:, i].std()) for i in range(4)])
    write_json(out / 'configs' / (dataset + '_TRAIN_PREPARATION.json'), report)
    print(json.dumps({key: report[key] for key in ['dataset', 'total_train_rows', 'fit_rows',
                     'inner_dev_rows', 'max_untruncated_tokens', 'prepared_train_path']}) , flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = prepare_tokenizer(args.out)
    for dataset in SPECS:
        prepare_dataset(args.out, dataset, tokenizer)


if __name__ == '__main__':
    main()
