"""Read-only XP04 input and split audit; deliberately performs no inference."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path('/data2/yb/paper/lyx-msa')
OUT = Path(__file__).resolve().parent
XP = ROOT / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
PRIOR = ROOT / 'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z'
DATA = ROOT / 'CMERC/data'
SEEDS = (1475, 67137, 2024, 3407, 42)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with path.open('rb') as stream:
        return pickle.load(stream, encoding='latin1')


def count_features(mapping, keys):
    values = [np.asarray(mapping[key]) for key in keys]
    assert values and len({v.shape[1] for v in values}) == 1
    assert all(v.ndim == 2 and np.isfinite(v).all() for v in values)
    return dict(dimension=int(values[0].shape[1]), utterances=sum(len(v) for v in values),
                all_zero=sum(int(np.all(v == 0, axis=1).sum()) for v in values))


files = {}
results = {}
for dataset in ('IEMOCAP', 'MELD'):
    raw_path = DATA / ('IEMOCAP_features.pkl' if dataset == 'IEMOCAP' else 'MELD_features_raw1.pkl')
    text_path = DATA / ('iemocap_features_roberta.pkl' if dataset == 'IEMOCAP' else 'meld_features_roberta.pkl')
    raw, text = load(raw_path), load(text_path)
    files[str(raw_path.resolve())] = digest(raw_path)
    files[str(text_path.resolve())] = digest(text_path)
    text_start = 2 if dataset == 'IEMOCAP' else 3
    feature_maps = {'audio': raw[4], 'visual': raw[5],
                    **{f'T{i + 1}': text[text_start + i] for i in range(4)}}
    split_rows = []
    for seed in SEEDS:
        cfg_path = XP / f'configs/effective/{dataset}_cmerc_validation_seed{seed}.json'
        cfg = json.loads(cfg_path.read_text())
        split = cfg['provenance']['split']
        split_sets = {name: set(map(str, split[f'{name}_ids']))
                      for name in ('train', 'validation', 'test')}
        assert not (split_sets['train'] & split_sets['validation'])
        assert not (split_sets['train'] & split_sets['test'])
        assert not (split_sets['validation'] & split_sets['test'])
        checkpoint = XP / f'runs/cmerc/{dataset}_validation_seed{seed}/best.pt'
        historical = XP / f'results/cmerc/{dataset}/validation/seed{seed}/predictions.npz'
        later = PRIOR / f'runs/{dataset}/seed{seed}/validation_predictions.npz'
        for path in (cfg_path, checkpoint, historical, later):
            assert path.is_file(), path
            files[str(path.resolve())] = digest(path)
        with np.load(historical) as z:
            historical_ids = set(map(str, z['dialogue_ids']))
            historical_n = len(z['labels'])
            historical_valid = len(historical_ids & split_sets['validation'])
            historical_test = len(historical_ids & split_sets['test'])
            assert len(z['labels']) == len(z['predictions']) == len(z['dialogue_ids']) == len(z['utterance_indices'])
        with np.load(later) as z:
            later_ids = set(map(str, z['dialogue_ids']))
            later_n = len(z['labels'])
            later_valid = len(later_ids & split_sets['validation'])
            later_test = len(later_ids & split_sets['test'])
            assert len(z['labels']) == len(z['predictions']) == len(z['dialogue_ids']) == len(z['utterance_indices'])
            assert np.array_equal(z['probabilities'].argmax(1), z['predictions'])
            assert np.isfinite(z['probabilities']).all()
            assert np.allclose(z['probabilities'].sum(1), 1, atol=1e-6)
            audio_ids = set(map(str, feature_maps['audio']))
            assert all(str(k) in audio_ids for k in z['dialogue_ids'])
        split_rows.append(dict(seed=seed, train_dialogues=len(split_sets['train']),
                               validation_dialogues=len(split_sets['validation']),
                               test_dialogues=len(split_sets['test']),
                               historical_utterances=historical_n,
                               historical_dialogues=len(historical_ids),
                               historical_validation_overlap=historical_valid,
                               historical_test_overlap=historical_test,
                               later_utterances=later_n, later_dialogues=len(later_ids),
                               later_validation_overlap=later_valid,
                               later_test_overlap=later_test,
                               historical_is_test=historical_ids == split_sets['test'],
                               later_is_validation=later_ids == split_sets['validation']))
    train_ids = split_sets['train']
    valid_ids = split_sets['validation']
    # The five effective configs carry the same split IDs; audit per-seed rows above.
    feature_summary = {}
    for name, mapping in feature_maps.items():
        lookup = {str(key): key for key in mapping}
        assert train_ids <= set(lookup) and valid_ids <= set(lookup)
        feature_summary[name] = {'train': count_features(mapping, [lookup[k] for k in sorted(train_ids)]),
                                 'validation': count_features(mapping, [lookup[k] for k in sorted(valid_ids)])}
    results[dataset] = dict(splits=split_rows, features=feature_summary)

for relative in ('code/dataloader.py', 'code/model.py', 'code/train_cm.py',
                 'code/scripts/evaluate_checkpoint.py'):
    path = XP / relative
    files[str(path.resolve())] = digest(path)

assert all(row['historical_is_test'] and row['later_is_validation']
           for data in results.values() for row in data['splits'])
assert all(row['historical_validation_overlap'] == 0
           for data in results.values() for row in data['splits'])

(OUT / 'provenance/source_hashes.json').write_text(json.dumps(files, indent=2) + '\n')
(OUT / 'verification.json').write_text(json.dumps({
    'stage': 'baseline_and_feature_audit', 'status': 'stopped_on_id_mismatch',
    'pilot_started': False, 'full_perturbation_started': False,
    'checkpoint_count': 10, 'historical_prediction_files': 10,
    'later_validation_prediction_files': 10, 'results': results}, indent=2) + '\n')

lines = ['# XP04 baseline and feature audit', '',
         '**Gate: stopped before Pilot.** The 10 XP04 prediction files under a path named `validation`'
         ' contain Test dialogue IDs, while the later independently inferred Validation files contain'
         ' the expected Validation IDs. The path component denotes the checkpoint selection protocol,'
         ' not the prediction split: the frozen evaluator calls `build_test_loader`.'
         ' These historical files cannot serve as original Validation predictions under the requested gate.', '',
         'The source is the frozen XP04 batch `20261016T050823Z`, not Protocol B.'
         ' All 10 best checkpoints and all four raw feature files exist and were SHA-256 hashed.'
         ' No checkpoint was loaded in this audit.', '',
         '| Dataset | Seed | Original `validation` rows/dialogues | Overlap with Validation/Test IDs | Later true Validation rows/dialogues | Later overlap with Validation/Test IDs |',
         '|---|---:|---:|---:|---:|---:|']
for dataset, data in results.items():
    for row in data['splits']:
        lines.append(f"| {dataset} | {row['seed']} | {row['historical_utterances']}/{row['historical_dialogues']} | {row['historical_validation_overlap']}/{row['historical_test_overlap']} | {row['later_utterances']}/{row['later_dialogues']} | {row['later_validation_overlap']}/{row['later_test_overlap']} |")
lines += ['', 'The overlaps are dialogue counts. For every seed, historical IDs exactly equal the registered Test IDs;'
          ' later IDs exactly equal the registered Validation IDs. Train, Validation and Test dialogue sets are disjoint.',
          '', '## Feature inventory', '',
          'The loader reads T1–T4 as four separate 1024-dimensional arrays; their source layer identity is not inferred.'
          ' The loader pads each batch with `pad_sequence`; `umask` is one on real utterances and zero on padding.'
          ' `qmask` is constructed separately and must remain unchanged. The model applies the configured BN to text'
          ' after input; the loader does not standardize audio or visual. Original feature zero is therefore not assumed'
          ' to equal the Training mean. No noise statistics were calculated or applied after the mismatch gate.', '',
          '| Dataset | Feature | Dimension | Train utterances | Validation utterances | Train all-zero | Validation all-zero |',
          '|---|---|---:|---:|---:|---:|---:|']
for dataset, data in results.items():
    for name, summary in data['features'].items():
        tr, va = summary['train'], summary['validation']
        lines.append(f"| {dataset} | {name} | {tr['dimension']} | {tr['utterances']} | {va['utterances']} | {tr['all_zero']} | {va['all_zero']} |")
lines += ['', 'Audio all-zero features and visual all-zero features are counted as missing evidence for future perturbation masking;'
          ' they are not deleted. Feature arrays in both splits are finite. The model produces log-probabilities;'
          ' later saved full probabilities are finite, normalized and agree with argmax predictions.', '',
          '## Stop reason and required resolution', '',
          'The historical file path cannot be used as evidence that its rows are Validation. The code confirms'
          ' it evaluates Test after Validation-based checkpoint selection. Establish whether the later'
          ' Validation inference artifact is accepted as the authoritative prediction source under the'
          ' XP04-only requirement, and independently verify its probabilities against fresh checkpoint inference.'
          ' Only then can the requested Pilot start. No Test label entered a calibration fit.', '',
          'See `verification.json` for per-seed machine-readable checks and `provenance/source_hashes.json` for input hashes.']
(OUT / 'XP04_BASELINE_AUDIT.md').write_text('\n'.join(lines) + '\n')
(OUT / 'XP04_FEATURE_AUDIT.md').write_text('\n'.join(lines[lines.index('## Feature inventory'):]) + '\n')
print(OUT)
