"""XP04 seed-1475 Validation-only feature perturbation Pilot."""
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/data2/yb/paper/lyx-msa')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z/scripts'
sys.path.insert(0, str(OLD))
import run_pilot as xp
from pilot_core import behavior_metrics, rng_for
from graph_index_cache import install_graph_cache

SEED = 1475
REPEATS = 10
SPECS = [('gaussian', 0.05), ('dropout', 0.05)]
MODALITIES = {'text': [0, 1, 2, 3], 'audio': [5], 'visual': [4],
              'joint': [0, 1, 2, 3, 5, 4]}
NAMES = {0: 'text0', 1: 'text1', 2: 'text2', 3: 'text3', 5: 'audio', 4: 'visual'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def probabilities(log_prob):
    p = np.exp(np.asarray(log_prob, dtype=np.float64))
    assert np.isfinite(p).all() and np.allclose(p.sum(-1), 1, atol=1e-5)
    return p


def corrupt(base, indices, kind, strength, dataset, dialogue, utterance, modality, repeat, stats):
    variant = [x.clone() if hasattr(x, 'clone') else x for x in base]
    if 4 in indices and np.all(np.asarray(base[4][utterance]) == 0):
        indices = [i for i in indices if i != 4]
    changed = []
    for index in indices:
        name = NAMES[index]
        x = base[index][utterance].numpy().astype(np.float64)
        rng = rng_for(dataset, SEED, dialogue, utterance, name, kind, strength, repeat)
        if kind == 'gaussian':
            value = x + strength * stats[name + '_std'].astype(np.float64) * rng.normal(size=x.shape)
        else:
            eligible = np.flatnonzero(stats[name + '_std'] > 0)
            count = int(round(strength * len(eligible)))
            mask = np.zeros(x.shape, dtype=bool)
            mask[rng.choice(eligible, size=count, replace=False)] = True
            value = np.where(mask, stats[name + '_mean'].astype(np.float64), x)
        value = value.astype(np.float32)
        assert np.isfinite(value).all() and value.shape == x.shape
        variant[index][utterance] = variant[index][utterance].new_tensor(value)
        changed.append(index)
    for index in range(9):
        if index not in changed:
            assert np.array_equal(variant[index].numpy(), base[index].numpy())
        elif len(base[index]) > 1:
            other = np.ones(len(base[index]), dtype=bool)
            other[utterance] = False
            assert np.array_equal(variant[index].numpy()[other], base[index].numpy()[other])
    return tuple(variant), bool(changed)


def dataset_run(dataset_name, api, resource):
    torch = api[0]
    target = OUT / dataset_name
    target.mkdir(exist_ok=True)
    dataset = xp.dataset_for(dataset_name, 'validation', api[2])
    true_n = 527 if dataset_name == 'IEMOCAP' else 1109
    true_d = 12 if dataset_name == 'IEMOCAP' else 114
    assert len(dataset) == true_d
    model = xp.load_model(dataset_name, SEED, api)
    original = xp.infer(model, dataset, api)
    assert len(original['labels']) == true_n
    prior_path = (ROOT / 'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z/runs'
                  / dataset_name / f'seed{SEED}/validation_predictions.npz')
    with np.load(prior_path) as f:
        prior = {name: f[name] for name in f.files}
    parity = xp.compare_alignment(original, prior)
    assert parity['passed'], parity
    cache = install_graph_cache(model.graph_model)
    np.savez_compressed(target / 'original_predictions.npz', **original)
    stats = xp.training_stats(OUT, dataset_name, api)
    assert all(int(stats[name + '_count']) == (5283 if dataset_name == 'IEMOCAP' else 9989)
               for name in ('text0', 'text1', 'text2', 'text3', 'audio', 'visual'))
    check = {'dataset': dataset_name, 'seed': SEED, 'dialogues': true_d,
             'utterances': true_n, 'checkpoint': str(xp.BASE / f'runs/cmerc/{dataset_name}_validation_seed{SEED}/best.pt'),
             'checkpoint_sha256': sha(xp.BASE / f'runs/cmerc/{dataset_name}_validation_seed{SEED}/best.pt'),
             'prior_validation_file': str(prior_path), 'prior_sha256': sha(prior_path),
             'parity': parity, 'train_stat_counts': {n: int(stats[n + '_count']) for n in NAMES.values()},
             'gpu': resource, 'target_forwards': 0, 'skipped_visual_targets': 0}
    write_json(target / 'validation_parity.json', check)
    orig_fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'label', 'prediction',
                   'confidence', 'probabilities', 'log_probabilities']
    pert_fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'modality', 'corruption',
                   'strength', 'repeat_id', 'rng_key', 'visual_missing_skipped', 'prediction',
                   'confidence', 'probabilities']
    behavior_fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'modality', 'corruption',
                       'strength', 'original_confidence', 'mu', 'delta', 'js', 'flip_rate',
                       'confidence_change_mean', 'confidence_change_std', 'visual_missing_skipped']
    with (target / 'original_predictions.csv').open('w', newline='') as fo, \
         (target / 'perturbed_predictions.csv').open('w', newline='') as fp, \
         (target / 'behavior_features.csv').open('w', newline='') as fb:
        wo, wp, wb = (csv.DictWriter(f, fieldnames=fields) for f, fields in
                      ((fo, orig_fields), (fp, pert_fields), (fb, behavior_fields)))
        for w in (wo, wp, wb):
            w.writeheader()
        row = 0
        started = time.monotonic()
        dialogue_cursor = 0
        for batch in xp.loader_for(dataset, torch):
            tensors = [x.cuda() for x in batch[:-1]]
            lengths = [int(x.sum().item()) for x in tensors[7]]
            size = sum(lengths)
            baseline = [x.clone() for x in tensors]
            cache.hyper_indices.clear()
            cache.gnn_indices.clear()
            with torch.no_grad():
                raw_baseline = xp.forward(model, tensors, lengths).detach().cpu().numpy()
            assert np.allclose(raw_baseline, original['log_probabilities'][row:row + size],
                               atol=1e-6, rtol=1e-5), 'Mixed-batch baseline parity failed'
            offset = 0
            for di, (dialogue, length) in enumerate(zip(batch[-1], lengths)):
                base = dataset[dialogue_cursor + di]
                dialogue = str(dialogue)
                assert str(base[-1]) == dialogue
                for ui in range(length):
                    assert original['dialogue_ids'][row] == dialogue
                    assert original['utterance_indices'][row] == ui
                    op = original['probabilities'][row].astype(np.float64)
                    oc = float(op.max())
                    wo.writerow(dict(dataset=dataset_name, seed=SEED, dialogue_id=dialogue,
                                     utterance_index=ui, label=int(original['labels'][row]),
                                     prediction=int(original['predictions'][row]), confidence=oc,
                                     probabilities=json.dumps(op.tolist()),
                                     log_probabilities=json.dumps(original['log_probabilities'][row].tolist())))
                    pred_probs, descriptors = [], []
                    for modality, indices in MODALITIES.items():
                        missing = 4 in indices and bool(np.all(base[4][ui].numpy() == 0))
                        if missing:
                            check['skipped_visual_targets'] += 1
                        for kind, strength in SPECS:
                            for repeat in range(REPEATS):
                                variant, changed = corrupt(base, indices, kind, strength, dataset_name,
                                                           dialogue, ui, modality, repeat, stats)
                                active = [j for j in indices if not (j == 4 and missing)]
                                if active:
                                    try:
                                        for j in active:
                                            tensors[j][ui, di].copy_(variant[j][ui].to(tensors[j].device))
                                        with torch.no_grad():
                                            logp = xp.forward(model, tensors, lengths)[offset + ui].detach().cpu().numpy()
                                        p = probabilities(logp)
                                    finally:
                                        for j in active:
                                            tensors[j][ui, di].copy_(baseline[j][ui, di])
                                    check['target_forwards'] += 1
                                else:
                                    p = op.copy()
                                pred_probs.append(p)
                                descriptors.append((modality, kind, strength, repeat, missing, changed))
                    pred_probs = np.asarray(pred_probs)
                    assert pred_probs.shape == (len(descriptors), len(op)) and np.isfinite(pred_probs).all()
                    for index, (modality, kind, strength, repeat, missing, changed) in enumerate(descriptors):
                        p = pred_probs[index]
                        key = [[dataset_name, SEED, dialogue, ui, NAMES[j], kind, strength, repeat]
                               for j in MODALITIES[modality]
                               if not (j == 4 and missing)]
                        wp.writerow(dict(dataset=dataset_name, seed=SEED, dialogue_id=dialogue,
                                         utterance_index=ui, modality=modality, corruption=kind,
                                         strength=strength, repeat_id=repeat, rng_key=json.dumps(key),
                                         visual_missing_skipped=missing, prediction=int(p.argmax()),
                                         confidence=float(p.max()), probabilities=json.dumps(p.tolist())))
                    for modality in MODALITIES:
                        for kind, strength in SPECS:
                            selected = [pred_probs[i] for i, desc in enumerate(descriptors)
                                        if desc[0] == modality and desc[1] == kind]
                            q = np.stack(selected)
                            metric = behavior_metrics(op[None, :], q[None, :, :])
                            conf = q.max(axis=1)
                            wb.writerow(dict(dataset=dataset_name, seed=SEED, dialogue_id=dialogue,
                                             utterance_index=ui, modality=modality, corruption=kind,
                                             strength=strength, original_confidence=oc,
                                             mu=float(metric['mu'][0]), delta=float(metric['delta_original'][0]),
                                             js=float(metric['js_divergence'][0]),
                                             flip_rate=float(metric['flip_rate'][0]),
                                             confidence_change_mean=float(np.mean(conf - oc)),
                                             confidence_change_std=float(np.std(conf - oc)),
                                             visual_missing_skipped=4 in MODALITIES[modality] and
                                             bool(np.all(base[4][ui].numpy() == 0))))
                    row += 1
                    if row % 25 == 0:
                        print(json.dumps({'dataset': dataset_name, 'utterances_done': row,
                                          'seconds': round(time.monotonic() - started, 1)}), flush=True)
                        for stream in (fo, fp, fb):
                            stream.flush()
                        write_json(target / 'progress.json', {'dialogues_done': dialogue_cursor,
                                                              'utterances_done': row})
                offset += length
            assert offset == size
            assert all(torch.equal(x, y) for x, y in zip(tensors, baseline)), 'Batch inputs were not restored'
            dialogue_cursor += len(lengths)
            if dialogue_cursor % 16 == 0 or dialogue_cursor == len(dataset):
                print(json.dumps({'dataset': dataset_name, 'dialogues_done': dialogue_cursor,
                                  'utterances_done': row, 'seconds': round(time.monotonic() - started, 1)}), flush=True)
                write_json(target / 'progress.json', {'dialogues_done': dialogue_cursor, 'utterances_done': row})
    assert row == true_n
    check['seconds'] = time.monotonic() - started
    check['completed'] = True
    write_json(target / 'validation_parity.json', check)
    del model, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return check


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '3')
    api = xp.imports()
    torch = api[0]
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    os.chdir(xp.CODE)
    resource = xp.resource_snapshot()
    write_json(OUT / 'provenance/gpu_before.json', resource)
    dataset_name = sys.argv[1]
    assert dataset_name in ('IEMOCAP', 'MELD')
    result = dataset_run(dataset_name, api, resource)
    print(json.dumps({'complete': dataset_name, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
