"""Frozen XP04 joint-only perturbations, original batches, paired RNG across model seeds."""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / 'provenance/inference_helpers'
sys.path.insert(0, str(HELPERS))
import run_pilot as xp
from graph_index_cache import install_graph_cache
from pilot_core import rng_for

RNG_SEED = 1475
INDICES = [0, 1, 2, 3, 5, 4]
NAMES = {0: 'text0', 1: 'text1', 2: 'text2', 3: 'text3', 5: 'audio', 4: 'visual'}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def original_reference(dataset, seed):
    path = xp.WORKSPACE / f'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z/runs/{dataset}/seed{seed}/validation_predictions.npz'
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return path, arrays


def changed_vectors(vectors, stats, dataset, dialogue, utterance, kind, repeat):
    output, rng_keys = {}, []
    for index, vector in vectors.items():
        name = NAMES[index]
        rng = rng_for(dataset, RNG_SEED, dialogue, utterance, name, kind, .05, repeat)
        x = np.asarray(vector, dtype=np.float64)
        std = stats[name + '_std'].astype(np.float64)
        if kind == 'gaussian':
            value = x + .05 * std * rng.normal(size=x.shape)
        else:
            eligible = np.flatnonzero(std > 0)
            count = int(round(.05 * len(eligible)))
            mask = np.zeros(x.shape, bool)
            mask[rng.choice(eligible, size=count, replace=False)] = True
            value = np.where(mask, stats[name + '_mean'].astype(np.float64), x)
            assert int(mask.sum()) == count
        output[index] = value.astype(np.float32)
        assert output[index].shape == np.asarray(vector).shape and np.isfinite(output[index]).all()
        assert np.array_equal(output[index][std == 0], np.asarray(vector, np.float32)[std == 0])
        rng_keys.append([dataset, RNG_SEED, dialogue, utterance, name, kind, .05, repeat])
    return output, rng_keys


def run(args):
    dest = ROOT / f'runs/{args.dataset}/seed{args.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'complete.json').exists(), 'Completed run exists; refusing to overwrite'
    assert not (dest / 'perturbed_predictions.csv').exists(), 'Partial run exists; archive or recover explicitly'
    reference_path, reference = original_reference(args.dataset, args.seed)
    stats_path = ROOT / f'provenance/{args.dataset}_train_statistics.npz'
    with np.load(stats_path) as z:
        stats = {k: z[k] for k in z.files}
    expected_train = 5283 if args.dataset == 'IEMOCAP' else 9989
    assert all(int(stats[n + '_count']) == expected_train for n in NAMES.values())
    checkpoint = xp.BASE / f'runs/cmerc/{args.dataset}_validation_seed{args.seed}/best.pt'
    initial_hashes = {str(p): xp.sha(p) for p in [checkpoint, reference_path, stats_path,
                      Path(__file__), HELPERS/'run_pilot.py', HELPERS/'pilot_core.py', HELPERS/'graph_index_cache.py']}
    resource = xp.resource_snapshot()
    write_json(dest/'resource_before.json', resource)
    api = xp.imports()
    torch = api[0]
    assert torch.__version__ == '1.13.1+cu117' and torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    os.chdir(xp.CODE)
    data = xp.dataset_for(args.dataset, 'validation', api[2])
    expected_n, expected_d = (527, 12) if args.dataset == 'IEMOCAP' else (1109, 114)
    assert len(data) == expected_d and len(reference['labels']) == expected_n
    assert set(map(str, data.keys)) == set(reference['dialogue_ids'])
    model = xp.load_model(args.dataset, args.seed, api)
    before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fresh = xp.infer(model, data, api)
    parity = xp.compare_alignment(fresh, reference)
    assert parity['passed'], parity
    write_json(dest/'validation_parity.json', parity)
    np.savez_compressed(dest/'fresh_validation_predictions.npz', **fresh)
    original_fields = ['dataset','seed','dialogue_id','utterance_index','label','prediction','confidence','probabilities','log_probabilities']
    with (dest/'original_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=original_fields); writer.writeheader()
        for i in range(expected_n):
            p = reference['probabilities'][i].astype(np.float64)
            writer.writerow({'dataset':args.dataset,'seed':args.seed,'dialogue_id':str(reference['dialogue_ids'][i]),
                'utterance_index':int(reference['utterance_indices'][i]),'label':int(reference['labels'][i]),
                'prediction':int(reference['predictions'][i]),'confidence':float(p.max()),
                'probabilities':json.dumps(p.tolist()),'log_probabilities':json.dumps(reference['log_probabilities'][i].tolist())})
    guard = xp.GraphGuard(model, torch)
    cache = install_graph_cache(model.graph_model)
    fields = ['dataset','seed','dialogue_id','utterance_index','modality','corruption','strength','repeat_id',
              'perturbation_seed','rng_key','visual_missing_skipped','prediction','confidence','probabilities','log_probabilities']
    rowbase, forwards, missing_count, cached_checks, mask_checks = 0, 0, 0, 0, 0
    started = time.monotonic()
    with (dest/'perturbed_predictions.csv').open('w', newline='') as stream, torch.no_grad():
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for batch_id, batch in enumerate(xp.loader_for(data, torch)):
            tensors = [x.cuda() for x in batch[:-1]]
            lengths = [int(x.sum().item()) for x in tensors[7]]
            baseline = [x.clone() for x in tensors]
            guard.reset(); cache.hyper_indices.clear(); cache.gnn_indices.clear(); cache.restore()
            raw_uncached = xp.forward(model, tensors, lengths)
            cache.enable()
            raw_cached = xp.forward(model, tensors, lengths)
            assert torch.allclose(raw_uncached, raw_cached, atol=1e-6, rtol=1e-5)
            assert np.allclose(raw_cached.cpu().numpy(), reference['log_probabilities'][rowbase:rowbase+sum(lengths)], atol=1e-6, rtol=1e-5)
            cached_checks += 1
            offset = 0
            for di, (dialogue, length) in enumerate(zip(batch[-1], lengths)):
                dialogue = str(dialogue)
                for ui in range(length):
                    row = rowbase + offset + ui
                    assert dialogue == str(reference['dialogue_ids'][row]) and ui == int(reference['utterance_indices'][row])
                    assert tensors[7][di,ui].item() == 1
                    missing = bool(torch.all(baseline[4][ui,di] == 0).item())
                    missing_count += int(missing)
                    active = [j for j in INDICES if not (j == 4 and missing)]
                    vectors = {j: baseline[j][ui,di].cpu().numpy().copy() for j in active}
                    for kind in ['gaussian','dropout']:
                        for repeat in range(10):
                            values, keys = changed_vectors(vectors, stats, args.dataset, dialogue, ui, kind, repeat)
                            try:
                                for j, v in values.items():
                                    tensors[j][ui,di].copy_(torch.from_numpy(v).to(tensors[j].device))
                                if ui == 0 and repeat == 0:
                                    # Full tensors verify that no context, mask or padding was altered.
                                    for j in range(9):
                                        expected = baseline[j].clone()
                                        if j in values: expected[ui,di].copy_(tensors[j][ui,di])
                                        assert torch.equal(expected,tensors[j]), 'Non-target feature or mask changed'
                                    mask_checks += 1
                                all_logp = xp.forward(model, tensors, lengths)
                                if di == 0 and ui == 0 and repeat == 0:
                                    cache.restore()
                                    uncached = xp.forward(model,tensors,lengths)
                                    cache.enable()
                                    assert torch.allclose(uncached,all_logp,atol=1e-6,rtol=1e-5)
                                    cached_checks += 1
                                lp = all_logp[offset+ui].detach().cpu().numpy()
                                p = np.exp(lp.astype(np.float64))
                                assert np.isfinite(p).all() and abs(p.sum()-1) < 1e-5
                            finally:
                                for j in active: tensors[j][ui,di].copy_(baseline[j][ui,di])
                            writer.writerow({'dataset':args.dataset,'seed':args.seed,'dialogue_id':dialogue,
                                'utterance_index':ui,'modality':'joint','corruption':kind,'strength':.05,'repeat_id':repeat,
                                'perturbation_seed':RNG_SEED,'rng_key':json.dumps(keys),'visual_missing_skipped':missing,
                                'prediction':int(p.argmax()),'confidence':float(p.max()),'probabilities':json.dumps(p.tolist()),
                                'log_probabilities':json.dumps(lp.tolist())})
                            forwards += 1
                    if (row+1) % 10 == 0:
                        stream.flush()
                        elapsed = time.monotonic()-started
                        status = {'dataset':args.dataset,'seed':args.seed,'utterances':row+1,'total':expected_n,
                                  'forwards':forwards,'seconds':elapsed,'forwards_per_second':forwards/elapsed,
                                  'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
                        print(json.dumps(status),flush=True); write_json(dest/'progress.json',status)
                offset += length
            assert all(torch.equal(x,y) for x,y in zip(tensors,baseline)), 'Batch restore mismatch'
            rowbase += sum(lengths)
    assert rowbase == expected_n and forwards == expected_n * 20
    assert missing_count == (1 if args.dataset == 'IEMOCAP' else 95)
    assert all(torch.equal(v.detach().cpu(),before_state[k]) for k,v in model.state_dict().items())
    guard.close()
    assert all(xp.sha(p)==h for p,h in initial_hashes.items()), 'Frozen source changed'
    complete = {'status':'completed','dataset':args.dataset,'seed':args.seed,'utterances':expected_n,'dialogues':expected_d,
                'perturbed_rows':forwards,'seconds':time.monotonic()-started,'visual_missing_retained':missing_count,
                'graph_hook_checks':guard.count,'cached_uncached_parity_checks':cached_checks,'target_mask_checks':mask_checks,
                'weights_unchanged':True,'batch_inputs_restored':True,'source_sha256':initial_hashes,
                'parity':parity,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'graph_cache_stats':cache.stats,
                'resource':resource}
    write_json(dest/'complete.json',complete)
    print(json.dumps({'completed':args.dataset,'seed':args.seed,'seconds':complete['seconds']}),flush=True)
    del model, data; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['IEMOCAP','MELD'],required=True)
    parser.add_argument('--seed',type=int,choices=[67137,2024,3407,42],required=True)
    args=parser.parse_args()
    run(args)
