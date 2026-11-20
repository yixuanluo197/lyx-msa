#!/usr/bin/env python3
"""Compare native Fairseq 0.9 and HF on ONE identical public backbone (Train Fit only)."""
import argparse
import csv
import gc
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaModel


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--cosmic', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    vendor = args.out / 'local_only/framework_vendor/fairseq-0.9.0'
    # Fairseq 0.9 imports aliases removed in NumPy 1.24. No encoder forward is changed.
    for key, value in [('float', float), ('int', int), ('bool', bool)]:
        if key not in np.__dict__:
            setattr(np, key, value)
    sys.path.insert(0, str(vendor))
    import fairseq
    from fairseq.data import Dictionary
    from fairseq.models.roberta.model import RobertaModel as FairseqRobertaModel
    from fairseq.models.roberta.model import RobertaEncoder, roberta_large_architecture

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1475)
    # Freeze criteria and selection algorithm before either forward pass.
    protocol = dict(
        status='FROZEN_BEFORE_OUTPUTS', samples_per_dataset=32,
        sample_rule='First unique token-ID sequences in existing fit_indices order; no target-based selection',
        dataset_parts=['IEMOCAP Train Fit', 'MELD Train Fit'],
        official_validation_accessed=False, test_accessed=False, labels_accessed=False,
        allclose_atol=1e-4, allclose_rtol=1e-4, dtype='float32', mode='eval',
        layers=[-1, -2, -3, -4], pooling='BOS position 0', batch_size=4,
        architecture=dict(layers=24, hidden=1024, ffn=4096, heads=16,
                          layernorm_eps=1e-5, activation='exact GELU',
                          max_total_tokens=512, position_embedding_rows=514,
                          position_ids='padding_idx + cumulative non-padding token count',
                          pad_id=1, bos_id=0, eos_id=2,
                          attention_mask='Identical token-ID tensor; HF nonpad mask; native Fairseq derives pad mask from token ID 1',
                          dropout='All disabled by eval mode',
                          hf_attention_implementation='eager'),
        fairseq_tag='v0.9.0', fairseq_commit='df2f84ce619edddb80e720a56abc74d5490fed99',
        framework_scope='Native official Fairseq reference version; historical COSMIC runtime version unconfirmed',
        source_url='https://github.com/facebookresearch/fairseq/tree/v0.9.0',
        source_tar_sha256=sha(vendor.parent / 'fairseq-v0.9.0.tar.gz'),
        import_only_compatibility='Restore NumPy float/int/bool aliases at runtime; no source edits',
        pretrained=str(args.pretrained), pretrained_sha256=sha(args.pretrained / 'model.safetensors'),
        same_weights_method='Exact HF backbone state tensors copied without numerical conversion to native Fairseq encoder; all pairs checked torch.equal after load',
        cpu_or_gpu=str(args.device), torch_version=torch.__version__, numpy_version=np.__version__,
        fairseq_import_path=fairseq.__file__, source_files={
            str(f.relative_to(vendor)): sha(f) for f in [
                vendor / 'fairseq/models/roberta/model.py',
                vendor / 'fairseq/modules/transformer_sentence_encoder.py',
                vendor / 'fairseq/modules/transformer_sentence_encoder_layer.py',
                vendor / 'fairseq/modules/multihead_attention.py',
                vendor / 'fairseq/modules/layer_norm.py',
                vendor / 'fairseq/modules/gelu.py']})
    write(args.out / 'provenance/framework_protocol.json', protocol)

    prepared = {}
    selected = {}
    manifest = []
    for dataset in ['IEMOCAP', 'MELD']:
        path = args.parent / 'local_only' / dataset / 'train_prepared.pt'
        data = torch.load(path, map_location='cpu')
        fit = set(map(int, data['fit_indices']))
        seen = set()
        rows = []
        for index in map(int, data['fit_indices']):
            ids = tuple(data['input_ids'][index])
            if ids in seen:
                continue
            seen.add(ids)
            assert data['rows'][index]['inner_subset'] == 'fit'
            assert index in fit and index not in set(map(int, data['dev_indices']))
            rows.append((index, list(ids)))
            row = data['rows'][index]
            manifest.append(dict(dataset=dataset, row_index=index,
                                 dialogue_id=row['dialogue_id'], utterance_index=row['utterance_index'],
                                 token_count=len(ids), token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
            if len(rows) == 32:
                break
        assert len(rows) == 32
        selected[dataset] = rows
        prepared[dataset] = dict(path=str(path), sha256=sha(path), samples=32)
        del data
    with (args.out / 'results/framework_sample_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)

    hf = RobertaModel.from_pretrained(str(args.pretrained), local_files_only=True,
                                    add_pooling_layer=False, attn_implementation='eager').float().eval()
    hf_state = hf.state_dict()
    token_type = hf_state['embeddings.token_type_embeddings.weight']
    if torch.count_nonzero(token_type).item() != 0:
        raise RuntimeError('BLOCKED: HF token-type embedding is nonzero and has no native Fairseq counterpart')
    dictionary = Dictionary.load(str(args.cosmic / 'dict.txt'))
    dictionary.add_symbol('<mask>')
    assert (len(dictionary), dictionary.bos(), dictionary.pad(), dictionary.eos()) == (50265, 0, 1, 2)
    fs_args = Namespace(max_positions=512)
    roberta_large_architecture(fs_args)
    fs = FairseqRobertaModel(fs_args, RobertaEncoder(fs_args, dictionary)).float().eval()
    fs_encoder = fs.decoder.sentence_encoder
    mapping = {
        'embed_tokens.weight': 'embeddings.word_embeddings.weight',
        'embed_positions.weight': 'embeddings.position_embeddings.weight',
        'emb_layer_norm.weight': 'embeddings.LayerNorm.weight',
        'emb_layer_norm.bias': 'embeddings.LayerNorm.bias',
    }
    layer_map = {
        'self_attn.q_proj': 'attention.self.query',
        'self_attn.k_proj': 'attention.self.key',
        'self_attn.v_proj': 'attention.self.value',
        'self_attn.out_proj': 'attention.output.dense',
        'self_attn_layer_norm': 'attention.output.LayerNorm',
        'fc1': 'intermediate.dense', 'fc2': 'output.dense',
        'final_layer_norm': 'output.LayerNorm',
    }
    for i in range(24):
        for fs_name, hf_name in layer_map.items():
            for suffix in ['weight', 'bias']:
                mapping[f'layers.{i}.{fs_name}.{suffix}'] = f'encoder.layer.{i}.{hf_name}.{suffix}'
    assert set(mapping) == set(fs_encoder.state_dict()), 'Unmapped native Fairseq backbone tensor'
    assert set(mapping.values()) | {'embeddings.token_type_embeddings.weight'} == set(hf_state), 'Unmapped HF backbone tensor'
    fs_encoder.load_state_dict({key: hf_state[value] for key, value in mapping.items()}, strict=True)
    fs_state = fs_encoder.state_dict()
    audit = []
    for fs_key, hf_key in mapping.items():
        a, b = fs_state[fs_key], hf_state[hf_key]
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)
        audit.append(dict(fairseq_key=fs_key, hf_key=hf_key, shape=list(a.shape),
                          elements=a.numel(), dtype=str(a.dtype), bit_exact=True,
                          tensor_sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest()))
    weights = dict(status='PASS', mapped_backbone_tensors=len(audit),
                   mapped_backbone_elements=sum(x['elements'] for x in audit),
                   complete_fairseq_backbone_coverage=True, complete_hf_backbone_coverage=True,
                   hf_only_zero_token_type_embedding=True,
                   zero_token_type_shape=list(token_type.shape),
                   qkv_mapping='Three separate native Fairseq q_proj/k_proj/v_proj weight and bias tensors match HF query/key/value tensor-for-tensor',
                   excluded_heads='Neither framework MLM head is invoked; no pooler; output is encoder hidden states only',
                   all_pairs_bit_exact=True, tensors=audit)
    write(args.out / 'provenance/framework_weight_mapping.json', weights)
    del hf_state, fs_state, token_type
    gc.collect()

    # Keep both independent implementations on the SAME physical device.
    hf = hf.to(args.device)
    fs = fs.to(args.device)
    metrics = []
    outputs = {}
    for dataset in ['IEMOCAP', 'MELD']:
        hf_outputs, fs_outputs = [], []
        with torch.inference_mode():
            for start in range(0, 32, 4):
                group = selected[dataset][start:start + 4]
                length = max(len(tokens) for _, tokens in group)
                ids = torch.ones((len(group), length), dtype=torch.long, device=args.device)
                for row, (_, tokens) in enumerate(group):
                    ids[row, :len(tokens)] = torch.tensor(tokens, device=args.device)
                h = hf(input_ids=ids, attention_mask=ids.ne(1), output_hidden_states=True, return_dict=True)
                _, extra = fs(ids, features_only=True, return_all_hiddens=True)
                states = extra['inner_states']
                assert len(h.hidden_states) == len(states) == 25
                # Native Fairseq return_all_hiddens is [sequence, batch, hidden].
                hf_outputs.append(torch.stack([h.hidden_states[-k][:, 0, :] for k in (1, 2, 3, 4)], dim=1).cpu())
                fs_outputs.append(torch.stack([states[-k][0, :, :] for k in (1, 2, 3, 4)], dim=1).cpu())
        a = torch.cat(hf_outputs).numpy().astype(np.float64)
        b = torch.cat(fs_outputs).numpy().astype(np.float64)
        assert a.shape == b.shape == (32, 4, 1024)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        outputs[dataset] = dict(huggingface=a.astype(np.float32), fairseq=b.astype(np.float32))
        for slot in range(4):
            diff = a[:, slot] - b[:, slot]
            cosine = (a[:, slot] * b[:, slot]).sum(1) / (np.linalg.norm(a[:, slot], axis=1) * np.linalg.norm(b[:, slot], axis=1))
            metrics.append(dict(dataset=dataset, layer_from_end=slot + 1, samples=32,
                                mae=float(np.abs(diff).mean()), rmse=float(np.sqrt((diff ** 2).mean())),
                                max_abs_error=float(np.abs(diff).max()), mean_cosine=float(cosine.mean()),
                                min_cosine=float(cosine.min()), allclose=bool(np.allclose(a[:, slot], b[:, slot], atol=1e-4, rtol=1e-4))))
        print(json.dumps(dict(dataset=dataset, metrics=metrics[-4:])), flush=True)
    np.savez_compressed(args.out / 'local_only/framework_outputs.npz', **{
        dataset + '_' + framework: array for dataset, arrays in outputs.items() for framework, array in arrays.items()})
    with (args.out / 'results/framework_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    summary = dict(status='PASS' if all(x['allclose'] for x in metrics) else 'FAIL',
                   all_eight_layer_comparisons_pass=all(x['allclose'] for x in metrics),
                   actual_same_weights_verified=True, metrics=metrics, prepared_inputs=prepared,
                   protocol_sha256=sha(args.out / 'provenance/framework_protocol.json'),
                   weight_mapping_sha256=sha(args.out / 'provenance/framework_weight_mapping.json'),
                   script_sha256=sha(__file__), original_pkl_opened=False,
                   official_validation_accessed=False, test_accessed=False)
    write(args.out / 'results/framework_summary.json', summary)


if __name__ == '__main__':
    main()
