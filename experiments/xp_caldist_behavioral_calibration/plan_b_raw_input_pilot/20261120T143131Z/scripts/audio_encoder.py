#!/usr/bin/env python3
"""Frozen Plan B waveform encoder and source-verified mono16k audio cache.

Only the local pinned facebook/wav2vec2-base checkpoint is accepted. Extraction
uses train/validation data only and writes new features outside historical assets.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'local_only/models/facebook_wav2vec2_base'
MODEL_REVISION = '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'
FFMPEG = '/home/yangbin/miniconda3/envs/multimodalerc310/bin/ffmpeg'
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE
MIN_SAMPLES = 400
PROTOCOL = {
    'model': 'facebook/wav2vec2-base', 'revision': MODEL_REVISION,
    'feature': 'last_hidden_state', 'dimension': 768,
    'sample_rate': SAMPLE_RATE, 'channels': 1, 'decode_dtype': 'float32le',
    'decode': 'ffmpeg full source audio stream0, no trimming, mono average, resample16000',
    'window_samples': WINDOW_SAMPLES, 'window_overlap': 0,
    'short_final_window': 'merge residual<400samples into preceding window',
    'waveform_normalization': 'official Wav2Vec2FeatureExtractor over complete utterance before windowing',
    'pooling': 'sum last hidden valid frames across nonoverlapping windows divided by total real output frames',
    'batch_size': 1, 'precision': 'float32', 'mode': 'eval_inference_all_parameters_frozen',
    'silence_or_missing': 'zero768 plus audio_missing_mask=True',
    'missing_criteria': 'None/nonexistent, decode failure/timeout, <400 samples, nonfinite, exact zero RMS or variance',
    'padding': 'none; no attention mask for group-norm pretrained base on single unpadded example',
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def waveform_quality(samples):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError('samples16k must be one-dimensional mono waveform')
    result = {'samples': int(len(samples)), 'sample_rate': SAMPLE_RATE,
              'duration_seconds': len(samples)/SAMPLE_RATE, 'finite': bool(np.isfinite(samples).all())}
    if len(samples) and result['finite']:
        result.update(rms=float(np.sqrt(np.mean(samples.astype(np.float64)**2))),
                      peak=float(np.max(np.abs(samples))),
                      clipped_fraction=float(np.mean(np.abs(samples) >= .999)),
                      variance=float(np.var(samples.astype(np.float64))))
    else:
        result.update(rms=0., peak=0., clipped_fraction=0., variance=0.)
    if not result['finite']:
        result['missing_reason'] = 'nonfinite_audio'
    elif len(samples) < MIN_SAMPLES:
        result['missing_reason'] = 'empty_or_too_short_for_convolution'
    elif result['rms'] == 0. or result['variance'] == 0.:
        result['missing_reason'] = 'zero_or_constant_audio'
    else:
        result['missing_reason'] = None
    result['audio_missing'] = result['missing_reason'] is not None
    return result


@lru_cache(maxsize=1)
def ffmpeg_sha256():
    return sha256_file(FFMPEG)


def read_audio_16k(path, cache_dir=None):
    """Decode complete original audio to float32 mono16k; return waveform, metadata.

    Cache entries are keyed by source bytes and exact decoding protocol. Existing
    historical caches are not trusted without their source/config/hash metadata.
    """
    if path is None or not Path(path).is_file():
        return np.empty(0, np.float32), {'source_path': str(path) if path else None,
            'audio_missing': True, 'missing_reason': 'source_missing', 'samples': 0}
    path = Path(path)
    source_sha = sha256_file(path)
    decode_config = {'ffmpeg_binary_sha256': ffmpeg_sha256(), 'sample_rate': SAMPLE_RATE,
                     'channels': 1, 'format': 'f32le', 'stream': '0:a:0', 'no_trim': True}
    key = stable_hash({'source_sha256': source_sha, 'decode_config': decode_config})
    cache_path = Path(cache_dir) / (key + '.npy') if cache_dir else None
    meta_path = cache_path.with_suffix('.json') if cache_path else None
    if cache_path and cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get('source_sha256') == source_sha and meta.get('decode_config') == decode_config
                and meta.get('cache_sha256') == sha256_file(cache_path)):
            audio = np.load(cache_path, allow_pickle=False)
            assert audio.dtype == np.float32 and audio.ndim == 1
            return audio, {**meta, 'source_path': str(path.resolve()), 'cache_hit': True}
    command = [FFMPEG, '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0',
               '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', 'pipe:1']
    try:
        proc = subprocess.run(command, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return np.empty(0, np.float32), {'source_path': str(path.resolve()),
            'source_sha256': source_sha, 'decode_config': decode_config, 'command': command,
            'audio_missing': True, 'missing_reason': 'decode_timeout', 'samples': 0}
    meta = {'source_path': str(path.resolve()), 'source_sha256': source_sha,
            'decode_config': decode_config, 'command': command, 'decode_returncode': proc.returncode,
            'decode_stderr': proc.stderr.decode(errors='replace'), 'cache_hit': False}
    if proc.returncode != 0:
        return np.empty(0, np.float32), {**meta, 'audio_missing': True,
                                      'missing_reason': 'decode_failure', 'samples': 0}
    audio = np.frombuffer(proc.stdout, dtype='<f4').copy()
    meta.update(waveform_quality(audio))
    meta['waveform_sha256'] = hashlib.sha256(audio.tobytes()).hexdigest()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_temp = Path(str(cache_path) + f'.{os.getpid()}.{threading.get_ident()}.partial')
        with open(cache_temp, 'wb') as stream:
            np.save(stream, audio, allow_pickle=False)
        cache_temp.replace(cache_path)
        meta['cache_path'] = str(cache_path)
        meta['cache_sha256'] = sha256_file(cache_path)
        write_json(meta_path, meta)
    return audio, meta


class AudioEncoder:
    def __init__(self, device='cpu', model_dir=None):
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
        import transformers
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_dir = Path(model_dir or MODEL_DIR)
        manifest = json.loads((ROOT / 'provenance/audio_model_files.json').read_text())
        if manifest['repo'] != 'facebook/wav2vec2-base' or manifest['revision'] != MODEL_REVISION:
            raise ValueError('Unregistered checkpoint identity')
        required = {'config.json', 'preprocessor_config.json', 'pytorch_model.bin'}
        if not required.issubset({item['name'] for item in manifest['files']}):
            raise RuntimeError('Pinned model download is incomplete')
        for item in manifest['files']:
            if item['name'] in ['config.json', 'preprocessor_config.json', 'pytorch_model.bin']:
                if sha256_file(model_dir / item['name']) != item['sha256']:
                    raise ValueError('Checkpoint/config hash mismatch: ' + item['name'])
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        # Official checkpoint uses the legacy weight_norm g/v names. Torch2.2's
        # positional convolution uses parametrizations.original0/original1.
        # Rename only these state keys; strict-load every base tensor unchanged.
        config = Wav2Vec2Config.from_pretrained(model_dir, local_files_only=True)
        config._attn_implementation = 'eager'
        self.model = Wav2Vec2Model(config).float()
        source = torch.load(model_dir / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        prefix = 'wav2vec2.'
        renames = {
            'encoder.pos_conv_embed.conv.weight_g': 'encoder.pos_conv_embed.conv.parametrizations.weight.original0',
            'encoder.pos_conv_embed.conv.weight_v': 'encoder.pos_conv_embed.conv.parametrizations.weight.original1',
        }
        base_state = {renames.get(key[len(prefix):], key[len(prefix):]): value
                      for key, value in source.items() if key.startswith(prefix)}
        excluded = sorted(key for key in source if not key.startswith(prefix))
        assert set(excluded) == {'quantizer.codevectors', 'quantizer.weight_proj.weight',
            'quantizer.weight_proj.bias', 'project_q.weight', 'project_q.bias',
            'project_hid.weight', 'project_hid.bias'}
        loaded = self.model.load_state_dict(base_state, strict=True)
        exact = all(torch.equal(value, self.model.state_dict()[key]) for key, value in base_state.items())
        assert exact and not loaded.missing_keys and not loaded.unexpected_keys
        conv = self.model.encoder.pos_conv_embed.conv
        legacy_kernel = torch._weight_norm(source[prefix + 'encoder.pos_conv_embed.conv.weight_v'],
            source[prefix + 'encoder.pos_conv_embed.conv.weight_g'], dim=2)
        kernel_error = float((conv.weight.detach() - legacy_kernel).abs().max())
        assert kernel_error == 0.
        loading = {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': [],
            'excluded_pretraining_head_keys': excluded, 'state_key_mapping_only': renames,
            'base_state_tensors': len(base_state), 'all_base_state_tensors_exact': exact,
            'legacy_weight_norm_dim': 2, 'weight_norm_kernel_max_abs_error': kernel_error}
        del source, base_state, legacy_kernel
        assert self.model.config.hidden_size == 768
        assert self.processor.sampling_rate == SAMPLE_RATE and self.processor.do_normalize
        self.model.eval().requires_grad_(False).to(self.device)
        self.last_metadata = {}
        self.metadata = {'protocol': PROTOCOL, 'protocol_hash': stable_hash(PROTOCOL),
                         'checkpoint_manifest': manifest, 'processor': self.processor.to_dict(),
                         'encoder_code_sha256': sha256_file(__file__),
                         'torch': torch.__version__, 'transformers': transformers.__version__,
                         'device': str(self.device), 'loading_info': loading,
                         'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'gpu_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
                         'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    @torch.inference_mode()
    def extract_waveform(self, samples16k):
        """Return CPU float32[768]; missing details are in .last_metadata."""
        samples = np.asarray(samples16k, np.float32)
        quality = waveform_quality(samples)
        self.last_metadata = {**quality, 'windows': 0, 'valid_frames': 0}
        if quality['audio_missing']:
            return torch.zeros(768, dtype=torch.float32)
        normalized = self.processor(samples, sampling_rate=SAMPLE_RATE, padding=False,
                                    return_tensors='pt').input_values[0]
        windows = list(normalized.split(WINDOW_SAMPLES))
        if len(windows) > 1 and len(windows[-1]) < MIN_SAMPLES:
            windows[-2] = torch.cat([windows[-2], windows[-1]])
            windows.pop()
        total, count = torch.zeros(768, device=self.device, dtype=torch.float32), 0
        for window in windows:
            output = self.model(input_values=window[None].to(self.device), return_dict=True).last_hidden_state[0]
            assert output.dtype == torch.float32 and output.shape[1] == 768
            assert torch.isfinite(output).all()
            total += output.sum(dim=0)
            count += output.shape[0]
        embedding = (total/count).cpu()
        assert torch.isfinite(embedding).all()
        self.last_metadata.update(windows=len(windows), window_input_samples=[len(w) for w in windows],
                                  valid_frames=count, embedding_sha256=hashlib.sha256(embedding.numpy().tobytes()).hexdigest())
        return embedding


def load_audio_encoder(device='cpu', model_dir=None):
    return AudioEncoder(device=device, model_dir=model_dir)


load_encoder = load_audio_encoder


def extract_waveform(samples16k, encoder=None, device='cpu'):
    if encoder is None:
        raise ValueError('Pass a shared encoder=load_audio_encoder(...); implicit model loading is disabled')
    return encoder.extract_waveform(samples16k)


def save_features(path, features):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.partial')
    torch.save(features, temp)
    temp.replace(path)


def extract_dataset(args):
    data_path, output_path = Path(args.data), Path(args.output)
    dataset = torch.load(data_path, map_location='cpu')
    data_sha = sha256_file(data_path)
    encoder = load_audio_encoder(args.device, args.model_dir)
    expected = {'dataset_sha256': data_sha, 'protocol_hash': stable_hash(PROTOCOL),
                'encoder_code_sha256': sha256_file(__file__)}
    result = {'train': {}, 'validation': {}, 'audio_missing_mask': {'train': {}, 'validation': {}},
              'metadata': {**encoder.metadata, **expected, 'dataset': args.dataset,
                           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
              'utterance_metadata': {'train': {}, 'validation': {}}}
    if output_path.exists():
        old = torch.load(output_path, map_location='cpu')
        if any(old['metadata'].get(k) != v for k, v in expected.items()):
            raise ValueError('Refusing incompatible feature overwrite')
        result = old
    splits = dataset.get('splits', dataset)
    unsaved = 0
    for split in ['train', 'validation']:
        for dialogue in splits[split]:
            did = dialogue['dialogue_id']
            if did in result[split]:
                continue
            vectors, missing, records = [], [], []
            assert len(dialogue['audio_paths']) == len(dialogue['utterance_ids'])
            for uid, path in zip(dialogue['utterance_ids'], dialogue['audio_paths']):
                samples, decode = read_audio_16k(path, args.cache_dir)
                vector = encoder.extract_waveform(samples)
                info = {**decode, **encoder.last_metadata, 'utterance_id': uid}
                if decode.get('audio_missing'):
                    info['missing_reason'] = decode['missing_reason']
                    info['audio_missing'] = True
                vectors.append(vector)
                missing.append(info['audio_missing'])
                records.append(info)
            result[split][did] = torch.stack(vectors)
            result['audio_missing_mask'][split][did] = torch.tensor(missing, dtype=torch.bool)
            result['utterance_metadata'][split][did] = records
            unsaved += 1
            if unsaved >= 25:
                save_features(output_path, result)
                unsaved = 0
            print(json.dumps({'split': split, 'dialogue': did, 'utterances': len(vectors),
                              'missing': sum(missing), 'completed_dialogues': len(result[split])}), flush=True)
    result['metadata']['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    result['metadata']['counts'] = {s: {'dialogues': len(result[s]),
        'utterances': sum(len(v) for v in result[s].values()),
        'audio_missing': sum(int(v.sum()) for v in result['audio_missing_mask'][s].values())}
        for s in ['train', 'validation']}
    save_features(output_path, result)
    write_json(output_path.with_suffix('.metadata.json'), result['metadata'])
    print(json.dumps(result['metadata']['counts']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'decode-smoke', 'extract'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-dir', default=str(MODEL_DIR))
    parser.add_argument('--audio')
    parser.add_argument('--cache-dir')
    parser.add_argument('--report')
    parser.add_argument('--data')
    parser.add_argument('--output')
    parser.add_argument('--dataset', choices=['iemocap', 'meld'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'extract':
        if not all([args.data, args.output, args.dataset, args.cache_dir]):
            parser.error('extract requires --dataset --data --output --cache-dir')
        extract_dataset(args)
        return
    if args.audio:
        samples, decode = read_audio_16k(args.audio, args.cache_dir)
    else:
        if args.command == 'decode-smoke':
            parser.error('decode-smoke requires --audio')
        samples = (.05*np.sin(np.arange(SAMPLE_RATE, dtype=np.float32)*2*np.pi*220/SAMPLE_RATE)).astype(np.float32)
        decode = {'synthetic_smoke': True}
    report = {'decode': decode, 'quality': waveform_quality(samples), 'protocol': PROTOCOL}
    if args.command == 'smoke':
        encoder = load_audio_encoder(args.device, args.model_dir)
        a = encoder.extract_waveform(samples)
        first = dict(encoder.last_metadata)
        b = encoder.extract_waveform(samples)
        zero = encoder.extract_waveform(np.zeros(SAMPLE_RATE, np.float32))
        assert a.shape == (768,) and a.dtype == torch.float32 and torch.isfinite(a).all()
        assert torch.equal(a,b) and torch.equal(zero, torch.zeros(768))
        assert encoder.metadata['trainable_parameters'] == 0 and not encoder.model.training
        report.update(encoder=encoder.metadata, embedding=first, repeat_exact=True,
                      zero_waveform_returns_zero=True, zero_waveform_missing_mask=encoder.last_metadata['audio_missing'],
                      original_waveform_missing_mask=first['audio_missing'])
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
