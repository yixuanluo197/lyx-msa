#!/usr/bin/env python3
"""Train-only candidate IS13 extraction; these are NOT the original MELD front end.

Frozen 8-condition probe is selected without labels or Validation features.
Float WAV/cWaveSource avoids opensmile-python's int16 overflow for z-scored audio.
"""
import argparse
import concurrent.futures
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import numpy as np
from scipy.io import wavfile
import opensmile
from opensmile.core.lib import OpenSMILE

OUT = Path(__file__).resolve().parents[1]
PRIOR = OUT.parent / '20261117T014409Z'
ROOT = OUT / 'local_only/MELD'
CONF = OUT / 'configs/meld'
FFMPEG = '/LOCAL/USER/miniconda3/envs/msa_extract/bin/ffmpeg'
FFPROBE = '/LOCAL/USER/miniconda3/envs/msa_extract/bin/ffprobe'
RAW = Path('/LOCAL/MELD_ASSETS/raw/MELD/MELD.Raw/train_splits')
CONDITIONS = [f'{mix}_{sr}_{norm}' for mix in ('meanall', 'ffmpegmono')
              for sr in ('native48k', '16k') for norm in ('none', 'zscore')]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def savej(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare_config():
    source = PRIOR / 'sources/opensmile/v3.0.1/config'
    CONF.mkdir(parents=True, exist_ok=True)
    # Copy the actual complete shared/core include tree already audited; do not substitute 2016.
    for dirname in ('is09-13', 'shared'):
        shutil.copytree(source / dirname, CONF / dirname, dirs_exist_ok=True)
    path = CONF / 'is09-13/IS13_ComParE.conf'
    original = path.read_text()
    wrapped = original.replace('\\{../shared/standard_data_output.conf.inc}',
                               '\\{../shared/standard_external_data_output_single.conf.inc}')
    path.write_text(wrapped)
    installed = Path(opensmile.__file__).parent / 'core/config/shared/standard_external_data_output_single.conf.inc'
    shutil.copy2(installed, CONF / 'shared/standard_external_data_output_single.conf.inc')
    (CONF/'IS13_source_wrapper.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), wrapped.splitlines(True), fromfile='official_IS13', tofile='candidate_IS13')))
    manifest = []
    for p in sorted(CONF.rglob('*')):
        if p.is_file() and p.name not in ('manifest.json', 'feature_fields.json'):
            manifest.append({'path': str(p.relative_to(CONF)), 'sha256': sha(p), 'bytes': p.stat().st_size})
    savej(CONF / 'manifest.json', {'candidate_not_original': True, 'copied_from': str(source),
        'modification': 'Only output sink changed; original wave source and IS13 core retained.', 'files': manifest})


def smile_features(wav_path):
    instance = OpenSMILE()
    instance.initialize(config_file=str(CONF/'is09-13/IS13_ComParE.conf'),
        options={'inputfile': str(wav_path), 'sinkLevel': 'func'}, loglevel=0)
    names = [instance.external_sink_get_element_name('extsink', i)
             for i in range(instance.external_sink_get_num_elements('extsink'))]
    ys, starts, ends = [], [], []
    callback = opensmile.Smile._sink_callback(ys, starts, ends)
    instance.external_sink_set_callback_ex('extsink', callback)
    instance.run()
    instance.free()
    arr = np.vstack(ys)
    if arr.shape != (1, 6373):
        raise ValueError(f'Expected one 6373D utterance, got {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError('Nonfinite IS13 features')
    return arr[0], names


def get_id(row, candidates):
    for field in candidates:
        if field in row:
            return int(row[field])
    raise KeyError(f'No {candidates} in manifest columns {list(row)}')


def process_one(item):
    idx, row, selected_conditions = item
    dia = get_id(row, ('official_dialogue_id', 'raw_dialogue_id', 'dialogue_id', 'Dialogue_ID'))
    utt = get_id(row, ('raw_utterance_id', 'utterance_id', 'Utterance_ID', 'utterance_index'))
    media = Path(row.get('media_path') or row.get('audio_path') or row.get('raw_audio_path')
                 or row.get('source_path') or RAW / f'dia{dia}_utt{utt}.mp4')
    result = {'row_index': idx, 'dialogue_id': dia, 'utterance_id': utt, 'media_path': str(media),
              'media_sha256': None, 'conditions': {}, 'status': 'ok'}
    arrays = {}
    try:
        result['media_sha256'] = sha(media)
        probe_cmd = [FFPROBE, '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=sample_rate,channels,channel_layout,codec_name', '-of', 'json', str(media)]
        stream = json.loads(subprocess.run(probe_cmd, check=True, capture_output=True).stdout)['streams'][0]
        result['original_audio'] = stream
        native_sr = int(stream['sample_rate'])
        channels = int(stream['channels'])
        if channels < 1 or native_sr != 48000:
            raise ValueError(f'Predeclared native48k candidate incompatible with {stream}')
        with tempfile.TemporaryDirectory(prefix='meld_candidate_') as temp:
            for mix in ('meanall', 'ffmpegmono'):
                for srname, rate in (('native48k', 48000), ('16k', 16000)):
                    conds = [c for c in selected_conditions if c.startswith(f'{mix}_{srname}_')]
                    if not conds:
                        continue
                    command = [FFMPEG, '-v', 'error', '-nostdin', '-threads', '1', '-i', str(media),
                               '-map', '0:a:0', '-vn']
                    if mix == 'meanall':
                        expression = '+'.join(f'{1/channels:.17g}*c{i}' for i in range(channels))
                        command += ['-af', f'pan=mono|c0={expression}']
                    else:
                        command += ['-ac', '1']
                    command += ['-ar', str(rate), '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
                    raw = subprocess.run(command, check=True, capture_output=True).stdout
                    wave = np.frombuffer(raw, dtype='<f4').copy()
                    if len(wave) == 0 or not np.isfinite(wave).all():
                        raise ValueError('Empty or nonfinite decoded waveform')
                    for cond in conds:
                        norm = cond.rsplit('_', 1)[1]
                        mu, sd = float(wave.astype('float64').mean()), float(wave.astype('float64').std())
                        signal = wave if norm == 'none' else ((wave.astype('float64')-mu)/max(sd,1e-12)).astype('float32')
                        wavpath = Path(temp) / f'{cond}.wav'
                        wavfile.write(wavpath, rate, signal)
                        arr, names = smile_features(wavpath)
                        arrays[cond] = arr
                        result['conditions'][cond] = {'status': 'ok', 'decode_command': command,
                            'sample_rate': rate, 'samples': int(len(signal)), 'duration_seconds': len(signal)/rate,
                            'decode_mean': mu, 'decode_std': sd, 'normalization': norm,
                            'post_mean': float(signal.astype('float64').mean()), 'post_std': float(signal.astype('float64').std()),
                            'post_peak_abs': float(np.abs(signal).max()), 'float32_wav_sha256': sha(wavpath),
                            'feature_sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
                            'finite': True, 'feature_width': len(arr)}
        result['field_names'] = names
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'
    return idx, arrays, result


def select_subset(rows, n=256, seed=67137):
    """Round-robin dialogues after seeded permutations, without labels or targets."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, row in enumerate(rows):
        dia = get_id(row, ('official_dialogue_id', 'raw_dialogue_id', 'dialogue_id', 'Dialogue_ID'))
        groups.setdefault(dia, []).append(i)
    keys = rng.permutation(sorted(groups))
    indices = []
    for key in keys:
        indices.append(int(rng.choice(groups[key])))
        if len(indices) == min(n, len(rows)):
            break
    if len(indices) != min(n, len(rows)):
        raise ValueError('Fewer dialogues than frozen probe sample count')
    return np.sort(np.array(indices, dtype=np.int64))


def validate_float_interface():
    """Three already selected Train clips: source amplitude and quantization control."""
    records = json.loads((ROOT/'candidates/smoke_records.json').read_text())
    source = CONF/'is09-13/IS13_ComParE.conf'
    external = CONF/'is09-13/IS13_ComParE_external_control.conf'
    external.write_text(source.read_text().replace('\\{../shared/standard_wave_input.conf.inc}',
                                                  '\\{\\cm[source{?}:source include config]}'))
    extractor = opensmile.Smile(str(external), opensmile.FeatureLevel.Functionals, loglevel=0)
    checks = []
    for r in records:
        command = r['conditions']['meanall_16k_none']['decode_command']
        signal = np.frombuffer(subprocess.run(command,check=True,capture_output=True).stdout,dtype='<f4').copy()
        assert np.abs(signal).max() < 1, 'External int16 control would wrap'
        quantized = (signal*32768).astype('int16').astype('float32')/32768
        y = extractor.process_signal(signal,16000).to_numpy()[0]
        with tempfile.TemporaryDirectory(prefix='meld_interface_control_') as temp:
            p = Path(temp)/'float.wav'
            wavfile.write(p,16000,signal)
            yy,names = smile_features(p)
            p2 = Path(temp)/'quantized_float.wav'
            wavfile.write(p2,16000,quantized)
            yq,namesq = smile_features(p2)
            assert names == list(extractor.feature_names) == namesq
        err = np.abs(yq.astype('float64')-y.astype('float64'))
        err_unquantized = np.abs(yy.astype('float64')-y.astype('float64'))
        checks.append({'row_index':r['row_index'],'dialogue_id':r['dialogue_id'],'utterance_id':r['utterance_id'],
            'source_channels':r['original_audio']['channels'],'source_samples':len(signal),
            'external_pcm16_vs_float_wav_of_same_quantized_samples': {'shape':list(y.shape),'max_abs_error':float(err.max()),
                'mae':float(err.mean()),'array_equal':bool(np.array_equal(y,yq))},
            'unquantized_float_wav_vs_external_pcm16':{'max_abs_error':float(err_unquantized.max()),'mae':float(err_unquantized.mean())},
            'decoded_float_sha256':hashlib.sha256(signal.tobytes()).hexdigest(),
            'float_wave_values_retained_without_clipping':True})
    savej(OUT/'provenance/meld_extraction_float_interface.json',{'checks':checks,
        'interpretation':'Same quantized input controls cWaveSource vs external source. Unquantized float WAV is the frozen candidate interface, not original production.',
        'candidate_interface':'float32 IEEE WAV -> official cWaveSource; zscore values outside [-1,1] retained, avoiding int16 wrap.'})


def main():
    script_sha256_at_start = sha(Path(__file__).resolve())
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT/'manifest_train.csv'))
    parser.add_argument('--stage', choices=['smoke', 'probe', 'train', 'validation'], required=True)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--interface-check-only', action='store_true')
    args = parser.parse_args()
    if args.interface_check_only:
        validate_float_interface()
        return
    if args.stage in ('train', 'validation') and not args.condition:
        parser.error('Train/Validation expansion requires one explicitly locked condition.')
    manifest = Path(args.manifest)
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if args.stage != 'validation' and any(r.get('split', 'train').lower() not in ('train', 'training') for r in rows):
        raise ValueError('Only Train manifests allowed for fitting/probe')
    if args.stage == 'validation' and any(r.get('split', '').lower() not in ('validation', 'val', 'dev') for r in rows):
        raise ValueError('Validation requires an explicit Validation-only manifest')
    prepare_config()
    base = ROOT/'candidates'
    base.mkdir(parents=True, exist_ok=True)
    conditions = [args.condition] if args.condition else CONDITIONS
    subset_file = base/'subset_indices.npy'
    if args.stage == 'validation':
        subset = np.load(subset_file)
    else:
        subset = select_subset(rows)
        if subset_file.exists():
            assert np.array_equal(np.load(subset_file), subset), 'Frozen subset changed'
        else:
            np.save(subset_file, subset)
    selected = subset[:3] if args.stage == 'smoke' else subset if args.stage == 'probe' else np.arange(len(rows))
    arrays = {c: np.full((len(selected),6373), np.nan, dtype='float32') for c in conditions}
    records = []
    start = time.time()
    positions = {int(idx): i for i, idx in enumerate(selected)}
    jobs = [(int(i), rows[int(i)], conditions) for i in selected]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers,6)) as ex:
        for k, (idx, output, record) in enumerate(ex.map(process_one, jobs), 1):
            if 'field_names' in record:
                fields = record.pop('field_names')
                assert len(fields) == 6373
                header = CONF/'feature_fields.json'
                if header.exists():
                    assert json.loads(header.read_text()) == fields
                else:
                    savej(header, fields)
            for c, vector in output.items():
                arrays[c][positions[idx]] = vector
            records.append(record)
            if k % 20 == 0 or k == len(jobs):
                print(json.dumps({'stage':args.stage,'completed':k,'total':len(jobs),'elapsed_s':round(time.time()-start,2)}), flush=True)
    outputs = []
    for c, values in arrays.items():
        folder = base/c
        folder.mkdir(exist_ok=True)
        path = folder/f'{args.stage}.npy'
        np.save(path, values)
        valid_path = folder/f'{args.stage}_valid.npy'
        np.save(valid_path, np.isfinite(values).all(axis=1))
        outputs.append({'condition':c,'path':str(path),'sha256':sha(path),'shape':list(values.shape),
                        'valid_mask_path':str(valid_path),'valid_mask_sha256':sha(valid_path),
                        'finite_rows':int(np.isfinite(values).all(axis=1).sum()),'rows_with_nonfinite':int((~np.isfinite(values).all(axis=1)).sum())})
    private = base/f'{args.stage}_records.json'
    savej(private,records)
    installed = Path(opensmile.__file__).parent
    libraries = sorted(installed.rglob('libSMILEapi.so'))
    provenance = {'stage':args.stage,'candidate_not_original':True,'manifest':str(manifest),'manifest_sha256':sha(manifest),
        'script_sha256_at_start':script_sha256_at_start,'command':__import__('sys').argv,'pid':os.getpid(),
        'subset_seed':67137,'subset_selection':'Sorted Dialogue IDs -> seeded permutation -> one seeded-choice row per group -> sorted full-Train row indices; no targets/labels',
        'selected_rows':len(selected),'conditions':conditions,'maximum_workers':min(args.workers,6),'elapsed_seconds':time.time()-start,
        'intensity_threshold':'none; original threshold unknown','waveform_zscore':'per utterance (waveform-mean)/population_std, no clipping; scipy float32 IEEE WAV',
        'sample_rate':'native48k or16k through ffmpeg libswresample defaults, no guessed original parameters',
        'feature_interface':'original official cWaveSource; only output sink external; NOT opensmile-python process_signal int16 cast',
        'opensmile_python_version':opensmile.__version__,'reported_engine_version':'3.0.1',
        'shared_library':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in libraries],
        'ffmpeg_sha256':sha(FFMPEG),'ffmpeg_version':subprocess.run([FFMPEG,'-version'],capture_output=True,text=True).stdout,
        'config_manifest_sha256':sha(CONF/'manifest.json'),'feature_fields_sha256':sha(CONF/'feature_fields.json') if (CONF/'feature_fields.json').exists() else None,
        'subset_indices_sha256':sha(subset_file),'outputs':outputs,'records':str(private),'records_sha256':sha(private),
        'failed_rows':sum(r['status']!='ok' for r in records),'failure_examples':[r for r in records if r['status']!='ok'][:3],
        'test_features_or_labels_read':False,'validation_targets_read':False,'cmERC_or_converter_training':False}
    savej(OUT/f'provenance/meld_extraction_{args.stage}.json',provenance)
    print(json.dumps({'status':'complete','outputs':outputs,'failed_rows':provenance['failed_rows']}),flush=True)


if __name__ == '__main__':
    main()
