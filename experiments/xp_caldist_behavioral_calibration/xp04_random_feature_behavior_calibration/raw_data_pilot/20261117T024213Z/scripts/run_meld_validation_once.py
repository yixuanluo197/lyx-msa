#!/usr/bin/env python3
"""Authorize the single frozen-candidate Validation extraction after Train model lock."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PROV = OUT/'provenance'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')


def main():
    lockpath = PROV/'MELD_model_lock.json'
    lock = json.loads(lockpath.read_text())
    assert lock['status']=='MODEL_LOCKED' and lock['candidate']=='ffmpegmono_16k_none'
    assert lock['transform']=='none' and lock['selected']['method']=='best_column_affine'
    assert lock['fit_effective_train_only'] and lock['train_dialogue_group_cv_only'] and not lock['test_used']
    assert not Path(lock['validation_features_path']).exists()
    assert not Path(lock['validation_valid_path']).exists()
    assert not (PROV/'meld_extraction_validation.json').exists()
    fixed = {str(lockpath):sha(lockpath)}
    for key,value in lock.items():
        if not key.endswith('_path'):
            continue
        hashkey = key[:-5]+'_sha256'
        if hashkey in lock:
            observed = sha(value)
            assert observed==lock[hashkey], f'Lock mismatch: {key}'
            fixed[value]=observed
    script = OUT/'scripts/extract_meld_candidates.py'
    train = json.loads((PROV/'meld_extraction_train.json').read_text())
    assert sha(script)==train['script_sha256_at_start'], 'DSP code changed after Train'
    fixed[str(script)] = sha(script)
    fixed[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    for path in sorted((OUT/'configs/meld').rglob('*')):
        if path.is_file():
            fixed[str(path)]=sha(path)
    manifest = Path(lock['manifest_path'])
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==1109 and all(r['split'].lower() in ('val','dev','validation') for r in rows)
    command = [sys.executable,str(script),'--stage','validation','--manifest',str(manifest),
               '--condition','ffmpegmono_16k_none','--workers','6']
    launch = {'status':'FROZEN_BEFORE_VALIDATION_EXTRACTION','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command':command,'wrapper_pid':os.getpid(),'fixed_input_hashes':fixed,'validation_rows':1109,
        'validation_targets_read':False,'test_read':False,'no_preexisting_validation_outputs':True,
        'frozen_train_method':'ffmpegmono_16k_none/none/best_column_affine',
        'missing_media_policy':'Keep original row as NaN/false, no cross-split replacement'}
    write_exclusive(PROV/'meld_extraction_validation_launch.json',launch)
    with (PROV/'meld_extraction_validation_stdout.log').open('x') as stdout, (PROV/'meld_extraction_validation_stderr.log').open('x') as stderr:
        process = subprocess.Popen(command,stdout=stdout,stderr=stderr)
        write_exclusive(PROV/'meld_extraction_validation_process.json',{'pid':process.pid,'command':command,'parent_pid':os.getpid()})
        status = process.wait()
    assert status==0, f'Extraction exited {status}'
    checks = {'all_locked_inputs_unchanged':all(sha(p)==h for p,h in fixed.items())}
    array = np.load(lock['validation_features_path'],mmap_mode='r')
    valid = np.load(lock['validation_valid_path'])
    bad = np.flatnonzero(~valid)
    checks.update({'shape1109x6373':array.shape==(1109,6373),'valid_mask_shape_and_dtype':valid.shape==(1109,) and valid.dtype==bool,
        'mask_matches_finite':bool(np.array_equal(valid,np.isfinite(array).all(axis=1))),
        'valid_rows1108':int(valid.sum())==1108,'invalid_rows_all_nan':bool(np.isnan(array[bad]).all()),
        'only_expected_missing_dev110_utt7':len(bad)==1 and rows[int(bad[0])]['raw_dialogue_id']=='110' and rows[int(bad[0])]['raw_utterance_id']=='7'})
    result = {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'invalid_full_row_indices':bad.tolist(),
        'feature_sha256':sha(lock['validation_features_path']),'valid_mask_sha256':sha(lock['validation_valid_path']),
        'launch_record_sha256':sha(PROV/'meld_extraction_validation_launch.json'),
        'frozen_model_lock_sha256':fixed[str(lockpath)],'validation_targets_read':False,'test_read':False,
        'converter_or_model_fitted':False,'cmERC_forward':False,'single_locked_candidate_only':True}
    write_exclusive(PROV/'meld_extraction_validation_verification.json',result)
    print(json.dumps(result),flush=True)
    assert result['status']=='PASS'


if __name__=='__main__':
    main()
