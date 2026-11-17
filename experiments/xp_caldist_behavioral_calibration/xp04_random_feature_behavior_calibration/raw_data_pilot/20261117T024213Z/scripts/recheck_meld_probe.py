#!/usr/bin/env python3
"""Independently repeat the fixed Train probe with code/config frozen before launch.

This repairs reproducibility evidence; it does not invent the old launch code hash.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

OUT = Path(__file__).resolve().parents[1]
AREA = OUT/'local_only/MELD/probe_recheck'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    if AREA.exists():
        raise RuntimeError('Independent recheck directory exists: refusing overwrite')
    (AREA/'frozen_code').mkdir(parents=True)
    (AREA/'provenance').mkdir()
    original = OUT/'scripts/extract_meld_candidates.py'
    frozen = AREA/'frozen_code/extract_meld_candidates.py'
    shutil.copy2(original,frozen)
    runner_copy = AREA/'frozen_code/recheck_meld_probe.py'
    shutil.copy2(Path(__file__),runner_copy)
    spec = importlib.util.spec_from_file_location('frozen_meld_extractor',frozen)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = AREA
    module.ROOT = AREA
    module.CONF = AREA/'configs/meld'
    module.PRIOR = OUT.parent/'20261117T014409Z'
    module.prepare_config()
    files = [p for p in sorted(module.CONF.rglob('*')) if p.is_file()]
    launch = {'pid':os.getpid(),'timestamp_unix':time.time(),'original_probe_code_hash':'not recorded at original launch; not reconstructed',
        'repair':'Independent fixed-input same-condition repeat from this frozen source/config snapshot',
        'frozen_extractor_sha256':sha(frozen),'runner_sha256':sha(Path(__file__)),'frozen_runner_sha256':sha(runner_copy),
        'train_manifest_sha256':sha(OUT/'local_only/MELD/manifest_train.csv'),
        'original_subset_sha256':sha(OUT/'local_only/MELD/candidates/subset_indices.npy'),
        'config_files':{str(p.relative_to(module.CONF)):sha(p) for p in files},
        'test_or_validation_read':False,'data_or_condition_expansion':False}
    write(AREA/'provenance/frozen_launch.json',launch)
    sys.argv = [str(frozen),'--stage','probe','--manifest',str(OUT/'local_only/MELD/manifest_train.csv'),'--workers','6']
    module.main()
    original_root = OUT/'local_only/MELD/candidates'
    new_root = AREA/'candidates'
    comparisons = []
    subset_same = bool(np.array_equal(np.load(original_root/'subset_indices.npy'),np.load(new_root/'subset_indices.npy')))
    for cond in module.CONDITIONS:
        a,b = original_root/cond/'probe.npy', new_root/cond/'probe.npy'
        aa,bb = np.load(a), np.load(b)
        comparisons.append({'condition':cond,'original_sha256':sha(a),'recheck_sha256':sha(b),
            'shape':list(bb.shape),'byte_hash_equal':sha(a)==sha(b),'array_bitwise_equal':bool(np.array_equal(aa.view('uint32'),bb.view('uint32'))),
            'max_abs_error':float(np.max(np.abs(aa.astype('float64')-bb.astype('float64'))))})
    config_unchanged = all(sha(module.CONF/rel)==h for rel,h in launch['config_files'].items())
    result = {'status':'PASS' if subset_same and config_unchanged and all(c['byte_hash_equal'] and c['array_bitwise_equal'] for c in comparisons) else 'FAIL',
        'original_launch_hash_gap_retained':True,'new_run_frozen_source_and_config_before_execution':True,
        'subset_indices_equal':subset_same,'frozen_config_files_unchanged':config_unchanged,'comparisons':comparisons,
        'frozen_launch_sha256':sha(AREA/'provenance/frozen_launch.json'),'original_extractor_current_sha256':sha(original),
        'scope':'Same256Train utterances and8frozen candidates only; noValidation/Test/targets/training'}
    write(OUT/'provenance/meld_extraction_probe_recheck.json',result)
    print(json.dumps(result),flush=True)
    if result['status']!='PASS':
        raise RuntimeError('Deterministic recheck failed; stop before Validation')


if __name__=='__main__':
    main()
