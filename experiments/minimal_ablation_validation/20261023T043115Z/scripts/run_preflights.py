"""Four isolated first-step probes; never use held-out test predictions."""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from supervise import ROOT, digest, now, resource_snapshot, validate_capsule, write_json


def main():
    config = json.loads((ROOT/'configs/registry.json').read_text())
    env = os.environ.copy()
    for key in ['PYTHONHASHSEED','CMERC_GRADIENT_PROBE','CMERC_STOP_AFTER_FIRST_BEST']:
        env.pop(key, None)
    env.update(config['environment'])
    env['PATH'] = str(Path(config['python']).parent) + ':' + env.get('PATH','')
    jobs = [(config['runs'][0],True)] + [(run,False) for run in config['runs']]
    records = []
    for run, full_cc in jobs:
        validate_capsule()
        suffix = run['run_id'] + ('_full_cc' if full_cc else '')
        while True:
            resource = resource_snapshot(env['CUDA_VISIBLE_DEVICES'])
            if not resource['processes'] and resource['free_mib'] >= 11500 and resource['utilization_percent'] <= 10:
                break
            write_json(ROOT/'state/status.json', {'active':True,'phase':'preflight_waiting_gpu',
                'formal_started':0,'formal_completed':0,'waiting_probe':suffix,'resource':resource})
            time.sleep(10)
        env['CMERC_PROFILE_OUTPUT'] = str(ROOT/'logs'/('preflight_' + suffix + '_memory.jsonl'))
        argv = [config['python'],'-u',str(ROOT/'scripts/memory_observer.py'),
                str(ROOT/'scripts/observe_training.py'),'--run-id',run['run_id'],'--preflight']
        if full_cc:
            argv.append('--full-cc-control')
        row = {'probe':suffix,'argv':argv,'environment':{k:env[k] for k in config['environment']},
            'started_at_utc':now(),'admission':resource,'formal_training':False}
        log = ROOT/'logs'/('preflight_' + suffix + '.log')
        with log.open('x') as output:
            child = subprocess.Popen(argv,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            write_json(ROOT/'state/status.json', {'active':True,'phase':'preflight_running','formal_started':0,
                'formal_completed':0,'probe':suffix,'pid':child.pid})
            row['returncode'] = child.wait()
        row['finished_at_utc'] = now()
        records.append(row)
        write_json(ROOT/'provenance/preflight_invocations.json',records)
        if row['returncode']:
            raise RuntimeError('Preflight failed without retry: ' + suffix)
        status = json.loads((ROOT/'preflight'/suffix/'observer/status.json').read_text())
        assert status['phase'] == 'preflight_passed'
    base = ROOT/'preflight'
    off = base/'MELD_wo_cc_seed67137/observer'
    on = base/'MELD_wo_cc_seed67137_full_cc/observer'
    left = json.loads((on/'first_batch.json').read_text())
    right = json.loads((off/'first_batch.json').read_text())
    for key in ['initial_model_sha256','rng_before_train','rng_after_forward','rng_after_step','dialogue_ids','lengths']:
        assert left[key] == right[key], key
    for key in ['classification_loss','cc_loss','mscl_loss','cscl_loss']:
        assert np.isclose(left['components'][key],right['components'][key],rtol=1e-5,atol=1e-6),key
    a = np.load(on/'first_batch.npz')
    b = np.load(off/'first_batch.npz')
    assert set(a.files) == set(b.files)
    for key in a.files:
        if key == 'labels':
            assert np.array_equal(a[key],b[key])
        else:
            assert np.allclose(a[key],b[key],rtol=1e-5,atol=1e-6),key
    for obj, cc_on in [(left,True),(right,False)]:
        v=obj['components']
        expected=v['classification_loss']+.15*(v['mscl_loss']+v['cscl_loss'])
        if cc_on:
            expected += .002*v['cc_loss']
        assert np.isclose(expected,v['total_loss'],rtol=1e-5,atol=1e-6)
    gate = {'status':'passed','completed_at_utc':now(),'formal_runs_started':0,'gpu_probes':len(records),
        'first_step_all_finite_and_parameters_updated':True,'no_test_forward_in_preflight':True,
        'MELD_full_vs_wo_CC_first_batch_same_initialization_forward_targets_and_rng':True,
        'MELD_only_weighted_CC_removed_from_loss':True,
        'source_order_cpu_evidence_sha256':digest(ROOT/'preflight/source_order_cpu.json'),
        'observer_unit_tests':'three CPU tests passed before GPU probes',
        'limits':['first-batch check does not prove five-seed efficacy','historical control uses another physical card']}
    write_json(ROOT/'preflight/gate.json',gate)
    write_json(ROOT/'state/status.json',{'active':True,'phase':'preflight_passed','formal_started':0,'formal_completed':0})
    print(json.dumps(gate),flush=True)


if __name__=='__main__':
    main()
